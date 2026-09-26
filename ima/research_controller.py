"""Single-owner feedback controller for agentic research campaigns."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .openrouter_orchestrator import (
    OpenRouterConfig,
    OpenRouterError,
    choose_research_proposals,
    normalize_openrouter_usage,
)
from .mlflow_tracking import (
    MLflowConfig,
    log_optimizer_cycle_trace,
    log_research_package_version,
)
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_evaluation import build_expanding_folds
from .research_resources import admission_slots, observe_resources, resource_report
from .research_search import ProgramSearchController, RecipeSuggestion
from .research_specs import PipelineRecipe, ResearchProposal
from .research_store import ResearchLedger, utc_now
from .feature_sets import FEATURE_SCHEMAS, drop_feature_families


class PlannerSpendCapReached(RuntimeError):
    def __init__(self, spent: float, cap: float) -> None:
        self.spent = spent
        self.cap = cap
        super().__init__(f"OpenRouter planner spend cap reached: ${spent:.6f} of ${cap:.6f}")


class DatasetFeatureProfile:
    """Check transform inputs against the frozen development training folds."""

    def __init__(self, dataset_path: Path, protocol: dict[str, Any]) -> None:
        numeric = set().union(*(schema.numeric for schema in FEATURE_SCHEMAS.values()))
        required = {"race_id", "race_no", "date"}
        if dataset_path.suffix == ".parquet":
            frame = pd.read_parquet(dataset_path)
            frame = frame[[column for column in frame if column in numeric | required]]
        else:
            frame = pd.read_csv(
                dataset_path, usecols=lambda column: column in numeric | required,
                low_memory=False,
            )
        self.present_columns = set(frame.columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        present_numeric = sorted(numeric & self.present_columns)
        values = frame[present_numeric].apply(pd.to_numeric, errors="coerce")
        folds = build_expanding_folds(frame, **protocol)
        self.unavailable: dict[str, set[str]] = {}
        for window in ("all_history", "trailing_3_years"):
            unavailable = set(numeric - self.present_columns)
            for fold in folds:
                train = frame["race_id"].astype(str).isin(fold.train_race_ids)
                if window == "trailing_3_years":
                    calibration = frame["race_id"].astype(str).isin(fold.calibration_race_ids)
                    cutoff = frame.loc[calibration, "date"].min() - pd.DateOffset(years=3)
                    train &= frame["date"].ge(cutoff)
                if not train.any():
                    unavailable.update(present_numeric)
                    continue
                unavailable.update(values.columns[~values.loc[train].notna().any()].tolist())
            self.unavailable[window] = unavailable
        self.summary = {
            "numeric_feature_count": len(numeric),
            "present_numeric_count": len(present_numeric),
            "unavailable_transform_columns": sorted(self.unavailable["all_history"]),
            "unavailable_transform_columns_by_window": {
                window: sorted(columns) for window, columns in self.unavailable.items()
            },
            "unavailable_transform_column_count": len(self.unavailable["all_history"]),
            "rule": "Transforms need numeric observations in every protected training fold.",
        }

    def admission_error(self, recipe: PipelineRecipe) -> str | None:
        schema = drop_feature_families(
            FEATURE_SCHEMAS[recipe.feature_schema], recipe.drop_feature_families
        )
        for transform in recipe.transforms:
            for column in transform.parameters["columns"]:
                if column not in schema.numeric:
                    return f"{column} is not in the effective numeric feature schema"
                if column in self.unavailable[recipe.train_window]:
                    return f"{column} has no numeric observations in a training fold"
        return None


def run_research_campaign(config: Any) -> dict[str, Any]:
    config.validate()
    if config.protocol_path is None:
        raise ValueError(
            "Executable agentic campaigns require an explicit protocol_path"
        )
    campaign_dir = Path(config.campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(campaign_dir):
        dataset_path = _resolve_dataset(config, campaign_dir)
        dataset_hash = _hash_file(dataset_path)
        protocol_parameters = _protocol_parameters(config)
        code_revision = _code_revision()
        environment_hash = _environment_hash()
        _validate_campaign_identity(
            campaign_dir,
            dataset_hash=dataset_hash,
            protocol_parameters=protocol_parameters,
            code_revision=code_revision,
            environment_hash=environment_hash,
        )
        ledger = ResearchLedger(campaign_dir / "ledger.sqlite")
        search = ProgramSearchController(campaign_dir)
        profile = DatasetFeatureProfile(dataset_path, protocol_parameters)
        ledger.recover_running()
        _reconcile_tells(ledger, search)
        tracking_errors = _reconcile_tracking(config, ledger)
        cycles = 0
        cycle = _next_cycle_number(campaign_dir)
        new_results: list[dict[str, Any]] = []
        while config.max_trials is None or _success_count(ledger) < config.max_trials:
            if (campaign_dir / "STOP").exists():
                (campaign_dir / "STOP").unlink()
                return _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results, "stopped",
                    tracking_errors,
                )
            consecutive_failures = _consecutive_failure_count(ledger)
            if consecutive_failures >= config.max_consecutive_failed_trials:
                return _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results,
                    "blocked_failures", tracking_errors,
                )
            remaining = None if config.max_trials is None else config.max_trials - _success_count(ledger)
            batch_size = config.proposal_batch_size if remaining is None else min(
                config.proposal_batch_size, remaining
            )
            batch_size = min(
                batch_size,
                config.max_consecutive_failed_trials - consecutive_failures,
            )
            slots, resources = _slots(config, campaign_dir, batch_size)
            if slots == 0:
                payload = _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results, "paused_admission",
                    tracking_errors,
                )
                payload["resources"] = resources
                _write_json_atomic(campaign_dir / "status.json", payload)
                return payload
            try:
                suggestions, decision = _next_suggestions(
                    config, campaign_dir, ledger, search, profile, batch_size, cycle
                )
            except PlannerSpendCapReached as exc:
                payload = _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results,
                    "paused_spend", tracking_errors,
                )
                payload["planner_spend_usd"] = exc.spent
                payload["planner_spend_cap_usd"] = exc.cap
                _write_json_atomic(campaign_dir / "status.json", payload)
                return payload
            if not suggestions:
                return _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results, "paused",
                    tracking_errors,
                )
            _write_json_atomic(campaign_dir / "status.json", {
                "status": "training" if slots else "paused_admission",
                "updated_at": utc_now(),
                "cycles": cycles,
                "ledger": ledger.snapshot(),
                "search": search.snapshot(),
                "resources": resources,
                "latest_decision": decision,
            })
            requests: list[RecipeExecutionRequest] = []
            suggestions_by_attempt: dict[str, RecipeSuggestion] = {}
            for suggestion in suggestions[:batch_size]:
                signature = _execution_signature(
                    suggestion.recipe, dataset_hash, protocol_parameters,
                    code_revision, environment_hash,
                )
                payload = suggestion.serializable() | {
                    "signature": signature,
                    "dataset_hash": dataset_hash,
                    "protocol_parameters": protocol_parameters,
                    "code_revision": code_revision,
                    "environment_hash": environment_hash,
                }
                attempt = ledger.reserve_attempt(signature, payload)
                if attempt.status == "completed":
                    continue
                ledger.mark_running(attempt.attempt_id)
                suggestions_by_attempt[attempt.attempt_id] = suggestion
                requests.append(RecipeExecutionRequest(
                    attempt_id=attempt.attempt_id,
                    proposal_id=suggestion.proposal_id or suggestion.trial_id,
                    trial_number=suggestion.trial_number,
                    recipe=suggestion.recipe,
                    dataset_path=dataset_path,
                    output_dir=campaign_dir / "trials" / attempt.attempt_id,
                    protocol_parameters=protocol_parameters,
                    dataset_hash=dataset_hash,
                    code_revision=code_revision,
                    environment_hash=environment_hash,
                ))
            if not requests:
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger))
                cycles += 1
                cycle += 1
                continue
            cycle_results: list[dict[str, Any]] = []
            completed = _execute_requests(requests, slots)
            for result in completed:
                row = result.serializable()
                lineage = suggestions_by_attempt[result.attempt_id]
                row["trial_id"] = lineage.trial_id
                row["program_id"] = lineage.program_id
                row["target_parameters"] = lineage.recipe.target.model_dump(mode="json")["parameters"]
                cycle_results.append(row)
                ledger.complete_attempt(result.attempt_id, row, status=result.status)
                _append_jsonl(campaign_dir / "trials.jsonl", row)
                new_results.append(row)
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger))
            trace_error = _trace_completed_cycle(
                config, campaign_dir, decision, cycle_results
            )
            if trace_error:
                tracking_errors.append(trace_error)
            cycles += 1
            cycle += 1
        return _finish_payload(
            campaign_dir, ledger, search, cycles, new_results, "complete",
            tracking_errors,
        )


def campaign_status(campaign_dir: Path) -> dict[str, Any]:
    ledger_path = campaign_dir / "ledger.sqlite"
    status = _read_json(campaign_dir / "status.json")
    if ledger_path.exists():
        status["ledger"] = ResearchLedger(ledger_path).snapshot()
    status["controller_online"] = _controller_lock_held(campaign_dir / "controller.lock")
    status["controller_lock"] = _read_lock_metadata(campaign_dir / "controller.lock")
    status["campaign_dir"] = str(campaign_dir)
    return status


def request_campaign_stop(campaign_dir: Path) -> Path:
    if not campaign_dir.exists():
        raise FileNotFoundError(campaign_dir)
    marker = campaign_dir / "STOP"
    marker.write_text(utc_now() + "\n", encoding="utf-8")
    return marker


def _next_suggestions(
    config: Any,
    campaign_dir: Path,
    ledger: ResearchLedger,
    search: ProgramSearchController,
    profile: "DatasetFeatureProfile",
    count: int,
    cycle: int,
) -> tuple[list[RecipeSuggestion], dict[str, Any]]:
    terminal = ledger.terminal_results()
    successes = [row for row in terminal if row["status"] == "completed"]
    if not successes:
        search.bootstrap(count)
        suggestions = search.ask(count)
        decision = {
            "cycle": cycle,
            "source": "bootstrap",
            "evidence_id": None,
            "suggestions": [item.serializable() for item in suggestions],
        }
        _persist_decision(campaign_dir, cycle, decision)
        return suggestions, decision
    evidence = _build_evidence(terminal, search, profile)
    evidence_path = campaign_dir / "evidence" / f"cycle-{cycle:04d}.json"
    _write_json_atomic(evidence_path, evidence)
    if config.planner_mode in {"fixture", "openrouter"} and search.has_capacity() and not _planner_due(
        campaign_dir, len(successes), config.replan_every_terminal_trials
    ):
        suggestions = search.ask(count)
        decision = {
            "cycle": cycle,
            "source": "approved_space_optuna",
            "evidence_id": evidence["evidence_id"],
            "completed_trial_count": len(successes),
            "evidence_trial_ids": [row["attempt_id"] for row in successes[-32:]],
            "suggestions": [item.serializable() for item in suggestions],
        }
        _persist_decision(campaign_dir, cycle, decision)
        return suggestions, decision
    if config.planner_mode == "openrouter":
        spent = _planner_spend(campaign_dir)
        if spent >= float(config.max_total_cost_usd):
            raise PlannerSpendCapReached(spent, float(config.max_total_cost_usd))
        remote = OpenRouterConfig.from_env(
            model=config.model,
            service_tier=config.service_tier,
            max_output_tokens=config.max_output_tokens,
            timeout_seconds=config.planner_timeout_seconds,
            provider_endpoint=config.provider_endpoint,
            reasoning_effort=config.planner_reasoning_effort,
        )
        _write_json_atomic(campaign_dir / "status.json", {
            "status": "provider_planning",
            "updated_at": utc_now(),
            "cycle": cycle,
            "ledger": ledger.snapshot(),
            "search": search.snapshot(),
            "evidence_id": evidence["evidence_id"],
            "planner_model": config.model,
            "planner_spend_usd": spent,
        })
        try:
            response = choose_research_proposals(evidence, min(count, 4), remote)
        except OpenRouterError as exc:
            if search.remaining_capacity() < count:
                for fallback in _fixture_proposals(evidence, min(count, 2), cycle):
                    if profile.admission_error(fallback.recipe):
                        continue
                    search.register(fallback)
                    if search.remaining_capacity() >= count:
                        break
            suggestions = search.ask(count)
            decision = {
                "cycle": cycle,
                "source": "degraded_local",
                "planner_error": str(exc),
                "evidence_id": evidence["evidence_id"],
                "completed_trial_count": len(successes),
                "evidence_trial_ids": [row["attempt_id"] for row in successes[-32:]],
                "suggestions": [item.serializable() for item in suggestions],
            }
            _persist_decision(campaign_dir, cycle, decision)
            return suggestions, decision
        proposals = [ResearchProposal.model_validate(row) for row in response["proposals"]]
        _write_json_atomic(campaign_dir / "planner" / f"cycle-{cycle:04d}.json", {
            "evidence_id": evidence["evidence_id"],
            "requested_model": config.model,
            "response": response,
        })
        source = "openrouter"
    elif config.planner_mode == "fixture":
        proposals = _fixture_proposals(evidence, min(count, 2), cycle)
        source = "fixture"
    else:
        proposals = _fixture_proposals(evidence, min(count, 2), cycle)
        source = "local_adaptive"
    completed_ids = {row["attempt_id"] for row in successes}
    program_ids = []
    rejected_proposals = list(response.get("rejected_proposals", [])) if source == "openrouter" else []
    known_recipes = set(search.snapshot()["recipe_hashes"])
    for proposal in proposals:
        if proposal.evidence_ids != (evidence["evidence_id"],):
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "stale_evidence_id",
            })
            continue
        if not proposal.parent_trial_ids:
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "missing_parent_trials",
            })
            continue
        unknown_parents = sorted(set(proposal.parent_trial_ids) - completed_ids)
        if unknown_parents:
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "unknown_parent_trials",
                "unknown_parent_trial_ids": unknown_parents,
            })
            continue
        if not proposal.search_space and proposal.recipe.recipe_hash() in known_recipes:
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "duplicate_recipe",
            })
            continue
        admission_error = profile.admission_error(proposal.recipe)
        if admission_error:
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "unavailable_transform_input",
                "detail": admission_error,
            })
            continue
        try:
            program_id = search.register(proposal)
        except ValueError as exc:
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": str(exc),
            })
            continue
        program_ids.append(program_id)
    if search.remaining_capacity() < count:
        for fallback in _fixture_proposals(evidence, min(count, 2), cycle):
            if profile.admission_error(fallback.recipe):
                continue
            search.register(fallback)
            if search.remaining_capacity() >= count:
                break
    suggestions = search.ask(count, preferred_program_ids=program_ids)
    local_refill = [item.trial_id for item in suggestions if item.program_id not in program_ids]
    decision = {
        "cycle": cycle,
        "source": source,
        "evidence_id": evidence["evidence_id"],
        "completed_trial_count": len(successes),
        "evidence_trial_ids": [row["attempt_id"] for row in successes[-32:]],
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "rejected_proposals": rejected_proposals,
        "approved_program_ids": program_ids,
        "local_refill_trial_ids": local_refill,
        "suggestions": [item.serializable() for item in suggestions],
    }
    if source == "openrouter":
        decision.update({
            "planner_model": config.model,
            "service_tier": response.get("service_tier") or config.service_tier,
            "planner_usage": response.get("usage") or normalize_openrouter_usage(
                response.get("raw_response", {})
            ),
        })
    _persist_decision(campaign_dir, cycle, decision)
    return suggestions, decision


def _fixture_proposals(
    evidence: dict[str, Any], count: int, cycle: int
) -> list[ResearchProposal]:
    completed = evidence["completed_trials"]
    parents = tuple(row["attempt_id"] for row in completed[-min(3, len(completed)):])
    winners = [
        row for row in completed if row.get("target_kind", "win_probability") == "win_probability"
    ]
    best = min(float(row["objective_value"]) for row in winners or completed)
    improving = any(
        row.get("mean_selected_minus_market") is not None
        and float(row["mean_selected_minus_market"]) <= 0
        for row in winners
    )
    proposals: list[ResearchProposal] = []
    for index in range(count):
        if improving:
            recipe = PipelineRecipe(
                feature_schema="benter-rich-v1",
                model={"kind": "logit", "parameters": {
                    "C": round(
                        0.03 * (index + 1) * max(best, 0.1) * (1 + cycle * 0.01),
                        6,
                    ),
                    "class_weight": None,
                    "max_iter": 1200,
                }},
            )
            axes = ("feature_schema", "hyperparameters")
            hypothesis = "Retain the current evidence-backed direction with a richer schema."
        elif "horse_rating" not in evidence.get("feature_profile", {}).get(
            "unavailable_transform_columns", []
        ):
            recipe = PipelineRecipe(
                transforms=({
                    "kind": "race_relative_rank",
                    "parameters": {"columns": ["horse_rating"]},
                },),
                model={"kind": "boosted", "parameters": {
                    "learning_rate": round(0.025 + index * 0.01 + cycle * 0.0001, 4),
                    "max_iter": 100 + index * 20,
                    "max_leaf_nodes": 15,
                }},
            )
            axes = ("transform", "model_family", "hyperparameters")
            hypothesis = "Revise the current direction with race-relative structure."
        else:
            recipe = PipelineRecipe(
                feature_schema="benter-rich-v1",
                model={"kind": "boosted", "parameters": {
                    "learning_rate": round(0.025 + index * 0.01 + cycle * 0.0001, 4),
                    "max_iter": 100 + index * 20,
                    "max_leaf_nodes": 15,
                }},
            )
            axes = ("feature_schema", "model_family", "hyperparameters")
            hypothesis = "Revise the current direction with boosted rich features."
        proposals.append(ResearchProposal(
            proposal_id=f"fixture-cycle-{cycle:04d}-{index:02d}",
            parent_trial_ids=parents,
            evidence_ids=(evidence["evidence_id"],),
            hypothesis=hypothesis,
            changed_axes=axes,
            recipe=recipe,
            expected_observation="Development objective changes against the cited parents.",
            falsification_rule="Reject when the comparable development objective does not improve.",
            search_space={
                "C": {"kind": "float", "low": 0.01, "high": 2.0, "log": True}
            } if recipe.model.kind == "logit" else {
                "learning_rate": {"kind": "float", "low": 0.02, "high": 0.1, "log": True}
            },
            max_trials=3,
        ))
    return proposals


def _build_evidence(
    terminal: list[dict[str, Any]],
    search: ProgramSearchController,
    profile: DatasetFeatureProfile,
) -> dict[str, Any]:
    successes = [row for row in terminal if row["status"] == "completed"]
    rows = []
    for row in successes[-32:]:
        result = row["result"]
        recipe = row["payload"]["recipe"]
        rows.append({
            "attempt_id": row["attempt_id"],
            "program_id": row["payload"].get("program_id"),
            "recipe_hash": result["recipe_hash"],
            "target_kind": result["target_kind"],
            "target_parameters": recipe["target"]["parameters"],
            "objective_name": result["objective_name"],
            "objective_value": result["objective_value"],
            "feature_schema": recipe["feature_schema"],
            "model_kind": recipe["model"]["kind"],
            "model_parameters": recipe["model"]["parameters"],
            "train_window": recipe["train_window"],
            "drop_feature_families": recipe["drop_feature_families"],
            "transforms": recipe["transforms"],
            "calibration_kind": recipe["calibration"]["kind"],
            "blend_kind": recipe["blend"]["kind"],
            "mean_selected_minus_market": result.get("metrics", {}).get(
                "mean_selected_minus_market"
            ),
        })
    best_by_target: dict[str, list[dict[str, Any]]] = {}
    target_keys = {
        (row["result"]["target_kind"], json.dumps(
            row["payload"]["recipe"]["target"]["parameters"], sort_keys=True
        )) for row in successes
    }
    for target_kind, parameters in sorted(target_keys):
        matching = sorted(
            (row for row in successes if row["result"]["target_kind"] == target_kind
             and json.dumps(row["payload"]["recipe"]["target"]["parameters"], sort_keys=True)
             == parameters),
            key=lambda row: float(row["result"]["objective_value"]),
        )[:3]
        key = target_kind if parameters == "{}" else f"{target_kind}:{parameters}"
        best_by_target[key] = [{
            "attempt_id": row["attempt_id"],
            "program_id": row["payload"].get("program_id"),
            "objective_name": row["result"]["objective_name"],
            "objective_value": row["result"]["objective_value"],
            "recipe": row["payload"]["recipe"],
        } for row in matching]
    by_id = {row["attempt_id"]: row for row in successes}
    program_outcomes = []
    for program_id, proposal in search.programs.items():
        children = [
            row for row in successes if row["payload"].get("program_id") == program_id
        ]
        parent_rows = [by_id[parent] for parent in proposal.parent_trial_ids if parent in by_id]
        comparable_parents = [
            row for row in parent_rows
            if row["result"]["target_kind"] == proposal.recipe.target.kind
            and row["payload"]["recipe"]["target"]["parameters"]
            == proposal.recipe.canonical_payload()["target"]["parameters"]
        ]
        child_best = min(
            (float(row["result"]["objective_value"]) for row in children), default=None
        )
        parent_best = min(
            (float(row["result"]["objective_value"]) for row in comparable_parents), default=None
        )
        used = len(search.studies[program_id].trials)
        verdict = "inconclusive"
        if child_best is not None and parent_best is not None:
            verdict = "keep" if child_best < parent_best else (
                "reject" if used >= proposal.max_trials else "revise"
            )
        program_outcomes.append({
            "program_id": program_id,
            "proposal_id": proposal.proposal_id,
            "target_kind": proposal.recipe.target.kind,
            "objective_name": children[0]["result"]["objective_name"] if children else None,
            "hypothesis": proposal.hypothesis,
            "changed_axes": proposal.changed_axes,
            "parent_trial_ids": proposal.parent_trial_ids,
            "budget": proposal.max_trials,
            "used": used,
            "completed": len(children),
            "parent_best": parent_best,
            "program_best": child_best,
            "development_direction": verdict,
        })
    failures = [{
        "attempt_id": row["attempt_id"],
        "program_id": row["payload"].get("program_id"),
        "error": str(row["result"].get("error", ""))[:300],
    } for row in terminal if row["status"] == "failed"][-10:]
    payload = {
        "schema_version": 2,
        "completed_trials": rows,
        "best_by_target": best_by_target,
        "program_outcomes": program_outcomes[-20:],
        "recent_failures": failures,
        "feature_profile": profile.summary,
        "notes": ["Development evidence only; holdout metrics are excluded."],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {"evidence_id": digest, **payload}


def _execute_requests(
    requests: list[RecipeExecutionRequest], slots: int
) -> Iterator[Any]:
    if slots <= 1 or len(requests) == 1:
        for request in requests:
            yield execute_recipe(request)
        return
    with ProcessPoolExecutor(max_workers=min(slots, len(requests))) as pool:
        futures = {pool.submit(execute_recipe, request): request for request in requests}
        for future in as_completed(futures):
            yield future.result()


def _reconcile_tells(ledger: ResearchLedger, search: ProgramSearchController) -> None:
    for effect in ledger.pending_tells():
        trial_number = int(effect["payload"]["trial_number"])
        program_id = str(effect["payload"]["program_id"])
        state = search.trial_state(program_id, trial_number)
        result = effect["result"]
        if state == "running":
            if result.get("status") == "completed":
                search.tell(program_id, trial_number, float(result["objective_value"]), result.get("metrics"))
            else:
                search.tell_failed(program_id, trial_number)
        elif state not in {"complete", "fail", "pruned"}:
            raise RuntimeError(f"Unexpected Optuna state during reconciliation: {state}")
        ledger.mark_told(effect["attempt_id"])


def _reconcile_tracking(config: Any, ledger: ResearchLedger) -> list[str]:
    if not config.mlflow_tracking_uri:
        return []
    errors: list[str] = []
    tracking = MLflowConfig.from_values(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name="ima-agentic-v2",
        register_models=True,
        registered_model_name="ima-agentic-candidates",
    )
    for item in ledger.pending_outbox():
        result = item["result"]
        package = result.get("artifacts", {}).get("package")
        if not package:
            continue
        try:
            linkage = log_research_package_version(
                Path(package), tracking,
                attempt_id=item["attempt_id"],
                result=result,
            )
            if linkage is not None:
                ledger.mark_uploaded(item["attempt_id"], json.dumps(linkage, sort_keys=True))
        except Exception as exc:  # tracking retries on the next controller pass.
            errors.append(f"{item['attempt_id']}: {type(exc).__name__}: {exc}")
    return errors


def _trace_completed_cycle(
    config: Any,
    campaign_dir: Path,
    decision: dict[str, Any],
    results: list[dict[str, Any]],
) -> str | None:
    if not config.mlflow_tracking_uri:
        return None
    tracking = MLflowConfig.from_values(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name="ima-agentic-v2",
        register_models=True,
        registered_model_name="ima-agentic-candidates",
    )
    try:
        log_optimizer_cycle_trace(campaign_dir, decision, results, tracking)
    except Exception as exc:  # tracing is retriable and must not stop training.
        return f"cycle-{decision.get('cycle')}: trace: {type(exc).__name__}: {exc}"
    return None


def _slots(config: Any, campaign_dir: Path, requested: int) -> tuple[int, dict[str, Any]]:
    snapshot = observe_resources(str(campaign_dir))
    ceiling = requested
    if isinstance(config.max_concurrent_trials, int):
        ceiling = min(ceiling, config.max_concurrent_trials)
    slots = admission_slots(snapshot, requested=ceiling)
    return max(0, slots), resource_report(snapshot, max(0, slots))


def _resolve_dataset(config: Any, campaign_dir: Path) -> Path:
    if config.dataset_path is not None:
        path = Path(config.dataset_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cached = campaign_dir / "inputs" / "rich-history.csv.gz"
    if cached.exists():
        return cached
    from .rich_features import load_full_rich_history
    frame = load_full_rich_history(
        Path("track/hkracing 2/runs.csv"),
        Path("track/hkracing 2/races.csv"),
        Path("data/processed/historical/runners.csv.gz"),
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(cached.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, cached)
    return cached


def _protocol_parameters(config: Any) -> dict[str, Any]:
    if config.protocol_path is None:
        return {}
    payload = json.loads(Path(config.protocol_path).read_text(encoding="utf-8"))
    allowed = {"min_train_races", "calibration_races", "score_races", "max_folds"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown protocol parameters: {unknown}")
    return payload


def _execution_signature(
    recipe: PipelineRecipe,
    dataset_hash: str,
    protocol: dict[str, Any],
    code_revision: str,
    environment_hash: str,
) -> str:
    payload = {
        "recipe": recipe.canonical_payload(),
        "dataset_hash": dataset_hash,
        "protocol": protocol,
        "code_revision": code_revision,
        "environment_hash": environment_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_campaign_identity(
    campaign_dir: Path,
    *,
    dataset_hash: str,
    protocol_parameters: dict[str, Any],
    code_revision: str,
    environment_hash: str,
) -> None:
    identity = {
        "schema_version": 1,
        "dataset_hash": dataset_hash,
        "protocol_hash": hashlib.sha256(
            json.dumps(
                protocol_parameters, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "code_revision": code_revision,
        "environment_hash": environment_hash,
        "target_contract_version": "research-targets-v2",
        "metric_version": "protected-development-v2",
    }
    path = campaign_dir / "campaign-identity.json"
    if path.exists():
        stored = _read_json(path)
        if stored != identity:
            changed = sorted(
                key for key in set(stored) | set(identity)
                if stored.get(key) != identity.get(key)
            )
            raise RuntimeError(
                "Campaign scientific identity changed; start a new campaign: "
                + ", ".join(changed)
            )
        return
    _write_json_atomic(path, identity)


def _success_count(ledger: ResearchLedger) -> int:
    return sum(row["status"] == "completed" for row in ledger.terminal_results())


def _consecutive_failure_count(ledger: ResearchLedger) -> int:
    count = 0
    for row in reversed(ledger.terminal_results()):
        if row["status"] == "completed":
            break
        count += 1
    return count


def _finish_payload(
    campaign_dir: Path,
    ledger: ResearchLedger,
    search: ProgramSearchController,
    cycles: int,
    results: list[dict[str, Any]],
    mode: str,
    tracking_errors: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "campaign_dir": str(campaign_dir),
        "cycles": cycles,
        "results": results,
        "ledger": ledger.snapshot(),
        "search": search.snapshot(),
        "updated_at": utc_now(),
    }
    if tracking_errors:
        payload["tracking_errors"] = sorted(set(tracking_errors))
    _write_json_atomic(campaign_dir / "status.json", payload)
    return payload


def _persist_decision(campaign_dir: Path, cycle: int, decision: dict[str, Any]) -> None:
    _write_json_atomic(campaign_dir / "decisions" / f"cycle-{cycle:04d}.json", decision)
    _append_jsonl(campaign_dir / "decisions.jsonl", decision)


def _next_cycle_number(campaign_dir: Path) -> int:
    observed: set[int] = set()
    decisions = campaign_dir / "decisions.jsonl"
    if decisions.exists():
        for line_number, line in enumerate(
            decisions.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                cycle = row["cycle"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid durable decision at {decisions}:{line_number}"
                ) from exc
            if not isinstance(cycle, int) or cycle < 0:
                raise RuntimeError(
                    f"Invalid cycle number at {decisions}:{line_number}"
                )
            observed.add(cycle)
    for directory in ("decisions", "evidence", "planner"):
        for path in (campaign_dir / directory).glob("cycle-*.json"):
            try:
                observed.add(int(path.stem.removeprefix("cycle-")))
            except ValueError as exc:
                raise RuntimeError(f"Invalid cycle artifact name: {path}") from exc
    return max(observed, default=-1) + 1


def _code_revision() -> str:
    configured = os.environ.get("IMA_CODE_REVISION")
    if configured:
        return configured.strip()
    revision_file = Path("REVISION")
    if revision_file.is_file():
        revision = revision_file.read_text(encoding="utf-8").strip()
        if revision:
            return revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _environment_hash() -> str:
    text = json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
    }, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def campaign_lock(campaign_dir: Path) -> Iterator[None]:
    path = campaign_dir / "controller.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Campaign controller already running: {campaign_dir}") from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}) + "\n")
        handle.flush()
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "not_started"}
    return json.loads(path.read_text(encoding="utf-8"))


def _planner_due(campaign_dir: Path, completed: int, interval: int) -> bool:
    decisions_path = campaign_dir / "decisions.jsonl"
    last_completed = 0
    if decisions_path.exists():
        for line in decisions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") in {"fixture", "openrouter"}:
                last_completed = max(
                    last_completed,
                    int(row.get("completed_trial_count", len(row.get("evidence_trial_ids", ())))),
                )
    first_threshold = min(3, interval)
    if last_completed == 0:
        return completed >= first_threshold
    return completed - last_completed >= interval


def _controller_lock_held(path: Path) -> bool:
    if not path.exists():
        return False
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return True
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    return False


def _read_lock_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _planner_spend(campaign_dir: Path) -> float:
    total = 0.0
    for path in (campaign_dir / "planner").glob("cycle-*.json"):
        payload = _read_json(path)
        response = payload.get("response", {})
        usage = response.get("usage") or normalize_openrouter_usage(
            response.get("raw_response", {})
        )
        value = usage.get("total_cost_usd")
        if value is not None:
            total += float(value)
    return total
