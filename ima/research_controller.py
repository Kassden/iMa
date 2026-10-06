"""Single-owner feedback controller for agentic research campaigns."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .openrouter_orchestrator import (
    OpenRouterConfig,
    OpenRouterError,
    choose_cycle_trial_budget,
    choose_research_proposals,
    normalize_openrouter_usage,
)
from .mlflow_tracking import (
    MLflowConfig,
    log_optimizer_cycle_trace,
    log_optimizer_planner_trace,
    log_research_package_version,
)
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_evaluation import build_expanding_folds
from .research_resources import admission_slots, observe_resources, resource_report
from .research_search import ProgramSearchController, RecipeSuggestion
from .research_specs import (
    FEATURE_DISCOVERY_PORTFOLIO_VERSION,
    FUNDAMENTAL_FIRST_PORTFOLIO_VERSION,
    PipelineRecipe,
    ResearchProposal,
    V5_PORTFOLIO_VERSION,
)
from .research_store import ResearchLedger, utc_now
from .feature_sets import FEATURE_SCHEMAS, drop_feature_families


class PlannerSpendCapReached(RuntimeError):
    def __init__(self, spent: float, cap: float) -> None:
        self.spent = spent
        self.cap = cap
        super().__init__(f"OpenRouter planner spend cap reached: ${spent:.6f} of ${cap:.6f}")


class DatasetFeatureProfile:
    """Check transform inputs against the frozen development training folds."""

    def __init__(self, dataset_path: Path, protocol: dict[str, Any], *,
                 extra_numeric_features: tuple[str, ...] = (),
                 prepared_frame: pd.DataFrame | None = None) -> None:
        numeric = set().union(*(schema.numeric for schema in FEATURE_SCHEMAS.values()))
        numeric.update(extra_numeric_features)
        self.numeric_columns = numeric
        required = {"race_id", "race_no", "date", "target_win", "source", "horse_id", "jockey_id", "jockey_key", "trainer_id", "trainer_key", "finish_seconds", "lengths_raw", "actual_weight"}
        if prepared_frame is not None:
            frame = prepared_frame[[column for column in prepared_frame if column in numeric | required]].copy()
        elif dataset_path.suffix == ".parquet":
            frame = pd.read_parquet(dataset_path)
            frame = frame[[column for column in frame if column in numeric | required]]
        else:
            frame = pd.read_csv(
                dataset_path, usecols=lambda column: column in numeric | required,
                low_memory=False,
            )
        if "target_win" in frame:
            winners = pd.to_numeric(frame["target_win"], errors="coerce").groupby(
                frame["race_id"]
            ).sum(min_count=1)
            valid_races = winners.index[winners.eq(1)]
            frame = frame[frame["race_id"].isin(valid_races)].copy()
        self.present_columns = set(frame.columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        present_numeric = sorted(numeric & self.present_columns)
        values = frame[present_numeric].apply(pd.to_numeric, errors="coerce")
        folds = build_expanding_folds(frame, **protocol)
        scored_races = set().union(*(set(fold.score_race_ids) for fold in folds))
        score_rows = frame["race_id"].astype(str).isin(scored_races)
        score_coverage = values.loc[score_rows].notna().mean()
        development_races = set().union(*(set(fold.train_race_ids) | set(fold.calibration_race_ids) | set(fold.score_race_ids) for fold in folds))
        full_coverage = values.loc[frame["race_id"].astype(str).isin(development_races)].notna().mean()
        development = frame[frame["race_id"].astype(str).isin(development_races)]
        entities = []
        for entity in ("horse","jockey","trainer"):
            key,fallback=f"{entity}_id",f"{entity}_key"
            identities=development[key] if key in development else pd.Series(index=development.index,dtype=object)
            if fallback in development:
                identities=identities.fillna(development[fallback])
            if identities.notna().all():
                entities.append(entity)
        measurements = [name for name,column in (("speed_mps","finish_seconds"),("beaten_lengths","lengths_raw"),("carried_weight","actual_weight")) if column in development and development[column].notna().any()]
        self.discovery_sources = {"entities":entities,"measurements":measurements,"history_sources":sorted(development.source.dropna().astype(str).unique()) if "source" in development else [],"scope":"development only; source availability checked before feature construction"}
        coverage_drift = (score_coverage - full_coverage).sort_values()
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
            "discovery_sources": self.discovery_sources,
            "numeric_feature_count": len(numeric),
            "present_numeric_count": len(present_numeric),
            "unavailable_transform_columns": sorted(self.unavailable["all_history"]),
            "unavailable_transform_columns_by_window": {
                window: sorted(columns) for window, columns in self.unavailable.items()
            },
            "unavailable_transform_column_count": len(self.unavailable["all_history"]),
            "score_era_coverage": {
                column: round(float(value), 3)
                for column, value in score_coverage.items()
                if value < 0.75
            },
            "largest_coverage_declines": {
                column: round(float(value), 3)
                for column, value in coverage_drift.head(25).items()
                if value < -0.05
            },
            "rule": "Transforms need numeric observations in every protected training fold.",
        }

    def admission_error(self, recipe: PipelineRecipe) -> str | None:
        if recipe.feature_discovery:
            spec=recipe.feature_discovery
            if not set(spec.entities)<=set(self.discovery_sources["entities"]):
                return "Discovery entity identities are incomplete in development data"
            if not set(spec.measurements)<=set(self.discovery_sources["measurements"]):
                return "Discovery measurement has no validated development observations"
            if not set(spec.history_sources)<=set(self.discovery_sources["history_sources"]):
                return "Discovery source is not registered in development data"
        schema = drop_feature_families(
            FEATURE_SCHEMAS[recipe.feature_schema], recipe.drop_feature_families
        )
        effective_numeric = set(schema.numeric) | (
            set(recipe.extra_numeric_features or ()) & self.numeric_columns
        )
        for transform in recipe.transforms:
            for column in transform.parameters["columns"]:
                if recipe.feature_discovery and re.fullmatch(r"dfs_[a-f0-9]{24}",column):
                    continue
                if column not in effective_numeric:
                    return f"{column} is not in the effective numeric feature schema"
                if column in self.unavailable[recipe.train_window]:
                    return f"{column} has no numeric observations in a training fold"
        return None


def run_research_campaign(config: Any) -> dict[str, Any]:
    if config.research_policy == "expansion_v6":
        from .research_expansion import run_expansion_campaign
        return run_expansion_campaign(config)
    config.validate()
    if config.research_policy == "discovery_v5":
        from .research_v5 import run_v5_campaign
        return run_v5_campaign(config)
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
            research_policy=config.research_policy,
        )
        ledger = ResearchLedger(campaign_dir / "ledger.sqlite")
        search = ProgramSearchController(campaign_dir)
        profile = DatasetFeatureProfile(dataset_path, protocol_parameters)
        ledger.recover_running()
        _reconcile_tells(ledger, search)
        tracking_errors = _reconcile_tracking(config, ledger, dataset_path)
        cycles = 0
        cycle = _next_cycle_number(campaign_dir)
        new_results: list[dict[str, Any]] = []
        pending = ledger.pending_attempts()
        if pending:
            slots, resources = _slots(config, campaign_dir, len(pending))
            if slots == 0:
                payload = _finish_payload(
                    campaign_dir, ledger, search, cycles, new_results,
                    "paused_admission", tracking_errors,
                )
                payload["resources"] = resources
                return payload
            requests = []
            for row in pending:
                suggestion = _suggestion_from_payload(row["payload"])
                ledger.mark_running(row["attempt_id"])
                requests.append(RecipeExecutionRequest(
                    attempt_id=row["attempt_id"],
                    proposal_id=suggestion.proposal_id or suggestion.trial_id,
                    trial_number=suggestion.trial_number,
                    recipe=suggestion.recipe,
                    dataset_path=dataset_path,
                    output_dir=campaign_dir / "trials" / row["attempt_id"],
                    protocol_parameters=protocol_parameters,
                    dataset_hash=dataset_hash,
                    code_revision=code_revision,
                    environment_hash=environment_hash,
                    portfolio_version=(
                        _portfolio_version(config.research_policy)
                    ),
                ))
            resumed_results = []
            for result in _execute_requests(requests, slots):
                row = result.serializable()
                original = next(item for item in pending if item["attempt_id"] == result.attempt_id)
                row["trial_id"] = original["payload"]["trial_id"]
                row["program_id"] = original["payload"]["program_id"]
                row["target_parameters"] = original["payload"]["recipe"]["target"]["parameters"]
                if _is_portfolio_policy(config.research_policy):
                    row.update(_portfolio_identity(
                        PipelineRecipe.model_validate(original["payload"]["recipe"]),
                        config.research_policy,
                    ))
                ledger.complete_attempt(result.attempt_id, row, status=result.status)
                _append_jsonl(campaign_dir / "trials.jsonl", row)
                resumed_results.append(row)
                new_results.append(row)
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger, dataset_path))
            decision = {
                "cycle": cycle, "source": "resume_pending",
                "suggestions": [item["payload"] for item in pending],
            }
            _persist_decision(campaign_dir, cycle, decision)
            trace_error = _trace_completed_cycle(
                config, campaign_dir, decision, resumed_results, ledger=ledger,
            )
            if trace_error:
                tracking_errors.append(trace_error)
            cycles += 1
            cycle += 1
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
                if _is_portfolio_policy(config.research_policy):
                    suggestions, decision = _next_v3_suggestions(
                        config, campaign_dir, ledger, search, profile, batch_size, cycle
                    )
                else:
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
            slots = min(slots, len(suggestions))
            planner_trace_error = _trace_planner_decision(config, campaign_dir, decision)
            if planner_trace_error:
                tracking_errors.append(planner_trace_error)
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
                if _is_portfolio_policy(config.research_policy):
                    payload.update(_portfolio_identity(suggestion.recipe, config.research_policy))
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
                    portfolio_version=(
                        _portfolio_version(config.research_policy)
                    ),
                ))
            if not requests:
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger, dataset_path))
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
                if _is_portfolio_policy(config.research_policy):
                    row.update(_portfolio_identity(lineage.recipe, config.research_policy))
                cycle_results.append(row)
                ledger.complete_attempt(result.attempt_id, row, status=result.status)
                _append_jsonl(campaign_dir / "trials.jsonl", row)
                new_results.append(row)
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger, dataset_path))
            trace_error = _trace_completed_cycle(
                config, campaign_dir, decision, cycle_results, ledger=ledger,
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


V3_PORTFOLIO_VERSION = FUNDAMENTAL_FIRST_PORTFOLIO_VERSION
V3_MLFLOW_EXPERIMENT = "ima-agentic-v3-fundamental"
V3_REGISTERED_MODEL_PREFIX = "ima-agentic-v3-fundamental-candidates"
V4_PORTFOLIO_VERSION = FEATURE_DISCOVERY_PORTFOLIO_VERSION
V4_MLFLOW_EXPERIMENT = "ima-agentic-v4-features"
V4_REGISTERED_MODEL_PREFIX = "ima-agentic-v4-feature-candidates"
V3_EXPERIMENTS = ("E1", "E2", "E3")
V3_CONTRACTS = {
    "B": ("win_probability", "benter_conditional_logit"),
    "E1": ("ranking_strength", "lightgbm_lambdarank"),
    "E2": ("placing_top_k", "catboost_classifier"),
    "E3": ("recorded_final_win_odds", "catboost_regressor"),
}
V4_EXPERIMENTS = ("E1", "E2", "E3", "E4")
V4_CONTRACTS = {
    "B": ("win_probability", "benter_conditional_logit"),
    "E1": ("win_probability", "boosted"),
    "E2": ("ranking_strength", "lightgbm_lambdarank"),
    "E3": ("placing_top_k", "catboost_classifier"),
    "E4": ("recorded_final_win_odds", "catboost_regressor"),
}


def _is_portfolio_policy(policy: str) -> bool:
    return policy in {"benter_v3", "feature_v4", "discovery_v5", "expansion_v6"}


def _portfolio_version(policy: str) -> str | None:
    if policy == "expansion_v6":
        from .research_specs import V6_PORTFOLIO_VERSION
        return V6_PORTFOLIO_VERSION
    return {
        "benter_v3": V3_PORTFOLIO_VERSION,
        "feature_v4": V4_PORTFOLIO_VERSION,
        "discovery_v5": V5_PORTFOLIO_VERSION,
    }.get(policy)


def _portfolio_identity(recipe: PipelineRecipe, policy: str) -> dict[str, str]:
    if policy == "expansion_v6":
        benter = (recipe.target.kind == "win_probability"
                  and recipe.model.kind == "benter_conditional_logit"
                  and recipe.performance_distribution is None)
        if recipe.pipeline_graph is not None:
            from .pipeline_graph import PipelineGraph
            graph = PipelineGraph.from_dict(recipe.pipeline_graph).validate()
            nodes = {node.node_id: node for node in graph.nodes}
            reachable = set()
            def visit(node_id):
                if node_id is None or node_id in reachable:
                    return
                reachable.add(node_id)
                for parent in nodes[node_id].inputs:
                    visit(parent)
            for root in (graph.output_node_id, graph.fundamental_node_id, graph.joint_node_id):
                visit(root)
            predictive = [nodes[key] for key in reachable if nodes[key].kind == "estimator"]
            benter = benter and bool(predictive) and all(
                node.kind == "estimator" and node.parameters.get("model_kind", recipe.model.kind) == "benter_conditional_logit"
                for node in predictive)
        return {"lane": "benter" if benter else "experimental",
                "experiment_id": "B" if benter else f"E:{recipe.target.kind}:{recipe.model.kind}",
                "portfolio_version": _portfolio_version(policy)}
    if policy == "benter_v3":
        return _v3_identity(recipe)
    pair = (recipe.target.kind, recipe.model.kind)
    for experiment_id, contract in V4_CONTRACTS.items():
        if pair == contract:
            return {
                "lane": "benter" if experiment_id == "B" else "experimental",
                "experiment_id": experiment_id,
                "portfolio_version": _portfolio_version(policy),
            }
    raise ValueError(f"Recipe is outside the v4 portfolio: {pair}")


def _portfolio_slot(index: int, policy: str) -> str:
    if policy == "expansion_v6":
        return "B" if round(4*(index+1)/5)>round(4*index/5) else "E"
    if policy == "benter_v3":
        return _v3_slot(index)
    if round(4 * (index + 1) / 5) > round(4 * index / 5):
        return "B"
    experimental_ordinal = index - round(4 * index / 5)
    return V4_EXPERIMENTS[experimental_ordinal % len(V4_EXPERIMENTS)]


def _portfolio_contracts(policy: str) -> dict[str, tuple[str, str]]:
    return V3_CONTRACTS if policy == "benter_v3" else V4_CONTRACTS


def _tracking_names(policy: str) -> tuple[str, str]:
    if policy == "expansion_v6":
        return "ima-agentic-v6-research", "ima-agentic-v6-research-candidates"
    if policy == "discovery_v5":
        return "ima-agentic-v5-discovery", "ima-agentic-v5-discovery-candidates"
    if policy == "feature_v4":
        return V4_MLFLOW_EXPERIMENT, V4_REGISTERED_MODEL_PREFIX
    if policy == "benter_v3":
        return V3_MLFLOW_EXPERIMENT, V3_REGISTERED_MODEL_PREFIX
    return "ima-agentic-v2", "ima-agentic-candidates"


def _v3_slot(index: int) -> str:
    if round(4 * (index + 1) / 5) > round(4 * index / 5):
        return "B"
    experimental_ordinal = index - round(4 * index / 5)
    return V3_EXPERIMENTS[experimental_ordinal % len(V3_EXPERIMENTS)]


def _v3_identity(recipe: PipelineRecipe) -> dict[str, str]:
    pair = (recipe.target.kind, recipe.model.kind)
    for experiment_id, contract in V3_CONTRACTS.items():
        if pair == contract:
            return {
                "lane": "benter" if experiment_id == "B" else "experimental",
                "experiment_id": experiment_id,
                "portfolio_version": V3_PORTFOLIO_VERSION,
            }
    raise ValueError(f"Recipe is outside the v3 portfolio: {pair}")


def _v3_seed_proposals() -> tuple[ResearchProposal, ...]:
    choices = (
        ("B", PipelineRecipe(
            feature_schema="benter-rich-v1",
            model={"kind": "benter_conditional_logit", "parameters": {"l2": 0.1}},
        ), {"l2": {"kind": "float", "low": 0.001, "high": 5.0, "log": True}}, 26),
        ("E1", PipelineRecipe(
            target={"kind": "ranking_strength"}, feature_schema="benter-rich-v1",
            model={"kind": "lightgbm_lambdarank", "parameters": {"n_estimators": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"learning_rate": {"kind": "float", "low": 0.02, "high": 0.12, "log": True}}, 8),
        ("E2", PipelineRecipe(
            target={"kind": "placing_top_k", "parameters": {"top_k": 3}},
            feature_schema="benter-rich-v1",
            model={"kind": "catboost_classifier", "parameters": {"iterations": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"depth": {"kind": "int", "low": 3, "high": 7}}, 8),
        ("E3", PipelineRecipe(
            target={"kind": "recorded_final_win_odds"},
            feature_schema="benter-rich-v1",
            model={"kind": "catboost_regressor", "parameters": {"iterations": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"depth": {"kind": "int", "low": 3, "high": 7}}, 8),
    )
    return tuple(ResearchProposal(
        proposal_id=f"v3-seed-{experiment_id}",
        hypothesis=f"Test the registered {experiment_id} contract against its target baseline.",
        changed_axes=("hyperparameters",),
        recipe=recipe,
        search_space=space,
        expected_observation="Measure held-out development metrics within this target only.",
        falsification_rule="Reject if the target-specific baseline is not improved.",
        max_trials=budget,
    ) for experiment_id, recipe, space, budget in choices)


def _v4_seed_proposals() -> tuple[ResearchProposal, ...]:
    choices = (
        ("B", PipelineRecipe(
            feature_schema="notebook-rich-v2",
            model={"kind": "benter_conditional_logit", "parameters": {"l2": 0.006}},
            transforms=({"kind": "race_relative_center", "parameters": {
                "columns": ["last_speed_ratio", "prior_win_rate"],
            }},),
        ), {"l2": {"kind": "float", "low": 0.001, "high": 0.05, "log": True}}, 4),
        ("E1", PipelineRecipe(
            feature_schema="benter-rich-v1",
            model={"kind": "boosted", "parameters": {"max_iter": 160}},
        ), {"l2_regularization": {"kind": "float", "low": 0.01,
                                      "high": 10.0, "log": True}}, 8),
        ("E2", PipelineRecipe(
            target={"kind": "ranking_strength"}, feature_schema="benter-rich-v1",
            model={"kind": "lightgbm_lambdarank", "parameters": {"n_estimators": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"learning_rate": {"kind": "float", "low": 0.02,
                               "high": 0.12, "log": True}}, 8),
        ("E3", PipelineRecipe(
            target={"kind": "placing_top_k", "parameters": {"top_k": 3}},
            feature_schema="benter-rich-v1",
            model={"kind": "catboost_classifier", "parameters": {"iterations": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"depth": {"kind": "int", "low": 3, "high": 7}}, 8),
        ("E4", PipelineRecipe(
            target={"kind": "recorded_final_win_odds"},
            feature_schema="benter-rich-v1",
            model={"kind": "catboost_regressor", "parameters": {"iterations": 120}},
            calibration={"kind": "none"}, blend={"kind": "none"},
        ), {"depth": {"kind": "int", "low": 3, "high": 7}}, 8),
    )
    return tuple(ResearchProposal(
        proposal_id=f"v4-seed-{experiment_id}",
        hypothesis=f"Test the registered v4 {experiment_id} contract on matched folds.",
        changed_axes=("feature_schema", "transform", "hyperparameters")
        if experiment_id == "B" else ("model_family", "hyperparameters"),
        recipe=recipe, search_space=space,
        expected_observation="Measure target-specific development metrics and feature coverage.",
        falsification_rule="Reject if the comparable objective does not improve.",
        max_trials=budget,
    ) for experiment_id, recipe, space, budget in choices)


def _suggestion_from_payload(payload: dict[str, Any]) -> RecipeSuggestion:
    return RecipeSuggestion(
        trial_id=str(payload["trial_id"]),
        trial_number=int(payload["trial_number"]),
        recipe=PipelineRecipe.model_validate(payload["recipe"]),
        hypothesis=str(payload["hypothesis"]),
        changed_axes=tuple(payload["changed_axes"]),
        program_id=payload.get("program_id"),
        proposal_id=payload.get("proposal_id"),
    )


def _next_v3_suggestions(
    config: Any, campaign_dir: Path, ledger: ResearchLedger,
    search: ProgramSearchController, profile: DatasetFeatureProfile,
    count: int, cycle: int,
) -> tuple[list[RecipeSuggestion], dict[str, Any]]:
    policy = config.research_policy
    version = _portfolio_version(policy)
    contracts = _portfolio_contracts(policy)
    if not search.programs:
        for proposal in (_v3_seed_proposals() if policy == "benter_v3"
                         else _v4_seed_proposals()):
            if policy == "feature_v4" and profile is not None:
                admission_error = profile.admission_error(proposal.recipe)
                if admission_error:
                    raise ValueError(
                        f"v4 seed {proposal.proposal_id} is unavailable: {admission_error}"
                    )
            search.register(proposal)
    suggestions: list[RecipeSuggestion] = []
    slot_decisions: list[dict[str, Any]] = []
    budget_decision = None
    if config.planner_mode == "openrouter":
        spent = _planner_spend(campaign_dir)
        if spent >= float(config.max_total_cost_usd):
            raise PlannerSpendCapReached(spent, float(config.max_total_cost_usd))
        evidence = _build_evidence(ledger.terminal_results(), search, profile)
        remote = OpenRouterConfig.from_env(
            model=config.model, service_tier=config.service_tier,
            max_output_tokens=config.max_output_tokens,
            timeout_seconds=config.planner_timeout_seconds,
            provider_endpoint=config.provider_endpoint,
            reasoning_effort=config.planner_reasoning_effort,
        )
        try:
            budget_decision = choose_cycle_trial_budget(evidence, count, remote)
        except OpenRouterError as exc:
            decision = {
                "cycle": cycle, "source": policy, "portfolio_version": version,
                "budget_ceiling": count, "budget_error": str(exc), "suggestions": [],
            }
            _persist_decision(campaign_dir, cycle, decision)
            return [], decision
        count = budget_decision["trial_budget"]
    base = len(ledger.reserved_attempts())
    for offset in range(count):
        experiment_id = _portfolio_slot(base + offset, policy)
        target_kind, model_kind = contracts[experiment_id]
        allowed = [program_id for program_id, proposal in reversed(list(search.programs.items()))
                   if proposal.recipe.target.kind == target_kind
                   and proposal.recipe.model.kind == model_kind]
        candidate = search.ask(1, allowed_program_ids=allowed)
        if not candidate:
            candidate, planner_decision = _v3_replenish(
                config, campaign_dir, ledger, search, profile, experiment_id, cycle
            )
            slot_decisions.append(planner_decision)
        if not candidate:
            slot_decisions.append({"experiment_id": experiment_id, "status": "paused_no_program"})
            break
        suggestions.extend(candidate)
    decision = {
        "cycle": cycle, "source": policy, "portfolio_version": version,
        "trial_budget": count, "budget_ceiling": budget_decision["ceiling"] if budget_decision else count,
        "budget_decision": budget_decision,
        "slot_decisions": slot_decisions,
        "suggestions": [item.serializable() | _portfolio_identity(item.recipe, policy)
                        for item in suggestions],
    }
    planner_calls = [row for row in slot_decisions if row.get("status") == "openrouter"]
    if budget_decision:
        planner_calls.insert(0, {"planner_usage": budget_decision["usage"]})
    if planner_calls:
        decision["planner_model"] = config.model
        decision["planner_usage"] = {
            key: sum(float(row.get("planner_usage", {}).get(key) or 0) for row in planner_calls)
            for key in ("input_tokens", "output_tokens", "total_tokens", "total_cost_usd")
        }
        decision["planner_usage"]["cost_status"] = (
            "reported" if all(row.get("planner_usage", {}).get("total_cost_usd") is not None
                              for row in planner_calls) else "unavailable"
        )
    _persist_decision(campaign_dir, cycle, decision)
    return suggestions, decision


def _v3_replenish(
    config: Any, campaign_dir: Path, ledger: ResearchLedger,
    search: ProgramSearchController, profile: DatasetFeatureProfile,
    experiment_id: str, cycle: int,
) -> tuple[list[RecipeSuggestion], dict[str, Any]]:
    policy = config.research_policy
    contracts = _portfolio_contracts(policy)
    feature_reference = None
    require_feature_change = False
    if policy == "feature_v4" and experiment_id == "B":
        benter_programs = sum(
            proposal.recipe.target.kind == "win_probability"
            and proposal.recipe.model.kind == "benter_conditional_logit"
            for proposal in search.programs.values()
        )
        require_feature_change = benter_programs % 2 == 1
        benter_results = [
            row for row in ledger.terminal_results()
            if row["status"] == "completed"
            and row["payload"]["recipe"]["target"]["kind"] == "win_probability"
            and row["payload"]["recipe"]["model"]["kind"] == "benter_conditional_logit"
        ]
        if benter_results:
            best = min(benter_results, key=lambda row: row["result"]["objective_value"])
            feature_reference = PipelineRecipe.model_validate(best["payload"]["recipe"])
    if config.planner_mode != "openrouter":
        return [], {"experiment_id": experiment_id, "status": "planner_not_configured"}
    spent = _planner_spend(campaign_dir)
    if spent >= float(config.max_total_cost_usd):
        raise PlannerSpendCapReached(spent, float(config.max_total_cost_usd))
    evidence = _build_evidence(ledger.terminal_results(), search, profile)
    evidence["v3_assignment"] = {
        "experiment_id": experiment_id,
        "target_kind": contracts[experiment_id][0],
        "model_kind": contracts[experiment_id][1],
        "portfolio_version": _portfolio_version(policy),
        "rule": "Propose exactly this target/model; choose features, transforms, and bounds only.",
        "required_feature_change": require_feature_change,
        "reference_feature_recipe": {
            "feature_schema": feature_reference.feature_schema,
            "drop_feature_families": list(feature_reference.drop_feature_families),
            "transforms": [spec.model_dump(mode="json") for spec in feature_reference.transforms],
            "train_window": feature_reference.train_window,
        } if feature_reference is not None else None,
    }
    remote = OpenRouterConfig.from_env(
        model=config.model, service_tier=config.service_tier,
        max_output_tokens=config.max_output_tokens,
        timeout_seconds=config.planner_timeout_seconds,
        provider_endpoint=config.provider_endpoint,
        reasoning_effort=config.planner_reasoning_effort,
    )
    try:
        response = choose_research_proposals(evidence, 1, remote)
    except OpenRouterError as exc:
        return [], {"experiment_id": experiment_id, "status": "planner_error", "error": str(exc)}
    _write_json_atomic(
        campaign_dir / "planner" / f"cycle-{cycle:04d}-{experiment_id}-{time.time_ns()}.json",
        {"evidence_id": evidence["evidence_id"], "requested_model": config.model,
         "response": response},
    )
    completed = {row["attempt_id"] for row in ledger.terminal_results()
                 if row["status"] == "completed"}
    rejected = []
    for payload in response["proposals"]:
        proposal = ResearchProposal.model_validate(payload)
        if (proposal.recipe.target.kind, proposal.recipe.model.kind) != contracts[experiment_id]:
            rejected.append("wrong_assigned_contract")
            continue
        if proposal.evidence_ids != (evidence["evidence_id"],):
            rejected.append("stale_evidence_id")
            continue
        if not proposal.parent_trial_ids or not set(proposal.parent_trial_ids) <= completed:
            rejected.append("invalid_parent_trials")
            continue
        if profile.admission_error(proposal.recipe):
            rejected.append("unavailable_transform_input")
            continue
        if require_feature_change and feature_reference is not None and (
            proposal.recipe.feature_program_id("same-dataset")
            == feature_reference.feature_program_id("same-dataset")
        ):
            rejected.append("unchanged_feature_program")
            continue
        program_id = search.register(proposal)
        suggestion = search.ask(1, allowed_program_ids=[program_id])
        if suggestion:
            return suggestion, {
                "experiment_id": experiment_id, "status": "openrouter",
                "evidence_id": evidence["evidence_id"],
                "planner_model": config.model,
                "planner_usage": response.get("usage"),
                "program_id": program_id,
                "rejected": rejected,
            }
    return [], {"experiment_id": experiment_id, "status": "rejected", "reasons": rejected}


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
    if search.has_capacity():
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
            "feature_program_id": result.get("lineage", {}).get("feature_program_id"),
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
            "mean_selected_minus_calibrated_market": result.get("metrics", {}).get(
                "mean_selected_minus_calibrated_market"
            ),
            "zero_fundamental_weight_fold_fraction": result.get("metrics", {}).get(
                "zero_fundamental_weight_fold_fraction"
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
        used = sum(
            trial.state.name != "PRUNED" for trial in search.studies[program_id].trials
        )
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
    payload["benter_incremental_candidates"] = sorted([
        {
            "attempt_id": row["attempt_id"],
            "paired_delta_log_loss": row["result"]["metrics"]["mean_selected_minus_calibrated_market"],
            "zero_fundamental_weight_fold_fraction": row["result"]["metrics"].get(
                "zero_fundamental_weight_fold_fraction"
            ),
        }
        for row in successes
        if row["payload"]["recipe"]["model"]["kind"] == "benter_conditional_logit"
        and row["result"].get("metrics", {}).get("mean_selected_minus_calibrated_market") is not None
    ], key=lambda row: row["paired_delta_log_loss"])[:5]
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


def _reconcile_tracking(
    config: Any, ledger: ResearchLedger, dataset_path: Path
) -> list[str]:
    if not config.mlflow_tracking_uri:
        return []
    errors: list[str] = []
    experiment_name, registered_model_name = _tracking_names(config.research_policy)
    tracking = MLflowConfig.from_values(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=experiment_name,
        register_models=True,
        registered_model_name=registered_model_name,
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
                dataset_path=dataset_path,
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
    *,
    ledger: ResearchLedger | None = None,
) -> str | None:
    if not config.mlflow_tracking_uri:
        return None
    experiment_name, registered_model_name = _tracking_names(config.research_policy)
    tracking = MLflowConfig.from_values(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=experiment_name,
        register_models=True,
        registered_model_name=registered_model_name,
    )
    try:
        log_optimizer_cycle_trace(
            campaign_dir, decision, results, tracking,
            campaign_results=(
                [item["result"] for item in ledger.terminal_results()]
                if ledger is not None else None
            ),
        )
    except Exception as exc:  # tracing is retriable and must not stop training.
        return f"cycle-{decision.get('cycle')}: trace: {type(exc).__name__}: {exc}"
    return None


def _trace_planner_decision(
    config: Any, campaign_dir: Path, decision: dict[str, Any]
) -> str | None:
    if not config.mlflow_tracking_uri or not decision.get("planner_model"):
        return None
    tracking = MLflowConfig.from_values(
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=_tracking_names(config.research_policy)[0],
    )
    try:
        log_optimizer_planner_trace(campaign_dir, decision, tracking)
    except Exception as exc:  # tracing is retriable and must not stop training.
        return f"cycle-{decision.get('cycle')}: planner trace: {type(exc).__name__}: {exc}"
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
    return protocol_spec_parameters(payload)


def protocol_spec_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Protocol must be an object")
    registered = "spec" in payload
    payload = payload.get("spec", payload)
    if not isinstance(payload, dict):
        raise ValueError("Protocol specification must be an object")
    allowed = {"min_train_races", "calibration_races", "score_races", "max_folds", "whole_meeting_boundaries", "fold_selection"}
    registry_fields = {"final_confirmation_races", "final_confirmation_race_ids", "final_confirmation_start"} if registered else set()
    unknown = sorted(set(payload) - allowed - registry_fields)
    if unknown:
        raise ValueError(f"unknown protocol parameters: {unknown}")
    return {key: value for key, value in payload.items() if key in allowed}


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
    research_policy: str = "legacy",
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
    if _is_portfolio_policy(research_policy):
        identity["research_policy"] = research_policy
        identity["portfolio_version"] = _portfolio_version(research_policy)
        identity["target_contract_version"] = "research-targets-v3"
        identity["metric_version"] = "protected-development-v3-fundamental-v1"
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
            match = re.fullmatch(r"cycle-(\d+)(?:-(?:B|E\d+)-\d+)?", path.stem)
            if match is None:
                raise RuntimeError(f"Invalid cycle artifact name: {path}")
            observed.add(int(match.group(1)))
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
    calls = list((campaign_dir/"planner-calls").glob("*.json"))
    if calls:
        return sum(float(normalize_openrouter_usage(_read_json(path)["response"]).get("total_cost_usd") or 0) for path in calls)
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
