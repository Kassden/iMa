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

from .openrouter_orchestrator import OpenRouterConfig, OpenRouterError, choose_research_proposals
from .mlflow_tracking import MLflowConfig, log_research_package_version
from .research_executor import RecipeExecutionRequest, execute_recipe
from .research_resources import admission_slots, observe_resources, resource_report
from .research_search import RecipeSearchController, RecipeSuggestion
from .research_specs import PipelineRecipe, ResearchProposal
from .research_store import ResearchLedger, utc_now


class PlannerSpendCapReached(RuntimeError):
    def __init__(self, spent: float, cap: float) -> None:
        self.spent = spent
        self.cap = cap
        super().__init__(f"OpenRouter planner spend cap reached: ${spent:.6f} of ${cap:.6f}")


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
        search = RecipeSearchController(campaign_dir)
        if search.study is None:
            raise RuntimeError(
                "Executable agentic campaigns require Optuna; install the research dependencies"
            )
        ledger.recover_running()
        _reconcile_tells(ledger, search)
        tracking_errors = _reconcile_tracking(config, ledger)
        cycles = 0
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
                    config, campaign_dir, ledger, search, batch_size, cycles
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
                requests.append(RecipeExecutionRequest(
                    attempt_id=attempt.attempt_id,
                    proposal_id=suggestion.trial_id,
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
                continue
            completed = _execute_requests(requests, slots)
            for result in completed:
                row = result.serializable()
                ledger.complete_attempt(result.attempt_id, row, status=result.status)
                _append_jsonl(campaign_dir / "trials.jsonl", row)
                new_results.append(row)
                _reconcile_tells(ledger, search)
                tracking_errors.extend(_reconcile_tracking(config, ledger))
            cycles += 1
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
    search: RecipeSearchController,
    count: int,
    cycle: int,
) -> tuple[list[RecipeSuggestion], dict[str, Any]]:
    successes = [row for row in ledger.terminal_results() if row["status"] == "completed"]
    if not successes:
        suggestions = search.ask(count)
        decision = {
            "cycle": cycle,
            "source": "bootstrap",
            "evidence_id": None,
            "suggestions": [item.serializable() for item in suggestions],
        }
        _persist_decision(campaign_dir, cycle, decision)
        return suggestions, decision
    evidence = _build_evidence(successes)
    evidence_path = campaign_dir / "evidence" / f"cycle-{cycle:04d}.json"
    _write_json_atomic(evidence_path, evidence)
    if config.planner_mode in {"fixture", "openrouter"} and not _planner_due(
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
            response = choose_research_proposals(evidence, count, remote)
        except OpenRouterError as exc:
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
        proposals = _fixture_proposals(evidence, count, cycle)
        source = "fixture"
    else:
        suggestions = search.ask(count)
        decision = {
            "cycle": cycle,
            "source": "local_optuna",
            "evidence_id": evidence["evidence_id"],
            "completed_trial_count": len(successes),
            "evidence_trial_ids": [row["attempt_id"] for row in successes[-32:]],
            "suggestions": [item.serializable() for item in suggestions],
        }
        _persist_decision(campaign_dir, cycle, decision)
        return suggestions, decision
    completed_ids = {row["attempt_id"] for row in successes}
    suggestions = []
    rejected_proposals = list(response.get("rejected_proposals", [])) if source == "openrouter" else []
    for proposal in proposals:
        if proposal.evidence_ids != (evidence["evidence_id"],):
            raise ValueError("planner proposal must cite the current evidence_id")
        if not proposal.parent_trial_ids or not set(proposal.parent_trial_ids).issubset(completed_ids):
            raise ValueError("planner proposal cites unknown or missing parent trials")
        try:
            suggestion = search.reserve_recipe(
                proposal.recipe, proposal.hypothesis, proposal.changed_axes
            )
        except ValueError as exc:
            if not str(exc).startswith("Recipe already reserved:"):
                raise
            rejected_proposals.append({
                "proposal_id": proposal.proposal_id,
                "recipe_hash": proposal.recipe.recipe_hash(),
                "reason": "duplicate_recipe",
            })
            continue
        suggestions.append(suggestion)
    local_refill = []
    if len(suggestions) < count:
        local_refill = search.ask(count - len(suggestions))
        suggestions.extend(local_refill)
    decision = {
        "cycle": cycle,
        "source": source,
        "evidence_id": evidence["evidence_id"],
        "completed_trial_count": len(successes),
        "evidence_trial_ids": [row["attempt_id"] for row in successes[-32:]],
        "proposals": [proposal.model_dump(mode="json") for proposal in proposals],
        "rejected_proposals": rejected_proposals,
        "local_refill_trial_ids": [item.trial_id for item in local_refill],
        "suggestions": [item.serializable() for item in suggestions],
    }
    _persist_decision(campaign_dir, cycle, decision)
    return suggestions, decision


def _fixture_proposals(
    evidence: dict[str, Any], count: int, cycle: int
) -> list[ResearchProposal]:
    completed = evidence["completed_trials"]
    parents = tuple(row["attempt_id"] for row in completed[-min(3, len(completed)):])
    best = min(float(row["objective_value"]) for row in completed)
    improving = min(float(row.get("mean_selected_minus_market", 0.0)) for row in completed) <= 0
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
        else:
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
        proposals.append(ResearchProposal(
            proposal_id=f"fixture-cycle-{cycle:04d}-{index:02d}",
            parent_trial_ids=parents,
            evidence_ids=(evidence["evidence_id"],),
            hypothesis=hypothesis,
            changed_axes=axes,
            recipe=recipe,
            expected_observation="Development objective changes against the cited parents.",
            falsification_rule="Reject when the comparable development objective does not improve.",
        ))
    return proposals


def _build_evidence(successes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for row in successes[-32:]:
        result = row["result"]
        rows.append({
            "attempt_id": row["attempt_id"],
            "recipe_hash": result["recipe_hash"],
            "target_kind": result["target_kind"],
            "objective_name": result["objective_name"],
            "objective_value": result["objective_value"],
            "mean_selected_minus_market": result.get("metrics", {}).get(
                "mean_selected_minus_market"
            ),
        })
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": 1,
        "evidence_id": digest,
        "completed_trials": rows,
        "notes": ["Development evidence only; holdout metrics are excluded."],
    }


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


def _reconcile_tells(ledger: ResearchLedger, search: RecipeSearchController) -> None:
    for effect in ledger.pending_tells():
        trial_number = int(effect["payload"]["trial_number"])
        state = search.trial_state(trial_number)
        result = effect["result"]
        if state == "running":
            if result.get("status") == "completed":
                search.tell(trial_number, float(result["objective_value"]), result.get("metrics"))
            else:
                search.tell_failed(trial_number)
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
    search: RecipeSearchController,
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
        usage = payload.get("response", {}).get("raw_response", {}).get("usage", {})
        value = usage.get("cost", usage.get("total_cost", 0.0))
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total
