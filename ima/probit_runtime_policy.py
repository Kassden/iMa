"""Exact-workload probit budgets and finite, objective-driven extensions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .research_resources import JobWorkload


@dataclass(frozen=True)
class ProbitRuntimePolicy:
    initial_seconds: float = 14400.
    hard_seconds: float = 21600.
    extension_seconds: float = 1800.
    max_extensions: int = 4
    progress_max_age_seconds: float = 300.
    extension_window_seconds: float = 30.
    objective_min_delta: float = 1e-9
    workload_fingerprint: str = ""

    def __post_init__(self):
        for name in ("initial_seconds", "hard_seconds", "extension_seconds",
                     "progress_max_age_seconds", "extension_window_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite runtime policy required: " + name)
        if self.hard_seconds < self.initial_seconds:
            raise ValueError("Hard cap must cover the initial allowance")
        if isinstance(self.max_extensions, bool) or not isinstance(self.max_extensions, int) or self.max_extensions < 0:
            raise ValueError("Nonnegative finite extension count required")
        if isinstance(self.objective_min_delta, bool) or not math.isfinite(self.objective_min_delta) or self.objective_min_delta <= 0:
            raise ValueError("Positive finite objective improvement required")

    def to_dict(self):
        return asdict(self)


def resolve_probit_budget(workload, samples=(), requested_seconds=14400.) -> dict:
    """Use JobEstimator's exact samples, including failed/censored lower bounds.

    Cold resource-admission extrapolations are not attempt deadlines.
    The four-hour floor applies even with successful history.
    A failed/censored runtime remains a lower bound, never a successful sample.
    """
    if isinstance(requested_seconds, bool) or not isinstance(requested_seconds, (int, float)) or not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise ValueError("Positive finite requested allowance required")
    # Qualified deployment adapters may reload the resource model owner.
    # Cross-release model instances are validated through their structured payload.
    workload = workload if isinstance(workload, JobWorkload) else JobWorkload.model_validate(
        workload.model_dump() if hasattr(workload, "model_dump") else workload)
    graph_probit = workload.family.startswith("graph:") and any(
        isinstance(node, dict) and node.get("model") == "gaussian_probit"
        for node in (workload.search_settings.get("graph_structure") or []))
    if workload.family.removeprefix("graph:") != "gaussian_probit" and not graph_probit:
        raise ValueError("Probit budget requires a probit workload")
    fingerprint = workload.fingerprint()
    matching = [sample for sample in samples if sample.workload_fingerprint == fingerprint]
    successful = [sample.wall_seconds for sample in matching if not sample.failed and not sample.censored]
    bounds = [sample.wall_seconds for sample in matching if sample.failed or sample.censored]
    measured = float(np.quantile(successful, .95))*1.25 if successful else 0.
    lower_bound = max(bounds)*1.25 if bounds else 0.
    initial = max(14400., float(requested_seconds), measured, lower_bound)
    return ProbitRuntimePolicy(initial_seconds=initial, hard_seconds=initial+7200.,
                               workload_fingerprint=fingerprint).to_dict()


def normalize_progress_policy(policy):
    return policy if isinstance(policy, ProbitRuntimePolicy) else ProbitRuntimePolicy(**policy)


def observe_optimizer_progress(state, progress, policy, now):
    """Observe optimizer, verified batches, or successful fold completion.

    Fresh but nonproductive work is distinct from a stalled, stale producer.
    CPU heartbeat, stage timing, and gradient change alone cannot extend a fit.
    """
    policy = normalize_progress_policy(policy)

    def reject(health):
        return {"productive_progress": None, "progress_health": health}

    if not isinstance(progress, dict):
        return reject("invalid_progress")
    try:
        stamp = float(progress["updated_at_epoch"])
        if not math.isfinite(stamp) or stamp > now:
            return reject("invalid_progress")
        if now-stamp > policy.progress_max_age_seconds:
            return reject("stalled_stale")
    except (KeyError, TypeError, ValueError):
        return reject("invalid_progress")
    # Old files from another task, and pre-resume progress, cannot renew a lease.
    if progress.get("pid") != state.get("pid") or stamp < state.get("progress_not_before_epoch", now):
        return reject("invalid_progress")
    stage, fold = progress.get("stage"), progress.get("fold_id")
    if not isinstance(fold, str) or not fold:
        return reject("invalid_progress")
    if stage not in {"optimizing", "verifying_quadrature", "fit_completed", "fold_completed"}:
        return reject("stage_active")
    try:
        if progress.get("gradient_norm") is not None and not math.isfinite(float(progress["gradient_norm"])):
            return reject("invalid_progress")
        if stage in {"fit_completed", "fold_completed"} and progress.get("objective") is not None and not math.isfinite(float(progress["objective"])):
            return reject("invalid_progress")
    except (TypeError, ValueError):
        return reject("invalid_progress")
    identity = [fold, progress.get("optimization_round", 0), progress.get("quadrature_order"), stage]
    previous = state.get("optimizer_observation") or {}
    evidence = state.get("productive_progress")
    if previous.get("identity") != identity:
        evidence = None
        previous = {}
    elif stamp <= previous.get("updated_at_epoch", 0):
        return {}
    observation = {"identity": identity, "updated_at_epoch": stamp}
    changes, productive = {}, False
    if stage in {"fit_completed", "fold_completed"}:
        if progress.get("converged") is False or (stage == "fit_completed" and progress.get("converged") is not True):
            return reject("invalid_progress")
        completed = state.get("completed_progress_folds", [])
        productive = fold not in completed
        if productive:
            changes["completed_progress_folds"] = completed+[fold]
    else:
        try:
            objective = float(progress["objective"])
            if not math.isfinite(objective):
                return reject("invalid_progress")
            if stage == "optimizing":
                iteration, evaluations = progress["iteration"], progress["objective_evaluations"]
                if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (iteration, evaluations)):
                    return reject("invalid_progress")
                productive = bool(previous and iteration > previous["iteration"]
                    and evaluations > previous["evaluations"]
                    and objective < previous["best_objective"]-policy.objective_min_delta)
                observation.update(iteration=iteration, evaluations=evaluations,
                    best_objective=min(objective, previous.get("best_objective", objective)))
            else:
                done, total = progress["verification_batches_completed"], progress["verification_batches_total"]
                if (any(isinstance(n, bool) or not isinstance(n, int) for n in (done, total))
                        or not 0 <= done <= total or total < 1):
                    return reject("invalid_progress")
                productive = bool(previous and total == previous["batches_total"] and done > previous["batches_completed"])
                observation.update(batches_completed=done, batches_total=total)
        except (KeyError, TypeError, ValueError):
            return reject("invalid_progress")
    if productive:
        evidence = {"observed_at_epoch": now, "updated_at_epoch": stamp,
                    "identity": identity, "kind": stage, "observation": observation}
    changes.update(optimizer_observation=observation, productive_progress=evidence,
                   progress_health="productive" if productive else "slow_nonproductive")
    return changes
