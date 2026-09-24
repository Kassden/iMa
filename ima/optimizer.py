"""Terminal-first optimizer contracts and local policy for iMa experiments."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

from .experiments import ExperimentSpec, experiment_specs
from .openrouter_orchestrator import OpenRouterConfig, choose_proposals, submit_proposal_batch
from .research_search import RecipeSearchController


ALLOWED_CHANGED_SURFACES = {
    "hyperparameters",
    "feature_schema",
    "feature_family",
    "transform",
    "dataset_window",
    "model_family",
    "calibration",
    "market_blend",
}
FORBIDDEN_TERMS = {
    "final_odds",
    "dividend",
    "dividends",
    "result",
    "results",
    "target_win",
    "target_probability",
    "live_execution",
    "promotion",
    "promote",
    "hkjc_credentials",
}
SPEC_PROFILES = {"default", "long", "adaptive"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MetricGates:
    primary_metric: str = "test_blended.race_log_loss"
    require_market_improvement: bool = True
    min_vote_margin: int = 1


@dataclass(frozen=True)
class CampaignConfig:
    campaign_dir: Path
    policy: str = "local"
    max_trials: int | None = 1
    proposal_batch_size: int = 1
    max_concurrent_trials: int | str = 1
    timeout_minutes: int | None = None
    service_tier: str | None = None
    openrouter_batch: bool = False
    model: str = "openrouter/local-policy"
    spec_profile: str = "default"
    champion_run_id: str | None = None
    gates: MetricGates = field(default_factory=MetricGates)

    def validate(self) -> None:
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be positive or None for unlimited")
        if self.proposal_batch_size <= 0:
            raise ValueError("proposal_batch_size must be positive")
        if isinstance(self.max_concurrent_trials, str):
            if self.max_concurrent_trials != "auto":
                raise ValueError("max_concurrent_trials must be positive or 'auto'")
        elif self.max_concurrent_trials <= 0:
            raise ValueError("max_concurrent_trials must be positive")
        if self.max_trials is not None and self.proposal_batch_size > self.max_trials:
            raise ValueError("proposal_batch_size cannot exceed max_trials")
        if (
            self.max_trials is not None
            and isinstance(self.max_concurrent_trials, int)
            and self.max_concurrent_trials > self.max_trials
        ):
            raise ValueError("max_concurrent_trials cannot exceed max_trials")
        if self.service_tier and self.service_tier != "flex":
            raise ValueError("only OpenRouter service_tier='flex' is supported")
        if self.openrouter_batch and self.policy == "local":
            raise ValueError("openrouter batch mode requires a remote OpenRouter policy")
        if self.spec_profile not in SPEC_PROFILES:
            raise ValueError(f"Unknown experiment spec profile: {self.spec_profile}")

    def serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["campaign_dir"] = str(self.campaign_dir)
        return payload


@dataclass(frozen=True)
class ExperimentProposal:
    trial_id: str
    hypothesis: str
    changed_surface: str
    spec: ExperimentSpec
    parent_trial_id: str | None = None
    request_id: str | None = None

    def validate(self) -> None:
        if self.changed_surface not in ALLOWED_CHANGED_SURFACES:
            raise ValueError(f"unsupported changed surface: {self.changed_surface}")
        payload = json.dumps(asdict(self), sort_keys=True).lower()
        matches = sorted(term for term in FORBIDDEN_TERMS if term in payload)
        if matches:
            raise ValueError(f"proposal contains forbidden terms: {matches}")

    def serializable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return payload


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    run_id: str
    status: str
    started_at: str
    ended_at: str
    output_dir: str
    command: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class CandidateVote:
    metric: str
    winner: str
    reason: str


@dataclass(frozen=True)
class AgentDecision:
    decision: str
    rationale: str
    next_action: str
    created_at: str = field(default_factory=utc_now)
    votes: list[CandidateVote] = field(default_factory=list)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_campaign(config: CampaignConfig) -> Path:
    config.validate()
    config.campaign_dir.mkdir(parents=True, exist_ok=True)
    path = config.campaign_dir / "campaign.json"
    payload = config.serializable() | {"updated_at": utc_now()}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        payload = existing | payload
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def completed_run_ids(campaign_dir: Path) -> set[str]:
    return {
        str(row["run_id"])
        for row in read_jsonl(campaign_dir / "trials.jsonl")
        if row.get("status") == "completed"
    }


def _completed_trial_rows(campaign_dir: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in read_jsonl(campaign_dir / "trials.jsonl")
        if row.get("status") == "completed" and isinstance(row.get("metrics"), dict)
    ]


def _parameter_signature(kind: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"kind": kind, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _adaptive_generation(done: set[str]) -> int:
    prefix = "adaptive-g"
    generations = []
    for run_id in done:
        if not run_id.startswith(prefix):
            continue
        try:
            generations.append(int(run_id[len(prefix):len(prefix) + 4]))
        except ValueError:
            continue
    return (max(generations) + 1) if generations else 1


def _trial_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = row.get("metrics", {})
    blended = _metric(metrics, "test_blended.race_log_loss")
    fundamental = _metric(metrics, "test_fundamental.race_log_loss")
    top_pick = _metric(metrics, "test_blended.top_pick_win_rate")
    return (
        blended if blended is not None else float("inf"),
        fundamental if fundamental is not None else float("inf"),
        -(top_pick if top_pick is not None else 0.0),
        str(row.get("run_id", "")),
    )


def _add_spec_if_new(
    specs: list[ExperimentSpec],
    seen_signatures: set[str],
    generation: int,
    kind: str,
    parameters: dict[str, Any],
) -> None:
    signature = _parameter_signature(kind, parameters)
    if signature in seen_signatures:
        return
    seen_signatures.add(signature)
    specs.append(ExperimentSpec(
        f"adaptive-g{generation:04d}-{kind}-{signature}",
        kind,
        parameters,
    ))


def adaptive_experiment_specs(campaign_dir: Path, max_specs: int = 512) -> list[ExperimentSpec]:
    """Build a deterministic next catalogue from completed campaign history.

    The profile first drains the hand-authored long grid. Once that baseline
    catalogue is completed, it generates new neighborhoods around the best
    completed configurations by metric ordering.
    """
    base = experiment_specs("long")
    rows = _completed_trial_rows(campaign_dir)
    if not rows:
        return base
    done = completed_run_ids(campaign_dir)
    if not {spec.run_id for spec in base}.issubset(done):
        return base

    existing_signatures = {
        _parameter_signature(
            str(row.get("metrics", {}).get("kind", "")),
            row.get("metrics", {}).get("parameters", {}),
        )
        for row in rows
    }
    generation = _adaptive_generation(done)
    generated: list[ExperimentSpec] = []
    for row in sorted(rows, key=_trial_sort_key)[:24]:
        metrics = row.get("metrics", {})
        kind = str(metrics.get("kind", row.get("run_id", "")))
        parameters = dict(metrics.get("parameters") or {})
        if kind == "logit":
            c_value = float(parameters.get("C", 1.0))
            for multiplier in (0.4, 0.6, 0.8, 0.9, 1.1, 1.25, 1.6, 2.2):
                candidate = dict(parameters)
                candidate["C"] = max(1e-5, round(c_value * multiplier, 8))
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
            for key in ("class_weight", "fit_intercept"):
                if key in parameters:
                    candidate = dict(parameters)
                    if key == "class_weight":
                        candidate[key] = None if candidate[key] == "balanced" else "balanced"
                    else:
                        candidate[key] = not bool(candidate[key])
                    _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
            for tolerance in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2):
                candidate = dict(parameters)
                candidate["tol"] = tolerance
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
        elif kind == "boosted":
            learning_rate = float(parameters.get("learning_rate", 0.06))
            max_iter = int(parameters.get("max_iter", 180))
            leaf_nodes = int(parameters.get("max_leaf_nodes", 31))
            l2_value = float(parameters.get("l2_regularization", 1.0))
            for lr_multiplier in (0.5, 0.75, 0.9, 1.1, 1.35, 1.75):
                candidate = dict(parameters)
                candidate["learning_rate"] = max(0.001, round(learning_rate * lr_multiplier, 8))
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
            for iter_delta in (-80, -40, 40, 80, 140):
                candidate = dict(parameters)
                candidate["max_iter"] = max(40, max_iter + iter_delta)
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
            for leaf in sorted({7, 15, 31, 63, max(3, leaf_nodes // 2), leaf_nodes * 2}):
                candidate = dict(parameters)
                candidate["max_leaf_nodes"] = leaf
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
            for l2_multiplier in (0.0, 0.25, 0.5, 1.5, 3.0, 6.0):
                candidate = dict(parameters)
                candidate["l2_regularization"] = round(l2_value * l2_multiplier, 8)
                _add_spec_if_new(generated, existing_signatures, generation, kind, candidate)
        if len(generated) >= max_specs:
            break
    return [*base, *generated[:max_specs]]


def specs_for_config(config: CampaignConfig) -> list[ExperimentSpec]:
    if config.spec_profile == "adaptive":
        return adaptive_experiment_specs(config.campaign_dir)
    return experiment_specs(config.spec_profile)


def remaining_trial_budget(config: CampaignConfig) -> int | None:
    if config.max_trials is None:
        return None
    return max(0, config.max_trials - len(completed_run_ids(config.campaign_dir)))


def local_proposals(config: CampaignConfig) -> list[ExperimentProposal]:
    config.validate()
    done = completed_run_ids(config.campaign_dir)
    remaining = remaining_trial_budget(config)
    if remaining == 0:
        return []
    proposals: list[ExperimentProposal] = []
    for spec in specs_for_config(config):
        if spec.run_id in done:
            continue
        proposals.append(
            ExperimentProposal(
                trial_id=f"trial-{len(done) + len(proposals) + 1:04d}",
                hypothesis=f"Evaluate existing grid spec {spec.run_id} as a bounded baseline trial.",
                changed_surface="hyperparameters",
                spec=spec,
            )
        )
        limit = config.proposal_batch_size if remaining is None else min(config.proposal_batch_size, remaining)
        if len(proposals) >= limit:
            break
    return proposals


def available_specs(config: CampaignConfig) -> list[ExperimentSpec]:
    done = completed_run_ids(config.campaign_dir)
    remaining = remaining_trial_budget(config)
    if remaining == 0:
        return []
    return [spec for spec in specs_for_config(config) if spec.run_id not in done]


def research_recipe_proposals(campaign_dir: Path, count: int = 1) -> list[dict[str, Any]]:
    """Return persisted v2 recipe suggestions without touching legacy execution."""
    return [proposal.serializable() for proposal in RecipeSearchController(campaign_dir).ask(count)]


def _proposals_from_remote_payload(
    config: CampaignConfig,
    payload: dict[str, Any],
) -> list[ExperimentProposal]:
    specs_by_id = {spec.run_id: spec for spec in available_specs(config)}
    proposals: list[ExperimentProposal] = []
    for index, row in enumerate(payload.get("proposals", []), start=1):
        run_id = row.get("run_id")
        if run_id not in specs_by_id:
            raise ValueError(f"OpenRouter planner selected unknown or completed run_id: {run_id}")
        proposals.append(
            ExperimentProposal(
                trial_id=f"trial-{len(completed_run_ids(config.campaign_dir)) + index:04d}",
                hypothesis=str(row.get("hypothesis", "")).strip(),
                changed_surface=str(row.get("changed_surface", "")).strip(),
                spec=specs_by_id[run_id],
                request_id=str(row.get("request_id")) if row.get("request_id") else None,
            )
        )
    if not proposals:
        raise ValueError("OpenRouter planner returned no proposals")
    validate_proposal_batch(proposals)
    return proposals


def openrouter_proposals(config: CampaignConfig) -> tuple[list[ExperimentProposal], dict[str, Any]]:
    config.validate()
    remaining = remaining_trial_budget(config)
    if remaining == 0:
        return [], {"proposal_payload": {"proposals": []}}
    remote_config = OpenRouterConfig.from_env(
        model=config.model if config.model != "openrouter/local-policy" else None,
        service_tier=config.service_tier,
    )
    result = choose_proposals(
        available_specs(config),
        config.proposal_batch_size if remaining is None else min(config.proposal_batch_size, remaining),
        remote_config,
    )
    proposals = _proposals_from_remote_payload(config, result["proposal_payload"])
    return proposals, result


def validate_proposal_batch(proposals: Iterable[ExperimentProposal]) -> None:
    run_ids: set[str] = set()
    trial_ids: set[str] = set()
    for proposal in proposals:
        proposal.validate()
        if proposal.spec.run_id in run_ids:
            raise ValueError(f"duplicate proposal run_id: {proposal.spec.run_id}")
        if proposal.trial_id in trial_ids:
            raise ValueError(f"duplicate proposal trial_id: {proposal.trial_id}")
        run_ids.add(proposal.spec.run_id)
        trial_ids.add(proposal.trial_id)


def _metric(payload: dict[str, Any], dotted: str) -> float | None:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return float(current) if isinstance(current, int | float) else None


def voting_rank(trial_metrics: list[dict[str, Any]]) -> tuple[str | None, list[CandidateVote]]:
    """Rank trials by simple metric votes instead of one winner-take-all metric."""
    if not trial_metrics:
        return None, []
    lower_better = ("test_blended.race_log_loss", "test_fundamental.race_log_loss")
    higher_better = (
        "test_blended.top_pick_win_rate",
        "test_fundamental.top_pick_win_rate",
        "test_blended.winner_top3_rate",
        "incremental_pseudo_r2",
    )
    votes: list[CandidateVote] = []
    for metric in lower_better:
        scored = [(row["run_id"], _metric(row, metric)) for row in trial_metrics]
        scored = [(run_id, value) for run_id, value in scored if value is not None]
        if scored:
            run_id, value = min(scored, key=lambda item: item[1])
            votes.append(CandidateVote(metric, run_id, f"lowest {metric}={value:.6g}"))
    for metric in higher_better:
        scored = [(row["run_id"], _metric(row, metric)) for row in trial_metrics]
        scored = [(run_id, value) for run_id, value in scored if value is not None]
        if scored:
            run_id, value = max(scored, key=lambda item: item[1])
            votes.append(CandidateVote(metric, run_id, f"highest {metric}={value:.6g}"))
    if not votes:
        return None, []
    counts: dict[str, int] = {}
    for vote in votes:
        counts[vote.winner] = counts.get(vote.winner, 0) + 1
    winner = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return winner, votes


def decide_from_results(results: list[TrialResult]) -> AgentDecision:
    completed = [result for result in results if result.status == "completed"]
    if not completed:
        return AgentDecision(
            decision="continue",
            rationale="No completed trials are available yet.",
            next_action="run_next_trial",
        )
    winner, votes = voting_rank([
        result.metrics | {"run_id": result.run_id}
        for result in completed
    ])
    if winner is None:
        return AgentDecision(
            decision="continue",
            rationale="Trials completed but no rankable metrics were found.",
            next_action="inspect_metrics",
        )
    return AgentDecision(
        decision="continue",
        rationale=f"Voting ranker currently prefers {winner}; candidate remains unpromoted.",
        next_action="run_next_trial_or_review",
        votes=votes,
    )


def _run_trial_worker(
    proposal_payload: dict[str, Any],
    campaign_dir: str,
    template_path: str,
) -> dict[str, Any]:
    from .data import build_full_history_dataset
    from .experiments import run_experiments
    from .mlflow_tracking import MLflowConfig

    spec_payload = proposal_payload["spec"]
    proposal = ExperimentProposal(
        trial_id=proposal_payload["trial_id"],
        hypothesis=proposal_payload["hypothesis"],
        changed_surface=proposal_payload["changed_surface"],
        spec=ExperimentSpec(spec_payload["run_id"], spec_payload["kind"], spec_payload["parameters"]),
        parent_trial_id=proposal_payload.get("parent_trial_id"),
        request_id=proposal_payload.get("request_id"),
    )
    started = utc_now()
    output_dir = Path(campaign_dir) / "trials" / proposal.trial_id
    command = [
        sys.executable,
        "-m",
        "scripts.run_experiments",
        "--output",
        str(output_dir),
    ]
    try:
        frame = build_full_history_dataset(
            Path("track/hkracing 2/runs.csv"),
            Path("track/hkracing 2/races.csv"),
            Path("data/processed/historical/runners.csv.gz"),
        )
        summary = run_experiments(
            frame,
            output_dir,
            Path(template_path),
            specs=[proposal.spec],
            mlflow_config=MLflowConfig.from_values(),
        )
        run = summary["runs"][0]
        return asdict(TrialResult(
            trial_id=proposal.trial_id,
            run_id=proposal.spec.run_id,
            status="completed",
            started_at=started,
            ended_at=utc_now(),
            output_dir=str(output_dir),
            command=command,
            metrics=run,
        ))
    except Exception as exc:  # pragma: no cover - exercised by integration failures.
        return asdict(TrialResult(
            trial_id=proposal.trial_id,
            run_id=proposal.spec.run_id,
            status="failed",
            started_at=started,
            ended_at=utc_now(),
            output_dir=str(output_dir),
            command=command,
            error=str(exc),
        ))


def execute_proposals(
    config: CampaignConfig,
    proposals: list[ExperimentProposal],
    template_path: Path = Path("docs/model-results/dashboard-template.html"),
) -> list[TrialResult]:
    validate_proposal_batch(proposals)
    payloads = [proposal.serializable() for proposal in proposals]
    results: list[dict[str, Any]] = []
    concurrency = resolved_concurrency(config, len(payloads))
    if concurrency == 1 or len(payloads) <= 1:
        for payload in payloads:
            results.append(_run_trial_worker(payload, str(config.campaign_dir), str(template_path)))
    else:
        with ProcessPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_run_trial_worker, payload, str(config.campaign_dir), str(template_path)): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                results.append(future.result())
    ordered = sorted(results, key=lambda row: row["trial_id"])
    trial_results = [TrialResult(**row) for row in ordered]
    for row in ordered:
        append_jsonl(config.campaign_dir / "trials.jsonl", row)
    decision = decide_from_results(trial_results)
    append_jsonl(config.campaign_dir / "decisions.jsonl", asdict(decision))
    write_report(config.campaign_dir, trial_results, decision)
    return trial_results


def resolved_concurrency(config: CampaignConfig, available_trials: int | None = None) -> int:
    if isinstance(config.max_concurrent_trials, int):
        value = config.max_concurrent_trials
    else:
        cpu_count = os.cpu_count() or 1
        value = max(1, min(32, cpu_count - 2 if cpu_count > 4 else cpu_count))
    value = min(value, config.proposal_batch_size)
    if config.max_trials is not None:
        value = min(value, config.max_trials)
    if available_trials is not None:
        value = min(value, max(1, available_trials))
    return max(1, value)


def write_report(campaign_dir: Path, results: list[TrialResult], decision: AgentDecision) -> Path:
    lines = [
        "# Optimizer Campaign Report",
        "",
        f"Updated: {utc_now()}",
        "",
        "## Trials",
        "",
    ]
    for result in results:
        lines.append(f"- {result.trial_id} `{result.run_id}`: {result.status}")
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Decision: `{decision.decision}`",
        f"- Next action: `{decision.next_action}`",
        f"- Rationale: {decision.rationale}",
        "",
        "## Voting Ranker",
        "",
    ])
    if decision.votes:
        for vote in decision.votes:
            lines.append(f"- `{vote.metric}` -> `{vote.winner}` ({vote.reason})")
    else:
        lines.append("- No votes yet.")
    path = campaign_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def dry_run(config: CampaignConfig) -> dict[str, Any]:
    write_campaign(config)
    remaining = remaining_trial_budget(config)
    if config.policy == "openrouter" and config.openrouter_batch:
        remote_config = OpenRouterConfig.from_env(
            model=config.model if config.model != "openrouter/local-policy" else None,
            service_tier=config.service_tier,
        )
        batch = submit_proposal_batch(
            available_specs(config),
            config.proposal_batch_size if remaining is None else min(config.proposal_batch_size, remaining),
            remote_config,
        )
        (config.campaign_dir / "openrouter-batch.json").write_text(
            json.dumps(batch, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "mode": "openrouter_batch_submitted",
            "campaign": config.serializable(),
            "batch": batch,
        }
    if config.policy == "openrouter":
        proposals, planner_result = openrouter_proposals(config)
        (config.campaign_dir / "openrouter-planner.json").write_text(
            json.dumps(planner_result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        proposals = local_proposals(config)
    validate_proposal_batch(proposals)
    payload = {
        "mode": "dry_run",
        "campaign": config.serializable(),
        "proposals": [proposal.serializable() for proposal in proposals],
        "completed_run_ids": sorted(completed_run_ids(config.campaign_dir)),
        "remaining_trials": "unlimited" if remaining is None else remaining,
        "resolved_concurrency": resolved_concurrency(config, len(proposals)) if proposals else 0,
    }
    (config.campaign_dir / "dry-run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def run_campaign(config: CampaignConfig, dry: bool = False) -> dict[str, Any]:
    write_campaign(config)
    if dry:
        return dry_run(config)
    if config.policy == "openrouter" and config.openrouter_batch:
        return dry_run(config)
    deadline = None
    if config.timeout_minutes is not None:
        deadline = datetime.now(timezone.utc).timestamp() + (config.timeout_minutes * 60)
    all_results: list[TrialResult] = []
    cycles = 0
    while config.max_trials is None or len(completed_run_ids(config.campaign_dir)) < config.max_trials:
        if deadline is not None and datetime.now(timezone.utc).timestamp() >= deadline:
            decision = AgentDecision("continue", "Campaign time budget reached.", "resume_later")
            append_jsonl(config.campaign_dir / "decisions.jsonl", asdict(decision))
            write_report(config.campaign_dir, all_results, decision)
            return {
                "mode": "time_budget_reached",
                "results": [asdict(result) for result in all_results],
                "campaign_dir": str(config.campaign_dir),
                "cycles": cycles,
            }
        if config.policy == "openrouter":
            proposals, planner_result = openrouter_proposals(config)
            (config.campaign_dir / f"openrouter-planner-{cycles + 1:04d}.json").write_text(
                json.dumps(planner_result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            proposals = local_proposals(config)
        if not proposals:
            decision = AgentDecision("stop", "No remaining proposals in the selected spec profile.", "none")
            append_jsonl(config.campaign_dir / "decisions.jsonl", asdict(decision))
            write_report(config.campaign_dir, all_results, decision)
            return {
                "mode": "complete",
                "decision": asdict(decision),
                "results": [asdict(result) for result in all_results],
                "campaign_dir": str(config.campaign_dir),
                "cycles": cycles,
            }
        all_results.extend(execute_proposals(config, proposals))
        cycles += 1
    return {
        "mode": "executed",
        "results": [asdict(result) for result in all_results],
        "campaign_dir": str(config.campaign_dir),
        "cycles": cycles,
    }


def env_openrouter_config() -> dict[str, str | None]:
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": os.environ.get("IMA_OPTIMIZER_MODEL"),
        "service_tier": os.environ.get("IMA_OPTIMIZER_SERVICE_TIER"),
    }
