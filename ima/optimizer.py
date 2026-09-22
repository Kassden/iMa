"""Terminal-first optimizer contracts and local policy for iMa experiments."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .experiments import ExperimentSpec, default_experiment_specs
from .openrouter_orchestrator import OpenRouterConfig, choose_proposals, submit_proposal_batch


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
    max_trials: int = 1
    proposal_batch_size: int = 1
    max_concurrent_trials: int = 1
    timeout_minutes: int | None = None
    service_tier: str | None = None
    openrouter_batch: bool = False
    model: str = "openrouter/local-policy"
    champion_run_id: str | None = None
    gates: MetricGates = field(default_factory=MetricGates)

    def validate(self) -> None:
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.proposal_batch_size <= 0:
            raise ValueError("proposal_batch_size must be positive")
        if self.max_concurrent_trials <= 0:
            raise ValueError("max_concurrent_trials must be positive")
        if self.proposal_batch_size > self.max_trials:
            raise ValueError("proposal_batch_size cannot exceed max_trials")
        if self.max_concurrent_trials > self.max_trials:
            raise ValueError("max_concurrent_trials cannot exceed max_trials")
        if self.service_tier and self.service_tier != "flex":
            raise ValueError("only OpenRouter service_tier='flex' is supported")
        if self.openrouter_batch and self.policy == "local":
            raise ValueError("openrouter batch mode requires a remote OpenRouter policy")

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


def local_proposals(config: CampaignConfig) -> list[ExperimentProposal]:
    config.validate()
    done = completed_run_ids(config.campaign_dir)
    proposals: list[ExperimentProposal] = []
    for spec in default_experiment_specs():
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
        if len(proposals) >= min(config.proposal_batch_size, config.max_trials - len(done)):
            break
    return proposals


def available_specs(config: CampaignConfig) -> list[ExperimentSpec]:
    done = completed_run_ids(config.campaign_dir)
    return [spec for spec in default_experiment_specs() if spec.run_id not in done]


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
    remote_config = OpenRouterConfig.from_env(
        model=config.model if config.model != "openrouter/local-policy" else None,
        service_tier=config.service_tier,
    )
    result = choose_proposals(
        available_specs(config),
        min(config.proposal_batch_size, config.max_trials - len(completed_run_ids(config.campaign_dir))),
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
    if config.max_concurrent_trials == 1 or len(payloads) <= 1:
        for payload in payloads:
            results.append(_run_trial_worker(payload, str(config.campaign_dir), str(template_path)))
    else:
        with ProcessPoolExecutor(max_workers=config.max_concurrent_trials) as pool:
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
    if config.policy == "openrouter" and config.openrouter_batch:
        remote_config = OpenRouterConfig.from_env(
            model=config.model if config.model != "openrouter/local-policy" else None,
            service_tier=config.service_tier,
        )
        batch = submit_proposal_batch(
            available_specs(config),
            min(config.proposal_batch_size, config.max_trials - len(completed_run_ids(config.campaign_dir))),
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
    if config.policy == "openrouter":
        proposals, planner_result = openrouter_proposals(config)
        (config.campaign_dir / "openrouter-planner.json").write_text(
            json.dumps(planner_result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        proposals = local_proposals(config)
    if not proposals:
        decision = AgentDecision("stop", "No remaining local proposals.", "none")
        append_jsonl(config.campaign_dir / "decisions.jsonl", asdict(decision))
        write_report(config.campaign_dir, [], decision)
        return {"mode": "complete", "decision": asdict(decision)}
    results = execute_proposals(config, proposals)
    return {
        "mode": "executed",
        "results": [asdict(result) for result in results],
        "campaign_dir": str(config.campaign_dir),
    }


def env_openrouter_config() -> dict[str, str | None]:
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": os.environ.get("IMA_OPTIMIZER_MODEL"),
        "service_tier": os.environ.get("IMA_OPTIMIZER_SERVICE_TIER"),
    }
