from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.optimizer import CampaignConfig, run_campaign


_CONFIG_KEYS = {
    "schema_version", "policy", "max_trials", "proposal_batch_size",
    "max_concurrent_trials", "timeout_minutes", "service_tier",
    "openrouter_batch", "model", "spec_profile", "planner_mode",
    "max_total_cost_usd", "max_output_tokens", "replan_every_terminal_trials",
    "dataset_path", "protocol_path", "mlflow_tracking_uri",
}


def _parse_max_trials(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"0", "none", "unlimited", "forever"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max-trials must be positive or 'unlimited'")
    return parsed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run the iMa model self-optimizer loop")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run or dry-run an optimizer campaign")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--policy", choices=("local", "openrouter", "agentic"))
    run.add_argument(
        "--max-trials",
        type=_parse_max_trials,
        default=None,
        help="Completed trial budget; use 0 or 'unlimited' to drain the spec profile",
    )
    run.add_argument("--proposal-batch-size", type=int)
    run.add_argument(
        "--max-concurrent-trials",
        default=None,
        help="Parallel trial workers, or 'auto' to size from available CPU with headroom",
    )
    run.add_argument("--timeout-minutes", type=int)
    run.add_argument("--service-tier", choices=("flex",), help="OpenRouter service tier")
    run.add_argument("--openrouter-batch", action="store_true", default=None)
    run.add_argument("--model", help="Remote planner model, e.g. openai/gpt-5.6-luna")
    run.add_argument("--spec-profile", choices=("default", "long", "adaptive"))
    run.add_argument("--planner-mode", choices=("local", "fixture", "openrouter"))
    run.add_argument("--max-total-cost-usd", type=float)
    run.add_argument("--max-output-tokens", type=int)
    run.add_argument("--replan-every-terminal-trials", type=int)
    run.add_argument("--dataset-path", type=Path)
    run.add_argument("--protocol-path", type=Path)
    run.add_argument("--mlflow-tracking-uri")
    run.add_argument("--dry-run", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        values = _load_config(args.config)
        cli_values = {
            "policy": args.policy,
            "max_trials": args.max_trials,
            "proposal_batch_size": args.proposal_batch_size,
            "max_concurrent_trials": args.max_concurrent_trials,
            "timeout_minutes": args.timeout_minutes,
            "service_tier": args.service_tier,
            "openrouter_batch": args.openrouter_batch,
            "model": args.model,
            "spec_profile": args.spec_profile,
            "planner_mode": args.planner_mode,
            "max_total_cost_usd": args.max_total_cost_usd,
            "max_output_tokens": args.max_output_tokens,
            "replan_every_terminal_trials": args.replan_every_terminal_trials,
            "dataset_path": args.dataset_path,
            "protocol_path": args.protocol_path,
            "mlflow_tracking_uri": args.mlflow_tracking_uri,
        }
        values.update({key: value for key, value in cli_values.items() if value is not None})
        max_concurrent = values.get("max_concurrent_trials", 1)
        max_concurrent_trials = "auto" if max_concurrent == "auto" else int(max_concurrent)
        config = CampaignConfig(
            campaign_dir=args.campaign,
            policy=values.get("policy", "local"),
            max_trials=values.get("max_trials", 1),
            proposal_batch_size=int(values.get("proposal_batch_size", 1)),
            max_concurrent_trials=max_concurrent_trials,
            timeout_minutes=values.get("timeout_minutes"),
            service_tier=values.get("service_tier"),
            openrouter_batch=bool(values.get("openrouter_batch", False)),
            model=values.get("model", "openrouter/local-policy"),
            spec_profile=values.get("spec_profile", "default"),
            planner_mode=values.get("planner_mode", "local"),
            max_total_cost_usd=values.get("max_total_cost_usd"),
            max_output_tokens=int(values.get("max_output_tokens", 4000)),
            replan_every_terminal_trials=int(values.get("replan_every_terminal_trials", 32)),
            dataset_path=Path(values["dataset_path"]) if values.get("dataset_path") else None,
            protocol_path=Path(values["protocol_path"]) if values.get("protocol_path") else None,
            mlflow_tracking_uri=values.get("mlflow_tracking_uri"),
        )
        payload = run_campaign(config, dry=args.dry_run)
        print(_render(payload))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optimizer config must be a JSON object")
    unknown = sorted(set(payload) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown optimizer config keys: {unknown}")
    if payload.get("schema_version", 1) != 1:
        raise ValueError("optimizer config schema_version must be 1")
    return {key: value for key, value in payload.items() if key != "schema_version"}


def _render(payload: dict) -> str:
    if payload.get("mode") == "dry_run":
        proposals = payload["proposals"]
        lines = [
            f"Campaign: {payload['campaign']['campaign_dir']}",
            f"Mode: dry-run",
            f"Policy: {payload['campaign']['policy']}",
            f"Proposal batch: {len(proposals)}",
            "",
        ]
        for proposal in proposals:
            if "spec" in proposal:
                spec = proposal["spec"]
                lines.extend([
                    f"- {proposal['trial_id']} {spec['run_id']}",
                    f"  hypothesis: {proposal['hypothesis']}",
                    f"  surface: {proposal['changed_surface']}",
                    f"  parameters: {json.dumps(spec['parameters'], sort_keys=True)}",
                ])
            else:
                recipe = proposal["recipe"]
                lines.extend([
                    f"- {proposal['trial_id']} {proposal['recipe_hash']}",
                    f"  hypothesis: {proposal['hypothesis']}",
                    f"  axes: {', '.join(proposal['changed_axes'])}",
                    f"  recipe: {recipe['feature_schema']} / {recipe['model']['kind']}",
                ])
        lines.append("")
        lines.append("Wrote: dry-run.json")
        return "\n".join(lines)
    if payload.get("mode") == "executed":
        lines = [
            f"Campaign: {payload['campaign_dir']}",
            "Mode: executed",
            f"Cycles: {payload.get('cycles', 1)}",
            "",
        ]
        for result in payload["results"]:
            lines.append(f"- {result['trial_id']} {result['run_id']}: {result['status']}")
        lines.append("")
        lines.append("Wrote: trials.jsonl, decisions.jsonl, report.md")
        return "\n".join(lines)
    if payload.get("mode") == "time_budget_reached":
        return "\n".join([
            f"Campaign: {payload['campaign_dir']}",
            "Mode: time-budget-reached",
            f"Cycles: {payload.get('cycles', 0)}",
            f"Trials returned in this command: {len(payload.get('results', []))}",
            "",
            "Wrote: trials.jsonl, decisions.jsonl, report.md",
        ])
    if payload.get("mode") == "openrouter_batch_submitted":
        batch = payload["batch"]
        return "\n".join([
            f"Campaign: {payload['campaign']['campaign_dir']}",
            "Mode: openrouter-batch-submitted",
            f"Batch: {batch.get('id', '<unknown>')}",
            f"Status: {batch.get('status', '<unknown>')}",
            "",
            "Wrote: openrouter-batch.json",
        ])
    return json.dumps(payload, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
