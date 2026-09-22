from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.optimizer import CampaignConfig, run_campaign


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run the iMa model self-optimizer loop")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run or dry-run an optimizer campaign")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--policy", choices=("local", "openrouter"), default="local")
    run.add_argument("--max-trials", type=int, default=1)
    run.add_argument("--proposal-batch-size", type=int, default=1)
    run.add_argument("--max-concurrent-trials", type=int, default=1)
    run.add_argument("--timeout-minutes", type=int)
    run.add_argument("--service-tier", choices=("flex",), help="OpenRouter service tier")
    run.add_argument("--openrouter-batch", action="store_true")
    run.add_argument("--model", help="Remote planner model, e.g. openai/gpt-5.6-luna")
    run.add_argument("--dry-run", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "run":
        config = CampaignConfig(
            campaign_dir=args.campaign,
            policy=args.policy,
            max_trials=args.max_trials,
            proposal_batch_size=args.proposal_batch_size,
            max_concurrent_trials=args.max_concurrent_trials,
            timeout_minutes=args.timeout_minutes,
            service_tier=args.service_tier,
            openrouter_batch=args.openrouter_batch,
            model=args.model or "openrouter/local-policy",
        )
        payload = run_campaign(config, dry=args.dry_run)
        print(_render(payload))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


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
            spec = proposal["spec"]
            lines.extend([
                f"- {proposal['trial_id']} {spec['run_id']}",
                f"  hypothesis: {proposal['hypothesis']}",
                f"  surface: {proposal['changed_surface']}",
                f"  parameters: {json.dumps(spec['parameters'], sort_keys=True)}",
            ])
        lines.append("")
        lines.append("Wrote: dry-run.json")
        return "\n".join(lines)
    if payload.get("mode") == "executed":
        lines = [f"Campaign: {payload['campaign_dir']}", "Mode: executed", ""]
        for result in payload["results"]:
            lines.append(f"- {result['trial_id']} {result['run_id']}: {result['status']}")
        lines.append("")
        lines.append("Wrote: trials.jsonl, decisions.jsonl, report.md")
        return "\n".join(lines)
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
