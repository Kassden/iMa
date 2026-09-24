from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.research_evidence import (
    DEFAULT_PRIMARY_METRIC,
    mlflow_sample_readback,
    summarize_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize optimizer campaign evidence without mutating the campaign."
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize_campaign(args.campaign_dir, args.primary_metric)
    if args.mlflow_tracking_uri:
        summary["mlflow_readback"] = mlflow_sample_readback(
            summary["selected_trials"],
            args.mlflow_tracking_uri,
        )

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
