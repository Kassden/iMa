from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.data import build_full_history_dataset
from ima.experiments import run_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 1997-2025 iMa model experiment grid")
    parser.add_argument("--legacy-runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--legacy-races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument(
        "--canonical-runners", type=Path,
        default=Path("data/processed/historical/runners.csv.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/full-history"))
    parser.add_argument(
        "--template", type=Path, default=Path("docs/model-results/dashboard-template.html")
    )
    args = parser.parse_args()
    frame = build_full_history_dataset(
        args.legacy_runs, args.legacy_races, args.canonical_runners
    )
    summary = run_experiments(frame, args.output, args.template)
    best = summary["runs"][0]
    print(json.dumps({
        "output": str(args.output),
        "runs": len(summary["runs"]),
        "best_fundamental_run": best["run_id"],
        "best_fundamental": best["test_fundamental"],
        "market_test": summary["market_test"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
