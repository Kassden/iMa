from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.experiments import (
    ExperimentSpec, default_experiment_specs, render_dashboard, results_frame, run_experiments,
)
from ima.feature_sets import RICH_SCHEMA
from ima.rich_features import load_full_rich_history


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full parameter grid with benter-rich-v1")
    parser.add_argument("--legacy-runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--legacy-races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/processed/historical/runners.csv.gz"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/historical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/benter-rich-grid"))
    parser.add_argument("--public", type=Path, default=Path("public"))
    parser.add_argument(
        "--template", type=Path, default=Path("docs/model-results/dashboard-template.html"),
    )
    args = parser.parse_args()

    rich = load_full_rich_history(
        args.legacy_runs, args.legacy_races, args.canonical,
        args.processed / "trackwork.csv.gz", args.processed / "barrier-trials.csv.gz",
        args.processed / "sectionals.csv.gz",
    )
    specs = [
        ExperimentSpec(f"benter-{spec.run_id}", spec.kind, spec.parameters)
        for spec in default_experiment_specs()
    ]
    rich_summary = run_experiments(
        rich, args.output, args.template, specs=specs, feature_schema=RICH_SCHEMA,
    )

    public_results = args.public / "results.json"
    summary = json.loads(public_results.read_text(encoding="utf-8"))
    baseline_runs = [run for run in summary["runs"] if run.get("feature_schema") != RICH_SCHEMA.name]
    rich_runs = rich_summary["runs"]
    offset = len(baseline_runs)
    for run in rich_runs:
        run["sequence"] += offset
    summary["runs"] = [*baseline_runs, *rich_runs]
    summary["rich_grid"] = {
        "schema": RICH_SCHEMA.name,
        "run_count": len(rich_runs),
        "output": str(args.output),
    }
    public_results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_frame(summary).to_csv(args.public / "results.csv", index=False)
    render_dashboard(summary, args.template, args.public / "index.html")

    best = min(rich_runs, key=lambda run: run["test_fundamental"]["race_log_loss"])
    print(json.dumps({
        "runs": len(rich_runs),
        "best_run": best["run_id"],
        "top1": best["test_fundamental"]["top_pick_win_rate"],
        "top3": best["test_fundamental"]["winner_top3_rate"],
        "race_log_loss": best["test_fundamental"]["race_log_loss"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
