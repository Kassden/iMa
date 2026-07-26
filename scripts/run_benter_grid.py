from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from ima.auxiliary import train_auxiliary_bundle
from ima.data import chronological_race_split
from ima.experiments import (
    ExperimentSpec, default_experiment_specs, merge_run_history, render_dashboard, results_frame,
    run_experiments,
)
from ima.feature_sets import FEATURE_SCHEMAS, NOTEBOOK_RICH_SCHEMA
from ima.rich_features import load_full_rich_history


def selected_experiment_specs(run_ids: list[str] | None) -> list[ExperimentSpec]:
    defaults = {spec.run_id: spec for spec in default_experiment_specs()}
    selected = list(defaults.values()) if not run_ids else [defaults[run_id] for run_id in run_ids]
    return [
        ExperimentSpec(f"notebook-{spec.run_id}", spec.kind, spec.parameters)
        for spec in selected
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full rich-feature parameter grid")
    parser.add_argument("--legacy-runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--legacy-races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--canonical", type=Path, default=Path("data/processed/historical/runners.csv.gz"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/historical"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/benter-rich-grid"))
    parser.add_argument("--public", type=Path, default=Path("public"))
    parser.add_argument(
        "--template", type=Path, default=Path("docs/model-results/dashboard-template.html"),
    )
    parser.add_argument(
        "--schema", choices=sorted(FEATURE_SCHEMAS), default=NOTEBOOK_RICH_SCHEMA.name,
    )
    parser.add_argument(
        "--run-id", action="append", choices=[spec.run_id for spec in default_experiment_specs()],
        help="Run only the selected default-grid specification; repeat for multiple runs.",
    )
    parser.add_argument(
        "--skip-auxiliary", action="store_true",
        help="Keep the existing published auxiliary bundle instead of retraining it.",
    )
    args = parser.parse_args()
    feature_schema = FEATURE_SCHEMAS[args.schema]

    rich = load_full_rich_history(
        args.legacy_runs, args.legacy_races, args.canonical,
        args.processed / "trackwork.csv.gz", args.processed / "barrier-trials.csv.gz",
        args.processed / "sectionals.csv.gz",
        args.processed / "horse-snapshots.csv.gz",
    )
    specs = selected_experiment_specs(args.run_id)
    rich_summary = run_experiments(
        rich, args.output, args.template, specs=specs, feature_schema=feature_schema,
    )
    if not args.skip_auxiliary:
        auxiliary = train_auxiliary_bundle(chronological_race_split(rich), feature_schema)
        joblib.dump(auxiliary, args.output / "models" / "auxiliary.joblib")
        rich_summary["auxiliary_predictions"] = auxiliary.report()
    (args.output / "results.json").write_text(
        json.dumps(rich_summary, indent=2), encoding="utf-8"
    )
    render_dashboard(rich_summary, args.template, args.output / "dashboard.html")

    public_results = args.public / "results.json"
    summary = json.loads(public_results.read_text(encoding="utf-8"))
    previous_history = summary.get("run_history", summary["runs"])
    baseline_runs = [
        run for run in summary["runs"] if run.get("feature_schema") != feature_schema.name
    ]
    rich_runs = rich_summary["runs"]
    offset = len(baseline_runs)
    for run in rich_runs:
        run["sequence"] += offset
    summary["runs"] = [*baseline_runs, *rich_runs]
    summary["run_history"] = merge_run_history(
        previous_history,
        rich_summary.get("run_history", rich_runs),
        default_execution_id=summary.get("created_at"),
    )
    summary["updated_at"] = rich_summary["created_at"]
    if "auxiliary_predictions" in rich_summary:
        summary["auxiliary_predictions"] = rich_summary["auxiliary_predictions"]
    summary["rich_grid"] = {
        "schema": feature_schema.name,
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
