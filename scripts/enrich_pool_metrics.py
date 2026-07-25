from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from ima.data import build_full_history_dataset, chronological_race_split
from ima.experiments import merge_run_history, render_dashboard, results_frame
from ima.pools import evaluate_top_pool_selections
from ima.rich_features import load_full_rich_history


def pool_metrics_for_artifact(frame, artifact: dict) -> dict:
    splits = chronological_race_split(frame)
    fundamental = artifact["calibrator"].transform(
        artifact["model"].predict_proba(splits.test), splits.test["race_id"]
    )
    combined = artifact["blend"].transform(
        fundamental,
        splits.test["market_probability"].to_numpy(),
        splits.test["race_id"],
    )
    pool_frame = splits.test.assign(
        fundamental_probability=fundamental,
        blended_probability=combined,
    )
    exponents = artifact["order_exponents"]
    return {
        "fundamental": evaluate_top_pool_selections(
            pool_frame, "fundamental_probability", exponents,
        ),
        "combined": evaluate_top_pool_selections(
            pool_frame, "blended_probability", exponents,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Add held-out pool metrics to saved experiment runs")
    parser.add_argument("--results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--output", type=Path, default=Path("public"))
    parser.add_argument(
        "--template", type=Path, default=Path("docs/model-results/dashboard-template.html")
    )
    args = parser.parse_args()

    summary = json.loads(args.results.read_text(encoding="utf-8"))
    execution_by_schema = {}
    for schema, report_path in {
        "baseline-v1": Path("artifacts/experiments/full-history/results.json"),
        "benter-rich-v1": Path("artifacts/experiments/benter-rich-grid/results.json"),
    }.items():
        execution_by_schema[schema] = json.loads(
            report_path.read_text(encoding="utf-8")
        )["created_at"]
    for run in summary["runs"]:
        schema = run.get("feature_schema", "baseline-v1")
        run.setdefault("execution_id", execution_by_schema[schema])
        run.setdefault("run_key", f"{run['execution_id']}|{schema}|{run['run_id']}")
    summary["run_history"] = merge_run_history(
        summary.get("run_history", []), summary["runs"],
    )
    baseline = build_full_history_dataset(
        Path("track/hkracing 2/runs.csv"),
        Path("track/hkracing 2/races.csv"),
        Path("data/processed/historical/runners.csv.gz"),
    )
    rich = load_full_rich_history(
        Path("track/hkracing 2/runs.csv"),
        Path("track/hkracing 2/races.csv"),
        Path("data/processed/historical/runners.csv.gz"),
        Path("data/processed/historical/trackwork.csv.gz"),
        Path("data/processed/historical/barrier-trials.csv.gz"),
        Path("data/processed/historical/sectionals.csv.gz"),
    )
    model_roots = {
        "baseline-v1": Path("artifacts/experiments/full-history/models"),
        "benter-rich-v1": Path("artifacts/experiments/benter-rich-grid/models"),
    }
    frames = {"baseline-v1": baseline, "benter-rich-v1": rich}
    metrics_by_identity = {}
    for run in summary["runs"]:
        schema = run.get("feature_schema", "baseline-v1")
        artifact_path = model_roots[schema] / f"{run['run_id']}.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        metrics = pool_metrics_for_artifact(frames[schema], joblib.load(artifact_path))
        run["pool_metrics"] = metrics
        metrics_by_identity[(schema, run["run_id"])] = metrics

    for run in summary.get("run_history", []):
        identity = (run.get("feature_schema", "baseline-v1"), run["run_id"])
        if identity in metrics_by_identity:
            run["pool_metrics"] = metrics_by_identity[identity]

    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_frame(summary).to_csv(args.output / "results.csv", index=False)
    render_dashboard(summary, args.template, args.output / "index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
