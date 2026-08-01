from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ima.experiments import render_dashboard, results_frame
from ima.mlflow_tracking import DEFAULT_EXPERIMENT_NAME, MLflowUnavailableError
from ima.pipeline_transparency import pipeline_manifest


def _mlflow():
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore
    except Exception as exc:
        raise MLflowUnavailableError(
            "MLflow export requires the mlflow package. Install with python3.11 -m pip install -e ."
        ) from exc
    return mlflow, MlflowClient


def _read_json_artifact(mlflow: Any, tracking_uri: str, run_id: str, artifact_path: str) -> dict:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        tracking_uri=tracking_uri,
    )
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def export_mlflow_runs(
    tracking_uri: str,
    experiment_name: str,
    base_results_path: Path,
) -> dict:
    mlflow, client_class = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    client = client_class(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
    )
    exported_runs = []
    dataset = None
    market_test = None
    for active in runs:
        try:
            run = _read_json_artifact(mlflow, tracking_uri, active.info.run_id, "run.json")
        except Exception:
            continue
        run["mlflow_run_id"] = active.info.run_id
        run["mlflow_experiment_id"] = active.info.experiment_id
        exported_runs.append(run)
        if dataset is None or market_test is None:
            try:
                context = _read_json_artifact(mlflow, tracking_uri, active.info.run_id, "context.json")
            except Exception:
                context = {}
            dataset = dataset or context.get("dataset")
            market_test = market_test or context.get("market_test")

    base = json.loads(base_results_path.read_text(encoding="utf-8")) if base_results_path.exists() else {}
    if exported_runs:
        base["runs"] = exported_runs
        base["run_history"] = exported_runs
    base["dataset"] = dataset or base.get("dataset", {})
    base["market_test"] = market_test or base.get("market_test", {})
    base["pipeline_manifest"] = pipeline_manifest()
    base["mlflow"] = {
        "enabled": True,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "exported_runs": len(exported_runs),
    }
    return base


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export MLflow-tracked iMa runs into the static dashboard contract"
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--base-results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--csv", type=Path, default=Path("public/results.csv"))
    parser.add_argument("--template", type=Path, default=Path("docs/model-results/dashboard-template.html"))
    parser.add_argument("--output", type=Path, default=Path("public/index.html"))
    args = parser.parse_args()
    summary = export_mlflow_runs(args.tracking_uri, args.experiment, args.base_results)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_frame(summary).to_csv(args.csv, index=False)
    render_dashboard(summary, args.template, args.output)
    print(json.dumps(summary["mlflow"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

