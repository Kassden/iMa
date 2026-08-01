from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.experiments import render_dashboard, results_frame
from ima.pipeline_transparency import pipeline_manifest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_simulator_report(report: dict) -> dict:
    compact_races = []
    for race in report.get("races", []):
        compact_races.append({
            "race_no": race.get("race_no"),
            "prediction_basis": race.get("prediction_basis", {}),
            "candidate_formula": race.get("candidate_formula", {}),
            "summary": race.get("summary", {}),
            "recommendations": race.get("recommendations", []),
            "priced_candidates": race.get("priced_candidates", []),
            "auxiliary_predictions": race.get("auxiliary_predictions", []),
        })
    race = compact_races[0] if compact_races else None
    return {
        "requested_date": report.get("requested_date"),
        "displayed_date": report.get("displayed_date"),
        "date_status": report.get("date_status"),
        "meeting": report.get("meeting", {}),
        "races": compact_races,
        "race": race,
    }


def compact_notebook_history(report: dict) -> dict:
    runs = report.get("runs", [])
    source = runs[0] if runs else None
    run = None if source is None else {
        "run_id": source.get("run_id"),
        "kind": source.get("kind"),
        "parameters": source.get("parameters", {}),
        "duration_seconds": source.get("duration_seconds"),
        "feature_schema": source.get("feature_schema"),
        "feature_count": source.get("feature_count"),
        "fundamental_weight": source.get("fundamental_weight"),
        "market_weight": source.get("market_weight"),
        "place_market_weight": source.get("place_market_weight", 0.0),
        "incremental_pseudo_r2": source.get("incremental_pseudo_r2"),
        "test_fundamental": source.get("test_fundamental", {}),
        "test_blended": source.get("test_blended", {}),
    }
    return {
        "created_at": report.get("created_at"),
        "dataset": report.get("dataset", {}),
        "run": run,
    }


def publish_latest_dashboard(
    results_path: Path,
    template_path: Path,
    output_path: Path,
    csv_path: Path,
    auxiliary_path: Path,
    winner_path: Path,
    simulator_path: Path,
    notebook_history_path: Path | None = None,
) -> dict:
    summary = _read_json(results_path)
    summary["pipeline_manifest"] = pipeline_manifest()
    summary["auxiliary_predictions"] = _read_json(auxiliary_path)
    summary["notebook_rich_benchmark"] = _read_json(winner_path)
    summary["simulator"] = compact_simulator_report(_read_json(simulator_path))
    if notebook_history_path and notebook_history_path.exists():
        summary["notebook_full_history"] = compact_notebook_history(
            _read_json(notebook_history_path)
        )
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results_frame(summary).to_csv(csv_path, index=False)
    render_dashboard(summary, template_path, output_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the latest iMa static dashboard")
    parser.add_argument("--results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--template", type=Path, default=Path("docs/model-results/dashboard-template.html"))
    parser.add_argument("--output", type=Path, default=Path("public/index.html"))
    parser.add_argument("--csv", type=Path, default=Path("public/results.csv"))
    parser.add_argument(
        "--auxiliary", type=Path,
        default=Path("artifacts/smoke/notebook-rich-auxiliary/report.json"),
    )
    parser.add_argument(
        "--winner", type=Path,
        default=Path("artifacts/smoke/notebook-rich-winner/report.json"),
    )
    parser.add_argument(
        "--simulator", type=Path,
        default=Path("artifacts/simulations/next-available/2026-07-27.json"),
    )
    parser.add_argument(
        "--notebook-history", type=Path,
        default=Path("artifacts/experiments/notebook-rich-v2/results.json"),
    )
    args = parser.parse_args()
    publish_latest_dashboard(
        args.results, args.template, args.output, args.csv,
        args.auxiliary, args.winner, args.simulator, args.notebook_history,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
