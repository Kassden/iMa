from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.experiments import render_dashboard, results_frame
from ima.pipeline_transparency import pipeline_manifest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_simulator_report(report: dict) -> dict:
    races = report.get("races", [])
    race = races[0] if races else {}
    return {
        "requested_date": report.get("requested_date"),
        "displayed_date": report.get("displayed_date"),
        "date_status": report.get("date_status"),
        "meeting": report.get("meeting", {}),
        "race": {
            "race_no": race.get("race_no"),
            "prediction_basis": race.get("prediction_basis", {}),
            "summary": race.get("summary", {}),
            "recommendations": race.get("recommendations", []),
        } if race else None,
    }


def publish_latest_dashboard(
    results_path: Path,
    template_path: Path,
    output_path: Path,
    csv_path: Path,
    auxiliary_path: Path,
    winner_path: Path,
    simulator_path: Path,
) -> dict:
    summary = _read_json(results_path)
    summary["pipeline_manifest"] = pipeline_manifest()
    summary["auxiliary_predictions"] = _read_json(auxiliary_path)
    summary["notebook_rich_benchmark"] = _read_json(winner_path)
    summary["simulator"] = compact_simulator_report(_read_json(simulator_path))
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
    args = parser.parse_args()
    publish_latest_dashboard(
        args.results, args.template, args.output, args.csv,
        args.auxiliary, args.winner, args.simulator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
