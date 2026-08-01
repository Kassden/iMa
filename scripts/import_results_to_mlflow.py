from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.mlflow_tracking import DEFAULT_EXPERIMENT_NAME, import_runs_from_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import existing iMa dashboard run history into an MLflow experiment"
    )
    parser.add_argument("--results", type=Path, default=Path("public/results.json"))
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--models-root",
        type=Path,
        help="Optional directory containing <run_id>.joblib artifacts to log with imported runs.",
    )
    args = parser.parse_args()
    report = import_runs_from_results(
        args.results,
        args.tracking_uri,
        args.experiment,
        args.models_root,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

