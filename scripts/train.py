from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.training import train_and_report, train_canonical_and_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train race-aware iMa probability models")
    parser.add_argument("--runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/latest"))
    parser.add_argument(
        "--canonical-runners", type=Path,
        help="Consolidated runners.csv.gz; when set, ignores --runs and --races",
    )
    args = parser.parse_args()
    if args.canonical_runners:
        report = train_canonical_and_report(args.canonical_runners, args.output)
    else:
        report = train_and_report(args.runs, args.races, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
