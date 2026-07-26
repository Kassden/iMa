from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.operations import finalize_meeting


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize an HKJC meeting and train challengers")
    parser.add_argument("--date", required=True, help="YYYY/MM/DD")
    parser.add_argument("--venue", required=True, choices=("ST", "HV"))
    parser.add_argument("--output", type=Path, default=Path("data/meetings"))
    parser.add_argument("--runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    parser.add_argument("--races", type=Path, default=Path("track/hkracing 2/races.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/models/meetings"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    args = parser.parse_args()
    result = finalize_meeting(
        args.date, args.venue, args.output, args.runs, args.races, args.artifacts, args.registry
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
