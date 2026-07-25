from __future__ import annotations

import argparse
from pathlib import Path

from scrapper.historical.results import collect_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one official HKJC historical result")
    parser.add_argument("--date", required=True, help="YYYY/MM/DD")
    parser.add_argument("--venue", required=True, choices=("ST", "HV"))
    parser.add_argument("--race", required=True, type=int)
    parser.add_argument("--output", type=Path, default=Path("data/historical/hkjc"))
    args = parser.parse_args()
    result = collect_result(args.date, args.venue, args.race, args.output)
    print(f"runners={len(result['runners'])}")
    print(f"source={result['source_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
