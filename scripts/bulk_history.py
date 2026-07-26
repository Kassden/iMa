from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from scrapper.historical.bulk import BulkArchiveCollector


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume official HKJC historical archive collection")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2005, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--output", type=Path, default=Path("data/historical/hkjc-2005-2025"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--meeting-index", type=Path,
        help="CSV or CSV.GZ with race_date and venue columns; scan only known meetings",
    )
    args = parser.parse_args()
    collector = BulkArchiveCollector(args.output, args.workers, args.delay)
    if args.meeting_index:
        frame = pd.read_csv(args.meeting_index, usecols=["race_date", "venue"])
        frame["race_date"] = pd.to_datetime(frame["race_date"]).dt.date.astype(str)
        targets = list(frame[["race_date", "venue"]].drop_duplicates().itertuples(index=False, name=None))
        report = collector.run_targets(targets, args.limit, known_meetings=True)
    else:
        report = collector.run(args.start, args.end, args.limit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
