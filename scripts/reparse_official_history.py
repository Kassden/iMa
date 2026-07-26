from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from scrapper.historical.results import parse_race


def race_number(path: Path) -> int:
    return int(path.stem.split(".")[0].removeprefix("R"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild normalized HKJC results from immutable raw HTML")
    parser.add_argument(
        "--root", type=Path, default=Path("data/historical/hkjc-2005-2025")
    )
    args = parser.parse_args()
    database = sqlite3.connect(args.root / "checkpoint.sqlite")
    report = {"venue_dates": 0, "races": 0, "runners": 0, "dead_heat_races": 0, "errors": []}
    for venue_dir in sorted((args.root / "raw").glob("*/*")):
        if not venue_dir.is_dir():
            continue
        race_date = venue_dir.parent.name
        venue = venue_dir.name
        destination = args.root / "normalized" / race_date / f"{venue}.json"
        existing = {}
        if destination.exists():
            existing = {
                int(race["race_no"]): race.get("source_url")
                for race in json.loads(destination.read_text(encoding="utf-8"))
            }
        races = []
        for path in sorted(venue_dir.glob("R*.html.gz"), key=race_number):
            race_no = race_number(path)
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    race = parse_race(handle.read())
            except (OSError, ValueError) as exc:
                report["errors"].append({"path": str(path), "error": str(exc)})
                continue
            runners = race.runners
            winners = sum(runner.place == 1 for runner in runners)
            report["dead_heat_races"] += int(winners > 1)
            report["races"] += 1
            report["runners"] += len(runners)
            races.append({
                "race_date": race_date,
                "venue": venue,
                "race_no": race_no,
                "source_url": existing.get(race_no),
                "race_class": race.race_class,
                "distance": race.distance,
                "prize": race.prize,
                "going": race.going,
                "course": race.course,
                "race_name": race.race_name,
                "runners": [asdict(runner) for runner in runners],
            })
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(races, indent=2), encoding="utf-8")
        temporary.replace(destination)
        database.execute(
            "UPDATE scans SET race_count = ?, runner_count = ? "
            "WHERE race_date = ? AND venue = ? AND status = 'complete'",
            (len(races), sum(len(race["runners"]) for race in races), race_date, venue),
        )
        report["venue_dates"] += 1
    database.commit()
    database.close()
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
