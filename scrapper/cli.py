"""Command line entrypoint for live or fixture-backed race snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .history import HistoricalFeatureStore
from .pipeline import build_model_rows, normalize_snapshot, scrape_race, write_snapshot


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Fetch HKJC runners and odds into the iMa model schema")
    result.add_argument("--race", type=int, required=True, help="Race number")
    result.add_argument("--date", help="HKJC meeting date")
    result.add_argument("--venue", help="HKJC venue code, such as ST or HV")
    result.add_argument("--endpoint", help="Override the HKJC GraphQL endpoint")
    result.add_argument(
        "--provider",
        choices=("browser", "graphql"),
        default="browser",
        help="Odds provider; browser is required when HKJC rejects direct HTTP",
    )
    result.add_argument("--fixture", type=Path, help="Use a recorded GraphQL JSON response")
    result.add_argument("--output", type=Path, default=Path("scrapper/snapshots"))
    result.add_argument("--runs", type=Path, default=Path("track/hkracing 2/runs.csv"))
    result.add_argument("--races", type=Path, default=Path("track/hkracing 2/races.csv"))
    result.add_argument("--identity-map", type=Path)
    result.add_argument("--without-history", action="store_true")
    result.add_argument("--without-horse-pages", action="store_true")
    result.add_argument(
        "--all-pools", action="store_true",
        help="Interact with exotic and multi-race tabs and preserve their live odds tables",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    history = None
    if not args.without_history:
        history = HistoricalFeatureStore.from_csv(args.runs, args.races, args.identity_map)

    if args.fixture:
        payload = json.loads(args.fixture.read_text(encoding="utf-8"))
        runners = normalize_snapshot(payload, args.race)
        model_rows = build_model_rows(runners, history)
        result = write_snapshot(payload, runners, model_rows, args.output)
    else:
        result = scrape_race(
            race_no=args.race,
            output_dir=args.output,
            race_date=args.date,
            venue_code=args.venue,
            endpoint=args.endpoint,
            history=history,
            scrape_horse_pages=not args.without_horse_pages,
            provider=args.provider,
            include_all_pools=args.all_pools,
        )

    print(f"runners={result.runner_count}")
    print(f"prediction_ready={result.prediction_ready_count}")
    print(f"model_csv={result.model_path}")
    print(f"report={result.report_path}")
    print(f"horse_details={result.horse_details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
