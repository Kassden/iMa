from __future__ import annotations

import argparse
import json
from pathlib import Path

from ima.data import build_runner_dataset, chronological_race_split, validate_runner_dataset
from scrapper.browser_odds import BrowserOddsClient
from scrapper.pipeline import normalize_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the iMa runtime and optionally live HKJC access")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--race", type=int, default=1)
    args = parser.parse_args()
    frame = build_runner_dataset(
        Path("track/hkracing 2/runs.csv"), Path("track/hkracing 2/races.csv")
    )
    validate_runner_dataset(frame)
    splits = chronological_race_split(frame)
    result = {
        "dataset_runners": len(frame),
        "dataset_races": int(frame["race_id"].nunique()),
        "test_races": int(splits.test["race_id"].nunique()),
        "live": None,
    }
    if args.live:
        payload = BrowserOddsClient().fetch_race_snapshot(args.race)
        runners = normalize_snapshot(payload, args.race)
        result["live"] = {
            "runner_count": len(runners),
            "win_odds_count": sum(runner.win_odds is not None for runner in runners),
            "place_odds_count": sum(runner.place_odds is not None for runner in runners),
            "provider": payload.get("provider"),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
