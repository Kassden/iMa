from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from ima.inference import legacy_live_frame
from ima.simulator import simulate_race
from ima.strategy import RiskBudget
from scrapper.pipeline import scrape_race


def _tomorrow_hk() -> str:
    return (datetime.now(ZoneInfo("Asia/Hong_Kong")).date() + timedelta(days=1)).isoformat()


def _meeting_summary(races: list[dict], bankroll: float, requested_date: str) -> dict:
    cost = sum(race["summary"]["total_cost"] for race in races)
    gross = sum(race["summary"]["expected_gross_return"] for race in races)
    return {
        "requested_date": requested_date,
        "races_simulated": len(races),
        "starting_bankroll": bankroll,
        "total_cost": cost,
        "remaining_bankroll": bankroll - cost,
        "expected_wins": sum(race["summary"]["expected_wins"] for race in races),
        "expected_gross_return": gross,
        "expected_net_return": gross - cost,
        "expected_roi": (gross - cost) / cost if cost else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate $10-unit wagers for tomorrow's displayed HKJC meeting"
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("artifacts/experiments/benter-rich-grid/models/benter-boost-lr003-leaf15.joblib"),
    )
    parser.add_argument("--auxiliary-model", type=Path)
    parser.add_argument("--date", default=_tomorrow_hk())
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--max-race", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulations/tomorrow"))
    parser.add_argument("--with-horse-pages", action="store_true")
    parser.add_argument("--allow-next-available", action="store_true")
    args = parser.parse_args()
    if args.bankroll < 10:
        parser.error("--bankroll must be at least $10")
    if args.max_race <= 0:
        parser.error("--max-race must be positive")

    winner_artifact = joblib.load(args.model)
    auxiliary = joblib.load(args.auxiliary_model) if args.auxiliary_model else None
    budget = RiskBudget(
        max_race_fraction=0.05,
        max_combination_fraction=0.01,
        minimum_edge=0.02,
        minimum_stake=10.0,
        stake_increment=10.0,
    )
    snapshot_dir = args.output / "snapshots"
    races = []
    errors = []
    remaining = args.bankroll
    displayed_date = None

    for race_no in range(1, args.max_race + 1):
        try:
            scrape = scrape_race(
                race_no,
                snapshot_dir,
                history=None,
                scrape_horse_pages=args.with_horse_pages,
                include_all_pools=True,
            )
        except Exception as exc:
            errors.append({"race_no": race_no, "error": str(exc)})
            if race_no == 1:
                break
            continue
        raw = json.loads(scrape.raw_path.read_text(encoding="utf-8"))
        live = legacy_live_frame(pd.read_csv(scrape.model_path))
        race_date = json.loads(scrape.report_path.read_text(encoding="utf-8"))["race_date"]
        displayed_date = displayed_date or race_date
        if race_date != args.date and not args.allow_next_available:
            errors.append({
                "race_no": race_no,
                "error": f"Displayed meeting is {race_date}, not requested tomorrow {args.date}",
            })
            break
        report = simulate_race(
            live,
            winner_artifact,
            raw,
            remaining,
            args.model.stem,
            budget,
            auxiliary,
        )
        report["race_no"] = race_no
        report["race_date"] = race_date
        report["snapshot"] = {
            "raw": str(scrape.raw_path),
            "runners": str(scrape.runners_path),
            "model": str(scrape.model_path),
        }
        races.append(report)
        remaining -= report["summary"]["total_cost"]
        if remaining < 10:
            break

    payload = {
        "created_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "mode": "paper_simulation_only",
        "requested_date": args.date,
        "displayed_date": displayed_date,
        "date_status": (
            "matched" if displayed_date == args.date
            else "next_available_allowed" if displayed_date and args.allow_next_available
            else "requested_date_unavailable"
        ),
        "model": str(args.model),
        "auxiliary_model": str(args.auxiliary_model) if args.auxiliary_model else None,
        "meeting": _meeting_summary(races, args.bankroll, args.date),
        "races": races,
        "errors": errors,
        "submission": "disabled",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.date}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if races else 2


if __name__ == "__main__":
    raise SystemExit(main())
