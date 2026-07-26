from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ima.domain import PoolCandidate
from ima.execution import PaperExecutor
from ima.inference import legacy_live_frame, predict_live_win
from ima.strategy import RiskBudget, recommend_wagers
from scrapper.pipeline import scrape_race


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one live race through forward paper trading")
    parser.add_argument("--race", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/benchmark/logit.joblib"))
    parser.add_argument("--bankroll", type=float, default=10000.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper"))
    parser.add_argument("--without-horse-pages", action="store_true")
    args = parser.parse_args()
    scrape = scrape_race(
        args.race,
        Path("scrapper/snapshots"),
        history=None,
        scrape_horse_pages=not args.without_horse_pages,
        include_all_pools=True,
    )
    live = legacy_live_frame(pd.read_csv(scrape.model_path))
    artifact = joblib.load(args.model)
    probability = predict_live_win(live, artifact)
    candidates = [
        PoolCandidate(
            race_id=str(row.race_id), pool="WIN", combination=(str(row.horse_no),),
            probability=float(probability[index]), current_decimal_odds=float(row.win_odds),
            model_version=args.model.stem,
        )
        for index, row in live.iterrows()
        if row.win_odds > 1
    ]
    recommendations = recommend_wagers(candidates, args.bankroll, RiskBudget())
    executor = PaperExecutor(args.output / "orders.jsonl")
    receipts = [executor.submit(item) for item in recommendations]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "race_id": str(live["race_id"].iloc[0]),
        "model": str(args.model),
        "runners": [
            {
                "horse_no": candidate.combination[0],
                "ratable": bool(live.iloc[index]["ratable"]),
                "probability": candidate.probability,
                "market_probability": candidate.market_probability,
                "fair_odds": candidate.fair_odds,
                "market_odds": candidate.current_decimal_odds,
                "discrepancy": candidate.probability_discrepancy,
            }
            for index, candidate in enumerate(candidates)
        ],
        "recommendations": [item.__dict__ for item in recommendations],
        "receipts": [item.__dict__ for item in receipts],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{report['race_id'].replace(':', '_')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
