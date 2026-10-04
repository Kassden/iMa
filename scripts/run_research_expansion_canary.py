"""Run the V6 feedback controller on a small, explicitly synthetic race fixture."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ima.optimizer import CampaignConfig
from ima.research_expansion import run_expansion_campaign


def build_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in range(80):
        for horse in range(6):
            result = (horse - day) % 6 + 1
            rows.append({
                "race_id": f"r{day}", "horse_no": horse + 1,
                "horse_id": f"h{horse}", "jockey_id": f"j{horse % 3}",
                "trainer_id": f"t{horse % 2}", "race_no": 1,
                "date": (pd.Timestamp("2020-01-01") + pd.Timedelta(days=day * 2)).isoformat(),
                "finish_seconds": 70 + horse + np.sin(day),
                "finish_time": 70 + horse + np.sin(day), "distance": 1200,
                "lengths_raw": horse, "actual_weight": 120 + horse,
                "horse_age": 3 + horse % 4, "horse_rating": 50 + horse,
                "declared_weight": 1100 + horse * 20, "draw": horse + 1,
                "race_class": 3, "surface": 0, "venue": "ST", "course": "TURF",
                "going": "GOOD", "field_size": 6, "result": result,
                "target_win": int(result == 1), "target_probability": float(result == 1),
                "market_probability": 1 / 6, "win_odds": 5 + horse,
                "last_speed_ratio": horse / 10, "prior_win_rate": .1 + horse / 100,
            })
    dataset = root / "fixture.csv"
    pd.DataFrame(rows).to_csv(dataset, index=False)
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({
        "min_train_races": 30, "calibration_races": 10, "score_races": 10,
        "max_folds": 2, "whole_meeting_boundaries": True,
    }), encoding="utf-8")
    return dataset, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner", choices=("fixture", "openrouter"), default="fixture")
    parser.add_argument("--max-concurrent-trials", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--mlflow-tracking-uri")
    args = parser.parse_args()
    # Synthetic canaries intentionally omit many of the production history columns.
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    dataset, protocol = build_fixture(args.output)
    config = CampaignConfig(
        campaign_dir=args.output / "campaign", policy="agentic", research_policy="expansion_v6",
        planner_mode=args.planner, model=args.model, max_total_cost_usd=1,
        max_trials=args.max_trials, proposal_batch_size=args.max_trials,
        max_concurrent_trials=min(args.max_concurrent_trials, args.max_trials),
        dataset_path=dataset, protocol_path=protocol, cpu_thread_budget=2,
        ram_budget_gib=8, memory_budget_gb_decimal=10,
        host_reserve_cpu_threads=1, host_reserve_ram_gib=.5,
        planning_checkpoint_seconds=30, replan_every_terminal_trials=4,
        max_consecutive_failed_trials=3, mlflow_tracking_uri=args.mlflow_tracking_uri,
        timeout_minutes=10,
    )
    result = run_expansion_campaign(config)
    print(json.dumps(result, indent=2, default=str))
    counts = result.get("ledger", {})
    if result.get("mode") != "complete" or counts.get("completed", 0) != args.max_trials:
        raise SystemExit("V6 canary did not complete its declared attempt count")
    if (counts.get("failed", 0) or counts.get("pending_tells", 0)
            or (args.mlflow_tracking_uri and counts.get("pending_tracking", 0))):
        raise SystemExit("V6 canary contains failures or undelivered results")


if __name__ == "__main__":
    main()
