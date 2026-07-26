from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd

from ima.inference import legacy_live_frame, predict_live_pools
from ima.pools import SUPPORTED_POOLS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict ranked HKJC pool outcomes from a normalized live runner CSV"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runners", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--pools", nargs="+", default=list(SUPPORTED_POOLS))
    args = parser.parse_args()
    if args.top <= 0:
        parser.error("--top must be positive")

    artifact = joblib.load(args.model)
    live_frame = legacy_live_frame(pd.read_csv(args.runners))
    predictions = predict_live_pools(
        live_frame,
        artifact,
        pools=args.pools,
        top_n=args.top,
    )
    payload = {
        race_id: {
            pool: [asdict(item) | {"fair_odds": item.fair_odds} for item in combinations]
            for pool, combinations in pools.items()
        }
        for race_id, pools in predictions.items()
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
