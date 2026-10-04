"""Replay committed champion recipes without modifying their source campaigns."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_executor import RecipeExecutionRequest, execute_recipe, _write_json_atomic
from ima.research_specs import PipelineRecipe, V6_PORTFOLIO_VERSION
from ima.research_telemetry import comparison_key
from ima.research_controller import protocol_spec_parameters


def read_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return protocol_spec_parameters(payload)


def read_champions(campaign: Path, *, attempt_ids=()) -> list[dict]:
    ledger = (campaign / "ledger.sqlite").resolve()
    if not ledger.is_file():
        raise FileNotFoundError(ledger)
    with sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT attempt_id,payload_json,result_json FROM attempts WHERE status='completed'"
        ).fetchall()
    candidates = {}
    for attempt_id, payload_json, result_json in rows:
        if attempt_ids and attempt_id not in attempt_ids:
            continue
        payload, result = json.loads(payload_json), json.loads(result_json)
        value = result.get("objective_value")
        if value is None or not math.isfinite(value):
            continue
        recipe = payload["recipe"]
        # Group only within original evaluation contracts, never blend market/fundamental scores.
        key = comparison_key(dict(result, attempt_id=attempt_id), recipe) + ":" + recipe["model"]["kind"]
        if attempt_ids:
            key = attempt_id
        candidate = {"attempt_id": attempt_id, "recipe": recipe, "original_result": result}
        if key not in candidates or value < candidates[key]["original_result"]["objective_value"]:
            candidates[key] = candidate
    found = {row["attempt_id"] for row in candidates.values()}
    if attempt_ids and set(attempt_ids) - found:
        raise ValueError(f"Requested completed attempts unavailable: {sorted(set(attempt_ids) - found)}")
    return sorted(candidates.values(), key=lambda row: row["attempt_id"])


def paired_prediction_summary(left: Path, right: Path) -> dict:
    a, b = pd.read_csv(left), pd.read_csv(right)
    keys = ["fold_id", "race_id", "horse_no"]
    if not set(keys) <= set(a) or not set(keys) <= set(b):
        return {"comparable": False, "reason": "Prediction row-key contract unavailable"}
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate scored runner keys")
    a, b = a.set_index(keys).sort_index(), b.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        return {"comparable": False, "reason": "Scored populations differ"}
    column = "model_probability"
    if column not in a or column not in b:
        return {"comparable": False, "reason": "Not a win-probability replay"}
    return {
        "comparable": True, "scored_runners": len(a),
        "population_hash": hashlib.sha256(json.dumps(a.index.tolist()).encode()).hexdigest(),
        "max_absolute_prediction_delta": float(np.abs(a[column] - b[column]).max()),
        "prediction_parity": bool(np.allclose(a[column], b[column], rtol=1e-10, atol=1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--attempt-id", action="append", default=[])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original-dataset", type=Path)
    parser.add_argument("--original-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if bool(args.original_dataset) != bool(args.original_protocol):
        parser.error("Original dataset and protocol must be supplied together")
    champions = read_champions(args.source_campaign, attempt_ids=args.attempt_id)
    records = []
    protocol = read_protocol(args.protocol)
    for champion in champions[:args.limit]:
        recipe = PipelineRecipe.model_validate(champion["recipe"])
        output = args.output / champion["attempt_id"]
        replay = execute_recipe(RecipeExecutionRequest(
            attempt_id="replay-" + champion["attempt_id"], proposal_id="champion-replay",
            trial_number=0, recipe=recipe, dataset_path=args.dataset,
            output_dir=output / "successor", protocol_parameters=protocol,
            portfolio_version=V6_PORTFOLIO_VERSION,
        ))
        record = {**champion, "successor_replay": replay.serializable(),
                  "comparison": {"comparable": False, "reason": "Historical score only; no matched replay"}}
        if args.original_dataset:
            original = execute_recipe(RecipeExecutionRequest(
                attempt_id="original-replay-" + champion["attempt_id"], proposal_id="champion-replay",
                trial_number=0, recipe=recipe, dataset_path=args.original_dataset,
                output_dir=output / "original", protocol_parameters=read_protocol(args.original_protocol),
                portfolio_version=V6_PORTFOLIO_VERSION,
            ))
            record["original_replay"] = original.serializable()
            stored_value = champion["original_result"].get("objective_value")
            record["original_score_reproduction"] = {
                "stored_objective": stored_value, "replayed_objective": original.objective_value,
                "reproduced": bool(original.status == "completed" and stored_value is not None
                    and np.isclose(original.objective_value, stored_value, rtol=1e-10, atol=1e-12)),
                "policy": "Historical identity check; does not promote a changed-data score",
            }
            stored_predictions = champion["original_result"].get("artifacts", {}).get("predictions")
            if original.status == "completed" and stored_predictions and Path(stored_predictions).is_file():
                record["original_prediction_reproduction"] = paired_prediction_summary(
                    Path(stored_predictions), Path(original.artifacts["predictions"]),
                )
            if original.status == replay.status == "completed":
                record["comparison"] = paired_prediction_summary(
                    Path(original.artifacts["predictions"]), Path(replay.artifacts["predictions"]),
                )
                if record["comparison"]["comparable"]:
                    record["comparison"]["objective_delta"] = replay.objective_value - original.objective_value
                    record["comparison"]["interpretation"] = "Matched scoring population; training and data may differ"
        records.append(record)
        _write_json_atomic(args.output / "replay-report.json", {
            "source_campaign": str(args.source_campaign.resolve()), "records": records,
            "automatic_promotion": False,
        })
    print(json.dumps({"replayed": len(records), "report": str(args.output / "replay-report.json")}, indent=2))
    if not records or any(row["successor_replay"]["status"] != "completed" for row in records):
        raise SystemExit("Champion replay incomplete")


if __name__ == "__main__":
    main()
