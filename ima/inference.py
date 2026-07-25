"""Live WIN inference with public fallback for unratable runners."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import FEATURES
from .feature_sets import BASELINE_SCHEMA
from .modeling import apply_public_fallback, normalize_by_race
from .pools import (
    CombinationProbability,
    OrderExponents,
    SUPPORTED_POOLS,
    canonical_pool_name,
    rank_combinations,
)


def _class_number(value: object) -> float:
    match = re.search(r"\d+", str(value or ""))
    return float(match.group()) if match else np.nan


def legacy_live_frame(model_csv: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=model_csv.index)
    frame["race_id"] = model_csv["race_id"].astype(str)
    frame["horse_no"] = model_csv["horse_no_x"].astype(str)
    frame["horse_age"] = model_csv.get("horse_age")
    frame["horse_rating"] = model_csv.get("horse_rating")
    frame["declared_weight"] = model_csv.get("declared_weight")
    frame["actual_weight"] = model_csv.get("actual_weight")
    frame["draw"] = model_csv.get("draw")
    frame["distance"] = model_csv.get("prev_dist")
    frame["race_class"] = model_csv.get("race_class", pd.Series(index=frame.index)).map(_class_number)
    frame["surface"] = np.nan
    frame["prize"] = np.nan
    frame["field_size"] = model_csv.get("horse_no_y")
    frame["prior_starts"] = model_csv.get("exp")
    starts = pd.to_numeric(frame["prior_starts"], errors="coerce")
    wins = pd.to_numeric(model_csv.get("won_odds"), errors="coerce")
    frame["prior_win_rate"] = wins / starts.replace(0, np.nan)
    frame["prior_avg_result"] = model_csv.get("cum_avg_prev_resu")
    frame["prior_avg_odds"] = model_csv.get("prev_odds")
    frame["last_result"] = model_csv.get("prev_resu")
    frame["last_win_odds"] = model_csv.get("prev_odds")
    frame["last_finish_time"] = model_csv.get("fin_time")
    frame["prior_second_count"] = model_csv.get("second_count")
    frame["prior_third_count"] = model_csv.get("third_count")
    frame["prior_second_rate"] = pd.to_numeric(frame["prior_second_count"], errors="coerce") / starts.replace(0, np.nan)
    frame["prior_third_rate"] = pd.to_numeric(frame["prior_third_count"], errors="coerce") / starts.replace(0, np.nan)
    frame["debut_flag"] = starts.fillna(0).eq(0).astype(float)
    frame["last_place_odds"] = model_csv.get("place_odds")
    frame["avg_result_2"] = model_csv.get("avg_2last")
    frame["venue"] = "UNKNOWN"
    frame["config"] = "UNKNOWN"
    frame["going"] = "UNKNOWN"
    frame["horse_country"] = model_csv.get("horse_country")
    frame["horse_type"] = model_csv.get("horse_type")
    frame["horse_gear"] = model_csv.get("horse_gear")
    for feature in FEATURES:
        if feature not in frame:
            frame[feature] = np.nan
    odds = pd.to_numeric(model_csv["win_odds"], errors="coerce").to_numpy()
    raw_market = np.divide(1.0, odds, out=np.zeros_like(odds), where=odds > 1)
    frame["market_probability"] = normalize_by_race(raw_market, frame["race_id"])
    frame["win_odds"] = odds
    place_odds = pd.to_numeric(
        model_csv.get("place_odds", pd.Series(np.nan, index=model_csv.index)), errors="coerce"
    ).to_numpy()
    place_raw = np.divide(
        1.0, place_odds, out=np.full_like(place_odds, np.nan), where=place_odds > 1,
    )
    place_strength = np.where(np.isfinite(place_raw), place_raw, raw_market)
    frame["place_market_probability"] = normalize_by_race(place_strength, frame["race_id"])
    frame["place_odds"] = place_odds
    frame["ratable"] = model_csv["prediction_ready"].astype(str).str.lower().isin({"true", "1"})
    return frame


def _model_ready_frame(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    ready = frame.copy()
    schema = getattr(artifact.get("model"), "feature_schema", BASELINE_SCHEMA)
    for feature in schema.numeric:
        if feature not in ready:
            ready[feature] = np.nan
    for feature in schema.categorical:
        if feature not in ready:
            ready[feature] = "UNKNOWN"
    return ready


def predict_live_win(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    frame = _model_ready_frame(frame, artifact)
    market = frame["market_probability"].to_numpy()
    raw = artifact["model"].predict_proba(frame)
    calibrated = artifact["calibrator"].transform(raw, frame["race_id"])
    if hasattr(artifact["blend"], "place_market_weight"):
        combined = artifact["blend"].transform(
            calibrated,
            market,
            frame["race_id"],
            frame["place_market_probability"].to_numpy(),
        )
    else:
        combined = artifact["blend"].transform(calibrated, market, frame["race_id"])
    return apply_public_fallback(
        combined, market, frame["race_id"], frame["ratable"].to_numpy()
    )


def predict_live_pools(
    frame: pd.DataFrame,
    artifact: dict,
    pools: tuple[str, ...] | list[str] = SUPPORTED_POOLS,
    top_n: int | None = None,
) -> dict[str, dict[str, list[CombinationProbability]]]:
    """Expand market-combined runner probabilities into ranked pool outcomes."""
    combined = predict_live_win(frame, artifact)
    work = frame.assign(_combined_probability=combined)
    exponents = artifact.get("order_exponents", OrderExponents())
    canonical_pools = tuple(dict.fromkeys(canonical_pool_name(pool) for pool in pools))
    predictions: dict[str, dict[str, list[CombinationProbability]]] = {}
    for race_id, race in work.groupby("race_id", sort=False):
        runner_ids = race["horse_no"].astype(str).tolist()
        strengths = race["_combined_probability"].to_numpy(dtype=float)
        race_predictions = {}
        for pool in canonical_pools:
            ranked = rank_combinations(runner_ids, strengths, pool, exponents=exponents)
            race_predictions[pool] = ranked if top_n is None else ranked[:top_n]
        predictions[str(race_id)] = race_predictions
    return predictions
