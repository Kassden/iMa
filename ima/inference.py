"""Live WIN inference with public fallback for unratable runners."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data import FEATURES
from .modeling import apply_public_fallback, normalize_by_race


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
    frame["ratable"] = model_csv["prediction_ready"].astype(str).str.lower().isin({"true", "1"})
    return frame


def predict_live_win(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    market = frame["market_probability"].to_numpy()
    raw = artifact["model"].predict_proba(frame)
    calibrated = artifact["calibrator"].transform(raw, frame["race_id"])
    combined = artifact["blend"].transform(calibrated, market, frame["race_id"])
    return apply_public_fallback(
        combined, market, frame["race_id"], frame["ratable"].to_numpy()
    )
