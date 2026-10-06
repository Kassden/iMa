"""Distinct prior-race, individual-workout and individual-trial speed histories."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .historical_events import event_views, instant, runner_cutoffs, select_as_of


def finish_seconds(value):
    """HKJC M.SS.hh or M:SS.hh; bare numeric values already denote seconds."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.number)):
        result = float(value)
    else:
        text = str(value).strip()
        match = re.fullmatch(r"(\d+)[.:](\d{2})[.](\d{2})", text)
        if match:
            if int(match[2]) >= 60:
                return np.nan
            result = int(match[1]) * 60 + int(match[2]) + int(match[3]) / 100
        else:
            try:
                result = float(text)
            except ValueError:
                return np.nan
    return result if math.isfinite(result) and result > 0 else np.nan


def normalize_speed_measurement(values, family):
    """Return explicit individual (metres, seconds, m/s, reason); no batch fallback."""
    distance = values.get("distance_metres")
    if distance is None:
        unit = values.get("distance_unit")
        if unit not in {"m", "metres", "meters", "km"}:
            return (np.nan, np.nan, np.nan, "distance_unit_unknown")
        try:
            distance = float(values.get("distance")) * (1000 if unit == "km" else 1)
        except (ValueError, TypeError):
            return (np.nan, np.nan, np.nan, "distance_missing")
    specific = {"barrier_trials": "trial_finish_seconds", "trackwork": "workout_finish_seconds", "past_races": "finish_seconds"}
    time = values.get(specific.get(family, "individual_time_seconds"))
    if time is None and values.get("time_basis") == "individual" and values.get("time_unit") in {"s", "seconds", "ms", "minutes"}:
        multiplier = {"ms": .001, "minutes": 60}.get(values["time_unit"], 1)
        try:
            time = float(values.get("time_seconds", values.get("time"))) * multiplier
        except (TypeError, ValueError):
            time = None
    try:
        distance, time = float(distance), float(time)
    except (ValueError, TypeError):
        return (np.nan, np.nan, np.nan, "individual_time_missing")
    if not math.isfinite(distance) or not math.isfinite(time) or distance <= 0 or time <= 0:
        return (np.nan, np.nan, np.nan, "invalid_measurement")
    return (distance, time, distance / time, None)


SPEED_FAMILIES = ("past_race", "workout", "trial")
SPEED_STATS = ("last_mps", "mean_3_mps", "mean_5_mps", "std_5_mps", "trend_mps", "recency_weighted_mps",
               "support", "available", "days_since", "same_distance_mean_mps", "same_surface_mean_mps", "same_venue_mean_mps")


def _statistics(rows, cutoff, distance, surface, venue):
    if not rows:
        return {"support": 0., "available": 0.}
    values = np.asarray([row["speed"] for row in rows], dtype=float)
    ages = np.asarray([(cutoff - row["occurred"]).total_seconds() / 86400 for row in rows])
    weights = np.exp(-np.minimum(ages / 90, 700))
    result = {"last_mps": values[-1], "mean_3_mps": values[-3:].mean(), "mean_5_mps": values[-5:].mean(),
              "std_5_mps": values[-5:].std(ddof=1) if len(values) > 1 else np.nan,
              "trend_mps": values[-1] - values[-2] if len(values) > 1 else np.nan,
              "recency_weighted_mps": np.average(values, weights=weights),
              "support": float(len(rows)), "available": 1., "days_since": ages[-1]}
    for key, context in (("distance", distance), ("surface", surface), ("venue", venue)):
        matched = [row["speed"] for row in rows if context is not None and row.get(key) == context and str(context).upper() not in {"NAN", "UNKNOWN", ""}]
        result[f"same_{key}_mean_mps"] = float(np.mean(matched)) if matched else np.nan
    return result


def build_speed_features(runners, events=(), *, policy="strict", retrospective_lags=None):
    """Index-aligned histories; never emit current-race speed as a predictor.

    Previous result dates use conservative next-day availability, an explicitly
    disclosed completed-race assumption, not a fabricated publication timestamp.
    Event families use the historical_events publication/capture policy.
    """
    names = [f"{family}_speed_{stat}" for family in SPEED_FAMILIES for stat in SPEED_STATS]
    output = pd.DataFrame(np.nan, index=runners.index, columns=names)
    cutoffs = runner_cutoffs(runners)
    race_history = {}
    for raw in runners.to_dict("records"):
        seconds = finish_seconds(raw.get("finish_seconds", raw.get("finish_time")))
        distance = pd.to_numeric(raw.get("distance"), errors="coerce")
        if not np.isfinite(seconds) or not np.isfinite(distance) or distance <= 0:
            continue
        occurred = instant(raw.get("race_finished_at") or str(raw["date"])[:10])
        eligible = instant(raw["outcome_available_at"]) if raw.get("outcome_available_at") else instant(str(raw["date"])[:10], date_only_next_day=True)
        eligible = max(occurred, eligible)
        race_history.setdefault(raw["horse_id"], []).append({"occurred": occurred, "eligible": eligible,
            "speed": distance / seconds, "race_id": raw["race_id"], "distance": distance,
            "surface": raw.get("surface", raw.get("course")), "venue": raw.get("venue")})
    for rows in race_history.values():
        rows.sort(key=lambda r: (r["occurred"], r["race_id"]))
    views = event_views(events, policy=policy, retrospective_lags=retrospective_lags)
    for position, raw in enumerate(runners.to_dict("records")):
        cutoff = cutoffs[position]
        families = {"past_race": [r for r in race_history.get(raw["horse_id"], ()) if r["eligible"] < cutoff and r["race_id"] != raw["race_id"]]}
        for source, family in (("trackwork", "workout"), ("barrier_trials", "trial")):
            timed = []
            for event in select_as_of(views.get((raw["horse_id"], source), ()), cutoff):
                distance, seconds, speed, reason = normalize_speed_measurement(event["typed_values"], source)
                if reason:
                    continue
                values = event["typed_values"]
                timed.append({"occurred": event["_occurred"], "speed": speed, "distance": distance,
                              "surface": values.get("surface", values.get("track")), "venue": values.get("venue")})
            families[family] = timed
        for family, rows in families.items():
            stats = _statistics(rows, cutoff, raw.get("distance"), raw.get("surface", raw.get("course")), raw.get("venue"))
            for statistic, value in stats.items():
                output.iat[position, output.columns.get_loc(f"{family}_speed_{statistic}")] = value
    return output


def speed_feature_catalog():
    return {f"{family}_speed_{stat}": {"family": family, "unit": "m/s" if stat.endswith("mps") else ("days" if stat == "days_since" else "count" if stat == "support" else "indicator"),
            "cutoff_rule": "occurrence_and_availability_strictly_before_cutoff", "fit_required": False,
            "source_semantics": "individual timed observations; source families never pooled"}
            for family in SPEED_FAMILIES for stat in SPEED_STATS}
