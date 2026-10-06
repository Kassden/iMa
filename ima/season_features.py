"""Raw official history -> complete, untimed point-in-time racecard features.

No filesystem access or model fitting. Pass official runner DataFrames (not an
already filtered feature matrix). Queries never contribute outcomes or markets
to history, including when evaluating completed races. Dates use a conservative
strict prior-calendar-day cutoff; all earlier races on the meeting day are held
out. Missing ratings/ages are not inferred from future horse profiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_sets import FEATURE_SCHEMAS, NOTEBOOK_RICH_SCHEMA
from .historical_events import instant
from .rich_features import _late_gain, _running_positions, _series, _surface, parse_lengths
from .speed_features import SPEED_STATS, _statistics, finish_seconds


SUPPORTED_SCHEMAS = tuple(FEATURE_SCHEMAS)
SPEED_EXTRA_FEATURES = tuple(
    f"past_race_speed_{name}" for name in (
        "mean_3_mps", "mean_5_mps", "recency_weighted_mps", "std_5_mps",
        "support", "trend_mps",
    )
)
_SPEED_COLUMNS = tuple(f"past_race_speed_{name}" for name in SPEED_STATS)
_NONFINISHERS = {"PU", "UR", "FE", "DNF", "DISQ", "TNP"}
_OFFICIAL_GOINGS = {
    "GOOD", "GOOD TO FIRM", "GOOD TO YIELDING", "YIELDING", "YIELDING TO SOFT",
    "SOFT", "HEAVY", "FAST", "WET FAST", "SLOW", "WET SLOW", "SEALED", "UNKNOWN",
}
_ALIASES = {
    "date": ("race_date",), "horse_id": ("horse_page_id",),
    "horse_rating": ("rating",), "horse_age": ("age",),
    "result": ("place", "placing"),
    "horse_country": ("country",), "horse_colour": ("colour",),
    "jockey_name": ("jockey",), "trainer_name": ("trainer",),
    "course": ("config",),
}
_QUERY_OUTCOMES = (
    "result", "place", "placing", "place_raw", "placing_raw", "dead_heat",
    "finish_time", "finish_seconds", "finishing_status", "win_odds", "place_odds",
    "lengths_behind", "running_position", "won", "target_win", "target_top3",
    "target_second", "target_third", "target_probability", "market_raw",
    "market_probability", "place_market_raw", "place_market_probability",
    "speed_ratio_raw", "speed_mps", "lengths_raw", "late_gain_raw",
    *(f"position_sec{i}_raw" for i in range(1, 5)),
    *(f"position_sec{i}" for i in range(1, 5)),
)
_LAGS = {
    "last_result": "result", "last_speed_ratio": "speed_ratio_raw",
    "last_lengths_behind": "lengths_raw", "last_late_position_gain": "late_gain_raw",
    "last_distance": "distance", "last_carried_weight": "actual_weight",
    "last_body_weight": "declared_weight", "last_rating": "horse_rating",
    "last_win_odds": "win_odds", "last_place_odds": "place_odds",
    "last_finish_time": "finish_seconds",
    **{f"last_position_sec{i}": f"position_sec{i}_raw" for i in range(1, 5)},
}
_HISTORIES = (
    (("horse_id", "distance_band"), "distance_band"), (("horse_id", "venue"), "venue"),
    (("horse_id", "course"), "course"), (("horse_id", "going"), "going"),
    (("horse_id", "surface"), "surface"), (("jockey_key",), "jockey"),
    (("trainer_key",), "trainer"), (("jockey_key", "trainer_key"), "jockey_trainer"),
    (("horse_id", "jockey_key"), "horse_jockey"),
    (("venue", "surface", "distance_band", "draw"), "draw_bias"),
)


def _normalize(raw: pd.DataFrame, *, query: bool) -> pd.DataFrame:
    frame = raw.copy().reset_index(drop=True)
    for destination, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in frame:
                frame[destination] = _series(frame, destination).fillna(frame[alias])
    _normalize_labels(frame)
    for name in ("date", "race_no", "horse_no", "horse_id"):
        if name not in frame or frame[name].isna().any():
            raise ValueError(f"Missing required runner identity: {name}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Missing runner date")
    for name in ("race_no", "horse_no"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if (frame[name].le(0) | frame[name].mod(1).ne(0)).any():
            raise ValueError(f"Invalid runner identity: {name}")
    if "race_id" not in frame:
        if "venue" not in frame or frame["venue"].isna().any():
            raise ValueError("race_id or venue is required")
        frame["race_id"] = (
            "HKJC:" + frame.date.dt.strftime("%Y-%m-%d") + ":" + frame.venue.astype(str)
            + ":R" + frame.race_no.astype(int).astype(str)
        )
    for name in ("race_id", "horse_id"):
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing required runner identity: {name}")
    if query and (frame.duplicated(["date", "race_id", "horse_no"]).any() or frame.duplicated(["date", "race_id", "horse_id"]).any()):
        raise ValueError("Duplicate runner identity")
    for name in ("race_id", "horse_id"):
        if query and name == "race_id":
            continue
        frame[name] = frame[name].astype(str)
    for name in (
        "horse_age", "horse_rating", "declared_weight", "actual_weight", "draw", "distance",
        "prize", "result", "win_odds", "place_odds",
    ):
        frame[name] = pd.to_numeric(_series(frame, name), errors="coerce")
    frame["race_class"] = pd.to_numeric(
        _series(frame, "race_class").astype("string").str.extract(r"(\d+)")[0], errors="coerce",
    )
    for name in NOTEBOOK_RICH_SCHEMA.categorical:
        if name != "surface":
            frame[name] = _series(frame, name).fillna("UNKNOWN").astype(str)
    frame["surface"] = [_surface(c, s) for c, s in zip(frame.course, _series(frame, "surface"))]
    frame["distance_band"] = (frame.distance / 200).round().mul(200)
    for entity in ("jockey", "trainer"):
        frame[f"{entity}_key"] = _series(frame, f"{entity}_id").fillna(_series(frame, f"{entity}_name"))
    frame["official_source"] = _series(frame, "source", "").fillna("").astype(str).str.startswith("official:hkjc").astype(float)
    frame["season_id"] = frame.date.dt.year - frame.date.dt.month.lt(7).astype(int)
    if query:
        # Evaluation query rows can contain outcomes, but these are never inputs.
        for name in (
            "result", "finish_time", "finish_seconds", "win_odds", "place_odds",
            "lengths_behind", "running_position", "finishing_status", "target_win",
            "target_probability", "market_raw", "market_probability", "place_market_probability",
        ):
            frame[name] = np.nan
        for name in frame.columns.intersection(_QUERY_OUTCOMES):
            frame[name] = np.nan
    else:
        frame["finish_seconds"] = _series(frame, "finish_seconds").combine_first(
            _series(frame, "finish_time").map(finish_seconds)
        ).map(finish_seconds)
        frame["lengths_raw"] = _series(frame, "lengths_behind").map(parse_lengths)
        frame["late_gain_raw"] = [_late_gain(p, r) for p, r in zip(_series(frame, "running_position"), frame.result)]
        positions = _series(frame, "running_position").map(_running_positions)
        for i in range(1, 5):
            frame[f"position_sec{i}_raw"] = positions.map(lambda p, i=i: p[i - 1] if len(p) >= i else np.nan)
    return frame


def _normalize_labels(frame):
    frame["going_raw"] = _series(frame, "going_raw").fillna(_series(frame, "going"))
    going = _series(frame, "going").fillna("UNKNOWN").astype(str).str.strip().str.upper()
    unrecognized = going[~going.isin(_OFFICIAL_GOINGS)]
    if not unrecognized.empty:
        raise ValueError(f"Unrecognized going labels: {sorted(unrecognized.unique())}")
    frame["going"] = going
    jockey = _series(frame, "jockey_name").astype("string")
    frame["jockey_name_raw"] = _series(frame, "jockey_name_raw").fillna(_series(frame, "jockey_raw")).fillna(jockey)
    annotation = r"\s*\(-\s*(\d+)\)\s*$"
    allowance = pd.to_numeric(jockey.str.extract(annotation)[0], errors="coerce")
    canonical = jockey.str.replace(annotation, "", regex=True)
    raw_label = frame["jockey_name_raw"].astype("string")
    raw_allowance = pd.to_numeric(raw_label.str.extract(annotation)[0], errors="coerce")
    raw_allowance = raw_allowance.where(raw_label.str.replace(annotation, "", regex=True).eq(canonical))
    if (allowance.notna() & raw_allowance.notna() & allowance.ne(raw_allowance)).any():
        raise ValueError("Jockey label conflicts with raw allowance annotation")
    allowance = allowance.fillna(raw_allowance)
    supplied = pd.to_numeric(_series(frame, "apprentice_allowance_lbs"), errors="coerce")
    if (allowance.notna() & supplied.notna() & allowance.ne(supplied)).any():
        raise ValueError("Jockey label conflicts with apprentice allowance metadata")
    frame["apprentice_allowance_lbs"] = supplied.fillna(allowance)
    frame["jockey_name"] = canonical
    if frame["jockey_name"].str.strip().eq("").fillna(False).any():
        raise ValueError("Missing jockey name after allowance annotation")


def _merge_exports(frame):
    keys = ["date", "venue", "race_no", "horse_id"]
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        overlaps = frame[duplicated]
        measurements = ["horse_no", "result", "finish_seconds", "win_odds", "place_odds",
                        "distance", "horse_rating", "horse_age", "actual_weight", "declared_weight", "draw"]
        conflicts = overlaps.groupby(keys, dropna=False)[measurements].nunique().gt(1)
        if conflicts.any().any():
            raise ValueError("Conflicting historical runner exports; resolve overlaps before building")
        # Coalesce only compatible observed fields, including optional form data.
        frame = pd.concat([frame[~duplicated], overlaps.groupby(keys, dropna=False, as_index=False).first()], ignore_index=True)
    if frame.duplicated(["date", "race_id", "horse_no"]).any():
        raise ValueError("Duplicate historical runner numbers")
    return frame


def _rates(rows: pd.DataFrame) -> dict[str, float]:
    results = rows.result
    known = results.notna() | _series(rows, "finishing_status", "").isin(_NONFINISHERS)
    denominator = int(known.sum())
    return {
        "starts": float(len(rows)),
        "win_rate": float(results.eq(1).sum() / denominator) if denominator else np.nan,
        "top3_rate": float(results.le(3).sum() / denominator) if denominator else np.nan,
    }


def _time_values(past, column, day, days):
    # Preserve rich_features' shift-then-time-roll training convention: each
    # prior value is indexed by the next start, including the untimed query.
    if past.empty:
        return past[column]
    threshold = (day - pd.Timedelta(days=days)).to_datetime64()
    mask = past.date.to_numpy()[1:] >= threshold
    indices = np.append(np.flatnonzero(mask), len(past) - 1)
    return past[column].iloc[indices]


def _horse_features(past: pd.DataFrame, row: pd.Series) -> dict:
    features = {f"prior_{k}": v for k, v in _rates(past).items()}
    features["debut_flag"] = float(past.empty)
    features["prior_avg_result"] = past.result.mean()
    features["prior_avg_odds"] = past.win_odds.mean()
    known = past.result.notna() | _series(past, "finishing_status", "").isin(_NONFINISHERS)
    for result, name in ((2, "second"), (3, "third")):
        features[f"prior_{name}_count"] = float(past.result.eq(result).sum()) if past.empty or known.any() else np.nan
        features[f"prior_{name}_rate"] = past.result.eq(result).sum() / known.sum() if known.any() else np.nan
    last = past.iloc[-1] if not past.empty else pd.Series(dtype=object)
    features.update({destination: last.get(source, np.nan) for destination, source in _LAGS.items()})
    features["days_since_last_race"] = (row.date - last.date).days if not past.empty else np.nan
    for windows, source, prefix in (
        ((2, 3, 5, 6), "result", "avg_result"),
        ((2, 3, 4, 5, 6), "speed_ratio_raw", "avg_speed_ratio"),
        ((2, 3, 4, 6), "win_odds", "avg_win_odds"),
        ((2, 3, 4, 6), "place_odds", "avg_place_odds"),
        ((3,), "lengths_raw", "avg_lengths_behind"),
        ((3,), "late_gain_raw", "avg_late_position_gain"),
    ):
        for window in windows:
            features[f"{prefix}_{window}"] = past[source].tail(window).mean()
    features["total_distance_4"] = past.distance.tail(4).sum(min_count=1)
    for days in (84, 112, 168, 183):
        recent = _time_values(past, "result", row.date, days)
        features[f"avg_result_{days}d"] = recent.mean()
    for operation, name in (("median", "median"), ("min", "best"), ("max", "worst")):
        values = recent.dropna()
        features[f"{name}_result_183d"] = getattr(values, operation)() if not values.empty else np.nan
    features["avg_speed_ratio_183d"] = _time_values(past, "speed_ratio_raw", row.date, 183).mean()
    same_distance = past[past.distance_band.eq(row.distance_band)]
    features["avg_same_distance_finish_time_183d"] = _time_values(same_distance, "finish_seconds", row.date, 183).mean()
    current = past[past.season_id.eq(row.season_id)]
    features["current_season_starts"] = float(len(current))
    features["current_season_avg_result"] = current.result.mean()
    features["previous_season_avg_result"] = past.loc[past.season_id.eq(row.season_id - 1), "result"].mean()
    for destination, source, lag in (
        ("distance_change", "distance", "last_distance"),
        ("carried_weight_change", "actual_weight", "last_carried_weight"),
        ("body_weight_change", "declared_weight", "last_body_weight"),
        ("rating_change", "horse_rating", "last_rating"),
    ):
        features[destination] = row[source] - features[lag]
    features["class_change"] = row.race_class - last.get("race_class", np.nan)
    weight = features["last_body_weight"]
    days = features["days_since_last_race"]
    features["body_weight_change_pct"] = features["body_weight_change"] / weight if pd.notna(weight) and weight != 0 else np.nan
    features["weight_change_per_day"] = features["body_weight_change"] / days if pd.notna(days) and days != 0 else np.nan
    timed = past[past.finish_seconds.gt(0) & past.distance.gt(0)]
    cutoff_value = row.get("cutoff_at")
    cutoff = instant(cutoff_value if pd.notna(cutoff_value) else row.date)
    observations = []
    for raw in timed.to_dict("records"):
        finished = raw.get("race_finished_at")
        occurred = instant(finished if pd.notna(finished) else raw["date"])
        available = raw.get("outcome_available_at")
        eligible = instant(available) if pd.notna(available) else instant(str(raw["date"])[:10], date_only_next_day=True)
        if max(occurred, eligible) < cutoff and str(raw["race_id"]) != str(row.race_id):
            observations.append(dict(occurred=occurred, speed=raw["distance"] / raw["finish_seconds"],
                                     distance=raw["distance"], surface=raw["surface"], venue=raw["venue"]))
    stats = _statistics(observations, cutoff, row.distance, row.surface, row.venue)
    features.update({name: stats.get(name.removeprefix("past_race_speed_"), np.nan) for name in _SPEED_COLUMNS})
    return features


def _event_features(row, events, family, date_column, windows, values=()):
    """Unknown feed coverage stays NaN; observed events use strict dates."""
    output = {f"{family}_available": 0.0, f"days_since_{family}": np.nan}
    output.update({f"{family}_{days}d": np.nan for days in windows})
    output.update({destination: np.nan for destination, _ in values})
    if events is None or events.empty:
        return output
    eligible = events[events.horse_id.eq(row.horse_id) & events[date_column].lt(row.date)]
    for availability in ("available_at", "first_seen_at"):
        if availability in eligible:
            dates = pd.to_datetime(eligible[availability], errors="coerce", utc=True)
            cutoff = row.date.tz_localize("Asia/Hong_Kong").tz_convert("UTC")
            eligible = eligible[dates.lt(cutoff)]
    if eligible.empty:
        return output
    output[f"{family}_available"] = 1.0
    latest = eligible.sort_values(date_column, kind="stable").iloc[-1]
    output[f"days_since_{family}"] = float((row.date - latest[date_column]).days)
    for days in windows:
        output[f"{family}_{days}d"] = float(eligible[date_column].ge(row.date - pd.Timedelta(days=days)).sum())
    for destination, source in values:
        output[destination] = latest.get(source, np.nan)
    return output


def build_season_features(
    history: pd.DataFrame,
    query: pd.DataFrame,
    *,
    season_results: pd.DataFrame | None = None,
    schema: str = "notebook-rich-v2",
    trackwork: pd.DataFrame | None = None,
    barriers: pd.DataFrame | None = None,
    sectionals: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
    veterinary: pd.DataFrame | None = None,
    movements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build frozen-package predictors without fabricated results or odds.

    Accepted schemas: baseline-v1, benter-rich-v1, notebook-rich-v2. All rich
    features and six speed extras are returned; baseline adds numeric surface
    and config/last_finish_time. Required raw identities: date (or race_date),
    race_no, horse_no, horse_id (or horse_page_id), race_id (or venue). IDs are
    matched exactly: callers must use the same namespace across all inputs.
    Rating/age aliases accept actual racecard/form values, never forward fill.
    Official going labels use their uppercase training spelling; unknown
    spellings raise rather than creating a new history bucket. Terminal jockey
    (-n) allowance annotations are metadata, not part of a rider identity.
    going_raw, jockey_name_raw and apprentice_allowance_lbs preserve evidence.
    Raw gear is retained as provenance; only explicit horse_gear populates that
    frozen categorical feature, matching prepare_rich_runner_dataset.
    Pass an entire query field; duplicate numbers/horses and conflicting history
    are rejected. Output retains query order and index. Query outcomes/odds are
    blanked, including placing/dead-heat aliases and raw outcome measurements,
    so attach observed settlement/market data separately after building.
    Optional feeds use the existing rich builder's column names and prior-day
    semantics. Unknown feed counts remain NaN rather than asserting zero events.
    Time windows preserve rich_features' shift-before-window convention. Speed
    extras follow build_speed_features' next-day availability and strict cutoff,
    so a date-only result is excluded at exactly the next day's midnight.
    """
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported feature schema: {schema}")
    raw = pd.concat([history, season_results], ignore_index=True) if season_results is not None else history.copy()
    # Deduplicate semantic runner fields after normalization, not arbitrary
    # collector metadata (which may contain lists or nested source cells).
    if raw.empty:
        raw = pd.DataFrame(columns=["date", "race_id", "race_no", "horse_no", "horse_id"])
    historical = _merge_exports(_normalize(raw, query=False))
    output = _normalize(query, query=True)
    if output.groupby("race_id")[["date", "venue", "race_no"]].nunique().gt(1).any().any():
        raise ValueError("A query race_id must identify one meeting race")
    output["field_size"] = output.groupby("race_id").horse_id.transform("size").astype(float)
    if "field_size" in query:
        supplied = pd.to_numeric(query.field_size, errors="coerce").to_numpy()
        if (pd.notna(supplied) & (supplied != output.field_size.to_numpy())).any():
            raise ValueError("Incomplete query field: supplied field_size differs from runner count")
    feed_specs = (
        (trackwork, "event_date"), (barriers, "event_date"), (sectionals, "race_date"),
        (profiles, "snapshot_at"), (veterinary, "event_date"), (movements, "event_date"),
    )
    feeds = []
    for feed, date_column in feed_specs:
        if feed is None or feed.empty:
            feeds.append(None)
            continue
        feed = feed.copy()
        if date_column == "event_date" and "arrival_date" in feed:
            feed[date_column] = _series(feed, date_column).fillna(feed.arrival_date)
        feed[date_column] = pd.to_datetime(feed[date_column], errors="raise").dt.normalize()
        feeds.append(feed)
    features = []
    for day, queries in output.groupby("date", sort=False):
        past = historical[historical.date.lt(day)].copy()
        if "outcome_available_at" in past:
            available = pd.to_datetime(past.outcome_available_at, errors="coerce", utc=True)
            cutoff = day.tz_localize("Asia/Hong_Kong").tz_convert("UTC")
            past = past[past.outcome_available_at.isna() | available.lt(cutoff)].copy()
        past = past.sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable")
        past["speed_ratio_raw"] = past.groupby(["date", "race_id"]).finish_seconds.transform("min") / past.finish_seconds
        horse_groups = past.groupby("horse_id", sort=False).indices
        history_groups = {
            prefix: past.groupby(keys[0] if len(keys) == 1 else list(keys), sort=False).indices
            for keys, prefix in _HISTORIES
        }
        for index, row in queries.iterrows():
            horse_past = past.iloc[horse_groups.get(row.horse_id, [])]
            item = _horse_features(horse_past, row)
            for keys, prefix in _HISTORIES:
                key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
                valid = all(pd.notna(row[k]) and str(row[k]) != "UNKNOWN" for k in keys)
                indices = history_groups[prefix].get(key, []) if valid else []
                matched = past.iloc[indices]
                item.update({f"{prefix}_{k}": v if valid else np.nan for k, v in _rates(matched).items()})
            for entity in ("jockey", "trainer"):
                matched = past.iloc[history_groups[entity].get(row[f"{entity}_key"], [])]
                latest = matched.iloc[-1] if not matched.empty else pd.Series(dtype=object)
                item[f"{entity}_last_result"] = latest.get("result", np.nan)
                item[f"{entity}_avg_result_90d"] = _time_values(matched, "result", day, 90).mean()
                if entity == "jockey":
                    item["jockey_last_speed_ratio"] = latest.get("speed_ratio_raw", np.nan)
            item.update(_auxiliary_features(row, feeds))
            features.append(pd.Series(item, name=index))
    computed = pd.DataFrame(features).reindex(output.index)
    # Overwrite any precomputed columns supplied by a query export.
    current_features = {"horse_age", "horse_rating", "declared_weight", "actual_weight", "draw", "distance", "race_class", "prize", "field_size", "official_source"}
    names = [f for f in NOTEBOOK_RICH_SCHEMA.numeric if f not in current_features]
    names.extend((*_SPEED_COLUMNS, "last_finish_time"))
    output = pd.concat([output.drop(columns=names, errors="ignore"), computed.reindex(columns=names)], axis=1).copy()
    output[list(NOTEBOOK_RICH_SCHEMA.numeric)] = output[list(NOTEBOOK_RICH_SCHEMA.numeric)].apply(pd.to_numeric, errors="coerce")
    output = output.copy()
    for name in ("horse_colour", "import_type", "sire", "dam_sire"):
        if name in computed:
            output[name] = output[name].where(output[name].ne("UNKNOWN"), computed[name]).fillna("UNKNOWN")
    ranks = {
        "age": ("horse_age", False), "rating": ("horse_rating", False),
        "carried_weight": ("actual_weight", True), "body_weight": ("declared_weight", True),
        "carried_weight_change": ("carried_weight_change", True),
        "prior_speed": ("last_speed_ratio", False), "distance_experience": ("distance_band_starts", False),
        "prior_form": ("prior_avg_result", True),
        **{f"last_position_sec{i}": (f"last_position_sec{i}", True) for i in range(1, 5)},
    }
    for name, (source, ascending) in ranks.items():
        output[f"{name}_rank"] = output.groupby("race_id")[source].rank(pct=True, ascending=ascending)
    for name, source in (("age", "horse_age"), ("rating", "horse_rating"), ("distance", "distance"), ("draw", "draw")):
        output[f"poly_{name}_sq"] = output[source].pow(2)
    for name, a, b in (
        ("rating_prior_win", "horse_rating", "prior_win_rate"), ("rating_recent_speed", "horse_rating", "last_speed_ratio"),
        ("age_distance", "horse_age", "distance"), ("draw_distance", "draw", "distance"),
        ("weight_change_distance", "carried_weight_change", "distance"), ("jockey_trainer_win", "jockey_win_rate", "trainer_win_rate"),
    ):
        output[f"poly_{name}"] = output[a] * output[b]
    if schema == "baseline-v1":
        output["config"] = output.course
        output["surface"] = output.surface.map({"TURF": 0.0, "AWT": 1.0})
    output.index = query.index
    output.attrs.update(feature_schema=schema, cutoff_rule="strict_before_query_calendar_day", speed_extra_features=SPEED_EXTRA_FEATURES)
    return output


def _auxiliary_features(row, feeds):
    trackwork, barriers, sectionals, profiles, veterinary, movements = feeds
    output = _event_features(row, trackwork, "trackwork", "event_date", (7, 14, 30))
    if barriers is not None:
        barriers = barriers.assign(last_trial_speed=pd.to_numeric(barriers.distance, errors="coerce") / barriers.finish_time.map(finish_seconds))
    trials = _event_features(row, barriers, "trial", "event_date", (90,), (("last_trial_placing", "placing"), ("last_trial_speed", "last_trial_speed")))
    output.update({"barrier_available": trials["trial_available"], "days_since_trial": trials["days_since_trial"], "trials_90d": trials["trial_90d"],
                   "last_trial_placing": trials["last_trial_placing"], "last_trial_speed": trials["last_trial_speed"]})
    if sectionals is not None:
        sectionals = sectionals.sort_values("section_index").groupby(["horse_id", "race_date"], as_index=False).tail(1)
    output.update(_event_features(row, sectionals, "sectionals", "race_date", (), (("last_sectional_time", "section_time"), ("last_sectional_position", "position"))))
    output.update(_event_features(row, profiles, "profile", "snapshot_at", (), (
        ("season_stakes", "season_stake"), ("total_stakes", "total_stake"),
        ("start_of_season_rating", "start_of_season_rating"), ("starts_past_10_meetings", "no_of_start_past_10_meetings"),
        ("horse_colour", "colour"), ("import_type", "import_type"), ("sire", "sire"), ("dam_sire", "dam_sire"),
    )))
    output["rating_change_from_season_start"] = row.horse_rating - pd.to_numeric(output["start_of_season_rating"], errors="coerce")
    vet = _event_features(row, veterinary, "veterinary", "event_date", (30, 90))
    output.update({"veterinary_available": vet["veterinary_available"], "days_since_veterinary": vet["days_since_veterinary"],
                   "veterinary_events_30d": vet["veterinary_30d"], "veterinary_events_90d": vet["veterinary_90d"]})
    for keyword in ("injury", "fracture", "surgery"):
        if vet["veterinary_available"]:
            eligible = veterinary[veterinary.horse_id.eq(row.horse_id) & veterinary.event_date.lt(row.date) & veterinary.event_date.ge(row.date - pd.Timedelta(days=365))]
            output[f"{keyword}_events_365d"] = float(_series(eligible, "details", "").str.contains(keyword, case=False, na=False).sum())
        else:
            output[f"{keyword}_events_365d"] = np.nan
    movement = _event_features(row, movements, "movement", "event_date", (365,))
    output.update({"movement_available": movement["movement_available"], "days_since_movement": movement["days_since_movement"], "movements_365d": movement["movement_365d"]})
    arrivals = movements[_series(movements, "to", "").str.contains("HONG KONG|SHA TIN", case=False, na=False)] if movements is not None else None
    output["days_since_hk_arrival"] = _event_features(row, arrivals, "hk_arrival", "event_date", ())["days_since_hk_arrival"]
    return output
