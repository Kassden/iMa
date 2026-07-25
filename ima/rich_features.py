"""Leakage-safe Benter-inspired feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import _finish_time_seconds
from .feature_sets import NOTEBOOK_RICH_SCHEMA, RICH_SCHEMA


def _series(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index, name=name)


def parse_lengths(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text == "---":
        return 0.0
    if text in {"", "-", "NAN"}:
        return np.nan
    named = {
        "NOSE": 0.05, "N": 0.05, "SH": 0.1, "S.H": 0.1,
        "HD": 0.2, "HEAD": 0.2, "NK": 0.3, "NECK": 0.3,
    }
    if text in named:
        return named[text]
    total = 0.0
    for token in text.replace("-", " ").split():
        try:
            if "/" in token:
                numerator, denominator = token.split("/", 1)
                total += float(numerator) / float(denominator)
            else:
                total += float(token)
        except ValueError:
            return np.nan
    return total


def _late_gain(value, result) -> float:
    if pd.isna(value) or pd.isna(result):
        return np.nan
    positions = [int(item) for item in str(value).split() if item.isdigit()]
    return float(positions[-2] - float(result)) if len(positions) >= 2 else np.nan


def _surface(course, surface) -> str:
    if pd.notna(course):
        text = str(course).upper()
        if "ALL WEATHER" in text or "AWT" in text:
            return "AWT"
        if "TURF" in text:
            return "TURF"
    if pd.notna(surface):
        try:
            return "TURF" if float(surface) == 0 else "AWT"
        except (TypeError, ValueError):
            return str(surface)
    return "UNKNOWN"


def _race_safe_history(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    race = frame[[*keys, "race_id", "date", "race_no", "target_win", "target_top3"]].groupby(
        [*keys, "race_id"], dropna=False, as_index=False
    ).agg(
        date=("date", "first"), race_no=("race_no", "first"),
        starts=("target_win", "size"), wins=("target_win", "sum"),
        top3=("target_top3", "sum"),
    )
    race = race.sort_values([*keys, "date", "race_no", "race_id"], kind="stable")
    grouped = race.groupby(keys, dropna=False, sort=False)
    starts_name = f"{prefix}_starts"
    race[starts_name] = grouped["starts"].cumsum() - race["starts"]
    prior_wins = grouped["wins"].cumsum() - race["wins"]
    prior_top3 = grouped["top3"].cumsum() - race["top3"]
    denominator = race[starts_name].replace(0, np.nan)
    race[f"{prefix}_win_rate"] = prior_wins / denominator
    race[f"{prefix}_top3_rate"] = prior_top3 / denominator
    return race[[*keys, "race_id", starts_name, f"{prefix}_win_rate", f"{prefix}_top3_rate"]]


def _merge_history(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    history = _race_safe_history(frame, keys, prefix)
    return frame.merge(history, on=[*keys, "race_id"], how="left", validate="many_to_one")


def _rolling_shift(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return frame.groupby("horse_id", sort=False)[column].transform(
        lambda values: values.shift().rolling(window, min_periods=1).mean()
    )


def _time_rolling_shift(
    frame: pd.DataFrame,
    column: str,
    days: int,
    operation: str = "mean",
    keys: tuple[str, ...] = ("horse_id",),
) -> pd.Series:
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(list(keys), sort=False, dropna=False):
        ordered = group.sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable")
        values = pd.to_numeric(ordered[column], errors="coerce")
        dated = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(ordered["date"]))
        rolling = dated.shift().rolling(f"{days}D", min_periods=1, closed="both")
        output.loc[ordered.index] = getattr(rolling, operation)().to_numpy()
    return output


def _within_race_rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    return frame.groupby("race_id")[column].rank(pct=True, ascending=ascending)


def _running_positions(value) -> list[float]:
    if pd.isna(value):
        return []
    return [float(item) for item in str(value).split() if item.isdigit()]


def _event_windows(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    event_date: str,
    windows: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output[f"{prefix}_available"] = 0.0
    output[f"days_since_{prefix}"] = np.nan
    for window in windows:
        output[f"{prefix}_{window}d"] = 0.0
    if events.empty:
        return output
    events = events[["horse_id", event_date]].dropna().copy()
    events[event_date] = pd.to_datetime(events[event_date]).dt.normalize()
    coverage_min, coverage_max = events[event_date].min(), events[event_date].max()
    output[f"{prefix}_available"] = (
        frame["date"].between(coverage_min, coverage_max)
    ).astype(float)
    dates_by_horse = {
        horse_id: np.sort(group[event_date].to_numpy(dtype="datetime64[D]"))
        for horse_id, group in events.groupby("horse_id", sort=False)
    }
    race_dates = frame["date"].to_numpy(dtype="datetime64[D]")
    for position, (horse_id, race_date) in enumerate(zip(frame["horse_id"], race_dates)):
        dates = dates_by_horse.get(horse_id)
        if dates is None:
            continue
        end = np.searchsorted(dates, race_date, side="left")
        if end:
            output.iat[position, output.columns.get_loc(f"days_since_{prefix}")] = float(
                (race_date - dates[end - 1]).astype("timedelta64[D]").astype(int)
            )
        for window in windows:
            start_date = race_date - np.timedelta64(window, "D")
            start = np.searchsorted(dates, start_date, side="left")
            output.iat[position, output.columns.get_loc(f"{prefix}_{window}d")] = float(end - start)
    return output


def _latest_event_values(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    event_date: str,
    values: dict[str, str],
    availability: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output[availability] = 0.0
    for destination, source in values.items():
        if source in events and not pd.api.types.is_numeric_dtype(events[source]):
            output[destination] = pd.Series(pd.NA, index=frame.index, dtype=object)
        else:
            output[destination] = np.nan
    if events.empty:
        return output
    events = events.copy()
    events[event_date] = pd.to_datetime(events[event_date]).dt.normalize()
    output[availability] = frame["date"].between(events[event_date].min(), events[event_date].max()).astype(float)
    grouped = {}
    for horse_id, group in events.dropna(subset=["horse_id", event_date]).groupby("horse_id", sort=False):
        ordered = group.sort_values(event_date, kind="stable")
        grouped[horse_id] = (ordered[event_date].to_numpy(dtype="datetime64[D]"), ordered)
    race_dates = frame["date"].to_numpy(dtype="datetime64[D]")
    for position, (horse_id, race_date) in enumerate(zip(frame["horse_id"], race_dates)):
        item = grouped.get(horse_id)
        if item is None:
            continue
        dates, group = item
        event_index = np.searchsorted(dates, race_date, side="left") - 1
        if event_index < 0:
            continue
        row = group.iloc[event_index]
        for destination, source in values.items():
            output.iat[position, output.columns.get_loc(destination)] = row.get(source)
    return output


def prepare_rich_runner_dataset(
    source: pd.DataFrame,
    trackwork: pd.DataFrame | None = None,
    barriers: pd.DataFrame | None = None,
    sectionals: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
    veterinary: pd.DataFrame | None = None,
    movements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = source.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame["win_odds"] = pd.to_numeric(frame["win_odds"], errors="coerce")
    frame["place_odds"] = pd.to_numeric(_series(frame, "place_odds"), errors="coerce")
    frame = frame[frame["result"].notna()].copy()
    frame["target_win"] = frame["result"].eq(1).astype(int)
    frame["target_top3"] = frame["result"].le(3).astype(int)
    frame["target_second"] = frame["result"].eq(2).astype(int)
    frame["target_third"] = frame["result"].eq(3).astype(int)
    winner_count = frame.groupby("race_id")["target_win"].transform("sum")
    valid_market = frame["win_odds"].gt(1).groupby(frame["race_id"]).transform("all")
    frame = frame[winner_count.gt(0) & valid_market].copy()
    frame["target_probability"] = frame["target_win"] / frame.groupby("race_id")["target_win"].transform("sum")
    frame["market_raw"] = 1.0 / frame["win_odds"]
    frame["market_probability"] = frame["market_raw"] / frame.groupby("race_id")["market_raw"].transform("sum")
    frame["field_size"] = frame.groupby("race_id")["horse_id"].transform("size").astype(float)
    frame["place_market_raw"] = np.where(frame["place_odds"].gt(1), 1.0 / frame["place_odds"], np.nan)
    paid_places = np.minimum(frame["field_size"], np.where(frame["field_size"].ge(7), 3.0, 2.0))
    place_total = frame.groupby("race_id")["place_market_raw"].transform("sum")
    frame["place_market_probability"] = frame["place_market_raw"] / place_total * paid_places
    frame["race_no"] = pd.to_numeric(frame["race_no"], errors="coerce").fillna(0).astype(int)
    frame["horse_no"] = pd.to_numeric(_series(frame, "horse_no"), errors="coerce")
    frame["horse_no"] = frame["horse_no"].fillna(frame.groupby("race_id").cumcount().add(100))
    for column in (
        "horse_age", "horse_rating", "declared_weight", "actual_weight", "draw", "distance", "prize",
    ):
        frame[column] = pd.to_numeric(_series(frame, column), errors="coerce")
    frame["race_class"] = pd.to_numeric(
        _series(frame, "race_class").astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    )
    for column in ("venue", "course", "going"):
        frame[column] = _series(frame, column, "UNKNOWN").fillna("UNKNOWN").astype(str)
    frame["horse_colour"] = _series(frame, "horse_colour", "UNKNOWN").fillna("UNKNOWN").astype(str)
    frame["surface"] = [
        _surface(course, surface)
        for course, surface in zip(frame["course"], _series(frame, "surface"))
    ]
    frame["official_source"] = _series(frame, "source", "").astype(str).eq("official:hkjc-results").astype(float)
    frame["jockey_key"] = _series(frame, "jockey_id").fillna(_series(frame, "jockey_name")).fillna("UNKNOWN").astype(str)
    frame["trainer_key"] = _series(frame, "trainer_id").fillna(_series(frame, "trainer_name")).fillna("UNKNOWN").astype(str)
    frame["distance_band"] = (frame["distance"] / 200).round().mul(200).fillna(-1).astype(int)
    frame["finish_seconds"] = _finish_time_seconds(_series(frame, "finish_time"))
    winner_time = frame.groupby("race_id")["finish_seconds"].transform("min")
    frame["speed_ratio_raw"] = winner_time / frame["finish_seconds"]
    frame["lengths_raw"] = _series(frame, "lengths_behind").map(parse_lengths)
    frame["late_gain_raw"] = [
        _late_gain(value, result)
        for value, result in zip(_series(frame, "running_position"), frame["result"])
    ]
    positions = _series(frame, "running_position").map(_running_positions)
    for section in range(4):
        frame[f"position_sec{section + 1}_raw"] = positions.map(
            lambda values, index=section: values[index] if len(values) > index else np.nan
        )
    frame = frame.sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable").reset_index(drop=True)

    horse = frame.groupby("horse_id", sort=False)
    frame["prior_starts"] = horse.cumcount().astype(float)
    denominator = frame["prior_starts"].replace(0, np.nan)
    frame["prior_win_rate"] = (horse["target_win"].cumsum() - frame["target_win"]) / denominator
    frame["prior_top3_rate"] = (horse["target_top3"].cumsum() - frame["target_top3"]) / denominator
    frame["prior_second_count"] = horse["target_second"].cumsum() - frame["target_second"]
    frame["prior_third_count"] = horse["target_third"].cumsum() - frame["target_third"]
    frame["prior_second_rate"] = frame["prior_second_count"] / denominator
    frame["prior_third_rate"] = frame["prior_third_count"] / denominator
    frame["debut_flag"] = frame["prior_starts"].eq(0).astype(float)
    frame["prior_avg_result"] = horse["result"].transform(lambda values: values.shift().expanding().mean())
    frame["prior_avg_odds"] = horse["win_odds"].transform(lambda values: values.shift().expanding().mean())
    frame["days_since_last_race"] = horse["date"].diff().dt.days
    lag_columns = {
        "last_result": "result", "last_speed_ratio": "speed_ratio_raw",
        "last_lengths_behind": "lengths_raw", "last_late_position_gain": "late_gain_raw",
        "last_distance": "distance", "last_carried_weight": "actual_weight",
        "last_body_weight": "declared_weight", "last_rating": "horse_rating",
        "last_class": "race_class", "last_win_odds": "win_odds",
        "last_place_odds": "place_odds",
        "last_position_sec1": "position_sec1_raw", "last_position_sec2": "position_sec2_raw",
        "last_position_sec3": "position_sec3_raw", "last_position_sec4": "position_sec4_raw",
    }
    for destination, column in lag_columns.items():
        frame[destination] = horse[column].shift()
    for window in (2, 3, 5, 6):
        frame[f"avg_result_{window}"] = _rolling_shift(frame, "result", window)
    for window in (2, 3, 4, 5, 6):
        frame[f"avg_speed_ratio_{window}"] = _rolling_shift(frame, "speed_ratio_raw", window)
    for window in (2, 3, 4, 6):
        frame[f"avg_win_odds_{window}"] = _rolling_shift(frame, "win_odds", window)
        frame[f"avg_place_odds_{window}"] = _rolling_shift(frame, "place_odds", window)
    frame["total_distance_4"] = horse["distance"].transform(
        lambda values: values.shift().rolling(4, min_periods=1).sum()
    )
    frame["avg_lengths_behind_3"] = _rolling_shift(frame, "lengths_raw", 3)
    frame["avg_late_position_gain_3"] = _rolling_shift(frame, "late_gain_raw", 3)
    frame["distance_change"] = frame["distance"] - frame["last_distance"]
    frame["carried_weight_change"] = frame["actual_weight"] - frame["last_carried_weight"]
    frame["body_weight_change"] = frame["declared_weight"] - frame["last_body_weight"]
    frame["body_weight_change_pct"] = frame["body_weight_change"] / frame["last_body_weight"].replace(0, np.nan)
    frame["rating_change"] = frame["horse_rating"] - frame["last_rating"]
    frame["class_change"] = frame["race_class"] - frame["last_class"]
    frame["weight_change_per_day"] = (
        frame["body_weight_change"] / frame["days_since_last_race"].replace(0, np.nan)
    )

    for days in (84, 112, 168, 183):
        frame[f"avg_result_{days}d"] = _time_rolling_shift(frame, "result", days)
    for operation, destination in (
        ("median", "median_result_183d"),
        ("min", "best_result_183d"),
        ("max", "worst_result_183d"),
    ):
        frame[destination] = _time_rolling_shift(frame, "result", 183, operation)
    frame["avg_speed_ratio_183d"] = _time_rolling_shift(frame, "speed_ratio_raw", 183)
    frame["avg_same_distance_finish_time_183d"] = _time_rolling_shift(
        frame, "finish_seconds", 183, keys=("horse_id", "distance_band")
    )

    frame["season_id"] = np.where(
        frame["date"].dt.month.ge(7), frame["date"].dt.year, frame["date"].dt.year - 1
    )
    horse_season = frame.groupby(["horse_id", "season_id"], sort=False)
    frame["current_season_starts"] = horse_season.cumcount().astype(float)
    frame["current_season_avg_result"] = horse_season["result"].transform(
        lambda values: values.shift().expanding().mean()
    )
    season_results = frame.groupby(["horse_id", "season_id"], as_index=False)["result"].mean()
    season_results["season_id"] += 1
    season_results = season_results.rename(columns={"result": "previous_season_avg_result"})
    frame = frame.merge(season_results, on=["horse_id", "season_id"], how="left", validate="many_to_one")

    frame["jockey_last_result"] = frame.groupby("jockey_key", sort=False)["result"].shift()
    frame["jockey_last_speed_ratio"] = frame.groupby("jockey_key", sort=False)["speed_ratio_raw"].shift()
    frame["jockey_avg_result_90d"] = _time_rolling_shift(
        frame, "result", 90, keys=("jockey_key",)
    )
    frame["trainer_last_result"] = frame.groupby("trainer_key", sort=False)["result"].shift()
    frame["trainer_avg_result_90d"] = _time_rolling_shift(
        frame, "result", 90, keys=("trainer_key",)
    )

    histories = (
        (["horse_id", "distance_band"], "distance_band"), (["horse_id", "venue"], "venue"),
        (["horse_id", "course"], "course"), (["horse_id", "going"], "going"),
        (["horse_id", "surface"], "surface"), (["jockey_key"], "jockey"),
        (["trainer_key"], "trainer"), (["jockey_key", "trainer_key"], "jockey_trainer"),
        (["horse_id", "jockey_key"], "horse_jockey"),
        (["venue", "surface", "distance_band", "draw"], "draw_bias"),
    )
    for keys, prefix in histories:
        frame = _merge_history(frame, keys, prefix)

    track = _event_windows(
        frame, trackwork if trackwork is not None else pd.DataFrame(),
        "event_date", (7, 14, 30), "trackwork",
    )
    frame[track.columns] = track

    barriers = barriers.copy() if barriers is not None else pd.DataFrame()
    if not barriers.empty:
        barriers["trial_finish_seconds"] = _finish_time_seconds(barriers["finish_time"])
        barriers["last_trial_speed"] = pd.to_numeric(barriers["distance"], errors="coerce") / barriers["trial_finish_seconds"]
    trial_counts = _event_windows(frame, barriers, "event_date", (90,), "trial")
    frame["barrier_available"] = trial_counts["trial_available"]
    frame["days_since_trial"] = trial_counts["days_since_trial"]
    frame["trials_90d"] = trial_counts["trial_90d"]
    trial_values = _latest_event_values(
        frame, barriers, "event_date",
        {"last_trial_placing": "placing", "last_trial_speed": "last_trial_speed"},
        "_trial_values_available",
    )
    frame[["last_trial_placing", "last_trial_speed"]] = trial_values[["last_trial_placing", "last_trial_speed"]]

    sectionals = sectionals.copy() if sectionals is not None else pd.DataFrame()
    if not sectionals.empty:
        sectionals["section_index"] = pd.to_numeric(sectionals["section_index"], errors="coerce")
        final = sectionals.sort_values("section_index").groupby(["horse_id", "race_date"], as_index=False).tail(1)
    else:
        final = sectionals
    sectional_values = _latest_event_values(
        frame, final, "race_date",
        {"last_sectional_time": "section_time", "last_sectional_position": "position"},
        "sectionals_available",
    )
    frame[sectional_values.columns] = sectional_values

    profiles = profiles.copy() if profiles is not None else pd.DataFrame()
    if not profiles.empty:
        profiles["profile_date"] = pd.to_datetime(profiles["snapshot_at"], errors="coerce").dt.normalize()
    profile_values = _latest_event_values(
        frame,
        profiles,
        "profile_date",
        {
            "season_stakes": "season_stake", "total_stakes": "total_stake",
            "start_of_season_rating": "start_of_season_rating",
            "starts_past_10_meetings": "no_of_start_past_10_meetings",
            "profile_colour": "colour", "import_type": "import_type",
            "sire": "sire", "dam_sire": "dam_sire",
        },
        "profile_available",
    )
    frame[profile_values.columns] = profile_values
    for column in (
        "season_stakes", "total_stakes", "start_of_season_rating", "starts_past_10_meetings",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["horse_colour"] = frame["horse_colour"].where(
        frame["horse_colour"].ne("UNKNOWN"), frame["profile_colour"]
    )
    frame["rating_change_from_season_start"] = (
        frame["horse_rating"] - pd.to_numeric(frame["start_of_season_rating"], errors="coerce")
    )

    veterinary = veterinary.copy() if veterinary is not None else pd.DataFrame()
    if not veterinary.empty:
        veterinary["event_date"] = pd.to_datetime(veterinary["event_date"], errors="coerce")
    vet_windows = _event_windows(frame, veterinary, "event_date", (30, 90), "veterinary")
    frame["veterinary_available"] = vet_windows["veterinary_available"]
    frame["days_since_veterinary"] = vet_windows["days_since_veterinary"]
    frame["veterinary_events_30d"] = vet_windows["veterinary_30d"]
    frame["veterinary_events_90d"] = vet_windows["veterinary_90d"]
    for keyword in ("injury", "fracture", "surgery"):
        matching = veterinary[
            _series(veterinary, "details", "").astype(str).str.contains(keyword, case=False, na=False)
        ] if not veterinary.empty else veterinary
        window = _event_windows(frame, matching, "event_date", (365,), keyword)
        frame[f"{keyword}_events_365d"] = window[f"{keyword}_365d"]

    movements = movements.copy() if movements is not None else pd.DataFrame()
    if not movements.empty:
        movements["event_date"] = pd.to_datetime(
            _series(movements, "event_date").fillna(_series(movements, "arrival_date")),
            errors="coerce",
        )
    movement_windows = _event_windows(frame, movements, "event_date", (365,), "movement")
    frame["movement_available"] = movement_windows["movement_available"]
    frame["days_since_movement"] = movement_windows["days_since_movement"]
    frame["movements_365d"] = movement_windows["movement_365d"]
    hk_arrivals = movements[
        _series(movements, "to", "").astype(str).str.contains("HONG KONG|SHA TIN", case=False, na=False)
    ] if not movements.empty else movements
    arrival_windows = _event_windows(frame, hk_arrivals, "event_date", (), "hk_arrival")
    frame["days_since_hk_arrival"] = arrival_windows["days_since_hk_arrival"]

    rank_specs = {
        "age_rank": ("horse_age", False),
        "rating_rank": ("horse_rating", False),
        "carried_weight_rank": ("actual_weight", True),
        "body_weight_rank": ("declared_weight", True),
        "carried_weight_change_rank": ("carried_weight_change", True),
        "prior_speed_rank": ("last_speed_ratio", False),
        "distance_experience_rank": ("distance_band_starts", False),
        "prior_form_rank": ("prior_avg_result", True),
        "last_position_sec1_rank": ("last_position_sec1", True),
        "last_position_sec2_rank": ("last_position_sec2", True),
        "last_position_sec3_rank": ("last_position_sec3", True),
        "last_position_sec4_rank": ("last_position_sec4", True),
    }
    for destination, (source_column, ascending) in rank_specs.items():
        frame[destination] = _within_race_rank(frame, source_column, ascending)

    frame["poly_age_sq"] = frame["horse_age"].pow(2)
    frame["poly_rating_sq"] = frame["horse_rating"].pow(2)
    frame["poly_distance_sq"] = frame["distance"].pow(2)
    frame["poly_draw_sq"] = frame["draw"].pow(2)
    frame["poly_rating_prior_win"] = frame["horse_rating"] * frame["prior_win_rate"]
    frame["poly_rating_recent_speed"] = frame["horse_rating"] * frame["last_speed_ratio"]
    frame["poly_age_distance"] = frame["horse_age"] * frame["distance"]
    frame["poly_draw_distance"] = frame["draw"] * frame["distance"]
    frame["poly_weight_change_distance"] = frame["carried_weight_change"] * frame["distance"]
    frame["poly_jockey_trainer_win"] = frame["jockey_win_rate"] * frame["trainer_win_rate"]

    for feature in NOTEBOOK_RICH_SCHEMA.features:
        if feature not in frame:
            frame[feature] = np.nan if feature in NOTEBOOK_RICH_SCHEMA.numeric else "UNKNOWN"
    for feature in NOTEBOOK_RICH_SCHEMA.categorical:
        frame[feature] = frame[feature].fillna("UNKNOWN").astype(str)
    return frame.drop(columns=[
        "target_top3", "target_second", "target_third", "last_class", "profile_colour",
    ])


def load_full_rich_history(
    legacy_runs_path: Path,
    legacy_races_path: Path,
    canonical_runners_path: Path,
    trackwork_path: Path | None = None,
    barriers_path: Path | None = None,
    sectionals_path: Path | None = None,
    profiles_path: Path | None = None,
    veterinary_path: Path | None = None,
    movements_path: Path | None = None,
) -> pd.DataFrame:
    runs = pd.read_csv(legacy_runs_path)
    races = pd.read_csv(legacy_races_path)
    legacy = runs.merge(races, on="race_id", how="inner", validate="many_to_one")
    legacy = legacy[pd.to_datetime(legacy["date"]).lt("2005-01-01")].copy()
    legacy["race_id"] = "legacy:" + legacy["race_id"].astype(str)
    legacy["horse_id"] = "legacy:" + legacy["horse_id"].astype(str)
    legacy["source"] = "kaggle:gdaley-hkracing"
    legacy["horse_age_reference_source"] = "kaggle:gdaley-hkracing-race-row"
    legacy["horse_age_reference_year"] = np.nan
    legacy["horse_age_reference_value"] = legacy["horse_age"]
    legacy["horse_age_year_offset"] = 0.0
    legacy["horse_age_identity_method"] = "source-anonymized-horse-id"
    legacy["course"] = legacy["config"]
    position_columns = [column for column in runs.columns if column.startswith("position_sec")]
    legacy["running_position"] = legacy[position_columns].apply(
        lambda row: " ".join(str(int(value)) for value in row.dropna()), axis=1
    )

    canonical = pd.read_csv(canonical_runners_path, low_memory=False)
    canonical["date"] = canonical["race_date"]
    canonical["horse_rating"] = canonical["rating"]
    canonical["race_id"] = (
        "canonical:" + canonical["race_date"].astype(str) + "|" + canonical["venue"].astype(str)
        + "|" + canonical["race_no"].astype(int).astype(str)
    )
    canonical["horse_id"] = "canonical:" + canonical["horse_id"].astype(str)
    columns = sorted(set(legacy.columns) | set(canonical.columns))
    source = pd.concat([legacy.reindex(columns=columns), canonical.reindex(columns=columns)], ignore_index=True)

    def optional(path: Path | None) -> pd.DataFrame:
        data = pd.read_csv(path, low_memory=False) if path and path.exists() else pd.DataFrame()
        if not data.empty:
            data["horse_id"] = "canonical:" + data["horse_id"].astype(str)
        return data

    return prepare_rich_runner_dataset(
        source,
        optional(trackwork_path),
        optional(barriers_path),
        optional(sectionals_path),
        optional(profiles_path),
        optional(veterinary_path),
        optional(movements_path),
    )
