"""Point-in-time historical dataset construction and race-grouped splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = (
    "horse_age", "horse_rating", "declared_weight", "actual_weight", "draw",
    "distance", "race_class", "surface", "prize", "field_size",
    "prior_starts", "prior_win_rate", "prior_avg_result", "prior_avg_odds",
    "last_result", "last_win_odds", "last_finish_time",
)
CATEGORICAL_FEATURES = ("venue", "config", "going", "horse_country", "horse_type", "horse_gear")
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class RaceSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _prior_expanding(series: pd.Series, operation: str) -> pd.Series:
    shifted = series.shift()
    expanding = shifted.expanding(min_periods=1)
    return getattr(expanding, operation)()


def build_runner_dataset(runs_path: Path, races_path: Path) -> pd.DataFrame:
    runs = pd.read_csv(runs_path)
    races = pd.read_csv(races_path)
    races["date"] = pd.to_datetime(races["date"], errors="raise")
    frame = runs.merge(races, on="race_id", how="inner", validate="many_to_one", suffixes=("", "_race"))
    frame = frame.sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable").reset_index(drop=True)
    frame["field_size"] = frame.groupby("race_id")["horse_no"].transform("size")
    horse_group = frame.groupby("horse_id", sort=False, group_keys=False)
    frame["prior_starts"] = horse_group.cumcount().astype(float)
    frame["prior_win_rate"] = horse_group["won"].transform(lambda s: _prior_expanding(s, "mean"))
    frame["prior_avg_result"] = horse_group["result"].transform(lambda s: _prior_expanding(s, "mean"))
    frame["prior_avg_odds"] = horse_group["win_odds"].transform(lambda s: _prior_expanding(s, "mean"))
    frame["last_result"] = horse_group["result"].shift()
    frame["last_win_odds"] = horse_group["win_odds"].shift()
    frame["last_finish_time"] = horse_group["finish_time"].shift()
    frame["target_win"] = (frame["result"] == 1).astype(int)
    winner_count = frame.groupby("race_id")["target_win"].transform("sum")
    frame["target_probability"] = frame["target_win"] / winner_count
    inverse_odds = np.where(frame["win_odds"] > 1, 1.0 / frame["win_odds"], np.nan)
    frame["market_raw"] = inverse_odds
    market_total = frame.groupby("race_id")["market_raw"].transform("sum")
    frame["market_probability"] = frame["market_raw"] / market_total
    frame["horse_age_reference_source"] = "kaggle:gdaley-hkracing-race-row"
    frame["horse_age_reference_year"] = np.nan
    frame["horse_age_reference_value"] = frame["horse_age"]
    frame["horse_age_year_offset"] = 0.0
    frame["horse_age_identity_method"] = "source-anonymized-horse-id"
    return frame


def _finish_time_seconds(values: pd.Series) -> pd.Series:
    def parse(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace(":", ".")
        parts = text.split(".")
        try:
            if len(parts) == 3:
                return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 100
            return float(text)
        except ValueError:
            return np.nan

    return values.map(parse)


def build_canonical_runner_dataset(runners_path: Path) -> pd.DataFrame:
    source = pd.read_csv(runners_path, low_memory=False)
    source["date"] = pd.to_datetime(source["race_date"], errors="raise")
    source["result"] = pd.to_numeric(
        source["result"].astype("string").str.extract(r"^(\d+)")[0], errors="coerce"
    )
    source["win_odds"] = pd.to_numeric(source["win_odds"], errors="coerce")
    source = source[source["result"].notna()].copy()
    source["race_id"] = (
        source["date"].dt.strftime("%Y-%m-%d") + "|" + source["venue"].astype(str)
        + "|" + source["race_no"].astype(int).astype(str)
    )
    source["target_win"] = source["result"].eq(1).astype(int)
    winner_count = source.groupby("race_id")["target_win"].transform("sum")
    valid_market = source["win_odds"].gt(1).groupby(source["race_id"]).transform("all")
    source = source[winner_count.gt(0) & valid_market].copy()
    source["target_probability"] = (
        source["target_win"] / source.groupby("race_id")["target_win"].transform("sum")
    )

    source["horse_no"] = pd.to_numeric(source["horse_no"], errors="coerce")
    fallback_number = source.groupby("race_id").cumcount().add(100).astype(float)
    source["horse_no"] = source["horse_no"].fillna(fallback_number)
    source["field_size"] = source.groupby("race_id")["horse_id"].transform("size")
    source["won"] = source["target_win"]
    source["finish_time"] = _finish_time_seconds(source["finish_time"])

    source["horse_age"] = pd.to_numeric(
        source["horse_age"], errors="coerce"
    ) if "horse_age" in source else np.nan
    source["horse_rating"] = pd.to_numeric(source["rating"], errors="coerce")
    source["declared_weight"] = pd.to_numeric(source["declared_weight"], errors="coerce")
    source["actual_weight"] = pd.to_numeric(source["actual_weight"], errors="coerce")
    source["draw"] = pd.to_numeric(source["draw"], errors="coerce")
    source["distance"] = pd.to_numeric(source["distance"], errors="coerce")
    source["race_class"] = pd.to_numeric(
        source["race_class"].astype("string").str.extract(r"(\d+)")[0], errors="coerce"
    )
    course_text = source["course"].astype("string").str.upper()
    source["surface"] = np.where(
        course_text.str.contains("ALL WEATHER|AWT", regex=True, na=False), 1.0,
        np.where(course_text.notna(), 0.0, np.nan),
    )
    source["prize"] = pd.to_numeric(source["prize"], errors="coerce")
    source["config"] = source["course"].fillna("UNKNOWN").astype(str)
    source["going"] = source["going"].fillna("UNKNOWN").astype(str)
    source["horse_country"] = (
        source["horse_country"].fillna("UNKNOWN").astype(str)
        if "horse_country" in source else "UNKNOWN"
    )
    source["horse_type"] = (
        source["horse_type"].fillna("UNKNOWN").astype(str)
        if "horse_type" in source else "UNKNOWN"
    )
    source["horse_gear"] = (
        source["gear"].fillna("UNKNOWN").astype(str)
        if "gear" in source else "UNKNOWN"
    )

    frame = source.sort_values(
        ["date", "race_no", "race_id", "horse_no"], kind="stable"
    ).reset_index(drop=True)
    horse_group = frame.groupby("horse_id", sort=False, group_keys=False)
    frame["prior_starts"] = horse_group.cumcount().astype(float)
    frame["prior_win_rate"] = horse_group["won"].transform(lambda s: _prior_expanding(s, "mean"))
    frame["prior_avg_result"] = horse_group["result"].transform(
        lambda s: _prior_expanding(s, "mean")
    )
    frame["prior_avg_odds"] = horse_group["win_odds"].transform(
        lambda s: _prior_expanding(s, "mean")
    )
    frame["last_result"] = horse_group["result"].shift()
    frame["last_win_odds"] = horse_group["win_odds"].shift()
    frame["last_finish_time"] = horse_group["finish_time"].shift()
    frame["market_raw"] = 1.0 / frame["win_odds"]
    frame["market_probability"] = frame["market_raw"] / frame.groupby("race_id")[
        "market_raw"
    ].transform("sum")
    return frame


def build_full_history_dataset(
    legacy_runs_path: Path,
    legacy_races_path: Path,
    canonical_runners_path: Path,
) -> pd.DataFrame:
    legacy = build_runner_dataset(legacy_runs_path, legacy_races_path)
    legacy = legacy[legacy["date"].lt(pd.Timestamp("2005-01-01"))].copy()
    legacy["race_id"] = "legacy:" + legacy["race_id"].astype(str)
    legacy["horse_id"] = "legacy:" + legacy["horse_id"].astype(str)

    canonical = build_canonical_runner_dataset(canonical_runners_path)
    canonical["race_id"] = "canonical:" + canonical["race_id"].astype(str)
    canonical["horse_id"] = "canonical:" + canonical["horse_id"].astype(str)

    columns = sorted(set(legacy.columns) | set(canonical.columns))
    frame = pd.concat(
        [legacy.reindex(columns=columns), canonical.reindex(columns=columns)],
        ignore_index=True,
    )
    return frame.sort_values(
        ["date", "race_no", "race_id", "horse_no"], kind="stable"
    ).reset_index(drop=True)


def chronological_race_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> RaceSplits:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave non-empty train, validation, and test windows")
    race_order = (
        frame[["race_id", "date", "race_no"]]
        .drop_duplicates("race_id")
        .sort_values(["date", "race_no", "race_id"], kind="stable")
    )
    count = len(race_order)
    train_end = max(1, int(count * train_fraction))
    validation_end = max(train_end + 1, int(count * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, count - 1)
    ids = race_order["race_id"].to_numpy()
    train_ids = set(ids[:train_end])
    validation_ids = set(ids[train_end:validation_end])
    test_ids = set(ids[validation_end:])
    return RaceSplits(
        train=frame[frame["race_id"].isin(train_ids)].copy(),
        validation=frame[frame["race_id"].isin(validation_ids)].copy(),
        test=frame[frame["race_id"].isin(test_ids)].copy(),
    )


def validate_runner_dataset(frame: pd.DataFrame) -> None:
    winners = frame.groupby("race_id")["target_win"].sum()
    if winners.lt(1).any():
        bad = winners[winners.lt(1)].index.tolist()[:10]
        raise ValueError(f"Races without a recorded winner: {bad}")
    target_totals = frame.groupby("race_id")["target_probability"].sum()
    if not np.allclose(target_totals.to_numpy(), 1.0):
        raise ValueError("Outcome target probabilities do not sum to one by race")
    if frame.duplicated(["race_id", "horse_no"]).any():
        raise ValueError("Duplicate runner numbers within a race")
    if frame["market_probability"].notna().any():
        totals = frame.dropna(subset=["market_probability"]).groupby("race_id")["market_probability"].sum()
        if not np.allclose(totals.to_numpy(), 1.0):
            raise ValueError("Market probabilities do not sum to one by race")
