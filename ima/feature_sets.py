"""Versioned model feature contracts and research provenance."""

from __future__ import annotations

from dataclasses import dataclass

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return (*self.numeric, *self.categorical)


BASELINE_SCHEMA = FeatureSchema("baseline-v1", NUMERIC_FEATURES, CATEGORICAL_FEATURES)

RICH_NUMERIC_FEATURES = (
    "horse_rating", "declared_weight", "actual_weight", "draw", "distance",
    "race_class", "prize", "field_size", "official_source",
    "prior_starts", "prior_win_rate", "prior_top3_rate", "prior_avg_result",
    "prior_avg_odds", "days_since_last_race", "last_result", "avg_result_2",
    "avg_result_3", "avg_result_5", "last_speed_ratio", "avg_speed_ratio_3",
    "avg_speed_ratio_5", "last_lengths_behind", "avg_lengths_behind_3",
    "last_late_position_gain", "avg_late_position_gain_3", "last_distance",
    "distance_change", "last_carried_weight", "carried_weight_change",
    "last_body_weight", "body_weight_change", "body_weight_change_pct",
    "last_rating", "rating_change", "class_change",
    "distance_band_starts", "distance_band_win_rate", "distance_band_top3_rate",
    "venue_starts", "venue_win_rate", "course_starts", "course_win_rate",
    "going_starts", "going_win_rate", "surface_starts", "surface_win_rate",
    "jockey_starts", "jockey_win_rate", "jockey_top3_rate",
    "trainer_starts", "trainer_win_rate", "trainer_top3_rate",
    "jockey_trainer_starts", "jockey_trainer_win_rate",
    "horse_jockey_starts", "horse_jockey_win_rate",
    "draw_bias_starts", "draw_bias_win_rate", "draw_bias_top3_rate",
    "trackwork_available", "days_since_trackwork", "trackwork_7d",
    "trackwork_14d", "trackwork_30d", "barrier_available", "days_since_trial",
    "trials_90d", "last_trial_placing", "last_trial_speed",
    "sectionals_available", "last_sectional_time", "last_sectional_position",
)

RICH_CATEGORICAL_FEATURES = ("venue", "course", "going", "surface")
RICH_SCHEMA = FeatureSchema("benter-rich-v1", RICH_NUMERIC_FEATURES, RICH_CATEGORICAL_FEATURES)


FEATURE_FAMILIES = {
    "current_race": (
        "horse_rating", "declared_weight", "actual_weight", "draw", "distance",
        "race_class", "prize", "field_size", "venue", "course", "going", "surface",
    ),
    "source_quality": ("official_source",),
    "current_condition": (
        "days_since_last_race", "last_result", "avg_result_2", "avg_result_3",
        "avg_result_5", "trackwork_available", "days_since_trackwork",
        "trackwork_7d", "trackwork_14d", "trackwork_30d", "barrier_available",
        "days_since_trial", "trials_90d", "last_trial_placing", "last_trial_speed",
    ),
    "past_performance": (
        "prior_starts", "prior_win_rate", "prior_top3_rate", "prior_avg_result",
        "prior_avg_odds", "last_speed_ratio", "avg_speed_ratio_3",
        "avg_speed_ratio_5", "last_lengths_behind", "avg_lengths_behind_3",
        "last_late_position_gain", "avg_late_position_gain_3",
        "sectionals_available", "last_sectional_time", "last_sectional_position",
    ),
    "performance_adjustments": (
        "last_carried_weight", "carried_weight_change", "last_body_weight",
        "body_weight_change", "body_weight_change_pct", "last_rating",
        "rating_change", "class_change", "jockey_starts", "jockey_win_rate",
        "jockey_top3_rate", "trainer_starts", "trainer_win_rate",
        "trainer_top3_rate", "jockey_trainer_starts", "jockey_trainer_win_rate",
        "horse_jockey_starts", "horse_jockey_win_rate",
    ),
    "preferences": (
        "last_distance", "distance_change", "distance_band_starts",
        "distance_band_win_rate", "distance_band_top3_rate", "venue_starts",
        "venue_win_rate", "course_starts", "course_win_rate", "going_starts",
        "going_win_rate", "surface_starts", "surface_win_rate",
    ),
    "post_position": ("draw_bias_starts", "draw_bias_win_rate", "draw_bias_top3_rate"),
}


FEATURE_ORIGINS = {
    "benter": tuple(RICH_SCHEMA.features),
    "simplified_notebook": (
        "prior_starts", "last_result", "avg_result_2", "last_speed_ratio",
        "last_distance", "last_carried_weight", "last_body_weight",
    ),
    "trackx_notebook": (
        "days_since_last_race", "avg_result_3", "avg_result_5",
        "avg_speed_ratio_3", "avg_speed_ratio_5", "body_weight_change",
        "body_weight_change_pct", "distance_change", "jockey_win_rate",
        "trainer_win_rate", "prior_avg_odds",
    ),
    "official_hkjc_recovery": (
        "distance", "race_class", "prize", "course", "going",
        "last_lengths_behind", "avg_lengths_behind_3",
        "last_late_position_gain", "avg_late_position_gain_3",
    ),
}


BENTER_COVERAGE = (
    {"factor": "recent race performance", "status": "supported"},
    {"factor": "time since last race", "status": "supported"},
    {"factor": "recent workouts", "status": "partial", "note": "2015-2017 third-party coverage"},
    {"factor": "horse age", "status": "unsupported", "note": "not reliably point-in-time"},
    {"factor": "past finishing positions", "status": "supported"},
    {"factor": "lengths behind winner", "status": "supported"},
    {"factor": "normalized past times", "status": "supported"},
    {"factor": "strength of competition", "status": "partial", "note": "class and rating movement proxies"},
    {"factor": "past weight carried", "status": "supported"},
    {"factor": "jockey contribution", "status": "supported"},
    {"factor": "bad luck adjustment", "status": "unsupported", "note": "no validated pre-race coding"},
    {"factor": "post-position bias", "status": "supported"},
    {"factor": "current weight and jockey ability", "status": "supported"},
    {"factor": "distance preference", "status": "supported"},
    {"factor": "surface and going preference", "status": "supported"},
    {"factor": "specific track preference", "status": "supported"},
)


def validate_feature_contract() -> None:
    features = RICH_SCHEMA.features
    if len(features) != len(set(features)):
        raise ValueError("Rich feature names must be unique")
    assigned = {feature for values in FEATURE_FAMILIES.values() for feature in values}
    missing = set(features) - assigned
    if missing:
        raise ValueError(f"Rich features without a family: {sorted(missing)}")