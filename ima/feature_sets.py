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

    def contract(self) -> dict:
        return {
            "name": self.name,
            "count": len(self.features),
            "numeric_count": len(self.numeric),
            "categorical_count": len(self.categorical),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
        }


BASELINE_SCHEMA = FeatureSchema("baseline-v1", NUMERIC_FEATURES, CATEGORICAL_FEATURES)

RICH_NUMERIC_FEATURES = (
    "horse_age", "horse_rating", "declared_weight", "actual_weight", "draw", "distance",
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

RICH_CATEGORICAL_FEATURES = (
    "venue", "course", "going", "surface", "horse_country", "horse_type", "horse_gear",
)
RICH_SCHEMA = FeatureSchema("benter-rich-v1", RICH_NUMERIC_FEATURES, RICH_CATEGORICAL_FEATURES)

NOTEBOOK_NUMERIC_FEATURES = (
    "prior_second_count", "prior_third_count", "prior_second_rate", "prior_third_rate",
    "debut_flag", "last_win_odds", "last_place_odds",
    "avg_win_odds_2", "avg_win_odds_3", "avg_win_odds_4", "avg_win_odds_6",
    "avg_place_odds_2", "avg_place_odds_3", "avg_place_odds_4", "avg_place_odds_6",
    "avg_result_6", "avg_speed_ratio_2", "avg_speed_ratio_4", "avg_speed_ratio_6",
    "total_distance_4", "weight_change_per_day",
    "avg_result_84d", "avg_result_112d", "avg_result_168d", "avg_result_183d",
    "median_result_183d", "best_result_183d", "worst_result_183d",
    "avg_speed_ratio_183d", "avg_same_distance_finish_time_183d",
    "current_season_starts", "current_season_avg_result", "previous_season_avg_result",
    "jockey_last_result", "jockey_last_speed_ratio", "jockey_avg_result_90d",
    "trainer_last_result", "trainer_avg_result_90d",
    "last_position_sec1", "last_position_sec2", "last_position_sec3", "last_position_sec4",
    "age_rank", "rating_rank", "carried_weight_rank", "body_weight_rank",
    "carried_weight_change_rank", "prior_speed_rank", "distance_experience_rank",
    "prior_form_rank", "last_position_sec1_rank", "last_position_sec2_rank",
    "last_position_sec3_rank", "last_position_sec4_rank",
    "profile_available", "season_stakes", "total_stakes", "start_of_season_rating",
    "rating_change_from_season_start", "starts_past_10_meetings",
    "veterinary_available", "days_since_veterinary", "veterinary_events_30d",
    "veterinary_events_90d", "injury_events_365d", "fracture_events_365d",
    "surgery_events_365d", "movement_available", "days_since_movement",
    "movements_365d", "days_since_hk_arrival",
    "poly_age_sq", "poly_rating_sq", "poly_distance_sq", "poly_draw_sq",
    "poly_rating_prior_win", "poly_rating_recent_speed", "poly_age_distance",
    "poly_draw_distance", "poly_weight_change_distance", "poly_jockey_trainer_win",
)

NOTEBOOK_CATEGORICAL_FEATURES = (
    "horse_colour", "import_type", "sire", "dam_sire",
)

NOTEBOOK_RICH_SCHEMA = FeatureSchema(
    "notebook-rich-v2",
    (*RICH_NUMERIC_FEATURES, *NOTEBOOK_NUMERIC_FEATURES),
    (*RICH_CATEGORICAL_FEATURES, *NOTEBOOK_CATEGORICAL_FEATURES),
)

FEATURE_SCHEMAS = {
    BASELINE_SCHEMA.name: BASELINE_SCHEMA,
    RICH_SCHEMA.name: RICH_SCHEMA,
    NOTEBOOK_RICH_SCHEMA.name: NOTEBOOK_RICH_SCHEMA,
}


FEATURE_FAMILIES = {
    "current_race": (
        "horse_age", "horse_rating", "declared_weight", "actual_weight", "draw", "distance",
        "race_class", "prize", "field_size", "venue", "course", "going", "surface",
        "horse_country", "horse_type", "horse_gear",
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


NOTEBOOK_FEATURE_FAMILIES = {
    **FEATURE_FAMILIES,
    "extended_finishing_form": (
        "prior_second_count", "prior_third_count", "prior_second_rate", "prior_third_rate",
        "debut_flag", "avg_result_6", "avg_result_84d", "avg_result_112d",
        "avg_result_168d", "avg_result_183d", "median_result_183d",
        "best_result_183d", "worst_result_183d", "current_season_starts",
        "current_season_avg_result", "previous_season_avg_result",
    ),
    "historical_market_form": (
        "last_win_odds", "last_place_odds", "avg_win_odds_2", "avg_win_odds_3",
        "avg_win_odds_4", "avg_win_odds_6", "avg_place_odds_2",
        "avg_place_odds_3", "avg_place_odds_4", "avg_place_odds_6",
    ),
    "extended_speed_and_time": (
        "avg_speed_ratio_2", "avg_speed_ratio_4", "avg_speed_ratio_6",
        "total_distance_4", "weight_change_per_day", "avg_speed_ratio_183d",
        "avg_same_distance_finish_time_183d",
    ),
    "recent_connections": (
        "jockey_last_result", "jockey_last_speed_ratio", "jockey_avg_result_90d",
        "trainer_last_result", "trainer_avg_result_90d",
    ),
    "sectional_positions": (
        "last_position_sec1", "last_position_sec2", "last_position_sec3",
        "last_position_sec4",
    ),
    "within_race_ranks": (
        "age_rank", "rating_rank", "carried_weight_rank", "body_weight_rank",
        "carried_weight_change_rank", "prior_speed_rank", "distance_experience_rank",
        "prior_form_rank", "last_position_sec1_rank", "last_position_sec2_rank",
        "last_position_sec3_rank", "last_position_sec4_rank",
    ),
    "horse_profile": (
        "profile_available", "season_stakes", "total_stakes", "start_of_season_rating",
        "rating_change_from_season_start", "starts_past_10_meetings",
        "horse_colour", "import_type", "sire", "dam_sire",
    ),
    "veterinary_and_movement": (
        "veterinary_available", "days_since_veterinary", "veterinary_events_30d",
        "veterinary_events_90d", "injury_events_365d", "fracture_events_365d",
        "surgery_events_365d", "movement_available", "days_since_movement",
        "movements_365d", "days_since_hk_arrival",
    ),
    "polynomial_interactions": (
        "poly_age_sq", "poly_rating_sq", "poly_distance_sq", "poly_draw_sq",
        "poly_rating_prior_win", "poly_rating_recent_speed", "poly_age_distance",
        "poly_draw_distance", "poly_weight_change_distance", "poly_jockey_trainer_win",
    ),
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
        "horse_age", "horse_country", "horse_type", "horse_gear",
        "distance", "race_class", "prize", "course", "going",
        "last_lengths_behind", "avg_lengths_behind_3",
        "last_late_position_gain", "avg_late_position_gain_3",
    ),
    "notebook_rich_v2": (*NOTEBOOK_NUMERIC_FEATURES, *NOTEBOOK_CATEGORICAL_FEATURES),
}


BENTER_COVERAGE = (
    {"factor": "recent race performance", "status": "supported"},
    {"factor": "time since last race", "status": "supported"},
    {"factor": "recent workouts", "status": "partial", "note": "2015-2017 third-party coverage"},
    {"factor": "horse age", "status": "partial", "note": "timestamped HKJC and archived snapshot propagation"},
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
    notebook_features = NOTEBOOK_RICH_SCHEMA.features
    if len(notebook_features) != len(set(notebook_features)):
        raise ValueError("Notebook-rich feature names must be unique")
    assigned_notebook = [
        feature for values in NOTEBOOK_FEATURE_FAMILIES.values() for feature in values
    ]
    if set(assigned_notebook) != set(notebook_features):
        missing = sorted(set(notebook_features) - set(assigned_notebook))
        extra = sorted(set(assigned_notebook) - set(notebook_features))
        raise ValueError(f"Notebook-rich family mismatch: missing={missing}, extra={extra}")
    if len(assigned_notebook) != len(set(assigned_notebook)):
        raise ValueError("Notebook-rich features must belong to exactly one family")
