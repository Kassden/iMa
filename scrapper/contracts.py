"""Stable contracts between live providers and the legacy iMa models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MODEL_FEATURE_COLUMNS = (
    "horse_no_y",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "horse_age",
    "declared_weight",
    "actual_weight",
    "draw",
    "win_odds",
    "place_odds",
    "last_speed",
    "avg_2last",
    "won_odds",
    "second_count",
    "third_count",
    "exp",
    "raced",
    "fin_time",
    "prev_dist",
    "race_class",
    "horse_country",
    "horse_type",
    "horse_rating",
    "horse_gear",
    "cum_avg_prev_resu",
    "prev_resu",
    "prev_odds",
    "prev_wt",
    "prev_declar_wt",
)

COMPACT_MODEL_FEATURE_COLUMNS = (
    "horse_id",
    "horse_no_y",
    "horse_age",
    "declared_weight",
    "actual_weight",
    "draw",
    "win_odds",
    "place_odds",
    "cum_avg_prev_resu",
    "prev_resu",
)

LIVE_REQUIRED_COLUMNS = (
    "horse_no_y",
    "horse_id",
    "jockey_id",
    "trainer_id",
    "declared_weight",
    "actual_weight",
    "draw",
    "win_odds",
    "race_class",
    "horse_rating",
)

HISTORY_REQUIRED_COLUMNS = (
    "horse_age",
    "last_speed",
    "avg_2last",
    "won_odds",
    "second_count",
    "third_count",
    "exp",
    "raced",
    "fin_time",
    "prev_dist",
    "horse_country",
    "horse_type",
    "cum_avg_prev_resu",
    "prev_resu",
    "prev_odds",
    "prev_wt",
    "prev_declar_wt",
)


@dataclass(frozen=True)
class NormalizedRunner:
    race_id: str
    race_no: int
    race_date: str
    venue_code: str
    post_time: str | None
    race_status: str
    field_size: int
    horse_no: str
    horse_id: str
    horse_page_id: str
    horse_name: str
    runner_status: str
    jockey_id: str | None
    trainer_id: str | None
    declared_weight: float | None
    actual_weight: float | None
    draw: int | None
    win_odds: float | None
    place_odds: float | None
    race_class: str | None
    horse_rating: float | None
    horse_gear: str | None
    last6run: str | None
    odds_updated_at: str | None
    received_at: str


@dataclass
class ModelRow:
    race_id: str
    horse_no: str
    values: dict[str, Any]
    missing_features: list[str] = field(default_factory=list)
    prediction_ready: bool = False

    def csv_row(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "horse_no_x": self.horse_no,
            **{name: self.values.get(name) for name in MODEL_FEATURE_COLUMNS},
            "prediction_ready": self.prediction_ready,
            "missing_features": ";".join(self.missing_features),
        }
