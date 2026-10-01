"""Bounded declarative feature synthesis, independent of planner prose."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def content_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:24]


class DiscoverySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    entities: tuple[Literal["horse", "jockey", "trainer"], ...] = ("horse",)
    measurements: tuple[Literal["speed_mps", "beaten_lengths", "carried_weight"], ...] = ("speed_mps",)
    aggregates: tuple[Literal["count", "mean", "std", "min", "max"], ...] = ("count", "mean", "std")
    windows_days: tuple[int | None, ...] = (90, 365)
    max_depth: int = Field(default=1, ge=1, le=2)
    max_definitions: int = Field(default=500, ge=1, le=500)
    max_selected: int = Field(default=16, ge=1, le=32)
    missingness_limit: float = Field(default=.95, ge=0, lt=1)
    correlation_threshold: float = Field(default=.95, gt=0, le=1)
    selection: Literal["quality", "mutual_information", "embedded", "sequential"] = "mutual_information"
    seed: int = 17
    availability_rule: Literal["date_only_next_day"] = "date_only_next_day"
    history_sources: tuple[str, ...] = ()
    sequence_windows: tuple[int, ...] = ()
    race_relative: bool = False
    adjusted_speed_residuals: bool = False
    residual_shrinkage: float = Field(default=5, ge=0, le=100)
    domain_history: bool = False
    recency_decay_days: int = Field(default=180, ge=7, le=730)
    diagnostics: tuple[Literal["race_permutation", "learning_curve"], ...] = ()

    @model_validator(mode="after")
    def valid_grid(self):
        for field in (self.entities, self.measurements, self.aggregates, self.windows_days):
            if not field or len(field) != len(set(field)):
                raise ValueError("Discovery grid values must be nonempty and unique")
        if any(w is not None and (w < 1 or w > 3650) for w in self.windows_days):
            raise ValueError("History windows must be 1..3650 days or null")
        if any(w not in {3,5,10} for w in self.sequence_windows):
            raise ValueError("Sequence windows must be 3, 5, or 10 prior starts")
        if len(self.history_sources)>20 or any(not s or len(s)>100 for s in self.history_sources):
            raise ValueError("Invalid history sources")
        return self

    def discovery_id(self) -> str:
        return content_id(self.model_dump(mode="json"))
