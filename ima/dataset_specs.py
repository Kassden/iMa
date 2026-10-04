"""Typed requests for isolated, immutable research datasets."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetProtocolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_train_races: int = Field(default=9000, ge=1)
    calibration_races: int = Field(default=500, ge=1)
    score_races: int = Field(default=500, ge=1)
    max_folds: int = Field(default=3, ge=1)
    fold_selection: Literal["earliest", "latest"] = "latest"
    whole_meeting_boundaries: bool = True
    final_confirmation_races: int = Field(default=500, ge=0)
    final_confirmation_race_ids: tuple[str, ...] = ()
    final_confirmation_start: str | None = None


class RacePopulationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    start_date: str | None = None
    end_date: str | None = None
    venues: tuple[str, ...] = ()
    exclusion_policy: Literal["whole_race_audit"] = "whole_race_audit"


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    parent_dataset_id: str | None = None
    raw_corpus_manifest_id: str = Field(min_length=1)
    event_policy_id: Literal["strict", "assumed_retrospective"] = "strict"
    retrospective_lags: dict[str, float] = Field(default_factory=dict)
    race_population_spec: RacePopulationSpec = Field(default_factory=RacePopulationSpec)
    history_window: int | None = Field(default=None, ge=1, description="Explicit trailing days; null preserves full history")
    target_contracts: tuple[str, ...] = ("win",)
    feature_definition_ids: tuple[str, ...] = ()
    protocol_id: str = "expansion-fullhistory-9000-500-500-v1"
    protocol: DatasetProtocolSpec = Field(default_factory=DatasetProtocolSpec)
    evaluation_population_id: str = "expansion-development-v1"
    expected_preserved_keys: tuple[tuple[str, str], ...] = ()
    rationale: str = Field(min_length=1)
    budget: dict[str, float] = Field(default_factory=dict)
    evidence_watermark: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_policy(self):
        import math
        import pandas as pd

        if self.event_policy_id == "strict" and self.retrospective_lags:
            raise ValueError("Strict requests cannot use retrospective publication assumptions")
        if self.event_policy_id == "assumed_retrospective" and not self.retrospective_lags:
            raise ValueError("Retrospective requests require explicit family-specific lag days")
        families = {"trackwork", "barrier_trials", "veterinary", "veterinary_clearance", "movements", "incidents", "sectionals"}
        if any(k not in families or not math.isfinite(v) or v < 0 for k, v in self.retrospective_lags.items()):
            raise ValueError("Invalid retrospective family or lag")
        if any(not math.isfinite(v) or v <= 0 for v in self.budget.values()):
            raise ValueError("Dataset budgets must be finite and positive")
        for value in (self.race_population_spec.start_date, self.race_population_spec.end_date, self.protocol.final_confirmation_start):
            if value is not None:
                pd.Timestamp(value)
        if self.race_population_spec.start_date and self.race_population_spec.end_date:
            if pd.Timestamp(self.race_population_spec.start_date) > pd.Timestamp(self.race_population_spec.end_date):
                raise ValueError("Population interval is reversed")
        if not self.target_contracts or set(self.target_contracts) - {"win", "top3", "speed", "finish_time", "position"}:
            raise ValueError("Unsupported target contract; no silent target substitution")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
