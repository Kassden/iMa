"""Registered train-fitted transforms for research recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class TransformError(ValueError):
    """Raised when a transform request is invalid or unsafe."""


@dataclass(frozen=True)
class TransformSpec:
    kind: str
    parameters: dict[str, Any]


@dataclass
class FittedResearchTransforms:
    """Serializable, train-fitted transform state for one recipe fold."""

    specs: tuple[TransformSpec, ...]
    clip_bounds: dict[str, tuple[float, float]]

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        specs: tuple[TransformSpec, ...],
    ) -> "FittedResearchTransforms":
        bounds: dict[str, tuple[float, float]] = {}
        for spec in specs:
            validate_transform_spec(spec)
            columns = tuple(spec.parameters.get("columns", ()))
            missing = [column for column in columns if column not in train]
            if missing:
                raise TransformError(f"transform columns missing from train: {missing}")
            empty = [
                column for column in columns
                if pd.to_numeric(train[column], errors="coerce").notna().sum() == 0
            ]
            if empty:
                raise TransformError(f"transform columns have no training observations: {empty}")
            if spec.kind != "clip_numeric_quantiles":
                continue
            lower = float(spec.parameters.get("lower", 0.01))
            upper = float(spec.parameters.get("upper", 0.99))
            for column in tuple(spec.parameters["columns"]):
                values = pd.to_numeric(train[column], errors="coerce")
                bounds[column] = (float(values.quantile(lower)), float(values.quantile(upper)))
        return cls(specs=specs, clip_bounds=bounds)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for spec in self.specs:
            if spec.kind == "clip_numeric_quantiles":
                for column in tuple(spec.parameters["columns"]):
                    if column not in output:
                        raise TransformError(f"clip columns missing from frame: {[column]}")
                    lo, hi = self.clip_bounds[column]
                    output[column] = pd.to_numeric(output[column], errors="coerce").clip(lo, hi)
            elif spec.kind == "race_relative_rank":
                output = _apply_race_relative_rank([output], spec)[0]
            else:  # pragma: no cover - specs are validated during fit.
                raise TransformError(f"Unknown transform kind: {spec.kind}")
        return output


REGISTERED_TRANSFORMS = {
    "clip_numeric_quantiles",
    "race_relative_rank",
}


def validate_transform_spec(spec: TransformSpec) -> None:
    if spec.kind not in REGISTERED_TRANSFORMS:
        raise TransformError(f"Unknown transform kind: {spec.kind}")
    if spec.kind == "clip_numeric_quantiles":
        lower = float(spec.parameters.get("lower", 0.01))
        upper = float(spec.parameters.get("upper", 0.99))
        columns = tuple(spec.parameters.get("columns", ()))
        if not columns:
            raise TransformError("clip_numeric_quantiles requires columns")
        if not 0 <= lower < upper <= 1:
            raise TransformError("clip quantiles must satisfy 0 <= lower < upper <= 1")
    if spec.kind == "race_relative_rank":
        columns = tuple(spec.parameters.get("columns", ()))
        if not columns:
            raise TransformError("race_relative_rank requires columns")


def apply_research_transforms(
    train: pd.DataFrame,
    *frames: pd.DataFrame,
    specs: tuple[TransformSpec, ...],
) -> tuple[pd.DataFrame, ...]:
    """Fit registered transforms on train and apply them to train plus frames."""
    fitted = FittedResearchTransforms.fit(train, specs)
    return tuple(fitted.transform(frame) for frame in (train, *frames))


def _apply_clip(frames: list[pd.DataFrame], spec: TransformSpec) -> list[pd.DataFrame]:
    columns = tuple(spec.parameters["columns"])
    lower = float(spec.parameters.get("lower", 0.01))
    upper = float(spec.parameters.get("upper", 0.99))
    train = frames[0]
    missing = [column for column in columns if column not in train.columns]
    if missing:
        raise TransformError(f"clip columns missing from train: {missing}")
    bounds = {
        column: (
            pd.to_numeric(train[column], errors="coerce").quantile(lower),
            pd.to_numeric(train[column], errors="coerce").quantile(upper),
        )
        for column in columns
    }
    for frame in frames:
        missing_frame = [column for column in columns if column not in frame.columns]
        if missing_frame:
            raise TransformError(f"clip columns missing from frame: {missing_frame}")
        for column, (lo, hi) in bounds.items():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lo, hi)
    return frames


def _apply_race_relative_rank(frames: list[pd.DataFrame], spec: TransformSpec) -> list[pd.DataFrame]:
    columns = tuple(spec.parameters["columns"])
    suffix = str(spec.parameters.get("suffix", "_race_rank"))
    for frame in frames:
        missing = [column for column in ("race_id", *columns) if column not in frame.columns]
        if missing:
            raise TransformError(f"rank columns missing from frame: {missing}")
        for column in columns:
            frame[f"{column}{suffix}"] = (
                pd.to_numeric(frame[column], errors="coerce")
                .groupby(frame["race_id"])
                .rank(pct=True, method="average")
                .replace([np.inf, -np.inf], np.nan)
            )
    return frames
