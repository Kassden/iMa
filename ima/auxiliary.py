"""Validation-tuned predictions for race position, time, and final market odds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import RaceSplits
from .feature_sets import FeatureSchema, NOTEBOOK_RICH_SCHEMA


DEFAULT_REGRESSOR_GRID = (
    {"learning_rate": 0.04, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    {"learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"learning_rate": 0.08, "max_leaf_nodes": 31, "l2_regularization": 3.0},
)


@dataclass(frozen=True)
class AuxiliaryTarget:
    name: str
    source: str
    log_target: bool = False
    minimum: float | None = None


TARGETS = (
    AuxiliaryTarget("position", "result", minimum=1.0),
    AuxiliaryTarget("finish_time", "finish_seconds", minimum=1.0),
    AuxiliaryTarget("win_odds", "win_odds", log_target=True, minimum=1.01),
    AuxiliaryTarget("place_odds", "place_odds", log_target=True, minimum=1.01),
)


def _preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), list(schema.numeric)),
        (
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]),
            list(schema.categorical),
        ),
    ])


def _valid_rows(frame: pd.DataFrame, target: AuxiliaryTarget) -> pd.Series:
    values = pd.to_numeric(frame[target.source], errors="coerce")
    valid = values.notna()
    if target.minimum is not None:
        valid &= values.ge(target.minimum)
    return valid


def _target_values(frame: pd.DataFrame, target: AuxiliaryTarget) -> np.ndarray:
    values = pd.to_numeric(frame[target.source], errors="coerce").to_numpy(dtype=float)
    return np.log(values) if target.log_target else values


def _restore(values: np.ndarray, target: AuxiliaryTarget) -> np.ndarray:
    restored = np.exp(values) if target.log_target else np.asarray(values, dtype=float)
    if target.minimum is not None:
        restored = np.maximum(restored, target.minimum)
    return restored


def auxiliary_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    target_name: str,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    correlation = spearmanr(actual, predicted, nan_policy="omit").statistic
    report: dict[str, float | int] = {
        "rows": int(len(actual)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }
    if target_name == "position":
        work = pd.DataFrame({
            "race_id": np.asarray(race_ids), "actual": actual, "predicted": predicted,
        }).sort_values(["race_id", "predicted"], kind="stable")
        top = work.groupby("race_id", sort=False).head(1)
        report["top_pick_win_rate"] = float(top["actual"].eq(1).mean())
        work["predicted_rank"] = work.groupby("race_id", sort=False).cumcount().add(1)
        report["rank_mae"] = float(np.mean(np.abs(work["actual"] - work["predicted_rank"])))
    return report


@dataclass
class TunedAuxiliaryModel:
    target: AuxiliaryTarget
    feature_schema: FeatureSchema
    selected_parameters: dict[str, Any]
    candidate_scores: list[dict[str, Any]]
    pipeline: Pipeline

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict(frame[list(self.feature_schema.features)])
        return _restore(raw, self.target)


@dataclass
class AuxiliaryModelBundle:
    feature_schema: FeatureSchema
    models: dict[str, TunedAuxiliaryModel] = field(default_factory=dict)
    validation_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)
    test_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        output["race_id"] = frame["race_id"].astype(str)
        if "horse_no" in frame:
            output["horse_no"] = frame["horse_no"].astype(str)
        for name, model in self.models.items():
            output[f"predicted_{name}"] = model.predict(frame)
        if "predicted_position" in output:
            output["predicted_position_rank"] = output.groupby("race_id")[
                "predicted_position"
            ].rank(method="first", ascending=True).astype(int)
        return output

    def report(self) -> dict[str, Any]:
        return {
            "feature_schema": self.feature_schema.name,
            "feature_count": len(self.feature_schema.features),
            "targets": {
                name: {
                    "source": model.target.source,
                    "selected_parameters": model.selected_parameters,
                    "candidate_scores": model.candidate_scores,
                    "validation": self.validation_metrics[name],
                    "test": self.test_metrics[name],
                }
                for name, model in self.models.items()
            },
            "market_boundary": {
                "historical_odds_features": (
                    "Lagged and rolling prior WIN/PLACE odds are available before the current race."
                ),
                "current_prices": (
                    "Current WIN, PLACE, and exotic prices are excluded from these estimators and "
                    "remain separate market observations."
                ),
            },
        }


def _fit_target(
    splits: RaceSplits,
    target: AuxiliaryTarget,
    feature_schema: FeatureSchema,
    parameter_grid: tuple[dict[str, Any], ...],
    random_state: int,
) -> tuple[TunedAuxiliaryModel, dict[str, float | int], dict[str, float | int]]:
    train_mask = _valid_rows(splits.train, target)
    validation_mask = _valid_rows(splits.validation, target)
    test_mask = _valid_rows(splits.test, target)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(f"Target {target.name} does not have complete train/validation/test coverage")

    candidate_scores = []
    fitted = []
    for parameters in parameter_grid:
        estimator_parameters = {
            "max_iter": 180,
            "random_state": random_state,
            **parameters,
        }
        pipeline = Pipeline([
            ("features", _preprocessor(feature_schema)),
            ("model", HistGradientBoostingRegressor(**estimator_parameters)),
        ])
        pipeline.fit(
            splits.train.loc[train_mask, list(feature_schema.features)],
            _target_values(splits.train.loc[train_mask], target),
        )
        predicted = _restore(
            pipeline.predict(splits.validation.loc[validation_mask, list(feature_schema.features)]),
            target,
        )
        actual = pd.to_numeric(
            splits.validation.loc[validation_mask, target.source], errors="coerce"
        ).to_numpy(dtype=float)
        metrics = auxiliary_metrics(
            actual,
            predicted,
            splits.validation.loc[validation_mask, "race_id"],
            target.name,
        )
        candidate_scores.append({"parameters": dict(parameters), **metrics})
        fitted.append(pipeline)

    selected_index = int(np.argmin([score["mae"] for score in candidate_scores]))
    model = TunedAuxiliaryModel(
        target=target,
        feature_schema=feature_schema,
        selected_parameters=dict(parameter_grid[selected_index]),
        candidate_scores=candidate_scores,
        pipeline=fitted[selected_index],
    )
    validation_metrics = {
        key: value for key, value in candidate_scores[selected_index].items() if key != "parameters"
    }
    test_prediction = model.predict(splits.test.loc[test_mask])
    test_actual = pd.to_numeric(
        splits.test.loc[test_mask, target.source], errors="coerce"
    ).to_numpy(dtype=float)
    test_metrics = auxiliary_metrics(
        test_actual,
        test_prediction,
        splits.test.loc[test_mask, "race_id"],
        target.name,
    )
    return model, validation_metrics, test_metrics


def train_auxiliary_bundle(
    splits: RaceSplits,
    feature_schema: FeatureSchema = NOTEBOOK_RICH_SCHEMA,
    parameter_grid: tuple[dict[str, Any], ...] = DEFAULT_REGRESSOR_GRID,
    random_state: int = 17,
) -> AuxiliaryModelBundle:
    bundle = AuxiliaryModelBundle(feature_schema=feature_schema)
    for target in TARGETS:
        model, validation, test = _fit_target(
            splits, target, feature_schema, parameter_grid, random_state,
        )
        bundle.models[target.name] = model
        bundle.validation_metrics[target.name] = validation
        bundle.test_metrics[target.name] = test
    return bundle
