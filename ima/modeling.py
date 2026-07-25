"""Race-normalized probability models, calibration, and market blending."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .feature_sets import BASELINE_SCHEMA, FeatureSchema


def normalize_by_race(values: np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"race_id": np.asarray(race_ids), "value": np.asarray(values, dtype=float)})
    frame["value"] = frame["value"].clip(lower=1e-12)
    totals = frame.groupby("race_id")["value"].transform("sum")
    return (frame["value"] / totals).to_numpy()


def apply_public_fallback(
    fundamental: np.ndarray,
    market: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    ratable: np.ndarray,
) -> np.ndarray:
    """Use public probability for unratable runners and rescale rated runners.

    This preserves first-time runners in the race without pretending that the
    fundamental model has information about them.
    """
    result = np.zeros(len(fundamental), dtype=float)
    race_ids = np.asarray(race_ids)
    ratable = np.asarray(ratable, dtype=bool)
    market = normalize_by_race(np.asarray(market, dtype=float), race_ids)
    for race_id in pd.unique(race_ids):
        mask = race_ids == race_id
        rated_mask = mask & ratable
        unrated_mask = mask & ~ratable
        result[unrated_mask] = market[unrated_mask]
        remaining = max(0.0, 1.0 - result[unrated_mask].sum())
        if rated_mask.any():
            rated = np.clip(np.asarray(fundamental)[rated_mask], 1e-12, None)
            result[rated_mask] = remaining * rated / rated.sum()
        elif unrated_mask.any():
            result[unrated_mask] = market[unrated_mask] / market[unrated_mask].sum()
    return result


def _preprocessor(feature_schema: FeatureSchema = BASELINE_SCHEMA) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(feature_schema.numeric)),
        ("categorical", categorical, list(feature_schema.categorical)),
    ])


@dataclass
class RaceProbabilityModel:
    kind: str = "logit"
    random_state: int = 17
    parameters: dict[str, Any] | None = None
    pipeline: Pipeline | None = None
    feature_schema: FeatureSchema = BASELINE_SCHEMA

    def fit(self, frame: pd.DataFrame) -> "RaceProbabilityModel":
        if self.kind == "logit":
            parameters = {"max_iter": 1000, "C": 0.5, "class_weight": "balanced"}
            parameters.update(self.parameters or {})
            estimator: Any = LogisticRegression(**parameters)
        elif self.kind == "boosted":
            parameters = {
                "learning_rate": 0.06,
                "max_iter": 180,
                "max_leaf_nodes": 31,
                "l2_regularization": 1.0,
                "random_state": self.random_state,
            }
            parameters.update(self.parameters or {})
            estimator = HistGradientBoostingClassifier(**parameters)
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")
        self.pipeline = Pipeline([("features", _preprocessor(self.feature_schema)), ("model", estimator)])
        self.pipeline.fit(frame[list(self.feature_schema.features)], frame["target_win"])
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted")
        raw = self.pipeline.predict_proba(frame[list(self.feature_schema.features)])[:, 1]
        return normalize_by_race(raw, frame["race_id"])


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    @classmethod
    def fit(cls, probabilities: np.ndarray, frame: pd.DataFrame) -> "TemperatureCalibrator":
        result = minimize_scalar(
            lambda log_t: race_log_loss(
                temperature_scale(probabilities, frame["race_id"], float(np.exp(log_t))), frame
            ),
            bounds=(-2.5, 2.5),
            method="bounded",
        )
        return cls(float(np.exp(result.x)))

    def transform(self, probabilities: np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
        return temperature_scale(probabilities, race_ids, self.temperature)


def temperature_scale(
    probabilities: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    powered = np.power(np.clip(probabilities, 1e-12, 1.0), 1.0 / temperature)
    return normalize_by_race(powered, race_ids)


@dataclass(frozen=True)
class MarketBlend:
    fundamental_weight: float
    market_weight: float

    @classmethod
    def fit(
        cls,
        fundamental: np.ndarray,
        market: np.ndarray,
        frame: pd.DataFrame,
    ) -> "MarketBlend":
        def objective(weights: np.ndarray) -> float:
            combined = blend_probabilities(fundamental, market, frame["race_id"], *weights)
            return race_log_loss(combined, frame)

        result = minimize(objective, x0=np.array([1.0, 1.0]), bounds=((0.0, 4.0), (0.0, 4.0)))
        return cls(float(result.x[0]), float(result.x[1]))

    def transform(
        self,
        fundamental: np.ndarray,
        market: np.ndarray,
        race_ids: pd.Series | np.ndarray,
    ) -> np.ndarray:
        return blend_probabilities(
            fundamental, market, race_ids, self.fundamental_weight, self.market_weight
        )


def blend_probabilities(
    fundamental: np.ndarray,
    market: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    fundamental_weight: float,
    market_weight: float,
) -> np.ndarray:
    score = np.power(np.clip(fundamental, 1e-12, 1.0), fundamental_weight)
    score *= np.power(np.clip(market, 1e-12, 1.0), market_weight)
    return normalize_by_race(score, race_ids)


def race_log_loss(probabilities: np.ndarray, frame: pd.DataFrame) -> float:
    losses = -frame["target_probability"].to_numpy() * np.log(np.clip(probabilities, 1e-12, 1.0))
    return float(pd.Series(losses).groupby(frame["race_id"].to_numpy()).sum().mean())


def race_brier_score(probabilities: np.ndarray, frame: pd.DataFrame) -> float:
    squared = np.square(probabilities - frame["target_probability"].to_numpy())
    return float(pd.Series(squared).groupby(frame["race_id"].to_numpy()).sum().mean())


def expected_calibration_error(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    bins: int = 10,
) -> float:
    bucket = np.minimum((np.asarray(probabilities) * bins).astype(int), bins - 1)
    target = frame["target_probability"].to_numpy()
    total = len(target)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += mask.sum() / total * abs(probabilities[mask].mean() - target[mask].mean())
    return float(error)


def evaluate_probabilities(probabilities: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    predicted = frame.assign(_probability=probabilities).sort_values(
        ["race_id", "_probability"], ascending=[True, False]
    )
    top_pick_win_rate = float(predicted.groupby("race_id").head(1)["target_win"].mean())
    top_three = predicted.groupby("race_id").head(3).groupby("race_id")["target_win"].max()
    winner_top3_rate = float(top_three.mean())
    winner_rank = predicted.assign(
        _rank=predicted.groupby("race_id").cumcount().add(1)
    ).loc[lambda item: item["target_win"].eq(1)].groupby("race_id")["_rank"].min()
    return {
        "race_log_loss": race_log_loss(probabilities, frame),
        "race_brier": race_brier_score(probabilities, frame),
        "ece": expected_calibration_error(probabilities, frame),
        "top_pick_win_rate": top_pick_win_rate,
        "winner_top3_rate": winner_top3_rate,
        "mean_winner_rank": float(winner_rank.mean()),
        "pseudo_r2": pseudo_r2(probabilities, frame),
    }


def pseudo_r2(probabilities: np.ndarray, frame: pd.DataFrame) -> float:
    target = frame["target_probability"].to_numpy()
    model_ll = float(np.sum(target * np.log(np.clip(probabilities, 1e-12, 1.0))))
    random_probability = 1.0 / frame.groupby("race_id")["race_id"].transform("size").to_numpy()
    random_ll = float(np.sum(target * np.log(random_probability)))
    return 1.0 - model_ll / random_ll


def incremental_pseudo_r2(
    combined: np.ndarray,
    market: np.ndarray,
    frame: pd.DataFrame,
) -> float:
    return pseudo_r2(combined, frame) - pseudo_r2(market, frame)


def disagreement_report(
    fundamental: np.ndarray,
    market: np.ndarray,
    combined: np.ndarray,
    frame: pd.DataFrame,
    bins: int = 5,
) -> list[dict[str, float | int | str]]:
    log_ratio = np.log(np.clip(fundamental, 1e-12, 1.0)) - np.log(np.clip(market, 1e-12, 1.0))
    quantiles = pd.qcut(log_ratio, q=bins, duplicates="drop")
    work = pd.DataFrame({
        "bucket": quantiles.astype(str),
        "fundamental": fundamental,
        "market": market,
        "combined": combined,
        "actual": frame["target_probability"].to_numpy(),
    })
    report = []
    for bucket, group in work.groupby("bucket", observed=True):
        report.append({
            "bucket": str(bucket),
            "runners": int(len(group)),
            "fundamental_mean": float(group["fundamental"].mean()),
            "market_mean": float(group["market"].mean()),
            "combined_mean": float(group["combined"].mean()),
            "actual_win_frequency": float(group["actual"].mean()),
        })
    return report