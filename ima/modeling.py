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
    conditional: "RaceConditionalLogitModel | None" = None

    def fit(self, frame: pd.DataFrame) -> "RaceProbabilityModel":
        if self.kind == "benter_conditional_logit":
            self.conditional = RaceConditionalLogitModel(
                l2=float((self.parameters or {}).get("l2", 1.0)),
                max_iter=int((self.parameters or {}).get("max_iter", 300)),
                feature_schema=self.feature_schema,
            ).fit(frame)
            return self
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
        if self.kind == "benter_conditional_logit":
            if self.conditional is None:
                raise RuntimeError("Model has not been fitted")
            return self.conditional.predict_proba(frame)
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted")
        raw = self.pipeline.predict_proba(frame[list(self.feature_schema.features)])[:, 1]
        return normalize_by_race(raw, frame["race_id"])


@dataclass
class RaceConditionalLogitModel:
    """Race-level multinomial likelihood with no intercept or market features."""

    l2: float = 1.0
    max_iter: int = 300
    feature_schema: FeatureSchema = BASELINE_SCHEMA
    preprocessor: ColumnTransformer | None = None
    coefficients: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "RaceConditionalLogitModel":
        if self.l2 < 0 or self.max_iter < 1:
            raise ValueError("Invalid conditional-logit regularization or iteration count")
        winners = frame.groupby("race_id")["target_win"].sum()
        if len(winners) == 0 or not np.allclose(winners.to_numpy(dtype=float), 1.0):
            raise ValueError("Conditional logit requires exactly one winner per race")
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ])
        self.preprocessor = ColumnTransformer([
            ("numeric", numeric, list(self.feature_schema.numeric)),
            ("categorical", categorical, list(self.feature_schema.categorical)),
        ], sparse_threshold=1.0)
        x = self.preprocessor.fit_transform(frame[list(self.feature_schema.features)])
        race_codes = pd.factorize(frame["race_id"], sort=False)[0]
        y = frame["target_win"].to_numpy(dtype=float)
        race_count = len(winners)

        def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
            return conditional_loss_gradient(coef, x, y, race_codes, race_count, self.l2)

        result = minimize(
            objective, np.zeros(x.shape[1], dtype=float), jac=True,
            method="L-BFGS-B", options={"maxiter": self.max_iter},
        )
        if not result.success and result.status != 1:
            raise RuntimeError(f"Conditional-logit fit failed: {result.message}")
        self.coefficients = np.asarray(result.x, dtype=float)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.coefficients is None:
            raise RuntimeError("Conditional-logit model has not been fitted")
        x = self.preprocessor.transform(frame[list(self.feature_schema.features)])
        codes, races = pd.factorize(frame["race_id"], sort=False)
        return _conditional_softmax(np.asarray(x @ self.coefficients).ravel(), codes, len(races))


def _conditional_softmax(scores: np.ndarray, race_codes: np.ndarray, race_count: int) -> np.ndarray:
    maxima = np.full(race_count, -np.inf)
    np.maximum.at(maxima, race_codes, scores)
    weights = np.exp(scores - maxima[race_codes])
    totals = np.bincount(race_codes, weights=weights, minlength=race_count)
    return weights / totals[race_codes]


def conditional_loss_gradient(
    coefficients: np.ndarray, x: Any, target: np.ndarray,
    race_codes: np.ndarray, race_count: int, l2: float,
) -> tuple[float, np.ndarray]:
    scores = np.asarray(x @ coefficients).ravel()
    probabilities = _conditional_softmax(scores, race_codes, race_count)
    loss = -float(target @ np.log(np.clip(probabilities, 1e-12, 1.0))) / race_count
    loss += 0.5 * l2 * float(coefficients @ coefficients)
    gradient = np.asarray(x.T @ (probabilities - target)).ravel() / race_count
    return loss, gradient + l2 * coefficients


def _probability_log_features(columns, race_ids):
    """Arrays are positional: callers must align every column to the same runners."""
    ids = np.asarray(race_ids)
    if ids.ndim != 1 or not len(ids) or pd.isna(ids).any():
        raise ValueError("Race IDs must be a nonempty one-dimensional array without missing IDs")
    features = []
    for column in columns:
        values = np.asarray(column, dtype=float)
        if values.shape != ids.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Probabilities must be finite, nonnegative and aligned to race IDs")
        features.append(np.log(np.clip(values, 1e-12, 1.0)))
    codes, races = pd.factorize(ids, sort=False)
    return np.column_stack(features), codes, len(races)


def _group_log_softmax(scores, codes, race_count):
    if not np.isfinite(scores).all():
        raise ValueError("Log probability scores must be finite")
    maxima = np.full(race_count, -np.inf)
    np.maximum.at(maxima, codes, scores)
    shifted = scores - maxima[codes]
    totals = np.bincount(codes, weights=np.exp(shifted), minlength=race_count)
    return shifted - np.log(totals[codes])


def _calibration_target(frame, codes, race_count):
    target = frame["target_probability"].to_numpy(dtype=float)
    if (target.shape != codes.shape or not np.isfinite(target).all()
            or (target < 0).any()
            or not np.allclose(np.bincount(codes, weights=target, minlength=race_count),
                               1.0, rtol=0, atol=1e-10)):
        raise ValueError("Calibration targets must be finite, nonnegative and sum to one per race")
    return target


def _blend_loss_gradient(weights, features, target, codes, race_count):
    log_probability = _group_log_softmax(features @ weights, codes, race_count)
    loss = -float(target @ log_probability) / race_count
    gradient = features.T @ (np.exp(log_probability) - target) / race_count
    return loss, gradient


def _fit_blend(columns, frame, initial):
    features, codes, race_count = _probability_log_features(columns, frame["race_id"])
    target = _calibration_target(frame, codes, race_count)

    def objective(weights):
        return _blend_loss_gradient(weights, features, target, codes, race_count)

    result = minimize(objective, np.asarray(initial, dtype=float), jac=True,
                      method="L-BFGS-B", bounds=[(0.0, 4.0)] * len(initial),
                      options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000})
    weights = np.asarray(result.x, dtype=float)
    if (not result.success or weights.shape != (len(initial),)
            or not np.isfinite(weights).all() or not np.isfinite(result.fun)
            or (weights < 0).any() or (weights > 4).any()):
        raise RuntimeError(f"Market blend fit failed: {result.message}")
    loss, gradient = objective(weights)
    # This residual is zero at a valid constrained optimum, including zero weights.
    projected = weights - np.clip(weights - gradient, 0.0, 4.0)
    if (not np.isfinite(loss) or not np.isfinite(gradient).all()
            or not np.isclose(result.fun, loss, rtol=1e-8, atol=1e-10)
            or np.max(np.abs(projected)) > 1e-5):
        raise RuntimeError("Market blend fit failed finite/objective/projected-gradient checks")
    return weights


def _blend_transform(columns, race_ids, weights):
    features, codes, race_count = _probability_log_features(columns, race_ids)
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Blend weights must be finite")
    return np.exp(_group_log_softmax(features @ weights, codes, race_count))


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    @classmethod
    def fit(cls, probabilities: np.ndarray, frame: pd.DataFrame) -> "TemperatureCalibrator":
        features, codes, race_count = _probability_log_features([probabilities], frame["race_id"])
        target = _calibration_target(frame, codes, race_count)

        def objective(log_t):
            log_probability = _group_log_softmax(features[:, 0] * np.exp(-log_t), codes, race_count)
            return -float(target @ log_probability) / race_count

        result = minimize_scalar(
            objective,
            bounds=(-2.5, 2.5),
            method="bounded",
        )
        if (not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun)
                or not -2.5 <= result.x <= 2.5
                or not np.isclose(result.fun, objective(result.x), rtol=1e-8, atol=1e-10)):
            raise RuntimeError(f"Temperature calibration failed: {result.message}")
        return cls(float(np.exp(result.x)))

    def transform(self, probabilities: np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
        return temperature_scale(probabilities, race_ids, self.temperature)


def temperature_scale(
    probabilities: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    temperature: float,
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return _blend_transform([probabilities], race_ids, [1.0 / temperature])


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
        weights = _fit_blend([fundamental, market], frame, [1.0, 1.0])
        return cls(float(weights[0]), float(weights[1]))

    def transform(
        self,
        fundamental: np.ndarray,
        market: np.ndarray,
        race_ids: pd.Series | np.ndarray,
    ) -> np.ndarray:
        return blend_probabilities(
            fundamental, market, race_ids, self.fundamental_weight, self.market_weight
        )


@dataclass(frozen=True)
class MultiMarketBlend:
    fundamental_weight: float
    win_market_weight: float
    place_market_weight: float

    @classmethod
    def fit(
        cls,
        fundamental: np.ndarray,
        win_market: np.ndarray,
        place_market: np.ndarray,
        frame: pd.DataFrame,
    ) -> "MultiMarketBlend":
        columns = _multi_market_columns(fundamental, win_market, place_market, frame["race_id"])
        weights = _fit_blend(columns, frame, [1.0, 1.0, 0.5])
        return cls(float(weights[0]), float(weights[1]), float(weights[2]))

    def transform(
        self,
        fundamental: np.ndarray,
        win_market: np.ndarray,
        race_ids: pd.Series | np.ndarray,
        place_market: np.ndarray | None = None,
    ) -> np.ndarray:
        place = win_market if place_market is None else place_market
        return blend_multi_market_probabilities(
            fundamental,
            win_market,
            place,
            race_ids,
            self.fundamental_weight,
            self.win_market_weight,
            self.place_market_weight,
        )


def blend_probabilities(
    fundamental: np.ndarray,
    market: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    fundamental_weight: float,
    market_weight: float,
) -> np.ndarray:
    return _blend_transform([fundamental, market], race_ids, [fundamental_weight, market_weight])


def _multi_market_columns(fundamental, win_market, place_market, race_ids):
    _probability_log_features([fundamental, win_market], race_ids)
    win = normalize_by_race(np.asarray(win_market, dtype=float), race_ids)
    place = np.asarray(place_market, dtype=float)
    if place.shape != win.shape:
        raise ValueError("Place probabilities must be aligned to race IDs")
    valid_place = np.isfinite(place) & (place > 0)
    place = normalize_by_race(np.where(valid_place, place, win), race_ids)
    return [fundamental, win, place]


def blend_multi_market_probabilities(
    fundamental: np.ndarray,
    win_market: np.ndarray,
    place_market: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    fundamental_weight: float,
    win_market_weight: float,
    place_market_weight: float,
) -> np.ndarray:
    columns = _multi_market_columns(fundamental, win_market, place_market, race_ids)
    return _blend_transform(columns, race_ids, [fundamental_weight, win_market_weight, place_market_weight])


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
