"""Research-only model components for matched cohort experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_sets import FeatureSchema
from .modeling import normalize_by_race, race_log_loss
from .research_evaluation import placing_metrics, ranking_metrics, regression_metrics
from .research_targets import TargetContract, apply_target_contract, target_contract


@dataclass(frozen=True)
class OffsetFitResult:
    coefficients: tuple[float, ...]
    loss: float
    gradient_check_error: float


class RaceSoftmaxOffsetModel:
    """Linear market-offset model: softmax(log(market) + Xw) by race."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = float(l2)
        self.coefficients: np.ndarray | None = None
        self.feature_columns: tuple[str, ...] = ()
        self.fit_result: OffsetFitResult | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...] | list[str],
        *,
        market_column: str = "market_probability",
    ) -> "RaceSoftmaxOffsetModel":
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        self.feature_columns = tuple(feature_columns)
        x = _centered_numeric_features(frame, self.feature_columns)
        y = frame["target_probability"].to_numpy(dtype=float)
        market = np.log(np.clip(frame[market_column].to_numpy(dtype=float), 1e-12, 1.0))
        race_ids = frame["race_id"].to_numpy()

        def objective(weights: np.ndarray) -> float:
            probabilities = _race_softmax(market + x @ weights, race_ids)
            return _mean_race_nll(probabilities, y, race_ids) + 0.5 * self.l2 * float(weights @ weights)

        def gradient(weights: np.ndarray) -> np.ndarray:
            probabilities = _race_softmax(market + x @ weights, race_ids)
            residual = probabilities - y
            race_count = pd.Series(race_ids).nunique()
            return (x.T @ residual) / race_count + self.l2 * weights

        start = np.zeros(x.shape[1], dtype=float)
        result = minimize(objective, start, jac=gradient, method="L-BFGS-B")
        self.coefficients = np.asarray(result.x, dtype=float)
        grad_error = float(check_grad(objective, gradient, self.coefficients))
        self.fit_result = OffsetFitResult(
            coefficients=tuple(float(value) for value in self.coefficients),
            loss=float(result.fun),
            gradient_check_error=grad_error,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame, *, market_column: str = "market_probability") -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Offset model has not been fitted")
        x = _centered_numeric_features(frame, self.feature_columns)
        market = np.log(np.clip(frame[market_column].to_numpy(dtype=float), 1e-12, 1.0))
        return _race_softmax(market + x @ self.coefficients, frame["race_id"].to_numpy())


@dataclass
class ResearchRegressor:
    kind: str = "ridge_regressor"
    parameters: dict[str, Any] | None = None
    pipeline: Pipeline | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        feature_schema: FeatureSchema,
        label_column: str,
    ) -> "ResearchRegressor":
        numeric = list(feature_schema.numeric)
        categorical = list(feature_schema.categorical)
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ], remainder="drop")
        params = dict(self.parameters or {})
        if self.kind == "ridge_regressor":
            estimator: Any = Ridge(**({"alpha": 1.0} | params))
        elif self.kind == "hist_gradient_regressor":
            estimator = HistGradientBoostingRegressor(**({"random_state": 42} | params))
        elif self.kind == "pairwise_ranker":
            estimator = HistGradientBoostingRegressor(**({"random_state": 42} | params))
        else:
            raise ValueError(f"Unknown research regressor: {self.kind}")
        self.pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        self.pipeline.fit(frame[list(feature_schema.features)], frame[label_column])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Regressor has not been fitted")
        return np.asarray(self.pipeline.predict(frame), dtype=float)


@dataclass
class ResearchClassifier:
    kind: str = "logit"
    parameters: dict[str, Any] | None = None
    pipeline: Pipeline | None = None
    feature_schema: FeatureSchema | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        feature_schema: FeatureSchema,
        label_column: str,
    ) -> "ResearchClassifier":
        self.feature_schema = feature_schema
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]), list(feature_schema.numeric)),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), list(feature_schema.categorical)),
        ], remainder="drop")
        params = dict(self.parameters or {})
        if self.kind == "logit":
            estimator: Any = LogisticRegression(**({"max_iter": 1000, "C": 0.5} | params))
        elif self.kind == "boosted":
            from sklearn.ensemble import HistGradientBoostingClassifier
            estimator = HistGradientBoostingClassifier(**({"random_state": 42} | params))
        else:
            raise ValueError(f"Unknown research classifier: {self.kind}")
        self.pipeline = Pipeline([("features", preprocessor), ("model", estimator)])
        self.pipeline.fit(frame[list(feature_schema.features)], frame[label_column])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None or self.feature_schema is None:
            raise RuntimeError("Classifier has not been fitted")
        return np.asarray(
            self.pipeline.predict_proba(frame[list(self.feature_schema.features)])[:, 1],
            dtype=float,
        )


def secondary_target_diagnostics(
    frame: pd.DataFrame,
    contract: TargetContract,
    predictions: np.ndarray,
) -> dict[str, float]:
    labelled = apply_target_contract(frame, contract)
    values = np.asarray(predictions, dtype=float)
    if contract.kind == "adjusted_finish_time_or_speed":
        metrics = regression_metrics(values, labelled, label_column=contract.label_column)
        return metrics | {
            "mae": metrics["race_mae"],
            "spearman": metrics["race_spearman"],
        }
    if contract.kind == "ranking_strength":
        metrics = ranking_metrics(values, labelled, label_column=contract.label_column)
        return metrics | {"spearman": metrics["race_spearman"]}
    if contract.kind == "placing_top_k":
        metrics = placing_metrics(
            values,
            labelled,
            label_column=contract.label_column,
            top_k=int(contract.parameters["top_k"]),
        )
        return metrics | {"brier": metrics["race_brier"]}
    if contract.kind == "win_probability":
        return {"race_log_loss": race_log_loss(values, labelled)}
    if contract.kind == "market_odds_forecast":
        metrics = regression_metrics(
            values,
            labelled,
            label_column=contract.label_column,
            include_rank_metrics=False,
        )
        return metrics | {"mae": metrics["race_mae"]}
    raise ValueError(f"Unsupported target diagnostics: {contract.kind}")


def offset_gradient_check(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str],
    *,
    l2: float = 1.0,
) -> float:
    model = RaceSoftmaxOffsetModel(l2=l2).fit(frame, feature_columns)
    assert model.fit_result is not None
    return model.fit_result.gradient_check_error


def _centered_numeric_features(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if not columns:
        raise ValueError("At least one feature column is required")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing offset feature columns: {missing}")
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median()).fillna(0.0)
    matrix = values.to_numpy(dtype=float)
    means = pd.DataFrame(matrix).groupby(frame["race_id"].to_numpy()).transform("mean").to_numpy()
    return matrix - means


def _race_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"race_id": race_ids, "score": scores})
    max_by_race = frame.groupby("race_id")["score"].transform("max").to_numpy()
    return normalize_by_race(np.exp(scores - max_by_race), race_ids)


def _mean_race_nll(probabilities: np.ndarray, target: np.ndarray, race_ids: np.ndarray) -> float:
    losses = -target * np.log(np.clip(probabilities, 1e-12, 1.0))
    return float(pd.Series(losses).groupby(race_ids).sum().mean())


def target_contract_for_kind(kind: str, parameters: dict[str, Any] | None = None) -> TargetContract:
    return target_contract(kind, parameters)
