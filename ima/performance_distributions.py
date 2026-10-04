"""Observed-performance distributions with explicit physical/log support."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import Ridge


@dataclass
class PerformanceDistribution:
    location: np.ndarray
    scale: np.ndarray
    coordinate: str = "latent_strength"
    direction: str = "higher"
    row_keys: tuple = ()
    metadata: dict = field(default_factory=dict)
    uncertainty_kind: str = "aleatoric"

    def __post_init__(self):
        self.location = np.asarray(self.location, dtype=float)
        self.scale = np.asarray(self.scale, dtype=float)
        if self.location.ndim != 1 or self.scale.shape != self.location.shape or not len(self.scale):
            raise ValueError("Distribution location/scale shape mismatch")
        if not np.isfinite(self.location).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Distribution requires finite location and positive scale")
        if self.coordinate not in {"latent_strength", "log_speed_mps", "log_time_seconds"}:
            raise ValueError("Physical Gaussian support is unsafe; use log_speed_mps/log_time_seconds")
        expected = "lower" if self.coordinate == "log_time_seconds" else "higher"
        if self.direction != expected:
            raise ValueError("Coordinate/direction mismatch")
        if self.row_keys and len(self.row_keys) != len(self.location):
            raise ValueError("Distribution row-key mismatch")

    @property
    def variance(self):
        return self.scale ** 2

    @property
    def family(self):
        return "lognormal" if self.coordinate.startswith("log_") else "normal"

    @property
    def support(self):
        return "positive_physical_via_log" if self.coordinate.startswith("log_") else "real_latent"

    def physical_mean(self):
        if not self.coordinate.startswith("log_"):
            return self.location.copy()
        return _supported_exp(self.location + self.variance / 2)

    def physical_quantile(self, probability):
        values = self.location + self.scale * norm.ppf(probability)
        return np.exp(values) if self.coordinate.startswith("log_") else values

    def physical_samples(self, standard_normals):
        values = self.location + self.scale * np.asarray(standard_normals)
        return _supported_exp(values) if self.coordinate.startswith("log_") else values

    def diagnostics(self, observed, interval=0.9):
        if not 0 < interval < 1:
            raise ValueError("Coverage interval must be between zero and one")
        y = np.asarray(observed, dtype=float)
        if y.shape != self.location.shape or not np.isfinite(y).all():
            raise ValueError("Invalid distribution observations")
        logged = self.coordinate.startswith("log_")
        if logged and np.any(y <= 0):
            raise ValueError("Physical observations must be positive")
        z = (np.log(y) if logged else y) - self.location
        z /= self.scale
        nll = -norm.logpdf(z) + np.log(self.scale)
        if logged:
            nll += np.log(y)
        # Gaussian CRPS is reported in the fitted (log or latent) coordinate.
        crps = self.scale * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        lo, hi = self.physical_quantile((1-interval)/2), self.physical_quantile((1+interval)/2)
        return {"physical_nll": float(nll.mean()), "coordinate_crps": float(crps.mean()),
                "interval_coverage": float(np.mean((y >= lo) & (y <= hi))),
                "physical_interval_width": float(np.mean(hi-lo)), "pit_mean": float(norm.cdf(z).mean())}


def _physical_target(y, coordinate):
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite performance target")
    if coordinate.startswith("log_"):
        if np.any(values <= 0):
            raise ValueError("Physical performance must be positive")
        return np.log(values)
    if coordinate != "latent_strength":
        raise ValueError("Unsupported performance coordinate")
    return values


@dataclass
class SharedResidualPerformanceModel:
    """Point fit plus scale learned on a strictly later held-out past window."""

    feature_columns: tuple[str, ...]
    coordinate: str = "log_speed_mps"
    estimator: Any = None
    scale_floor: float = 1e-4
    fitted_mean: Any = None
    shared_scale: float | None = None
    metadata: dict = field(default_factory=dict)

    def fit(self, train, calibration, *, label_column, date_column="date"):
        if self.scale_floor <= 0 or not np.isfinite(self.scale_floor):
            raise ValueError("Invalid scale floor")
        if pd.to_datetime(train[date_column]).max() >= pd.to_datetime(calibration[date_column]).min():
            raise ValueError("Residual calibration must follow mean training")
        self.fitted_mean = clone(self.estimator if self.estimator is not None else Ridge())
        self.fitted_mean.fit(train[list(self.feature_columns)], _physical_target(train[label_column], self.coordinate))
        return self.calibrate_residuals(train, calibration, label_column=label_column, date_column=date_column)

    def calibrate_residuals(self, train, calibration, *, label_column, date_column="date"):
        if self.scale_floor <= 0 or not np.isfinite(self.scale_floor) or not len(train) or not len(calibration):
            raise ValueError("Invalid scale floor or empty residual fit population")
        if self.fitted_mean is None:
            raise RuntimeError("Mean must be fitted before residual calibration")
        if pd.to_datetime(train[date_column]).max() >= pd.to_datetime(calibration[date_column]).min():
            raise ValueError("Residual calibration must follow mean training")
        residuals = _physical_target(calibration[label_column], self.coordinate) - self.fitted_mean.predict(calibration[list(self.feature_columns)])
        self.shared_scale = max(self.scale_floor, float(np.sqrt(np.mean(residuals ** 2))))
        self.metadata = {"fit_regime": "observed_performance", "scale_fit_scope": "disjoint_past_residuals",
                         "training_cutoff": str(pd.to_datetime(train[date_column]).max()),
                         "calibration_cutoff": str(pd.to_datetime(calibration[date_column]).max()),
                         "scale_support_rows": len(calibration)}
        return self

    def predict_distribution(self, frame):
        if self.shared_scale is None:
            raise RuntimeError("Performance model is not fitted")
        return PerformanceDistribution(self.fitted_mean.predict(frame[list(self.feature_columns)]),
            np.full(len(frame), self.shared_scale), self.coordinate,
            "lower" if self.coordinate == "log_time_seconds" else "higher",
            row_keys=_row_keys(frame), metadata=self.metadata.copy())

    def predict(self, frame):
        return self.predict_distribution(frame).physical_mean()

    def predict_fundamental_proba(self, frame):
        from .probabilistic_adapters import distribution_to_win
        return distribution_to_win(self.predict_distribution(frame), frame["race_id"])

    predict_proba = predict_fundamental_proba


@dataclass
class CatBoostUncertaintyModel:
    feature_columns: tuple[str, ...]
    coordinate: str = "log_speed_mps"
    categorical_columns: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    scale_floor: float = 1e-4
    scale_shrinkage: float = 0.1
    model: Any = None
    scale_multiplier: float = 1.0
    shared_variance: float | None = None
    metadata: dict = field(default_factory=dict)

    def _features(self, frame):
        result = frame[list(self.feature_columns)].copy()
        for column in self.categorical_columns:
            result[column] = result[column].fillna("__missing__").astype(str)
        return result

    def fit(self, train, *, label_column, calibration=None, date_column="date"):
        from catboost import CatBoostRegressor
        self.scale_multiplier = 1.0
        if not 0 <= self.scale_shrinkage <= 1 or self.scale_floor <= 0 or not np.isfinite(self.scale_floor):
            raise ValueError("Invalid variance shrinkage/floor")
        if calibration is not None and (not len(calibration) or pd.to_datetime(train[date_column]).max() >= pd.to_datetime(calibration[date_column]).min()):
            raise ValueError("Scale calibration must be chronological and nonempty")
        try:
            worker_threads = int(os.environ.get("IMA_RESEARCH_THREADS", "1"))
        except ValueError as exc:
            raise ValueError("IMA_RESEARCH_THREADS must be a positive integer") from exc
        if worker_threads < 1:
            raise ValueError("IMA_RESEARCH_THREADS must be a positive integer")
        params = {"iterations": 160, "depth": 5, "thread_count": worker_threads, "verbose": False,
                  "random_seed": 42, "allow_writing_files": False} | self.parameters
        if params["thread_count"] != worker_threads:
            raise ValueError("CatBoost thread_count must match the worker IMA_RESEARCH_THREADS allocation")
        if params.get("loss_function", "RMSEWithUncertainty") != "RMSEWithUncertainty":
            raise ValueError("Conditional scale requires RMSEWithUncertainty")
        params["loss_function"] = "RMSEWithUncertainty"
        target = _physical_target(train[label_column], self.coordinate)
        self.model = CatBoostRegressor(**params)
        self.model.fit(self._features(train), target, cat_features=list(self.categorical_columns))
        # Training population prior shrinks context variances, not per-horse free parameters.
        self.shared_variance = max(self.scale_floor**2, float(np.var(target)))
        self.metadata = {"fit_regime": "observed_performance", "loss": "RMSEWithUncertainty",
                         "raw_output": "mean,log_sigma", "support": "positive_physical_via_log" if self.coordinate.startswith("log_") else "real_latent",
                         "training_cutoff": str(pd.to_datetime(train[date_column]).max()),
                         "scale_shrinkage": self.scale_shrinkage}
        if calibration is not None:
            if pd.to_datetime(train[date_column]).max() >= pd.to_datetime(calibration[date_column]).min():
                raise ValueError("Scale calibration must be chronological")
            raw = np.asarray(self.model.predict(self._features(calibration), prediction_type="RawFormulaVal"))
            residual = _physical_target(calibration[label_column], self.coordinate) - raw[:, 0]
            self.shared_variance = max(self.scale_floor**2, float(np.mean(residual**2)))
            dist = self.predict_distribution(calibration)
            self.scale_multiplier = max(self.scale_floor, float(np.sqrt(np.mean((residual/dist.scale)**2))))
            self.metadata["calibration_cutoff"] = str(pd.to_datetime(calibration[date_column]).max())
            self.metadata["shared_scale_fit_scope"] = "disjoint_past_residuals"
        else:
            self.metadata["shared_scale_fit_scope"] = "training_target_variance_prior_uncalibrated"
        return self

    def predict_distribution(self, frame, *, shared_scale=False):
        if self.model is None:
            raise RuntimeError("CatBoost uncertainty model is not fitted")
        raw = np.asarray(self.model.predict(self._features(frame), prediction_type="RawFormulaVal"))
        variance = np.exp(2 * raw[:, 1])
        variance = (1-self.scale_shrinkage)*variance + self.scale_shrinkage*self.shared_variance
        if shared_scale:
            variance[:] = self.shared_variance
        multiplier = 1.0 if shared_scale and "calibration_cutoff" in self.metadata else self.scale_multiplier
        scale = np.maximum(self.scale_floor, np.sqrt(variance)*multiplier)
        return PerformanceDistribution(raw[:, 0], scale, self.coordinate,
            "lower" if self.coordinate == "log_time_seconds" else "higher",
            row_keys=_row_keys(frame), metadata=self.metadata.copy())

    def predict(self, frame):
        return self.predict_distribution(frame).physical_mean()

    def predict_fundamental_proba(self, frame):
        from .probabilistic_adapters import distribution_to_win
        return distribution_to_win(self.predict_distribution(frame), frame["race_id"])

    predict_proba = predict_fundamental_proba


def _row_keys(frame):
    return tuple(zip(frame["race_id"].astype(str), frame["horse_id"].astype(str))) if {"race_id", "horse_id"} <= set(frame.columns) else ()


def _supported_exp(values):
    with np.errstate(over="raise", under="raise", invalid="raise"):
        return np.exp(values)
