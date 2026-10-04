"""Identified independent-Gaussian race winner likelihood (experimental)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, roots_hermitenorm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .performance_distributions import PerformanceDistribution, _row_keys


@lru_cache(maxsize=16)
def _quadrature(order):
    z, w = roots_hermitenorm(order)
    return z, np.log(w / np.sqrt(2*np.pi))


def gaussian_log_win_probabilities(location, scale, *, order=64):
    """Log P(X_i=max X), integrating conditional normal CDF products."""
    mu, sigma = np.asarray(location, float), np.asarray(scale, float)
    if mu.ndim != 1 or mu.shape != sigma.shape or not len(mu):
        raise ValueError("Invalid Gaussian race shape")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Nonfinite location or nonpositive scale")
    if order < 8:
        raise ValueError("Quadrature order must be at least eight")
    if len(mu) == 1:
        return np.zeros(1)
    if len(mu) == 2:
        difference = (mu[0]-mu[1])/np.hypot(*sigma)
        return np.array([log_ndtr(difference), log_ndtr(-difference)])
    z, log_weights = _quadrature(order)
    values = np.empty(len(mu))
    for i in range(len(mu)):
        mask = np.arange(len(mu)) != i
        thresholds = (mu[i] + sigma[i]*z[:, None] - mu[mask]) / sigma[mask]
        values[i] = logsumexp(log_weights + log_ndtr(thresholds).sum(axis=1))
    return values


def gaussian_win_probabilities(location, scale, *, order=64, tolerance=1e-6,
                               max_order=512, return_diagnostics=False):
    previous = None
    refinement_error = np.inf
    while True:
        logs = gaussian_log_win_probabilities(location, scale, order=order)
        normalization_error = abs(float(np.exp(logsumexp(logs)))-1)
        values = np.exp(logs-logsumexp(logs))
        if previous is not None:
            refinement_error = float(np.max(np.abs(values-previous)))
        converged = normalization_error <= tolerance and refinement_error <= tolerance
        if converged or order >= max_order or len(values) <= 2:
            if len(values) <= 2:
                refinement_error, converged = 0.0, True
            break
        previous = values
        order = min(max_order, 2*order)
    diagnostics = {"quadrature_order": order, "normalization_error": normalization_error,
                   "refinement_error": refinement_error, "converged": converged}
    underflows = int(np.count_nonzero(values == 0))
    diagnostics["float64_underflow_count"] = underflows
    diagnostics["probability_floor"] = float(np.finfo(float).tiny) if underflows else None
    if underflows:
        values = np.maximum(values, np.finfo(float).tiny)
        values /= values.sum()
        if not return_diagnostics:
            warnings.warn("Winner probabilities below float64 support were floored; use gaussian_log_win_probabilities for rare-outcome likelihoods", RuntimeWarning, stacklevel=2)
    if return_diagnostics:
        return values, diagnostics
    if not converged:
        raise RuntimeError(f"Gaussian quadrature did not converge: {diagnostics}")
    return values


@dataclass
class GaussianRaceProbit:
    feature_columns: tuple[str, ...]
    heteroscedastic: bool = False
    l2: float = 0.01
    scale_l2: float = 0.1
    max_iter: int = 200
    quadrature_order: int = 48
    preprocessor: object = None
    coefficients: np.ndarray | None = None
    scale_coefficients: np.ndarray | None = None
    fit_diagnostics: dict = field(default_factory=dict)

    def _matrix(self, frame, fit=False):
        raw = frame[list(self.feature_columns)].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.preprocessor = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
            matrix = self.preprocessor.fit_transform(raw)
        else:
            matrix = self.preprocessor.transform(raw)
        codes = pd.factorize(frame["race_id"], sort=False)[0]
        for code in np.unique(codes):
            selected = codes == code
            matrix[selected] -= matrix[selected].mean(axis=0)
        return matrix, codes

    def fit(self, frame):
        if self.l2 <= 0 or self.scale_l2 <= 0 or self.max_iter < 1:
            raise ValueError("Probit requires positive identification regularization")
        x, codes = self._matrix(frame, fit=True)
        y = frame["target_win"].to_numpy(float)
        groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
        if not groups or any(len(g) < 2 or not np.isin(y[g], [0, 1]).all() or y[g].sum() != 1 for g in groups):
            raise ValueError("Probit needs complete races and exactly one winner")
        if "field_size" in frame:
            observed = frame.groupby("race_id")["race_id"].transform("size").to_numpy()
            if not np.array_equal(observed, pd.to_numeric(frame["field_size"], errors="coerce").to_numpy()):
                raise ValueError("Probit requires complete pre-race fields")
        dimension = x.shape[1]

        def objective(parameters):
            means = x @ parameters[:dimension]
            scales = self._scales(x, parameters[dimension:], codes) if self.heteroscedastic else np.ones(len(x))
            losses = []
            for group in groups:
                logs = gaussian_log_win_probabilities(means[group], scales[group], order=self.quadrature_order)
                losses.append(-(logs-logsumexp(logs))[np.argmax(y[group])])
            return float(np.mean(losses) + self.l2*np.sum(parameters[:dimension]**2)/2
                         + self.scale_l2*np.sum(parameters[dimension:]**2)/2)

        result = minimize(objective, np.zeros(dimension*(2 if self.heteroscedastic else 1)),
                          method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": 1e-9})
        if not result.success:
            raise RuntimeError(f"Probit likelihood fit did not converge: {result.message}")
        self.coefficients = result.x[:dimension]
        self.scale_coefficients = result.x[dimension:] if self.heteroscedastic else None
        means = x @ self.coefficients
        scales = self._scales(x, self.scale_coefficients, codes) if self.heteroscedastic else np.ones(len(x))
        errors = []
        for group in groups:
            coarse = gaussian_log_win_probabilities(means[group], scales[group], order=self.quadrature_order)
            fine = gaussian_win_probabilities(means[group], scales[group], order=self.quadrature_order)
            errors.append(float(np.max(np.abs(np.exp(coarse-logsumexp(coarse))-fine))))
        if max(errors) > 1e-4:
            self.coefficients, self.scale_coefficients = None, None
            raise RuntimeError("Fitted probit likelihood quadrature needs a higher quadrature_order")
        self.fit_diagnostics = {"fit_regime": "winner_likelihood", "objective": float(result.fun),
            "iterations": int(result.nit), "converged": True, "lane": "experimental",
            "identification": "race-centered mean; race geometric-mean sigma=1; bounded log-scale link; no intercept",
            "quadrature_order": self.quadrature_order,
            "training_quadrature_max_refinement_error": max(errors)}
        return self

    @staticmethod
    def _scales(x, coefficients, codes):
        logged = 3*np.tanh((x @ coefficients)/3)
        for code in np.unique(codes):
            rows = codes == code
            logged[rows] -= logged[rows].mean()
        return np.exp(logged)

    def predict_distribution(self, frame):
        if self.coefficients is None:
            raise RuntimeError("Probit has not been fitted")
        x, codes = self._matrix(frame)
        scale = self._scales(x, self.scale_coefficients, codes) if self.heteroscedastic else np.ones(len(x))
        return PerformanceDistribution(x @ self.coefficients, scale, row_keys=_row_keys(frame), metadata=self.fit_diagnostics.copy())

    def predict_fundamental_proba(self, frame):
        dist = self.predict_distribution(frame)
        values = np.empty(len(frame))
        codes = pd.factorize(frame["race_id"], sort=False)[0]
        for code in np.unique(codes):
            rows = np.flatnonzero(codes == code)
            values[rows] = gaussian_win_probabilities(dist.location[rows], dist.scale[rows], order=self.quadrature_order)
        return values

    predict_proba = predict_fundamental_proba
    predict = predict_fundamental_proba


def fit_probit(train, feature_schema, parameters=None):
    """Parent dispatch for model kind gaussian_probit; numeric predictors only."""
    return GaussianRaceProbit(tuple(feature_schema.numeric), **dict(parameters or {})).fit(train)
