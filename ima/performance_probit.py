"""Identified independent-Gaussian race winner likelihood (experimental)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import erfcx, log_ndtr, logsumexp, roots_hermitenorm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .performance_distributions import PerformanceDistribution, _row_keys


def _race_groups(codes):
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order]))+1)


def _race_center(values, codes):
    counts = np.bincount(codes)
    if values.ndim == 1:
        return values - (np.bincount(codes, weights=values)/counts)[codes]
    centered = values.copy()
    for column in range(values.shape[1]):
        centered[:, column] -= (np.bincount(codes, weights=values[:, column])/counts)[codes]
    return centered


def _race_batches(groups, winners, order):
    by_size = {}
    for group, winner in zip(groups, winners):
        by_size.setdefault(len(group), []).append((group, winner))
    batches = []
    for size, items in sorted(by_size.items()):
        # Bound each square quadrature tensor, independent of corpus size.
        chunk = max(1, 262144//(size*size*order))
        for start in range(0, len(items), chunk):
            part = items[start:start+chunk]
            batches.append((np.stack([item[0] for item in part]),
                            np.array([item[1] for item in part])))
    return batches


def _batch_winner_loss_gradient(mu, sigma, winners, order):
    count, size = mu.shape
    if size <= 2 or size*size*order > 262144:
        losses, mean_grad, scale_grad = [], np.empty_like(mu), np.empty_like(mu)
        for row in range(count):
            logs, mean_jac, scale_jac = _gaussian_log_win_probabilities(
                mu[row], sigma[row], order=order, with_gradient=True)
            normalizer = logsumexp(logs)
            losses.append(normalizer-logs[winners[row]])
            weights = np.exp(logs-normalizer)
            weights[winners[row]] -= 1
            mean_grad[row], scale_grad[row] = weights@mean_jac, weights@scale_jac
        return np.asarray(losses), mean_grad, scale_grad
    z, log_weights = _quadrature(order)
    thresholds = (mu[:, :, None, None]+sigma[:, :, None, None]*z[None, None, :, None]
                  - mu[:, None, None, :])/sigma[:, None, None, :]
    diagonal = np.arange(size)
    cdf_logs = log_ndtr(thresholds)
    cdf_logs[:, diagonal, :, diagonal] = 0
    terms = log_weights[None, None, :]+cdf_logs.sum(axis=-1)
    logs = logsumexp(terms, axis=-1)
    normalizers = logsumexp(logs, axis=-1)
    weights = np.exp(logs-normalizers[:, None])
    weights[np.arange(count), winners] -= 1
    ratios = np.exp(terms-logs[:, :, None])[..., None]*_inverse_mills_ratio(thresholds)
    ratios[:, diagonal, :, diagonal] = 0
    locations = ratios/sigma[:, None, None, :]
    mean_grad = -(weights[:, :, None]*locations.sum(axis=2)).sum(axis=1)
    mean_grad += weights*locations.sum(axis=(2, 3))
    scale_grad = -(weights[:, :, None]*(ratios*thresholds).sum(axis=2)).sum(axis=1)
    scale_grad += weights*sigma*(locations*z[None, None, :, None]).sum(axis=(2, 3))
    losses = normalizers-logs[np.arange(count), winners]
    return losses, mean_grad, scale_grad


@lru_cache(maxsize=16)
def _quadrature(order):
    z, w = roots_hermitenorm(order)
    with np.errstate(divide="ignore"):
        log_weights = np.log(w / np.sqrt(2*np.pi))
    return z, log_weights


def gaussian_log_win_probabilities(location, scale, *, order=64):
    """Log P(X_i=max X), integrating conditional normal CDF products."""
    return _gaussian_log_win_probabilities(location, scale, order=order)


def _inverse_mills_ratio(values):
    """Normal density/CDF ratio without cancellation in the negative tail."""
    result = np.empty_like(values)
    negative = values < 0
    result[negative] = np.sqrt(2/np.pi) / erfcx(-values[negative]/np.sqrt(2))
    positive = values[~negative]
    result[~negative] = np.exp(-positive**2/2 - np.log(2*np.pi)/2 - log_ndtr(positive))
    return result


def _gaussian_log_win_probabilities(location, scale, *, order, with_gradient=False):
    """Optionally return Jacobians with respect to location and log(scale)."""
    mu, sigma = np.asarray(location, float), np.asarray(scale, float)
    if mu.ndim != 1 or mu.shape != sigma.shape or not len(mu):
        raise ValueError("Invalid Gaussian race shape")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Nonfinite location or nonpositive scale")
    if order < 8:
        raise ValueError("Quadrature order must be at least eight")
    if with_gradient:
        location_jacobian = np.zeros((len(mu), len(mu)))
        log_scale_jacobian = np.zeros_like(location_jacobian)
    if len(mu) == 1:
        values = np.zeros(1)
        return (values, location_jacobian, log_scale_jacobian) if with_gradient else values
    if len(mu) == 2:
        total_scale = np.hypot(*sigma)
        difference = (mu[0]-mu[1])/total_scale
        signed_difference = np.array([difference, -difference])
        values = log_ndtr(signed_difference)
        if with_gradient:
            ratios = _inverse_mills_ratio(signed_difference)
            location_jacobian = (ratios*np.array([1., -1.])/total_scale)[:, None] * np.array([1., -1.])
            log_scale_jacobian = (-ratios*signed_difference)[:, None] * (sigma/total_scale)**2
            return values, location_jacobian, log_scale_jacobian
        return values
    z, log_weights = _quadrature(order)
    values = np.empty(len(mu))
    for i in range(len(mu)):
        mask = np.arange(len(mu)) != i
        thresholds = (mu[i] + sigma[i]*z[:, None] - mu[mask]) / sigma[mask]
        log_terms = log_weights + log_ndtr(thresholds).sum(axis=1)
        values[i] = logsumexp(log_terms)
        if with_gradient:
            # Differentiate the same fixed quadrature, weighting nodes in log space.
            weighted_ratios = np.exp(log_terms-values[i])[:, None] * _inverse_mills_ratio(thresholds)
            location_terms = weighted_ratios / sigma[mask]
            location_jacobian[i, mask] = -location_terms.sum(axis=0)
            location_jacobian[i, i] = location_terms.sum()
            log_scale_jacobian[i, mask] = -(weighted_ratios*thresholds).sum(axis=0)
            log_scale_jacobian[i, i] = sigma[i]*np.sum(location_terms*z[:, None])
    return (values, location_jacobian, log_scale_jacobian) if with_gradient else values


def gaussian_win_probabilities(location, scale, *, order=64, tolerance=1e-6,
                               max_order=512, return_diagnostics=False):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Gaussian quadrature requires a positive finite tolerance")
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
                   "refinement_error": refinement_error, "converged": converged,
                   "integration_method": "gauss_hermite"}
    if not converged:
        diagnostics.update(integration_method="adaptive_quad",
                           hermite_normalization_error=normalization_error,
                           hermite_refinement_error=refinement_error)
        mu, sigma = np.asarray(location, float), np.asarray(scale, float)

        def adaptive_integral(budget):
            probabilities, errors = np.zeros(len(mu)), np.zeros(len(mu))
            for i in range(len(mu)):
                mask = np.arange(len(mu)) != i
                offsets = (mu[i]-mu[mask])/sigma[mask]
                ratios = sigma[i]/sigma[mask]
                centers = (mu[mask]-mu[i])/sigma[i]
                widths = sigma[mask]/sigma[i]
                transitions = np.r_[centers, centers-8*widths, centers+8*widths, 0.]
                points = np.unique(transitions[(transitions > -8) & (transitions < 8)])

                def integrand(z):
                    return float(np.exp(-z*z/2-np.log(2*np.pi)/2
                                        + log_ndtr(offsets+ratios*z).sum()))

                # Explicit transitions prevent narrow CDF changes being missed;
                # infinite tails retain the complete Gaussian winner likelihood.
                for lower, upper, breakpoints in ((-np.inf,-8.,None),
                                                   (-8.,8.,points), (8.,np.inf,None)):
                    result = quad(integrand, lower, upper, points=breakpoints,
                                  epsabs=budget/(3*len(mu)), epsrel=budget/(3*len(mu)),
                                  limit=200, full_output=1)
                    if len(result) != 3:
                        raise RuntimeError(f"QUADPACK did not converge: {result[3]}")
                    value, error, _ = result
                    if not np.isfinite([value,error]).all() or value < 0 or error < 0:
                        raise RuntimeError("QUADPACK returned invalid probability or error")
                    probabilities[i] += value
                    errors[i] += error
            total = probabilities.sum()
            if not np.isfinite(total) or total <= 0:
                raise RuntimeError("QUADPACK returned invalid total probability")
            # Include normalization's propagation of all per-runner error estimates.
            bound = float((errors.max()+errors.sum())/total)
            return probabilities/total, abs(float(total)-1), bound

        try:
            coarse, _, coarse_bound = adaptive_integral(tolerance/16)
            fine, normalization_error, fine_bound = adaptive_integral(tolerance/64)
            refinement_error = float(np.max(np.abs(fine-coarse)))
            adaptive_bound = max(coarse_bound,fine_bound)
            converged = (normalization_error <= tolerance and refinement_error <= tolerance
                         and adaptive_bound <= tolerance)
            values = fine
            diagnostics.update(normalization_error=normalization_error,
                               refinement_error=refinement_error,
                               adaptive_error_bound=adaptive_bound, converged=converged)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            diagnostics.update(adaptive_error=str(exc), converged=False)
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
        return _race_center(matrix, codes), codes

    def fit(self, frame):
        if self.l2 <= 0 or self.scale_l2 <= 0 or self.max_iter < 1:
            raise ValueError("Probit requires positive identification regularization")
        started = perf_counter()
        x, codes = self._matrix(frame, fit=True)
        y = frame["target_win"].to_numpy(float)
        groups = _race_groups(codes)
        if not groups or any(len(g) < 2 or not np.isin(y[g], [0, 1]).all() or y[g].sum() != 1 for g in groups):
            raise ValueError("Probit needs complete races and exactly one winner")
        if "field_size" in frame:
            observed = frame.groupby("race_id")["race_id"].transform("size").to_numpy()
            if not np.array_equal(observed, pd.to_numeric(frame["field_size"], errors="coerce").to_numpy()):
                raise ValueError("Probit requires complete pre-race fields")
        dimension = x.shape[1]

        winners = [int(np.argmax(y[group])) for group in groups]
        batches = _race_batches(groups, winners, self.quadrature_order)
        preparation_seconds = perf_counter()-started
        objective_seconds, evaluations = 0.0, 0

        def objective(parameters):
            nonlocal objective_seconds, evaluations
            tick = perf_counter()
            result = self._objective_and_gradient(parameters, x, codes, groups, winners, batches=batches)
            objective_seconds += perf_counter()-tick
            evaluations += 1
            return result

        optimizer_started = perf_counter()
        result = minimize(objective, np.zeros(dimension*(2 if self.heteroscedastic else 1)),
                          jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter, "ftol": 1e-9})
        optimizer_seconds = perf_counter()-optimizer_started
        if not result.success:
            raise RuntimeError(f"Probit likelihood fit did not converge: {result.message}")
        self.coefficients = result.x[:dimension]
        self.scale_coefficients = result.x[dimension:] if self.heteroscedastic else None
        means = x @ self.coefficients
        scales = self._scales(x, self.scale_coefficients, codes) if self.heteroscedastic else np.ones(len(x))
        verification_started = perf_counter()
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
            "objective_evaluations": evaluations, "objective_seconds": objective_seconds,
            "preparation_seconds": preparation_seconds, "optimizer_seconds": optimizer_seconds,
            "quadrature_verification_seconds": perf_counter()-verification_started,
            "fit_seconds": perf_counter()-started, "objective_engine": "bounded_numpy_batches_v1",
            "training_quadrature_max_refinement_error": max(errors)}
        return self

    def _objective_and_gradient(self, parameters, x, codes, groups, winners, *, batches=None):
        dimension = x.shape[1]
        means = x @ parameters[:dimension]
        scales = self._scales(x, parameters[dimension:], codes) if self.heteroscedastic else np.ones(len(x))
        mean_gradient = np.zeros(len(x))
        scale_gradient = np.zeros(len(x)) if self.heteroscedastic else None
        losses = []
        for indices, selected_winners in (batches if batches is not None else _race_batches(groups,winners,self.quadrature_order)):
            batch_loss, batch_mean, batch_scale = _batch_winner_loss_gradient(
                means[indices], scales[indices], selected_winners, self.quadrature_order)
            losses.extend(batch_loss)
            mean_gradient[indices] = batch_mean
            if self.heteroscedastic:
                # Race centering's adjoint precedes the bounded tanh link's derivative.
                scale_gradient[indices] = batch_scale-batch_scale.mean(axis=1, keepdims=True)
        gradient = np.empty_like(parameters)
        gradient[:dimension] = x.T @ mean_gradient / len(groups) + self.l2*parameters[:dimension]
        if self.heteroscedastic:
            scale_gradient *= 1-np.tanh((x @ parameters[dimension:])/3)**2
            gradient[dimension:] = x.T @ scale_gradient / len(groups) + self.scale_l2*parameters[dimension:]
        value = float(np.mean(losses) + self.l2*np.sum(parameters[:dimension]**2)/2
                      + self.scale_l2*np.sum(parameters[dimension:]**2)/2)
        return value, gradient

    @staticmethod
    def _scales(x, coefficients, codes):
        logged = 3*np.tanh((x @ coefficients)/3)
        return np.exp(_race_center(logged, codes))

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
