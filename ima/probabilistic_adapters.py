"""Explicit information contracts between rank, win and performance outputs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .modeling import TemperatureCalibrator, normalize_by_race
from .performance_probit import gaussian_win_probabilities
from .rank_distributions import plackett_luce_distribution, sample_plackett_luce_distribution, simulate_rank_distribution


def ranking_scores_to_probabilities(scores, race_ids):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or len(scores) != len(race_ids) or not np.isfinite(scores).all():
        raise ValueError("Invalid ranking scores")
    frame = pd.DataFrame({"score": scores, "race_id": np.asarray(race_ids)})
    maxima = frame.groupby("race_id")["score"].transform("max").to_numpy()
    return normalize_by_race(np.exp(scores-maxima), race_ids)


@dataclass
class RankingWinAdapter:
    calibrator: TemperatureCalibrator | None = None

    def fit(self, scores, calibration_frame):
        calibration_frame = calibration_frame.copy()
        if "target_probability" not in calibration_frame:
            calibration_frame["target_probability"] = calibration_frame["target_win"]
        self.calibrator = TemperatureCalibrator.fit(
            ranking_scores_to_probabilities(scores, calibration_frame["race_id"]), calibration_frame)
        return self

    def transform(self, scores, race_ids):
        p = ranking_scores_to_probabilities(scores, race_ids)
        return self.calibrator.transform(p, race_ids) if self.calibrator else p

    @property
    def information_contract(self):
        return {"input": "ranking_score", "output": "win_probability",
                "assumption": "existing race softmax plus chronological temperature",
                "not_inferred": "runner-specific variance or unique joint orders"}


def distribution_to_win(distribution, race_ids, *, return_diagnostics=False):
    if len(race_ids) != len(distribution.location):
        raise ValueError("Distribution race mismatch")
    result = np.empty(len(race_ids))
    diagnostics = []
    codes = pd.factorize(np.asarray(race_ids), sort=False)[0]
    for code in np.unique(codes):
        rows = np.flatnonzero(codes == code)
        means = distribution.location[rows]
        if distribution.direction == "lower":
            means = -means
        result[rows], diagnostic = gaussian_win_probabilities(means, distribution.scale[rows], return_diagnostics=True)
        diagnostic["race_id"] = str(np.asarray(race_ids)[rows[0]])
        diagnostics.append(diagnostic)
        if not diagnostic["converged"]:
            raise RuntimeError(f"Distribution winner quadrature did not converge: {diagnostic}")
    distribution.metadata["winner_quadrature"] = diagnostics
    return (result, diagnostics) if return_diagnostics else result


def win_to_joint(probabilities, runner_ids, *, method="auto", **simulation_settings):
    if method not in {"auto", "exact", "sampled"}:
        raise ValueError("Unknown win-to-joint adapter method")
    if method == "exact" or (method == "auto" and len(runner_ids) <= 8):
        if simulation_settings:
            raise ValueError("Exact PL adapter does not accept simulation settings")
        return plackett_luce_distribution(probabilities, runner_ids)
    return sample_plackett_luce_distribution(probabilities, runner_ids, **simulation_settings)


def distribution_to_joint(distribution, runner_ids, **simulation_settings):
    return simulate_rank_distribution(distribution, runner_ids, **simulation_settings)


def fit_performance_distribution(spec, train, calibration, feature_schema, *,
                                 label_column, seed=42, model_spec=None):
    """Data-independent parent executor interface; physical labels enter unlogged."""
    from .performance_distributions import SharedResidualPerformanceModel, CatBoostUncertaintyModel
    from .research_models import ResearchRegressor
    parameters = dict(spec or {})
    kind = parameters.pop("kind", "shared_residual")
    coordinate = parameters.pop("coordinate", "log_speed_mps")
    date_column = parameters.pop("date_column", "date")
    if kind == "catboost_uncertainty":
        model_parameters = dict(parameters.pop("parameters", {}))
        model_parameters.setdefault("random_seed", seed)
        model = CatBoostUncertaintyModel(tuple(feature_schema.features), coordinate,
            tuple(feature_schema.categorical), model_parameters, **parameters)
        return model.fit(train, label_column=label_column, calibration=calibration, date_column=date_column)
    if kind != "shared_residual":
        raise ValueError(f"Unsupported performance distribution: {kind}")
    model_kind = getattr(model_spec, "kind", "ridge_regressor") if not isinstance(model_spec, dict) else model_spec.get("kind", "ridge_regressor")
    model_parameters = getattr(model_spec, "parameters", {}) if not isinstance(model_spec, dict) else model_spec.get("parameters", {})
    # Use the existing mature regressor's preprocessing pipeline as the mean control.
    initial = ResearchRegressor(kind=model_kind, parameters=model_parameters)
    from .performance_distributions import _physical_target
    logged = train.copy()
    logged[label_column] = _physical_target(train[label_column], coordinate)
    initial.fit(logged, feature_schema, label_column)
    if initial.pipeline is None:
        raise ValueError("Shared residual adapter currently requires a sklearn mean pipeline")
    model = SharedResidualPerformanceModel(tuple(feature_schema.features), coordinate,
        **parameters)
    model.fitted_mean = initial.pipeline
    return model.calibrate_residuals(train, calibration, label_column=label_column, date_column=date_column)
