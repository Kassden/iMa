"""Statistical association, redundancy, and held-out feature contribution."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif

from .feature_sets import FeatureSchema
from .modeling import RaceProbabilityModel, race_log_loss


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    adjusted[order] = ranked
    return adjusted


def _mutual_information(values: pd.Series, target: pd.Series, discrete: bool) -> float:
    if discrete:
        encoded = pd.factorize(values.fillna("__MISSING__").astype(str), sort=True)[0]
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        encoded = numeric.fillna(numeric.median()).to_numpy()
    if np.unique(encoded).size < 2:
        return 0.0
    return float(mutual_info_classif(
        np.asarray(encoded).reshape(-1, 1), target.to_numpy(),
        discrete_features=discrete, random_state=17,
    )[0])


def association_report(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    rows = []
    target = frame["target_win"].astype(int)
    for feature in schema.numeric:
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = values.notna()
        correlation, p_value = (np.nan, np.nan)
        if valid.sum() >= 3 and values[valid].nunique() > 1 and target[valid].nunique() > 1:
            correlation, p_value = pointbiserialr(target[valid], values[valid])
        rows.append({
            "feature": feature, "kind": "numeric", "association": correlation,
            "p_value": p_value, "mutual_information": _mutual_information(values, target, False),
            "coverage": float(valid.mean()), "unique_values": int(values.nunique(dropna=True)),
        })
    for feature in schema.categorical:
        values = frame[feature].fillna("__MISSING__").astype(str)
        table = pd.crosstab(values, target)
        association, p_value = (np.nan, np.nan)
        if table.shape[0] > 1 and table.shape[1] > 1:
            chi2, p_value, _, _ = chi2_contingency(table)
            denominator = len(values) * max(1, min(table.shape[0] - 1, table.shape[1] - 1))
            association = float(np.sqrt(chi2 / denominator))
        rows.append({
            "feature": feature, "kind": "categorical", "association": association,
            "p_value": p_value, "mutual_information": _mutual_information(values, target, True),
            "coverage": float(frame[feature].notna().mean()),
            "unique_values": int(values.nunique(dropna=True)),
        })
    report = pd.DataFrame(rows)
    report["p_value_fdr"] = benjamini_hochberg(report["p_value"])
    report["significant_fdr_05"] = report["p_value_fdr"].lt(0.05)
    return report.sort_values(["mutual_information", "association"], ascending=False, kind="stable")


def correlation_report(
    frame: pd.DataFrame,
    numeric_features: Iterable[str],
    threshold: float = 0.80,
    sample_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(numeric_features)
    sample = frame[columns]
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=17)
    matrix = sample.corr(method="spearman", min_periods=100)
    pairs = []
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1:]:
            correlation = matrix.at[left, right]
            if np.isfinite(correlation) and abs(correlation) >= threshold:
                pairs.append({
                    "feature_a": left, "feature_b": right,
                    "spearman": float(correlation), "abs_spearman": float(abs(correlation)),
                })
    redundant = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman", "abs_spearman"])
    if not redundant.empty:
        redundant = redundant.sort_values("abs_spearman", ascending=False, kind="stable")
    return matrix, redundant


def permutation_importance_report(
    model: RaceProbabilityModel,
    frame: pd.DataFrame,
    feature_groups: dict[str, Iterable[str]],
    repeats: int = 2,
) -> pd.DataFrame:
    baseline = race_log_loss(model.predict_proba(frame), frame)
    rng = np.random.default_rng(17)
    rows = []
    for name, features in feature_groups.items():
        features = tuple(features)
        deltas = []
        for _ in range(repeats):
            changed = frame.copy()
            permutation = rng.permutation(len(changed))
            for feature in features:
                changed[feature] = changed[feature].to_numpy()[permutation]
            deltas.append(race_log_loss(model.predict_proba(changed), changed) - baseline)
        rows.append({
            "name": name, "features": list(features), "repeats": repeats,
            "baseline_race_log_loss": baseline,
            "delta_race_log_loss_mean": float(np.mean(deltas)),
            "delta_race_log_loss_std": float(np.std(deltas)),
        })
    return pd.DataFrame(rows).sort_values("delta_race_log_loss_mean", ascending=False, kind="stable")
