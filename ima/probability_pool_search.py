"""Bounded race-aware pooling search over frozen, aligned predictions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .modeling import race_log_loss


TEMPERATURES = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0)
METHODS = ("arithmetic", "geometric")


@dataclass(frozen=True)
class PoolRecipe:
    pair: str
    method: str
    benter_weight: float
    temperature: float

    def validate(self):
        if not self.pair or self.method not in METHODS:
            raise ValueError("Invalid pool pair or method")
        if not np.isfinite(self.benter_weight) or not 0 <= self.benter_weight <= 1:
            raise ValueError("Pool weight must be between zero and one")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Pool temperature must be finite and positive")

    @property
    def configuration_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def pool_probabilities(matrix, race_ids, recipe):
    recipe.validate()
    values = np.asarray(matrix, dtype=float)
    ids = np.asarray(race_ids)
    if values.ndim != 2 or values.shape != (len(ids), 2) or not len(ids):
        raise ValueError("Pooling needs two aligned component columns and nonempty race IDs")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Component probabilities must be finite and between zero and one")
    codes, races = pd.factorize(ids, sort=False)
    if (codes < 0).any():
        raise ValueError("Missing race identity")
    for column in range(2):
        totals = np.bincount(codes, weights=values[:, column], minlength=len(races))
        if not np.allclose(totals, 1, rtol=0, atol=1e-8):
            raise ValueError("Each component must sum to one in every race")
    weight = recipe.benter_weight
    if weight in (0, 1):
        raw = values[:, 0 if weight == 1 else 1]
    elif recipe.method == "arithmetic":
        raw = values @ np.array([weight, 1 - weight])
    else:
        raw = None
    if raw is not None and recipe.temperature == 1:
        return raw / np.bincount(codes, weights=raw)[codes]
    logs = (np.log(np.maximum(raw, np.finfo(float).tiny)) if raw is not None else
            np.log(np.maximum(values, np.finfo(float).tiny)) @ np.array([weight, 1 - weight]))
    scores = logs / recipe.temperature
    probabilities = np.empty(len(ids), dtype=float)
    for code in range(len(races)):
        mask = codes == code
        probabilities[mask] = np.exp(scores[mask] - logsumexp(scores[mask]))
    return probabilities


def _preseason_frame(frame):
    required = {"date", "race_id", "cohort"}
    if not required.issubset(frame) or frame.empty or not frame.cohort.eq("calibration").all():
        raise ValueError("Only nonempty preseason calibration rows are accepted")
    output = frame.copy()
    dates = pd.to_datetime(output.date, utc=True, errors="raise").dt.tz_convert(None)
    if dates.isna().any() or not dates.lt(pd.Timestamp("2026-09-01")).all():
        raise ValueError("Search dates must be strictly preseason")
    if output.race_id.isna().any() or output.groupby("race_id").date.nunique().gt(1).any():
        raise ValueError("Missing race identity or race spanning dates")
    if "horse_no" in output and output.duplicated(["race_id", "horse_no"]).any():
        raise ValueError("Duplicate runner identity")
    output["date"] = dates.dt.normalize()
    return output


def chronological_split(frame):
    output = _preseason_frame(frame)
    dates = np.sort(output.date.unique())
    if len(dates) < 5:
        raise ValueError("At least five preseason meeting dates are required")
    first, second = int(len(dates) * 0.6), int(len(dates) * 0.8)
    groups = {"search": dates[:first], "confirmation": dates[first:second], "calibration": dates[second:]}
    return {name: output[output.date.isin(days)].copy() for name, days in groups.items()}


def search_pool(frame, pairs, weights=None, temperatures=None):
    work = _preseason_frame(frame)
    if not pairs:
        raise ValueError("Search needs at least one aligned component pair")
    if "target_win" not in work:
        raise ValueError("Search requires observed winner labels")
    labels = pd.to_numeric(work.target_win, errors="raise")
    if not labels.isin([0, 1]).all() or not labels.groupby(work.race_id).sum().eq(1).all():
        raise ValueError("Every search race must have exactly one winner")
    work["target_probability"] = labels.astype(float)
    weights = tuple(np.linspace(0, 1, 21)) if weights is None else tuple(weights)
    temperatures = TEMPERATURES if temperatures is None else tuple(temperatures)
    if not weights or not temperatures or len(set(weights)) != len(weights) or len(set(temperatures)) != len(temperatures):
        raise ValueError("Search grids must be nonempty and unique")
    rows = []
    for pair, matrix in sorted(pairs.items()):
        for method in METHODS:
            for weight in weights:
                for temperature in temperatures:
                    recipe = PoolRecipe(pair, method, round(float(weight), 12), float(temperature))
                    probabilities = pool_probabilities(matrix, work.race_id, recipe)
                    rows.append(asdict(recipe) | {"configuration_id": recipe.configuration_id,
                        "search_race_log_loss": race_log_loss(probabilities, work)})
    table = pd.DataFrame(rows)
    ranked = table.assign(_loss=table.search_race_log_loss.round(12),
        _pair=table.pair.map(lambda pair: (0 if pair == "original_pair" else 1, pair)),
        _method=table.method.map({"arithmetic": 0, "geometric": 1}),
        _weight=(table.benter_weight - 0.6).abs(), _temperature=(table.temperature - 1).abs()
    ).sort_values(["_loss", "_pair", "_method", "_weight", "_temperature", "benter_weight", "temperature"])
    best = ranked.iloc[0]
    selected = PoolRecipe(str(best.pair), str(best.method), float(best.benter_weight), float(best.temperature))
    return ranked[table.columns].reset_index(drop=True), selected
