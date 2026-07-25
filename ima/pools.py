"""Coherent pool-combination probabilities from Plackett-Luce strengths."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ORDERED_POOLS = {"WIN": 1, "TIERCE": 3, "FIRST4": 4, "QUARTET": 4}
UNORDERED_POOLS = {"QIN": 2, "QPL": 2, "TRI": 3}


@dataclass(frozen=True)
class CombinationProbability:
    pool: str
    runners: tuple[str, ...]
    probability: float

    @property
    def fair_odds(self) -> float:
        return float("inf") if self.probability <= 0 else 1.0 / self.probability


@dataclass(frozen=True)
class OrderExponents:
    second: float = 1.0
    third: float = 1.0


def benter_order_probability(
    order: Iterable[int],
    strengths: np.ndarray,
    exponents: OrderExponents = OrderExponents(),
) -> float:
    remaining = list(range(len(strengths)))
    probability = 1.0
    for place, selected in enumerate(order):
        if selected not in remaining:
            return 0.0
        exponent = 1.0 if place == 0 else exponents.second if place == 1 else exponents.third
        adjusted = np.power(strengths[remaining], exponent)
        denominator = float(adjusted.sum())
        probability *= float(strengths[selected] ** exponent) / denominator
        remaining.remove(selected)
    return probability


def plackett_luce_order_probability(order: Iterable[int], strengths: np.ndarray) -> float:
    return benter_order_probability(order, strengths)


def fit_order_exponents(frame: pd.DataFrame, probability_column: str) -> OrderExponents:
    races = []
    for _, race in frame.groupby("race_id", sort=False):
        ordered = race.sort_values("result")
        if len(ordered) < 3 or ordered["result"].iloc[:3].duplicated().any():
            continue
        strengths = np.clip(race[probability_column].to_numpy(dtype=float), 1e-12, None)
        index_by_runner = {runner: index for index, runner in enumerate(race["horse_no"])}
        order = [index_by_runner[runner] for runner in ordered["horse_no"].iloc[:3]]
        races.append((strengths, order))

    def place_loss(exponent: float, place: int) -> float:
        loss = 0.0
        for strengths, order in races:
            remaining = [index for index in range(len(strengths)) if index not in order[:place]]
            adjusted = np.power(strengths[remaining], exponent)
            selected = remaining.index(order[place])
            loss -= np.log(np.clip(adjusted[selected] / adjusted.sum(), 1e-12, 1.0))
        return loss

    second = minimize_scalar(lambda value: place_loss(value, 1), bounds=(0.2, 2.0), method="bounded")
    third = minimize_scalar(lambda value: place_loss(value, 2), bounds=(0.2, 2.0), method="bounded")
    return OrderExponents(float(second.x), float(third.x))


def rank_combinations(
    runner_ids: list[str],
    probabilities: np.ndarray,
    pool: str,
    exponents: OrderExponents = OrderExponents(),
) -> list[CombinationProbability]:
    if len(runner_ids) != len(probabilities):
        raise ValueError("Runner ids and probabilities must have equal length")
    strengths = np.clip(np.asarray(probabilities, dtype=float), 1e-12, None)
    pool = pool.upper()
    results: list[CombinationProbability] = []
    if pool in ORDERED_POOLS:
        places = ORDERED_POOLS[pool]
        orders = itertools.permutations(range(len(runner_ids)), places)
        for order in orders:
            probability = benter_order_probability(order, strengths, exponents)
            results.append(CombinationProbability(pool, tuple(runner_ids[i] for i in order), probability))
    elif pool in UNORDERED_POOLS:
        places = UNORDERED_POOLS[pool]
        for combination in itertools.combinations(range(len(runner_ids)), places):
            probability = sum(
                benter_order_probability(order, strengths, exponents)
                for order in itertools.permutations(combination)
            )
            results.append(
                CombinationProbability(pool, tuple(sorted(runner_ids[i] for i in combination)), probability)
            )
    elif pool == "PLACE":
        places = min(3, len(runner_ids))
        for index, runner_id in enumerate(runner_ids):
            probability = sum(
                benter_order_probability(order, strengths, exponents)
                for order in itertools.permutations(range(len(runner_ids)), places)
                if index in order
            )
            results.append(CombinationProbability(pool, (runner_id,), probability))
    else:
        raise ValueError(f"Unsupported pool: {pool}")
    return sorted(results, key=lambda item: item.probability, reverse=True)
