"""Coherent pool-combination probabilities from Plackett-Luce strengths."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


ORDERED_POOLS = {"WIN": 1, "TIERCE": 3, "QUARTET": 4}
UNORDERED_POOLS = {"QIN": 2, "TRI": 3, "FIRST4": 4}
POOL_ALIASES = {
    "WIN": "WIN",
    "PLACE": "PLACE",
    "QIN": "QIN",
    "QUINELLA": "QIN",
    "QPL": "QPL",
    "QUINELLA PLACE": "QPL",
    "TRI": "TRI",
    "TRIO": "TRI",
    "TIERCE": "TIERCE",
    "FIRST4": "FIRST4",
    "FIRST 4": "FIRST4",
    "QUARTET": "QUARTET",
}
SUPPORTED_POOLS = ("WIN", "PLACE", "QIN", "QPL", "TRI", "TIERCE", "FIRST4", "QUARTET")


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


def canonical_pool_name(pool: str) -> str:
    normalized = " ".join(
        str(pool).strip().upper().replace("-", " ").replace("_", " ").split()
    )
    if normalized not in POOL_ALIASES:
        raise ValueError(f"Unsupported pool: {pool}")
    return POOL_ALIASES[normalized]


def paid_place_count(field_size: int) -> int:
    """Return HKJC standard paid places for single-race PLACE and QPL pools."""
    return min(field_size, 3 if field_size >= 7 else 2)


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
    pool = canonical_pool_name(pool)
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
    elif pool == "QPL":
        places = paid_place_count(len(runner_ids))
        for combination in itertools.combinations(range(len(runner_ids)), 2):
            selected = set(combination)
            probability = sum(
                benter_order_probability(order, strengths, exponents)
                for order in itertools.permutations(range(len(runner_ids)), places)
                if selected.issubset(order)
            )
            results.append(
                CombinationProbability(pool, tuple(sorted(runner_ids[i] for i in combination)), probability)
            )
    elif pool == "PLACE":
        places = paid_place_count(len(runner_ids))
        for index, runner_id in enumerate(runner_ids):
            probability = sum(
                benter_order_probability(order, strengths, exponents)
                for order in itertools.permutations(range(len(runner_ids)), places)
                if index in order
            )
            results.append(CombinationProbability(pool, (runner_id,), probability))
    return sorted(results, key=lambda item: item.probability, reverse=True)


def _unordered_probability(
    selected: tuple[int, ...], strengths: np.ndarray, exponents: OrderExponents,
) -> float:
    return float(sum(
        benter_order_probability(order, strengths, exponents)
        for order in itertools.permutations(selected)
    ))


def _place_probability(
    selected: int, strengths: np.ndarray, places: int, exponents: OrderExponents,
) -> float:
    others = [index for index in range(len(strengths)) if index != selected]
    probability = benter_order_probability((selected,), strengths, exponents)
    if places >= 2:
        probability += sum(
            benter_order_probability((first, selected), strengths, exponents)
            for first in others
        )
    if places >= 3:
        probability += sum(
            benter_order_probability((first, second, selected), strengths, exponents)
            for first, second in itertools.permutations(others, 2)
        )
    return float(probability)


def _qpl_probability(
    selected: tuple[int, int], strengths: np.ndarray, places: int, exponents: OrderExponents,
) -> float:
    if places == 2:
        return _unordered_probability(selected, strengths, exponents)
    other_runners = [index for index in range(len(strengths)) if index not in selected]
    probability = 0.0
    for other in other_runners:
        probability += sum(
            benter_order_probability(order, strengths, exponents)
            for order in itertools.permutations((*selected, other))
        )
    return float(probability)


def evaluate_top_pool_selections(
    frame: pd.DataFrame,
    probability_column: str,
    exponents: OrderExponents = OrderExponents(),
) -> dict[str, dict[str, float | int]]:
    """Evaluate the highest-strength selection for every supported single-race pool."""
    totals = {
        pool: {"races": 0, "hits": 0, "probability_sum": 0.0}
        for pool in SUPPORTED_POOLS
    }
    for _, race in frame.groupby("race_id", sort=False):
        race = race.dropna(subset=[probability_column, "result", "horse_no"]).copy()
        if len(race) < 2:
            continue
        race = race.sort_values(probability_column, ascending=False, kind="stable")
        strengths = np.clip(race[probability_column].to_numpy(dtype=float), 1e-12, None)
        strengths /= strengths.sum()
        results = pd.to_numeric(race["result"], errors="coerce").to_numpy(dtype=float)
        field_size = len(race)
        paid_places = paid_place_count(field_size)

        for pool in SUPPORTED_POOLS:
            count = {
                "WIN": 1, "PLACE": 1, "QIN": 2, "QPL": 2,
                "TRI": 3, "TIERCE": 3, "FIRST4": 4, "QUARTET": 4,
            }[pool]
            if field_size < count:
                continue
            selected = tuple(range(count))
            if pool == "WIN":
                probability = float(strengths[0])
                hit = results[0] == 1
            elif pool == "PLACE":
                probability = _place_probability(0, strengths, paid_places, exponents)
                hit = 1 <= results[0] <= paid_places
            elif pool == "QPL":
                probability = _qpl_probability((0, 1), strengths, paid_places, exponents)
                hit = bool(np.all((results[:2] >= 1) & (results[:2] <= paid_places)))
            elif pool in UNORDERED_POOLS:
                probability = _unordered_probability(selected, strengths, exponents)
                actual = set(np.flatnonzero((results >= 1) & (results <= count)))
                hit = len(actual) == count and set(selected) == actual
            else:
                probability = benter_order_probability(selected, strengths, exponents)
                hit = bool(np.array_equal(results[:count], np.arange(1, count + 1)))
            totals[pool]["races"] += 1
            totals[pool]["hits"] += int(hit)
            totals[pool]["probability_sum"] += probability

    report = {}
    for pool, values in totals.items():
        races = int(values["races"])
        hits = int(values["hits"])
        mean_probability = float(values["probability_sum"] / races) if races else 0.0
        hit_rate = float(hits / races) if races else 0.0
        report[pool] = {
            "races": races,
            "hits": hits,
            "hit_rate": hit_rate,
            "mean_top_probability": mean_probability,
            "calibration_gap": hit_rate - mean_probability,
        }
    return report
