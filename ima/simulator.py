"""Auditable $10-unit simulation against timestamped HKJC market prices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .auxiliary import AuxiliaryModelBundle
from .domain import PoolCandidate, WagerRecommendation
from .inference import predict_live_pools
from .pools import CombinationProbability
from .strategy import RiskBudget, recommend_wagers


PROVIDER_POOL_ALIASES = {
    "TCE": "TIERCE",
    "TRI": "TRI",
    "FF": "FIRST4",
    "QTT": "QUARTET",
}
UNORDERED_POOLS = {"QIN", "QPL", "TRI", "FIRST4"}


@dataclass(frozen=True)
class MarketPrice:
    pool: str
    combination: tuple[str, ...]
    decimal_odds: float
    updated_at: str | None = None


def _combination_key(pool: str, combination: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    runners = tuple(str(item) for item in combination)
    return tuple(sorted(runners, key=lambda item: (0, int(item)) if item.isdigit() else (1, item))) \
        if pool in UNORDERED_POOLS else runners


def extract_market_prices(raw_payload: dict[str, Any], live_frame: pd.DataFrame) -> list[MarketPrice]:
    prices: dict[tuple[str, tuple[str, ...]], MarketPrice] = {}
    for row in live_frame.itertuples():
        for pool, odds in (("WIN", row.win_odds), ("PLACE", getattr(row, "place_odds", None))):
            if pd.notna(odds) and float(odds) > 1:
                combination = (str(row.horse_no),)
                prices[(pool, combination)] = MarketPrice(pool, combination, float(odds))

    provider = raw_payload.get("provider") or {}
    for provider_pool, page in (provider.get("pool_pages") or {}).items():
        records = [
            *(page.get("normalized_combinations") or []),
            *(page.get("top_combinations") or []),
            *(page.get("matrix_combinations") or []),
        ]
        for record in records:
            pool = record.get("pool") or PROVIDER_POOL_ALIASES.get(provider_pool)
            if pool not in {"QIN", "QPL", "TRI", "TIERCE", "FIRST4", "QUARTET"}:
                continue
            try:
                odds = float(record["decimal_odds"])
            except (KeyError, TypeError, ValueError):
                continue
            if odds <= 1:
                continue
            combination = _combination_key(pool, record.get("combination") or [])
            if not combination:
                continue
            price = MarketPrice(pool, combination, odds, page.get("updated_at_display"))
            prices[(pool, combination)] = price
    return sorted(prices.values(), key=lambda item: (item.pool, item.combination))


def priced_candidates(
    predictions: dict[str, dict[str, list[CombinationProbability]]],
    prices: list[MarketPrice],
    model_version: str,
) -> list[PoolCandidate]:
    probability_index = {}
    for race_id, pools in predictions.items():
        for pool, combinations in pools.items():
            for item in combinations:
                probability_index[(pool, _combination_key(pool, item.runners))] = (
                    str(race_id), item.probability,
                )
    candidates = []
    for price in prices:
        matched = probability_index.get((price.pool, _combination_key(price.pool, price.combination)))
        if matched is None:
            continue
        race_id, probability = matched
        candidates.append(PoolCandidate(
            race_id=race_id,
            pool=price.pool,
            combination=price.combination,
            probability=float(probability),
            current_decimal_odds=price.decimal_odds,
            odds_updated_at=price.updated_at,
            model_version=model_version,
        ))
    return candidates


def summarize_recommendations(
    recommendations: list[WagerRecommendation],
    priced_count: int,
) -> dict[str, Any]:
    cost = float(sum(item.stake for item in recommendations))
    expected_gross = float(sum(
        item.stake * item.probability * item.current_decimal_odds
        for item in recommendations
    ))
    expected_wins = float(sum(item.probability for item in recommendations))
    max_payout = float(max(
        (item.stake * item.current_decimal_odds for item in recommendations), default=0.0,
    ))
    return {
        "priced_candidates": priced_count,
        "recommended_bets": len(recommendations),
        "stake_unit": 10.0,
        "total_cost": cost,
        "expected_wins": expected_wins,
        "expected_gross_return": expected_gross,
        "expected_net_return": expected_gross - cost,
        "expected_roi": (expected_gross - cost) / cost if cost else 0.0,
        "maximum_single_bet_payout": max_payout,
        "maximum_single_bet_profit_after_total_cost": max_payout - cost if cost else 0.0,
        "abstention": None if recommendations else (
            "No priced combination passed the conservative probability, edge, Kelly, and $10 minimum checks."
        ),
    }


def simulate_race(
    live_frame: pd.DataFrame,
    winner_artifact: dict,
    raw_payload: dict[str, Any],
    bankroll: float,
    model_version: str,
    budget: RiskBudget | None = None,
    auxiliary: AuxiliaryModelBundle | None = None,
) -> dict[str, Any]:
    budget = budget or RiskBudget(
        max_race_fraction=0.05,
        max_combination_fraction=0.01,
        minimum_edge=0.02,
        minimum_stake=10.0,
        stake_increment=10.0,
    )
    predictions = predict_live_pools(live_frame, winner_artifact, top_n=None)
    prices = extract_market_prices(raw_payload, live_frame)
    candidates = priced_candidates(predictions, prices, model_version)
    recommendations = recommend_wagers(candidates, bankroll, budget)
    auxiliary_predictions = auxiliary.predict(live_frame).to_dict("records") if auxiliary else []
    return {
        "race_ids": sorted(predictions),
        "model_version": model_version,
        "market_sources": {
            "WIN": "timestamped current runner WIN odds",
            "PLACE": "timestamped current runner PLACE odds",
            "exotics": "timestamped pool-specific HKJC combination odds when rendered",
        },
        "summary": summarize_recommendations(recommendations, len(candidates)),
        "recommendations": [asdict(item) for item in recommendations],
        "auxiliary_predictions": auxiliary_predictions,
        "priced_candidates": [
            {
                "race_id": item.race_id,
                "pool": item.pool,
                "combination": item.combination,
                "probability": item.probability,
                "fair_odds": item.fair_odds,
                "market_odds": item.current_decimal_odds,
                "expected_value_per_dollar": item.probability * item.current_decimal_odds - 1.0,
            }
            for item in sorted(
                candidates,
                key=lambda candidate: candidate.probability * candidate.current_decimal_odds,
                reverse=True,
            )
        ],
    }
