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
HKJC_WIN_TAKEOUT_RATE = 0.18


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


def _expected_value_per_dollar(probability: float, decimal_odds: float) -> float:
    return probability * decimal_odds - 1.0


def _takeout_adjusted_gain_per_dollar(
    probability: float,
    decimal_odds: float,
    takeout_rate: float = HKJC_WIN_TAKEOUT_RATE,
) -> float:
    return probability * decimal_odds * (1.0 - takeout_rate) - 1.0


def _finish_estimates_by_horse(
    auxiliary_predictions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float | int]]:
    estimates: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in auxiliary_predictions:
        horse_no = row.get("horse_no")
        race_id = row.get("race_id")
        if horse_no is None or race_id is None:
            continue
        estimates[(str(race_id), str(horse_no))] = {
            "predicted_finish_time": row.get("predicted_finish_time"),
            "predicted_position": row.get("predicted_position"),
            "predicted_position_rank": row.get("predicted_position_rank"),
        }
    return estimates


def _candidate_finish_detail(
    candidate: PoolCandidate,
    estimates: dict[tuple[str, str], dict[str, float | int]],
) -> dict[str, Any]:
    members = []
    times = []
    ranks = []
    positions = []
    for runner in candidate.combination:
        estimate = estimates.get((candidate.race_id, str(runner)), {})
        finish_time = estimate.get("predicted_finish_time")
        position = estimate.get("predicted_position")
        rank = estimate.get("predicted_position_rank")
        if finish_time is not None:
            times.append(float(finish_time))
        if position is not None:
            positions.append(float(position))
        if rank is not None:
            ranks.append(int(rank))
        members.append({
            "horse_no": str(runner),
            "predicted_finish_time": finish_time,
            "predicted_position": position,
            "predicted_position_rank": rank,
        })
    return {
        "estimated_finish_time": sum(times) / len(times) if times else None,
        "estimated_position": sum(positions) / len(positions) if positions else None,
        "best_estimated_position_rank": min(ranks) if ranks else None,
        "runner_estimates": members,
    }


def _basis_fields(probability: float, probability_basis: str) -> dict[str, float | str | None]:
    is_fallback = probability_basis == "public_win_market_fallback"
    return {
        "probability_basis": probability_basis,
        "model_probability": None if is_fallback else probability,
        "fallback_probability": probability if is_fallback else None,
        "staking_probability": probability,
    }


def serialize_candidate(
    item: PoolCandidate,
    finish_estimates: dict[tuple[str, str], dict[str, float | int]],
    probability_basis: str,
    stake_unit: float = 10.0,
) -> dict[str, Any]:
    probability = float(item.probability)
    market_probability = float(item.market_probability)
    market_odds = float(item.current_decimal_odds)
    ev_per_dollar = _expected_value_per_dollar(probability, market_odds)
    takeout_gain = _takeout_adjusted_gain_per_dollar(probability, market_odds)
    finish_detail = _candidate_finish_detail(item, finish_estimates)
    return {
        "race_id": item.race_id,
        "pool": item.pool,
        "combination": item.combination,
        **_basis_fields(probability, probability_basis),
        "market_probability": market_probability,
        "probability_edge": probability - market_probability,
        "our_odds": item.fair_odds,
        "market_odds": market_odds,
        "expected_value_per_dollar": ev_per_dollar,
        "expected_value_per_10": ev_per_dollar * stake_unit,
        "takeout_rate": HKJC_WIN_TAKEOUT_RATE,
        "takeout_adjusted_gain_per_dollar": takeout_gain,
        "takeout_adjusted_gain_per_10": takeout_gain * stake_unit,
        "odds_updated_at": item.odds_updated_at,
        **finish_detail,
    }


def serialize_recommendation(
    item: WagerRecommendation,
    finish_estimates: dict[tuple[str, str], dict[str, float | int]],
    probability_basis: str,
) -> dict[str, Any]:
    probability = float(item.probability)
    market_odds = float(item.current_decimal_odds)
    market_probability = 0.0 if market_odds <= 0 else 1.0 / market_odds
    takeout_gain = _takeout_adjusted_gain_per_dollar(probability, market_odds)
    candidate = PoolCandidate(
        race_id=item.race_id,
        pool=item.pool,
        combination=item.combination,
        probability=probability,
        current_decimal_odds=market_odds,
        model_version=item.model_version,
    )
    return {
        **asdict(item),
        **_basis_fields(probability, probability_basis),
        "market_probability": market_probability,
        "probability_edge": probability - market_probability,
        "our_odds": item.fair_odds,
        "market_odds": market_odds,
        "expected_value_per_10": item.expected_value_per_unit * item.stake,
        "takeout_rate": HKJC_WIN_TAKEOUT_RATE,
        "takeout_adjusted_gain_per_dollar": takeout_gain,
        "takeout_adjusted_gain": takeout_gain * item.stake,
        **_candidate_finish_detail(candidate, finish_estimates),
    }


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
    finish_estimates = _finish_estimates_by_horse(auxiliary_predictions)
    ratable_count = int(live_frame["ratable"].sum()) if "ratable" in live_frame else len(live_frame)
    probability_basis = "fundamental_plus_market" if ratable_count else "public_win_market_fallback"
    return {
        "race_ids": sorted(predictions),
        "model_version": model_version,
        "market_sources": {
            "WIN": "timestamped current runner WIN odds",
            "PLACE": "timestamped current runner PLACE odds",
            "exotics": "timestamped pool-specific HKJC combination odds when rendered",
        },
        "prediction_basis": {
            "runners": len(live_frame),
            "fundamental_ratable_runners": ratable_count,
            "basis": probability_basis,
            "warning": None if ratable_count else (
                "No runner had the complete live historical feature row required by the model; "
                "runner strengths therefore use the documented public WIN fallback."
            ),
        },
        "candidate_formula": {
            "model_probability": (
                "Independent model probability after calibration and market blending; unavailable when the "
                "race is fully public-fallback."
            ),
            "fallback_probability": "Public-derived staking probability used when runners are unratable.",
            "staking_probability": "The probability used for EV and Kelly sizing.",
            "market_probability": "Raw reciprocal of currently displayed decimal market odds: 1 / market_odds.",
            "our_odds": "Model fair decimal odds: 1 / model_probability.",
            "probability_edge": "model_probability - market_probability.",
            "expected_value_per_dollar": "model_probability * market_odds - 1.",
            "takeout_adjusted_gain_per_dollar": (
                "model_probability * market_odds * (1 - 0.18) - 1, using the requested 18% HKJC "
                "WIN takeout haircut."
            ),
        },
        "summary": summarize_recommendations(recommendations, len(candidates)),
        "recommendations": [
            serialize_recommendation(item, finish_estimates, probability_basis)
            for item in recommendations
        ],
        "auxiliary_predictions": auxiliary_predictions,
        "priced_candidates": [
            serialize_candidate(item, finish_estimates, probability_basis)
            for item in sorted(
                candidates,
                key=lambda candidate: candidate.probability * candidate.current_decimal_odds,
                reverse=True,
            )
        ],
    }
