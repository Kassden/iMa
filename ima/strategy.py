"""Uncertainty-aware, race-budget-constrained Kelly recommendations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .domain import PoolCandidate, WagerRecommendation


@dataclass(frozen=True)
class RiskBudget:
    max_race_fraction: float = 0.02
    max_combination_fraction: float = 0.005
    uncertainty_z: float = 1.64
    minimum_edge: float = 0.02
    minimum_stake: float = 10.0
    stake_increment: float = 10.0
    drawdown_multiplier: float = 1.0

    def validate(self) -> None:
        if not 0 < self.max_combination_fraction <= self.max_race_fraction < 1:
            raise ValueError("Risk fractions are inconsistent")
        if self.minimum_stake <= 0 or self.stake_increment <= 0:
            raise ValueError("Stake controls must be positive")
        if not 0 <= self.drawdown_multiplier <= 1:
            raise ValueError("Drawdown multiplier must be in [0, 1]")


def _kelly_fraction(probability: float, decimal_odds: float) -> float:
    if decimal_odds <= 1:
        return 0.0
    return max(0.0, (probability * decimal_odds - 1.0) / (decimal_odds - 1.0))


def recommend_wagers(
    candidates: list[PoolCandidate],
    bankroll: float,
    budget: RiskBudget,
) -> list[WagerRecommendation]:
    budget.validate()
    if bankroll <= 0:
        raise ValueError("Bankroll must be positive")
    scored = []
    for candidate in candidates:
        conservative_p = max(0.0, candidate.probability - budget.uncertainty_z * candidate.probability_std)
        edge = conservative_p - candidate.market_probability
        fraction = _kelly_fraction(conservative_p, candidate.current_decimal_odds)
        if edge < budget.minimum_edge or fraction <= 0:
            continue
        scored.append((candidate, conservative_p, fraction))
    if not scored:
        return []

    race_cap = budget.max_race_fraction * budget.drawdown_multiplier
    requested = sum(min(item[2], budget.max_combination_fraction) for item in scored)
    scale = min(1.0, race_cap / requested) if requested else 0.0
    recommendations = []
    for candidate, conservative_p, raw_fraction in scored:
        fraction = min(raw_fraction, budget.max_combination_fraction) * scale
        raw_stake = bankroll * fraction
        stake = (raw_stake // budget.stake_increment) * budget.stake_increment
        if stake < budget.minimum_stake:
            continue
        token = "|".join((candidate.race_id, candidate.pool, *candidate.combination, candidate.model_version))
        decision_id = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
        recommendations.append(WagerRecommendation(
            decision_id=decision_id,
            race_id=candidate.race_id,
            pool=candidate.pool,
            combination=candidate.combination,
            stake=stake,
            bankroll=bankroll,
            probability=candidate.probability,
            fair_odds=candidate.fair_odds,
            current_decimal_odds=candidate.current_decimal_odds,
            expected_value_per_unit=candidate.probability * candidate.current_decimal_odds - 1.0,
            kelly_fraction=fraction,
            model_version=candidate.model_version,
        ))
    return recommendations
