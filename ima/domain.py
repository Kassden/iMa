"""Auditable domain contracts shared by prediction, strategy, and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class PoolCandidate:
    race_id: str
    pool: str
    combination: tuple[str, ...]
    probability: float
    current_decimal_odds: float
    probability_std: float = 0.0
    projected_decimal_odds: float | None = None
    odds_updated_at: str | None = None
    model_version: str = "unknown"

    @property
    def fair_odds(self) -> float:
        return float("inf") if self.probability <= 0 else 1.0 / self.probability

    @property
    def market_probability(self) -> float:
        return 0.0 if self.current_decimal_odds <= 0 else 1.0 / self.current_decimal_odds

    @property
    def probability_discrepancy(self) -> float:
        return self.probability - self.market_probability


@dataclass(frozen=True)
class WagerRecommendation:
    decision_id: str
    race_id: str
    pool: str
    combination: tuple[str, ...]
    stake: float
    bankroll: float
    probability: float
    fair_odds: float
    current_decimal_odds: float
    expected_value_per_unit: float
    kelly_fraction: float
    model_version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class ExecutionReceipt:
    decision_id: str
    status: str
    external_id: str | None = None
    detail: str | None = None
