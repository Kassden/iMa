"""Market probability, fair odds, and discrepancy calculations."""

from __future__ import annotations

import numpy as np


def implied_probabilities(decimal_odds: np.ndarray, takeout: float | None = None) -> np.ndarray:
    odds = np.asarray(decimal_odds, dtype=float)
    if np.any(odds <= 1):
        raise ValueError("Decimal odds must be greater than one")
    raw = 1.0 / odds
    normalized = raw / raw.sum()
    if takeout is not None and not 0 <= takeout < 1:
        raise ValueError("Takeout must be in [0, 1)")
    return normalized


def fair_odds(probability: float, margin: float = 0.0) -> float:
    if not 0 < probability <= 1:
        raise ValueError("Probability must be in (0, 1]")
    if not 0 <= margin < 1:
        raise ValueError("Margin must be in [0, 1)")
    return 1.0 / (probability * (1.0 - margin))
