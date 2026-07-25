"""Immutable prediction records and post-race outcome reconciliation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .domain import WagerRecommendation


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  decision_id TEXT PRIMARY KEY,
  race_id TEXT NOT NULL,
  pool TEXT NOT NULL,
  combination_json TEXT NOT NULL,
  recommendation_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
  decision_id TEXT PRIMARY KEY REFERENCES predictions(decision_id),
  official_combination_json TEXT NOT NULL,
  dividend REAL,
  stake REAL NOT NULL,
  payout REAL NOT NULL,
  settled_at TEXT NOT NULL
);
"""


class PredictionLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def record(self, recommendation: WagerRecommendation) -> bool:
        try:
            self.connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    recommendation.decision_id,
                    recommendation.race_id,
                    recommendation.pool,
                    json.dumps(recommendation.combination),
                    json.dumps(asdict(recommendation), sort_keys=True),
                    recommendation.created_at,
                ),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def settle(
        self,
        decision_id: str,
        official_combination: tuple[str, ...],
        dividend: float | None,
        settled_at: str,
    ) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT combination_json, recommendation_json FROM predictions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown decision: {decision_id}")
        expected = tuple(json.loads(row[0]))
        recommendation = json.loads(row[1])
        stake = float(recommendation["stake"])
        won = expected == official_combination
        payout = stake * float(dividend) if won and dividend is not None else 0.0
        self.connection.execute(
            "INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?, ?, ?)",
            (decision_id, json.dumps(official_combination), dividend, stake, payout, settled_at),
        )
        self.connection.commit()
        return {"won": won, "stake": stake, "payout": payout, "profit": payout - stake}

    def performance(self) -> dict[str, float]:
        row = self.connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(stake), 0), COALESCE(SUM(payout), 0) FROM outcomes"
        ).fetchone()
        count, stake, payout = int(row[0]), float(row[1]), float(row[2])
        return {
            "settled_wagers": count,
            "stake": stake,
            "payout": payout,
            "profit": payout - stake,
            "roi": (payout - stake) / stake if stake else 0.0,
        }
