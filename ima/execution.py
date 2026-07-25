"""Paper and live wager execution boundaries."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .domain import ExecutionReceipt, WagerRecommendation


class WagerExecutor(Protocol):
    def submit(self, recommendation: WagerRecommendation) -> ExecutionReceipt: ...


class PaperExecutor:
    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self._submitted: set[str] = set()
        if ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._submitted.add(json.loads(line)["decision_id"])

    def submit(self, recommendation: WagerRecommendation) -> ExecutionReceipt:
        if recommendation.decision_id in self._submitted:
            return ExecutionReceipt(recommendation.decision_id, "duplicate")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(recommendation), sort_keys=True) + "\n")
        self._submitted.add(recommendation.decision_id)
        return ExecutionReceipt(recommendation.decision_id, "paper_accepted")


class HKJCWebExecutor:
    """Fail-closed boundary for future authenticated browser submission."""

    def __init__(self, live_enabled: bool = False):
        self.live_enabled = live_enabled

    def submit(self, recommendation: WagerRecommendation) -> ExecutionReceipt:
        if not self.live_enabled:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="live execution disabled")
        raise RuntimeError(
            "Live HKJC submission requires an authenticated session, credential vault, "
            "transaction confirmation parser, and operator-approved limits"
        )
