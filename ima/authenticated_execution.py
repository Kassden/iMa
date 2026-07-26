"""Guarded authenticated execution port for HKJC web transactions."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from .domain import ExecutionReceipt, WagerRecommendation


class AuthenticatedSession(Protocol):
    def is_authenticated(self) -> bool: ...
    def submit_and_confirm(self, recommendation: WagerRecommendation) -> tuple[str, str]: ...


@dataclass(frozen=True)
class TransactionLimits:
    max_stake_per_bet: float = 10.0
    max_stake_per_race: float = 10.0
    max_stake_per_day: float = 50.0
    allowed_pools: tuple[str, ...] = ("WIN",)


class MacOSKeychainCredentials:
    """Retrieve credentials without exposing them through files or logs."""

    def __init__(self, account: str, password_service: str = "ima-hkjc-password"):
        self.account = account
        self.password_service = password_service

    def password(self) -> str:
        completed = subprocess.run(
            ["security", "find-generic-password", "-w", "-s", self.password_service, "-a", self.account],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("HKJC password was not found in macOS Keychain")
        return completed.stdout.rstrip("\n")


class GuardedHKJCExecutor:
    def __init__(
        self,
        session: AuthenticatedSession,
        audit_path: Path,
        limits: TransactionLimits = TransactionLimits(),
        live_enabled: bool = False,
        operator_approved: bool = False,
    ):
        self.session = session
        self.audit_path = audit_path
        self.limits = limits
        self.live_enabled = live_enabled
        self.operator_approved = operator_approved
        self._accepted: set[str] = set()
        self._race_stakes: dict[str, float] = {}
        self._daily_stake = 0.0

    def submit(self, recommendation: WagerRecommendation) -> ExecutionReceipt:
        if recommendation.decision_id in self._accepted:
            return ExecutionReceipt(recommendation.decision_id, "duplicate")
        if not self.live_enabled or not self.operator_approved:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="live approval missing")
        if recommendation.pool not in self.limits.allowed_pools:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="pool not allowed")
        if recommendation.stake > self.limits.max_stake_per_bet:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="per-bet limit")
        race_total = self._race_stakes.get(recommendation.race_id, 0.0) + recommendation.stake
        if race_total > self.limits.max_stake_per_race:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="per-race limit")
        if self._daily_stake + recommendation.stake > self.limits.max_stake_per_day:
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="daily limit")
        if not self.session.is_authenticated():
            return ExecutionReceipt(recommendation.decision_id, "blocked", detail="session not authenticated")
        external_id, confirmation = self.session.submit_and_confirm(recommendation)
        if not external_id or not confirmation:
            return ExecutionReceipt(recommendation.decision_id, "failed", detail="missing confirmation")
        self._accepted.add(recommendation.decision_id)
        self._race_stakes[recommendation.race_id] = race_total
        self._daily_stake += recommendation.stake
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "day": date.today().isoformat(),
                "decision_id": recommendation.decision_id,
                "race_id": recommendation.race_id,
                "pool": recommendation.pool,
                "combination": recommendation.combination,
                "stake": recommendation.stake,
                "external_id": external_id,
                "confirmation": confirmation,
            }, sort_keys=True) + "\n")
        return ExecutionReceipt(recommendation.decision_id, "accepted", external_id, confirmation)
