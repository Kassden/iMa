"""Versioned champion/challenger registry with explicit promotion gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PromotionPolicy:
    minimum_test_races: int = 500
    maximum_log_loss_regression: float = 0.0
    maximum_ece_regression: float = 0.002
    require_manual_live_approval: bool = True


def eligible_for_promotion(
    champion_metrics: dict[str, float],
    challenger_metrics: dict[str, float],
    test_races: int,
    policy: PromotionPolicy,
) -> tuple[bool, list[str]]:
    reasons = []
    if test_races < policy.minimum_test_races:
        reasons.append("insufficient_test_races")
    if challenger_metrics["race_log_loss"] > champion_metrics["race_log_loss"] + policy.maximum_log_loss_regression:
        reasons.append("log_loss_regression")
    if challenger_metrics["ece"] > champion_metrics["ece"] + policy.maximum_ece_regression:
        reasons.append("calibration_regression")
    return not reasons, reasons


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def register(self, version: str, report: dict, artifact_paths: list[str]) -> Path:
        target = self.path / f"{version}.json"
        if target.exists():
            raise FileExistsError(f"Model version already exists: {version}")
        target.write_text(json.dumps({
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "report": report,
            "artifacts": artifact_paths,
        }, indent=2), encoding="utf-8")
        return target

    def promote(self, version: str, approved_by: str, live_approved: bool = False) -> Path:
        source = self.path / f"{version}.json"
        if not source.exists():
            raise FileNotFoundError(source)
        record = json.loads(source.read_text(encoding="utf-8"))
        pointer = self.path / "champion.json"
        pointer.write_text(json.dumps({
            "version": version,
            "approved_by": approved_by,
            "live_approved": live_approved,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "record": record,
        }, indent=2), encoding="utf-8")
        return pointer
