"""Explicit target contracts for research-only optimizer experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class TargetContractError(ValueError):
    """Raised when a target cannot be safely constructed from the frame."""


@dataclass(frozen=True)
class TargetContract:
    kind: str
    parameters: dict[str, Any] = field(default_factory=dict)
    label_column: str = ""
    metric_family: str = ""
    model_task: str = ""
    notes: tuple[str, ...] = ()


SUPPORTED_TARGETS = {
    "win_probability",
    "ranking_strength",
    "placing_top_k",
    "adjusted_finish_time_or_speed",
    "market_odds_forecast",
}


def target_contract(kind: str, parameters: dict[str, Any] | None = None) -> TargetContract:
    params = dict(parameters or {})
    if kind not in SUPPORTED_TARGETS:
        raise TargetContractError(f"Unknown target kind: {kind}")
    if kind == "win_probability":
        return TargetContract(kind, params, "target_probability", "race_log_loss", "classifier")
    if kind == "ranking_strength":
        return TargetContract(kind, params, "target_rank_score", "within_race_ranking", "ranker")
    if kind == "placing_top_k":
        top_k = int(params.get("top_k", 3))
        if top_k < 1:
            raise TargetContractError("placing_top_k requires top_k >= 1")
        return TargetContract(
            kind,
            {"top_k": top_k},
            f"target_top_{top_k}",
            "top_k_probability",
            "classifier",
            ("top_k depends on eligible field size",),
        )
    if kind == "adjusted_finish_time_or_speed":
        return TargetContract(
            kind,
            params,
            "target_speed",
            "regression_and_rank",
            "regressor",
            ("target is physical speed; condition adjustment is train-fitted by the executor",),
        )
    return TargetContract(kind, params, "target_future_market_probability", "odds_forecast", "regressor")


def apply_target_contract(frame: pd.DataFrame, contract: TargetContract) -> pd.DataFrame:
    """Return a copy with the contract label materialized and validated."""
    work = frame.copy()
    if contract.kind == "win_probability":
        _require_columns(work, ("race_id", "target_win", "target_probability"))
        _require_one_winner_per_race(work)
        _require_race_probability_totals(work, "target_probability")
        return work
    if contract.kind == "ranking_strength":
        _require_columns(work, ("race_id", "result"))
        result = pd.to_numeric(work["result"], errors="coerce")
        if result.isna().any():
            raise TargetContractError("ranking_strength requires numeric result values")
        field_size = work.groupby("race_id")["race_id"].transform("size").clip(lower=1)
        work["target_rank_score"] = 1.0 - (result - 1.0) / field_size
        return work
    if contract.kind == "placing_top_k":
        _require_columns(work, ("race_id", "result"))
        top_k = int(contract.parameters["top_k"])
        field_size = work.groupby("race_id")["race_id"].transform("size")
        if (field_size < top_k).any():
            raise TargetContractError("placing_top_k cannot exceed field size")
        result = pd.to_numeric(work["result"], errors="coerce")
        if result.isna().any():
            raise TargetContractError("placing_top_k requires numeric result values")
        work[contract.label_column] = result.le(top_k).astype(int)
        return work
    if contract.kind == "adjusted_finish_time_or_speed":
        _require_columns(work, ("race_id", "finish_time", "distance"))
        finish = pd.to_numeric(work["finish_time"], errors="coerce")
        distance = pd.to_numeric(work["distance"], errors="coerce")
        valid = finish.gt(0) & distance.gt(0)
        coverage = float(valid.mean()) if len(valid) else 0.0
        min_coverage = float(contract.parameters.get("min_coverage", 0.8))
        if coverage < min_coverage:
            raise TargetContractError(
                f"finish-time coverage {coverage:.3f} below required {min_coverage:.3f}"
            )
        work["target_speed"] = distance / finish
        return work
    if contract.kind == "market_odds_forecast":
        _require_columns(work, (
            "race_id", "odds_snapshot_at", "forecast_at", "future_market_probability",
        ))
        snapshot = pd.to_datetime(work["odds_snapshot_at"], errors="coerce", utc=True)
        forecast = pd.to_datetime(work["forecast_at"], errors="coerce", utc=True)
        if snapshot.isna().any() or forecast.isna().any() or not snapshot.lt(forecast).all():
            raise TargetContractError(
                "market_odds_forecast requires timestamped snapshots strictly before forecast_at"
            )
        _require_race_probability_totals(work, "future_market_probability")
        work["target_future_market_probability"] = work["future_market_probability"]
        return work
    raise TargetContractError(f"Unsupported target kind: {contract.kind}")


def validate_no_forbidden_label_features(
    contract: TargetContract,
    feature_columns: tuple[str, ...] | list[str],
) -> None:
    forbidden = {
        "target_win",
        "target_probability",
        "result",
        "won",
        "finish_time",
        contract.label_column,
    }
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        raise TargetContractError(
            f"Target {contract.kind} forbids post-race label features: {leaked}"
        )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TargetContractError(f"Missing required target columns: {missing}")


def _require_one_winner_per_race(frame: pd.DataFrame) -> None:
    winners = frame.groupby("race_id")["target_win"].sum()
    if not np.allclose(winners.to_numpy(), 1.0):
        bad = winners[~np.isclose(winners.to_numpy(), 1.0)].index.tolist()[:10]
        raise TargetContractError(f"Expected exactly one winner per race: {bad}")


def _require_race_probability_totals(frame: pd.DataFrame, column: str) -> None:
    totals = frame.groupby("race_id")[column].sum()
    if not np.allclose(totals.to_numpy(), 1.0):
        raise TargetContractError(f"{column} must sum to one by race")
