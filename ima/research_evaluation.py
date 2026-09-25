"""Protected research evaluation helpers for agentic optimizer campaigns."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .modeling import evaluate_probabilities, race_log_loss
from .research_targets import TargetContract, apply_target_contract, target_contract


class ResearchEvaluationError(ValueError):
    """Raised when a research evaluation protocol is invalid."""


@dataclass(frozen=True)
class ResearchFold:
    fold_id: str
    train_race_ids: tuple[str, ...]
    calibration_race_ids: tuple[str, ...]
    score_race_ids: tuple[str, ...]
    train_end_date: str
    calibration_end_date: str
    score_end_date: str


@dataclass(frozen=True)
class ProtocolManifest:
    protocol_id: str
    target_kind: str
    race_count: int
    runner_count: int
    first_date: str
    last_date: str
    folds: tuple[ResearchFold, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["folds"] = [asdict(fold) for fold in self.folds]
        return payload


def race_order(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("race_id", "date", "race_no"))
    work = frame[["race_id", "date", "race_no"]].drop_duplicates("race_id").copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise")
    return work.sort_values(["date", "race_no", "race_id"], kind="stable").reset_index(drop=True)


def build_expanding_folds(
    frame: pd.DataFrame,
    *,
    min_train_races: int,
    calibration_races: int,
    score_races: int,
    max_folds: int | None = None,
) -> tuple[ResearchFold, ...]:
    if min_train_races < 1 or calibration_races < 1 or score_races < 1:
        raise ResearchEvaluationError("Fold race counts must be positive")
    order = race_order(frame)
    required = min_train_races + calibration_races + score_races
    if len(order) < required:
        raise ResearchEvaluationError(
            f"Need at least {required} races for protected folds, found {len(order)}"
        )
    folds: list[ResearchFold] = []
    start = min_train_races
    while start + calibration_races + score_races <= len(order):
        train = order.iloc[:start]
        calibration = order.iloc[start:start + calibration_races]
        score = order.iloc[start + calibration_races:start + calibration_races + score_races]
        fold = ResearchFold(
            fold_id=f"fold-{len(folds) + 1:03d}",
            train_race_ids=tuple(train["race_id"].astype(str)),
            calibration_race_ids=tuple(calibration["race_id"].astype(str)),
            score_race_ids=tuple(score["race_id"].astype(str)),
            train_end_date=_date_text(train["date"].iloc[-1]),
            calibration_end_date=_date_text(calibration["date"].iloc[-1]),
            score_end_date=_date_text(score["date"].iloc[-1]),
        )
        validate_fold(frame, fold)
        folds.append(fold)
        if max_folds is not None and len(folds) >= max_folds:
            break
        start += score_races
    return tuple(folds)


def validate_fold(frame: pd.DataFrame, fold: ResearchFold) -> None:
    groups = [set(fold.train_race_ids), set(fold.calibration_race_ids), set(fold.score_race_ids)]
    if any(not group for group in groups):
        raise ResearchEvaluationError("Train, calibration and score groups must be non-empty")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ResearchEvaluationError(f"Fold {fold.fold_id} has overlapping race IDs")
    order = race_order(frame)
    position = {str(row.race_id): index for index, row in order.iterrows()}
    missing = sorted(set().union(*groups) - set(position))
    if missing:
        raise ResearchEvaluationError(f"Fold {fold.fold_id} references unknown races: {missing}")
    if max(position[race_id] for race_id in fold.train_race_ids) >= min(
        position[race_id] for race_id in fold.calibration_race_ids
    ):
        raise ResearchEvaluationError(f"Fold {fold.fold_id} violates train < calibration")
    if max(position[race_id] for race_id in fold.calibration_race_ids) >= min(
        position[race_id] for race_id in fold.score_race_ids
    ):
        raise ResearchEvaluationError(f"Fold {fold.fold_id} violates calibration < score")


def make_protocol_manifest(
    frame: pd.DataFrame,
    *,
    target: str | TargetContract = "win_probability",
    min_train_races: int = 24,
    calibration_races: int = 6,
    score_races: int = 6,
    max_folds: int | None = None,
) -> ProtocolManifest:
    contract = target_contract(target) if isinstance(target, str) else target
    work = apply_target_contract(frame, contract)
    folds = build_expanding_folds(
        work,
        min_train_races=min_train_races,
        calibration_races=calibration_races,
        score_races=score_races,
        max_folds=max_folds,
    )
    order = race_order(work)
    fingerprint = {
        "target": contract.kind,
        "races": order["race_id"].astype(str).tolist(),
        "folds": [asdict(fold) for fold in folds],
    }
    protocol_id = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return ProtocolManifest(
        protocol_id=protocol_id,
        target_kind=contract.kind,
        race_count=int(order["race_id"].nunique()),
        runner_count=int(len(work)),
        first_date=_date_text(order["date"].iloc[0]),
        last_date=_date_text(order["date"].iloc[-1]),
        folds=folds,
    )


def select_fold(frame: pd.DataFrame, race_ids: Iterable[str]) -> pd.DataFrame:
    ids = set(race_ids)
    selected = frame[frame["race_id"].astype(str).isin(ids)].copy()
    if set(selected["race_id"].astype(str)) != ids:
        missing = sorted(ids - set(selected["race_id"].astype(str)))
        raise ResearchEvaluationError(f"Missing fold race rows: {missing}")
    return selected.sort_values(["date", "race_no", "race_id"], kind="stable").reset_index(drop=True)


def baseline_probabilities(frame: pd.DataFrame, kind: str) -> np.ndarray:
    _require_columns(frame, ("race_id",))
    if kind == "uniform":
        field_size = frame.groupby("race_id")["race_id"].transform("size").to_numpy(dtype=float)
        return 1.0 / field_size
    if kind == "raw_market":
        _require_columns(frame, ("market_probability",))
        return _normalise_existing(frame["market_probability"].to_numpy(dtype=float), frame["race_id"])
    if kind == "calibrated_market":
        _require_columns(frame, ("market_probability",))
        market = _normalise_existing(frame["market_probability"].to_numpy(dtype=float), frame["race_id"])
        return _normalise_existing(np.sqrt(np.clip(market, 1e-12, 1.0)), frame["race_id"])
    raise ResearchEvaluationError(f"Unknown baseline kind: {kind}")


def per_race_log_losses(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    _validate_probability_vector(probabilities, frame)
    work = frame[["race_id", "date", "race_no", "target_probability"]].copy()
    work["_probability"] = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    work["_loss"] = -work["target_probability"] * np.log(work["_probability"])
    losses = work.groupby("race_id", sort=False).agg(
        date=("date", "first"),
        race_no=("race_no", "first"),
        race_log_loss=("_loss", "sum"),
        runners=("_loss", "size"),
    ).reset_index()
    losses.insert(0, "label", label)
    return losses


def evaluate_research_probabilities(
    probabilities: np.ndarray,
    frame: pd.DataFrame,
    *,
    label: str,
) -> dict:
    _validate_probability_vector(probabilities, frame)
    return {
        "label": label,
        "metrics": evaluate_probabilities(probabilities, frame),
        "race_weighted_log_loss": race_log_loss(probabilities, frame),
        "per_race_losses": per_race_log_losses(probabilities, frame, label=label).to_dict("records"),
    }


def _validate_probability_vector(probabilities: np.ndarray, frame: pd.DataFrame) -> None:
    if len(probabilities) != len(frame):
        raise ResearchEvaluationError("Probability vector length does not match frame")
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ResearchEvaluationError("Probabilities must be finite and non-negative")
    totals = pd.Series(values).groupby(frame["race_id"].to_numpy()).sum()
    if not np.allclose(totals.to_numpy(), 1.0):
        raise ResearchEvaluationError("Probabilities must sum to one by race")


def _normalise_existing(values: np.ndarray, race_ids: pd.Series) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    totals = pd.Series(clipped).groupby(race_ids.to_numpy()).transform("sum").to_numpy()
    return clipped / totals


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ResearchEvaluationError(f"Missing required evaluation columns: {missing}")
