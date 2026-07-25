"""Reproducible training, evaluation, and artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from .data import (
    build_canonical_runner_dataset, build_runner_dataset, chronological_race_split,
    validate_runner_dataset,
)
from .modeling import (
    MarketBlend,
    RaceProbabilityModel,
    TemperatureCalibrator,
    evaluate_probabilities,
    disagreement_report,
    incremental_pseudo_r2,
)
from .pools import fit_order_exponents


@dataclass(frozen=True)
class CandidateReport:
    kind: str
    validation: dict[str, float]
    test_fundamental: dict[str, float]
    test_blended: dict[str, float]
    temperature: float
    fundamental_weight: float
    market_weight: float
    incremental_pseudo_r2: float
    disagreement: list[dict]
    second_place_exponent: float
    third_place_exponent: float


def _evaluate_candidate(kind, splits, output_dir: Path) -> CandidateReport:
    model = RaceProbabilityModel(kind=kind).fit(splits.train)
    validation_raw = model.predict_proba(splits.validation)
    calibrator = TemperatureCalibrator.fit(validation_raw, splits.validation)
    validation_probability = calibrator.transform(validation_raw, splits.validation["race_id"])
    order_frame = splits.validation.assign(model_probability=validation_probability)
    order_exponents = fit_order_exponents(order_frame, "model_probability")
    blend = MarketBlend.fit(
        validation_probability,
        splits.validation["market_probability"].to_numpy(),
        splits.validation,
    )
    test_fundamental = calibrator.transform(model.predict_proba(splits.test), splits.test["race_id"])
    test_blended = blend.transform(
        test_fundamental,
        splits.test["market_probability"].to_numpy(),
        splits.test["race_id"],
    )
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "blend": blend,
        "order_exponents": order_exponents,
        "features": list(model.pipeline.feature_names_in_) if model.pipeline is not None else [],
    }
    joblib.dump(artifact, output_dir / f"{kind}.joblib")
    return CandidateReport(
        kind=kind,
        validation=evaluate_probabilities(validation_probability, splits.validation),
        test_fundamental=evaluate_probabilities(test_fundamental, splits.test),
        test_blended=evaluate_probabilities(test_blended, splits.test),
        temperature=calibrator.temperature,
        fundamental_weight=blend.fundamental_weight,
        market_weight=blend.market_weight,
        incremental_pseudo_r2=incremental_pseudo_r2(
            test_blended, splits.test["market_probability"].to_numpy(), splits.test
        ),
        disagreement=disagreement_report(
            test_fundamental,
            splits.test["market_probability"].to_numpy(),
            test_blended,
            splits.test,
        ),
        second_place_exponent=order_exponents.second,
        third_place_exponent=order_exponents.third,
    )


def _train_frame_and_report(
    frame,
    output_dir: Path,
    dataset_name: str,
    limitations: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_runner_dataset(frame)
    splits = chronological_race_split(frame)
    market_metrics = evaluate_probabilities(splits.test["market_probability"].to_numpy(), splits.test)
    candidates = [_evaluate_candidate(kind, splits, output_dir) for kind in ("logit", "boosted")]
    champion = min(candidates, key=lambda item: item.test_blended["race_log_loss"])
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "dataset": {
            "runners": len(frame),
            "races": int(frame["race_id"].nunique()),
            "date_min": frame["date"].min().date().isoformat(),
            "date_max": frame["date"].max().date().isoformat(),
            "train_races": int(splits.train["race_id"].nunique()),
            "validation_races": int(splits.validation["race_id"].nunique()),
            "test_races": int(splits.test["race_id"].nunique()),
            "train_date_min": splits.train["date"].min().date().isoformat(),
            "train_date_max": splits.train["date"].max().date().isoformat(),
            "validation_date_min": splits.validation["date"].min().date().isoformat(),
            "validation_date_max": splits.validation["date"].max().date().isoformat(),
            "test_date_min": splits.test["date"].min().date().isoformat(),
            "test_date_max": splits.test["date"].max().date().isoformat(),
        },
        "market_test": market_metrics,
        "candidates": [asdict(candidate) for candidate in candidates],
        "recommended_champion": champion.kind,
        "limitations": limitations,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_and_report(runs_path: Path, races_path: Path, output_dir: Path) -> dict:
    frame = build_runner_dataset(runs_path, races_path)
    return _train_frame_and_report(
        frame,
        output_dir,
        "legacy-hkracing-1997-2005",
        [
            "The archive ends in 2005 and does not establish current-market performance.",
            "Historical win_odds timestamp semantics are not documented; ROI claims require timestamped odds.",
            "Promotion to wagering requires recent forward paper-trading evidence.",
        ],
    )


def train_canonical_and_report(runners_path: Path, output_dir: Path) -> dict:
    frame = build_canonical_runner_dataset(runners_path)
    return _train_frame_and_report(
        frame,
        output_dir,
        "canonical-hkjc-2005-2025",
        [
            "Result-page WIN odds are final market information, not guaranteed pre-race decision prices.",
            "Official result rows lack many race-card fields, so this baseline relies heavily on weights, draw, venue, and lagged horse form.",
            "Winnerless archive races and races with incomplete final odds are excluded.",
            "Promotion to wagering requires forward paper trading with timestamped live odds.",
        ],
    )
