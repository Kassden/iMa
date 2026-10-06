"""Descriptive numerical diagnostics on saved score windows; no base-model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ima.modeling import (  # noqa: E402
    _blend_loss_gradient, _calibration_target, _probability_log_features,
    blend_probabilities, normalize_by_race, race_log_loss,
)

PREFIX = "market_blend_outlook_20261006_"
CANDIDATES = ("best_blend", "best_fundamental", "v6_fundamental", "v7_runtime_fundamental")


def audit(input_root: Path) -> dict:
    sources = {}

    def source(path):
        raw = path.read_bytes()
        sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return raw

    metadata = json.loads(source(input_root / (PREFIX + "analysis.json")))
    manifest = json.loads(source(input_root / (PREFIX + "dataset_manifest.json")))
    for name in ("ima/modeling.py", "scripts/audit_market_blend.py", "tests/test_modeling.py",
                 "ima/rich_features.py", "ima/research_executor.py", "ima/research_model_package.py"):
        source(ROOT / name)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_kind": "descriptive_saved_score_windows_only",
        "caveat": "Market-only exponents are optimized retrospectively on each saved score window. "
                  "These are numerical/boundary diagnostics, not calibration fits, out-of-sample "
                  "performance, current-season backtests, or proof of the original optimizer's behavior. "
                  "No baseline models, campaigns, or feature pipelines are trained or refitted.",
        "candidate_policy": "Four named historical shortlist artifacts are fixed inputs; no new selection.",
        "positional_alignment": "Probability arrays must share runner order. This script checks saved "
                                "keys, labels and fold membership, not unavailable original calibration arrays.",
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "objective": "Equal-race categorical log loss from stable group log-softmax; "
                     "gradient = X.T @ (p - y) / race_count, X = [log(fundamental), log(market)].",
        "bounds": [0, 4],
        "projected_gradient_acceptance_tolerance": 1e-5,
        "candidates": {},
    }
    reference = None
    for name in CANDIDATES:
        directory = input_root / (PREFIX + name)
        prediction_path = directory / "predictions.csv"
        source(prediction_path)
        detail = metadata["chosen"][name]
        if sources[str(prediction_path.resolve())] != detail["lineage"]["prediction_sha256"]:
            raise ValueError(f"Prediction lineage hash mismatch: {name}")
        protocol = json.loads(source(directory / "protocol.json"))
        source(directory / "package_recipe.json")
        replay = json.loads(source(input_root / (PREFIX + name + "_feature_replay.json")))
        frame = pd.read_csv(prediction_path, dtype={"race_id": str, "horse_no": str, "fold_id": str})
        keys = ["fold_id", "race_id", "horse_no"]
        population = frame[keys + ["date", "target_win"]].sort_values(keys).reset_index(drop=True)
        if frame.duplicated(keys).any() or not frame.groupby(keys[:2]).target_win.sum().eq(1).all():
            raise ValueError(f"Invalid saved population: {name}")
        if reference is None:
            reference = population
        else:
            pd.testing.assert_frame_equal(reference, population)
        for column in ("model_probability", "market_probability", "selected_probability",
                       "calibrated_market_probability"):
            if (not frame[column].between(0, 1).all()
                    or not np.allclose(frame.groupby(keys[:2])[column].sum(), 1, atol=1e-10, rtol=0)):
                raise ValueError(f"Invalid saved probabilities: {name}/{column}")
        candidate = {"campaign": detail["campaign"], "attempt_id": detail["attempt_id"],
                     "feature_replay_implementation_hashes": replay["implementation_hashes"], "folds": []}
        for fold in protocol["folds"]:
            groups = [set(fold[k]) for k in ("train_race_ids", "calibration_race_ids", "score_race_ids")]
            if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError(f"Overlapping fold populations: {name}/{fold['fold_id']}")
            sub = frame[frame.fold_id == fold["fold_id"]].copy()
            if set(sub.race_id) != groups[2]:
                raise ValueError("Saved score membership differs from protocol")
            sub["target_probability"] = sub.target_win
            features, codes, count = _probability_log_features(
                [sub.model_probability, sub.market_probability], sub.race_id)
            target = _calibration_target(sub, codes, count)

            def objective(weights):
                return _blend_loss_gradient(np.asarray(weights), features, target, codes, count)

            boundary = minimize_scalar(lambda b: objective([0, b])[0], bounds=(0, 4),
                                       method="bounded", options={"xatol": 1e-12})
            if not boundary.success or not np.isfinite([boundary.x, boundary.fun]).all():
                raise RuntimeError("Descriptive market-only optimization failed")
            loss, gradient = objective([0, boundary.x])
            calibration_rows = frame[frame.race_id.isin(groups[1])]
            row = {"fold_id": fold["fold_id"], "score_races": count, "score_runners": len(sub),
                   "score_date_min": sub.date.min(), "score_date_max": sub.date.max(),
                   "calibration_races_required": len(groups[1]),
                   "calibration_races_present_in_prior_score_exports": calibration_rows.race_id.nunique(),
                   "calibration_prediction_caveat": "Prior-fold fundamental predictions are from different "
                       "frozen models and do not reproduce this fold's original calibration predictions.",
                   "descriptive_score_boundary": {"market_exponent": float(boundary.x),
                       "score_nll": loss, "gradient": gradient.tolist(),
                       "score_nll_delta_at_a_0_001": objective([.001, boundary.x])[0] - loss},
                   "numerical_points": []}
            logged = next(f for f in detail["folds"] if f["fold_id"] == fold["fold_id"])
            if logged.get("blend_fundamental_weight") is not None:
                weights = [logged["blend_fundamental_weight"], logged["blend_market_weight"]]
                repaired = blend_probabilities(sub.model_probability.to_numpy(),
                                              sub.market_probability.to_numpy(), sub.race_id, *weights)
                row["saved_coefficient_numerical_effect"] = {
                    "weights": weights,
                    "max_probability_change_from_saved_selected": float(np.max(
                        np.abs(repaired - sub.selected_probability.to_numpy()))),
                    "original_fit_status": "Not recorded; original calibration predictions unavailable."}
            for weights in ([1., 1.], [4., 4.]):
                stable_loss, stable_gradient = objective(weights)
                probability = blend_probabilities(sub.model_probability.to_numpy(),
                                                  sub.market_probability.to_numpy(), sub.race_id, *weights)
                legacy_product = np.exp(features @ weights)
                legacy_probability = normalize_by_race(legacy_product, sub.race_id)
                step = np.eye(2) * 1e-5
                numerical = np.array([(objective(np.array(weights) + d)[0]
                                       - objective(np.array(weights) - d)[0]) / 2e-5 for d in step])
                error = float(np.max(np.abs(numerical - stable_gradient)))
                if error > 1e-8:
                    raise ValueError("Analytical gradient failed finite-difference check")
                row["numerical_points"].append({"weights": weights, "stable_score_nll": stable_loss,
                    "legacy_clipped_score_nll": race_log_loss(legacy_probability, sub),
                    "legacy_product_floor_rows": int((legacy_product < 1e-12).sum()),
                    "max_probability_change": float(np.max(np.abs(probability - legacy_probability))),
                    "analytical_gradient": stable_gradient.tolist(), "finite_difference_error": error})
            candidate["folds"].append(row)
        result["candidates"][name] = candidate

    features = np.log(np.array([[.999, .001], [.001, .999]]))
    objective = lambda w: _blend_loss_gradient(w, features, np.array([1., 0.]), np.array([0, 0]), 1)
    controls = []
    for start in ([1., 1.], [4., 4.]):
        fit = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0, 4)] * 2,
                       options={"gtol": 1e-8, "ftol": 1e-12})
        if not fit.success or fit.fun >= 1e-8:
            raise RuntimeError("Positive control did not recover")
        controls.append({"start": start, "weights": fit.x.tolist(), "synthetic_nll": float(fit.fun),
                         "success": bool(fit.success), "gradient": objective(fit.x)[1].tolist()})
    result["synthetic_positive_controls"] = controls
    protected = manifest["identity"]["protected_confirmation_races"]
    result["confirmation_availability"] = {
        "manifest_eligible_date_max": manifest["date_max"],
        "protected_race_ids": len(protected),
        "protected_ids_on_or_after_2026_09_01": sum(r.split(":")[1] >= "2026-09-01" for r in protected),
        "labels_published": manifest["validation"]["confirmation_labels_published"],
        "quarantined_rows": manifest["validation"]["confirmation_rows_quarantined"],
        "caveat": "IDs and manifest metadata only; protected outcomes and row completeness were not inspected."}
    result["feature_replay_hash_concern"] = (
        "Existing replay checks enforce "
        "only recorded hashes; they do not establish a hash guard for rich_features.py or modeling.py. "
        "Keep candidate recipes and history frozen and record both implementation hashes with replay evidence. "
        "Preparation's module hash list also omits these files; unchanged code_revision can retain a cache "
        "identity across a hotpatch. No replay/cache code is changed by this numerical repair.")
    for path, digest in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Audit source changed during execution: {path}")
    result["sources_sha256"] = sources
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / ".tmp")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/race-readiness-20261007/blend-diagnostic.json")
    args = parser.parse_args()
    result = audit(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote descriptive numerical audit: {args.output}")


if __name__ == "__main__":
    main()
