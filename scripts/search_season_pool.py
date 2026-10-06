"""Search cached preseason probability pools, confirm, and settle frozen forecasts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ima.modeling import race_log_loss
from ima.probability_pool_search import PoolRecipe, chronological_split, pool_probabilities, search_pool
from scripts.evaluate_season_2026 import evaluate, quote_lookup, save_json, target_frame
from scripts.forecast_season_market import build_forecast, verify_quote_sources


ROOT = Path("artifacts/race-readiness-20261007")
OUTPUT = Path("artifacts/pool-search-20261007")
KEYS = ["race_id", "horse_no", "horse_id"]
BASELINE = PoolRecipe("original_pair", "arithmetic", 0.6, 1.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_predictions(metadata, path, expected_hash):
    if sha(path) != expected_hash:
        raise ValueError(f"Prediction hash mismatch: {path}")
    frame = pd.read_csv(path)
    if metadata.duplicated(KEYS).any() or frame.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction identity")
    joined = metadata[KEYS].merge(frame[KEYS + ["model_probability"]], on=KEYS,
                                   how="left", validate="one_to_one")
    if len(frame) != len(metadata) or joined.model_probability.isna().any():
        raise ValueError("Prediction population differs from query metadata")
    return joined.model_probability.to_numpy(dtype=float)


def load_inputs(source, components):
    metadata_path, features_path = source / "query-metadata.parquet", source / "query-features.parquet"
    metadata = pd.read_parquet(metadata_path)
    metadata["date"] = pd.to_datetime(metadata.date)
    fresh = json.loads((source / "fresh-predictions/readback.json").read_text())
    component_receipt = json.loads((components / "readback.json").read_text())
    query_hash = sha(features_path)
    if fresh.get("input_sha256") != query_hash or component_receipt.get("input_sha256") != query_hash:
        raise ValueError("Cached predictions belong to a different query dataset")
    expected_fresh = {"benter_conditional_logit", "boosted", "pool", "gaussian_probit"}
    if set(fresh.get("models", {})) != expected_fresh:
        raise ValueError("Expected four frozen source candidates")
    expected_components = {"pool_benter", "pool_boosted"}
    if set(component_receipt.get("models", {})) != expected_components:
        raise ValueError("Expected the two original fitted pool components")
    manifest_path = source / "final-fits/fresh-candidates.json"
    manifest = json.loads(manifest_path.read_text())
    earliest = metadata.loc[metadata.cohort.eq("calibration"), "date"].min()
    if pd.isna(earliest):
        raise ValueError("Missing preseason population")
    for family, model in fresh["models"].items():
        candidate = manifest["candidates"][family]
        if candidate["package_id"] != model["package_id"]:
            raise ValueError("Cached package identity differs from frozen manifest")
        cutoffs = [pd.Timestamp(candidate["fit_date_max"]), pd.Timestamp(model["training_cutoff"])]
        if any(pd.isna(cutoff) or cutoff >= earliest for cutoff in cutoffs):
            raise ValueError("Base model fitting overlaps preseason search")
    pool = manifest["candidates"]["pool"]
    if component_receipt.get("package_id") != pool["package_id"]:
        raise ValueError("Component parent identity differs from frozen pool")
    for key in ("fit_date_max", "training_cutoff"):
        value = component_receipt.get(key)
        cutoff = pd.Timestamp(value)
        if value is None or pd.isna(cutoff) or cutoff >= earliest:
            raise ValueError("Component fitting overlaps preseason search")
        expected = pool.get(key, fresh["models"]["pool"]["training_cutoff"])
        if cutoff != pd.Timestamp(expected):
            raise ValueError("Component cutoff differs from frozen parent")
    if any(model.get("package_id") != pool["package_id"] for model in component_receipt["models"].values()):
        raise ValueError("Component model identity differs from frozen parent")
    paths = [metadata_path, features_path, manifest_path, source / "fresh-predictions/readback.json",
             components / "readback.json", Path(__file__),
             Path(__file__).with_name("evaluate_season_2026.py"),
             Path(__file__).with_name("forecast_season_market.py"),
             Path("ima/probability_pool_search.py"), Path("ima/modeling.py"),
             Path("ima/pools.py"), Path("ima/season_evaluation.py")]
    official = source / "official"
    paths.extend(path for path in (official / "races.json", official / "odds-quotes.json",
                                  official / "odds-unit-semantics.json") if path.exists())
    quote_path = official / "odds-quotes.json"
    if quote_path.exists():
        quote_records = json.loads(quote_path.read_text())
        if isinstance(quote_records, list):
            for filename in sorted({quote["source_file"] for quote in quote_records}):
                if Path(filename).name != filename:
                    raise ValueError("Official source path escapes capture directory")
                paths.append(official / filename)
    predictions = {}
    for family, receipt in fresh["models"].items():
        path = source / "fresh-predictions" / f"{family}.csv"
        predictions[family] = aligned_predictions(metadata, path, receipt["output_sha256"])
        paths.append(path)
    for family, receipt in component_receipt["models"].items():
        path = components / f"{family}.csv"
        predictions[family] = aligned_predictions(metadata, path, receipt["output_sha256"])
        paths.append(path)
    pairs = {"original_pair": np.column_stack([predictions["pool_benter"], predictions["pool_boosted"]]),
             "standalone_pair": np.column_stack([predictions["benter_conditional_logit"], predictions["boosted"]])}
    reconstruction = pool_probabilities(pairs["original_pair"], metadata.race_id, BASELINE)
    if not np.allclose(reconstruction, predictions["pool"], rtol=0, atol=1e-12):
        raise ValueError("Extracted components do not reproduce the original 60/40 pool")
    return metadata, pairs, {str(path.resolve()): sha(path) for path in paths}, fresh


def run_search(source, components, output):
    source, components, output = Path(source), Path(components), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; prior evidence is immutable")
    metadata, pairs, source_hashes, fresh = load_inputs(source, components)
    if len(metadata[metadata.cohort.eq("season")]) != 978 or len(metadata[metadata.cohort.eq("tomorrow")]) != 108:
        raise ValueError("Expected the frozen season and tomorrow runner populations")
    calibration = metadata[metadata.cohort.eq("calibration")].copy()
    rating = pd.to_numeric(calibration.horse_rating, errors="coerce")
    usable = np.isfinite(rating).groupby(calibration.race_id).transform("all")
    calibration = target_frame(calibration[usable].copy())
    if calibration.empty:
        raise ValueError("No whole finite-rating preseason fields")
    roles = chronological_split(calibration)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"weights": np.linspace(0, 1, 21).round(12).tolist(),
        "methods": ["arithmetic", "geometric"], "temperatures": [0.7, 0.85, 1, 1.15, 1.3, 1.5, 2],
        "pairs": list(pairs), "grid_count": 588,
        "split": "first60%,next20%,last20% of unique preseason meeting dates",
        "adoption": "One search-selected candidate must beat baseline confirmation race log loss by more than1e-12",
        "calibration": "market/order fitting exclusively in final disjoint window",
        "season": "already-inspected exploratory evaluation, not untouched confirmation",
        "source_sha256": source_hashes}
    save_json(output / "protocol.json", protocol)
    search_frame = roles["search"]
    table, selected = search_pool(search_frame, {name: matrix[search_frame.index] for name, matrix in pairs.items()})
    if len(table) != 588:
        raise ValueError("Unexpected bounded search grid population")
    table.to_csv(output / "search-grid.csv", index=False)
    confirmation = roles["confirmation"]
    baseline_p = pool_probabilities(pairs[BASELINE.pair][confirmation.index], confirmation.race_id, BASELINE)
    candidate_p = pool_probabilities(pairs[selected.pair][confirmation.index], confirmation.race_id, selected)
    baseline_loss = race_log_loss(baseline_p, confirmation)
    candidate_loss = race_log_loss(candidate_p, confirmation)
    adopted = candidate_loss < baseline_loss - 1e-12
    split_rows = pd.concat([frame.assign(role=role) for role, frame in roles.items()])
    split_rows[["race_id", "horse_no", "horse_id", "date", "role"]].to_csv(output / "splits.csv", index=False)
    split_stats = {role: {"races": int(frame.race_id.nunique()), "meetings": int(frame.date.nunique()),
        "date_min": str(frame.date.min()), "date_max": str(frame.date.max()), "runners": len(frame)}
        for role, frame in roles.items()}
    selection = {"selected_recipe": asdict(selected), "baseline_recipe": asdict(BASELINE),
        "confirmation_candidate_loss": candidate_loss, "confirmation_baseline_loss": baseline_loss,
        "confirmation_delta": candidate_loss - baseline_loss, "adopted": bool(adopted),
        "adopted_model": "candidate_pool" if adopted else "baseline_pool",
        "protocol_sha256": sha(output / "protocol.json"), "splits": split_stats,
        "search_grid_sha256": sha(output / "search-grid.csv"), "splits_sha256": sha(output / "splits.csv"),
        "source_sha256": source_hashes}
    save_json(output / "selection.json", selection)
    search_result = selection | {"grid_count": len(table), "search_best_loss": float(table.search_race_log_loss.iloc[0]),
        "search_baseline_loss": race_log_loss(pool_probabilities(pairs[BASELINE.pair][search_frame.index],
                                                                   search_frame.race_id, BASELINE), search_frame),
        "reserved_preseason_races": int(metadata.loc[metadata.cohort.eq("calibration"), "race_id"].nunique()),
        "usable_preseason_races": int(calibration.race_id.nunique()), "no_base_model_training": True,
        "component_receipt": str((components / "readback.json").resolve()),
        "policy": protocol["season"], "downstream_calibration_policy": protocol["calibration"]}
    save_json(output / "search.json", search_result)
    selection_hash = sha(output / "selection.json")
    derived = metadata.copy()
    derived.loc[derived.cohort.eq("calibration"), "cohort"] = "reserved_unused"
    for role, frame in roles.items():
        derived.loc[frame.index, "cohort"] = role
    derived.to_parquet(output / "query-metadata.parquet", index=False)
    shutil.copy2(source / "query-features.parquet", output / "query-features.parquet")
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    receipt = {"input_sha256": sha(output / "query-features.parquet"),
        "metadata_sha256": sha(output / "query-metadata.parquet"), "models": {},
        "selection_sha256": selection_hash, "source_sha256": source_hashes,
        "policy": "Deterministic composition of trusted cached inference; no model training"}
    candidates = {}
    for name, recipe in (("baseline_pool", BASELINE), ("candidate_pool", selected)):
        probabilities = pool_probabilities(pairs[recipe.pair], metadata.race_id, recipe)
        rows = metadata[["date", *KEYS, "race_no", "horse_name", "field_size"]].copy()
        rows["model_probability"] = probabilities
        rows["model"] = name
        path = predictions_dir / f"{name}.csv"
        rows.to_csv(path, index=False)
        receipt["models"][name] = {"package_id": recipe.configuration_id,
            "training_cutoff": max(model["training_cutoff"] for model in fresh["models"].values()),
            "output_sha256": sha(path), "rows": len(rows), "races": int(rows.race_id.nunique()),
            "recipe": asdict(recipe)}
        candidates[name] = {"recipe": asdict(recipe), "package_id": recipe.configuration_id}
    save_json(predictions_dir / "readback.json", receipt)
    save_json(output / "models/frozen-candidates.json", {"candidates": candidates,
        "selection_sha256": selection_hash, "source_sha256": source_hashes})
    evaluate(SimpleNamespace(output=output, official_dir=source / "official", predictions_dir=predictions_dir))
    if sha(output / "selection.json") != selection_hash:
        raise ValueError("Selection changed during season scoring")
    evaluation = output / "evaluation"
    frames = {name: pd.read_csv(evaluation / f"{name}-tomorrow-runners.csv", parse_dates=["date"])
              for name in candidates}
    quotes, units = quote_lookup(source / "official")
    source_paths = verify_quote_sources(quotes, source / "official")
    source_paths.extend([source / "official/odds-quotes.json", source / "official/odds-unit-semantics.json",
                         source / "official/races.json"])
    for path in source_paths:
        key, digest = str(path.resolve()), sha(path)
        if key in source_hashes and source_hashes[key] != digest:
            raise ValueError("Official source changed during evaluation")
        source_hashes.setdefault(key, digest)
    runners, combinations = build_forecast(frames, json.loads((evaluation / "calibration.json").read_text()),
        json.loads((evaluation / "summary.json").read_text()), quotes, units, families=tuple(candidates))
    if len(runners) != 216 or len(combinations) != 432:
        raise ValueError("Incomplete comparative tomorrow forecasts")
    runners.to_csv(evaluation / "tomorrow-market-blends.csv", index=False)
    combinations.to_csv(evaluation / "tomorrow-market-blend-combinations.csv", index=False)
    if any(sha(path) != digest for path, digest in source_hashes.items()):
        raise ValueError("Frozen source changed while computing results")
    outputs = [path for path in output.rglob("*") if path.is_file()]
    proof = {"ready_for_report": True, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes, "output_sha256": {str(path.resolve()): sha(path) for path in outputs},
        "selection_sha256": selection_hash, "grid_count": 588, "season_races": 78,
        "tomorrow_runners_per_variant": 108, "existing_campaigns_modified": False,
        "model_training": False, "wagers": False}
    save_json(output / "result-provenance.json", proof)
    return search_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_search(args.source, args.components, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
