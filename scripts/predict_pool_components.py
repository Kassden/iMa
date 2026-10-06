"""Extract two frozen pool components; never fit or change the original package."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "race_id", "race_no", "horse_id", "horse_no", "horse_name", "field_size"]
DEPENDENCIES = {"numpy", "pandas", "scipy", "scikit-learn", "joblib"}
PACKAGE_FILES = {"manifest.json", "recipe.json", "feature-replay.json", "refit.json", "model.joblib"}
CORE_SOURCE = {"__init__.py", "research_model_package.py", "research_executor.py",
               "pipeline_graph.py", "research_transforms.py", "modeling.py"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def before_query(value, earliest, label):
    cutoff = pd.to_datetime(value, utc=True, errors="raise")
    if value is None or pd.isna(cutoff) or cutoff >= earliest:
        raise ValueError(f"{label} must precede every query")


def check_cutoffs(payload, earliest, label):
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"fit_date_max", "training_cutoff"}:
                before_query(value, earliest, f"{label}/{key}")
            else:
                check_cutoffs(value, earliest, f"{label}/{key}")
    elif isinstance(payload, list):
        for value in payload:
            check_cutoffs(value, earliest, label)


def identity(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(KEYS).issubset(frame) or frame.empty or frame[KEYS].isna().any().any():
        raise ValueError("Missing or empty runner identity")
    result = frame[KEYS].copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(result.date, utc=True, errors="raise")
    if result.date.isna().any():
        raise ValueError("Missing query dates")
    for name in ("race_no", "horse_no", "field_size"):
        values = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError("Invalid numeric runner identity")
        result[name] = values.astype("int64")
    for name in ("race_id", "horse_id", "horse_name"):
        result[name] = result[name].astype(str)
        if result[name].str.strip().eq("").any():
            raise ValueError("Blank runner identity")
    if result.duplicated(["race_id", "horse_no"]).any() or result.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("Duplicate runner identity")
    grouped = result.groupby("race_id", sort=False)
    if not grouped.race_id.transform("size").eq(result.field_size).all():
        raise ValueError("Incomplete query race")
    if (grouped[["date", "race_no", "field_size"]].nunique() != 1).any().any():
        raise ValueError("Inconsistent race identity")
    return result


def probabilities(values, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.Series) and not values.index.equals(frame.index):
        raise ValueError("Component prediction index is misaligned")
    values = np.asarray(values, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid component probabilities")
    totals = pd.Series(values).groupby(frame.race_id.to_numpy()).sum()
    if not np.allclose(totals.to_numpy(), 1, rtol=0, atol=1e-10):
        raise ValueError("Component probabilities are not normalized")
    return values


def check_recipe(recipe: dict):
    graph = recipe.get("pipeline_graph") or {}
    nodes = graph.get("nodes", [])
    if (recipe.get("target", {}).get("kind") != "win_probability"
            or recipe.get("calibration", {}).get("kind") != "none"
            or recipe.get("blend", {}).get("kind") != "none"
            or graph.get("fundamental_node_id") != "pool" or graph.get("output_node_id") != "pool"
            or len(nodes) != 3 or len({n["node_id"] for n in nodes}) != 3):
        raise ValueError("Expected original three-node fundamental WIN pool")
    byid = {n["node_id"]: n for n in nodes}
    if set(byid) != {"benter", "boosted", "pool"}:
        raise ValueError("Unexpected original pool nodes")
    pool = byid["pool"]
    if (pool["kind"] != "weighted_probability_pool" or pool.get("inputs") != ["benter", "boosted"]
            or pool.get("parameters", {}).get("weights") != [0.6, 0.4]):
        raise ValueError("Expected fixed 0.6/0.4 Benter/boosted pool")
    for name, kind in (("benter", "benter_conditional_logit"), ("boosted", "boosted")):
        node = byid[name]
        if node["kind"] != "estimator" or node.get("inputs") or node.get("parameters", {}).get("model_kind") != kind:
            raise ValueError("Unexpected pool parent estimator")


def _load_runtime(directory: Path, source_root: Path):
    # Refuse already-imported implementation from another checkout, rather than reload it.
    for name, module in tuple(sys.modules.items()):
        if name == "ima" or name.startswith("ima."):
            path = getattr(module, "__file__", None)
            if path and not Path(path).resolve().is_relative_to(source_root / "ima"):
                raise ValueError("Imported iMa implementation is outside pinned source-root")
    sys.path.insert(0, str(source_root))
    package_module = importlib.import_module("ima.research_model_package")
    modeling = importlib.import_module("ima.modeling")
    if Path(package_module.__file__).resolve() != source_root / "ima/research_model_package.py":
        raise ValueError("Package loader is not the pinned implementation")
    if Path(modeling.__file__).resolve() != source_root / "ima/modeling.py":
        raise ValueError("Pool normalizer is not the pinned implementation")
    return package_module.load_research_package(directory), modeling.normalize_by_race


def run_inference(*, manifest: Path, queries: Path, output: Path, trusted_root: Path,
                  source_root: Path, pool_predictions: Path, inference_readback: Path) -> dict:
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Output must be new or empty; never overwrite evidence")
    source_root, trusted_root = source_root.resolve(), trusted_root.resolve()
    spec = read_json(manifest)
    environment = spec.get("environment", {})
    if not DEPENDENCIES.issubset(environment) or any(version(n) != v for n, v in environment.items()):
        raise ValueError("Manifest environment does not match original inference environment")
    implementations = spec.get("implementation_hashes", {})
    if not CORE_SOURCE.issubset(implementations):
        raise ValueError("Missing pinned implementation hashes")
    if set(implementations) != {path.name for path in (source_root / "ima").glob("*.py")}:
        raise ValueError("Pinned implementation inventory differs from source-root")
    bindings = {}

    def bind(path, expected=None):
        path = path.resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Source/package/input hash mismatch: {path}")
        bindings[str(path)] = digest

    for name, digest in implementations.items():
        path = (source_root / "ima" / name).resolve()
        if not path.is_relative_to(source_root / "ima"):
            raise ValueError("Implementation path outside pinned source-root")
        bind(path, digest)
    for path in (manifest, queries, inference_readback):
        bind(path)
    frame = pd.read_parquet(queries)
    keys = identity(frame)
    earliest = keys.date.min()
    for name, candidate in spec["candidates"].items():
        for field in ("fit_date_max", "training_cutoff"):
            if field not in candidate:
                raise ValueError(f"Missing {name}/{field}")
            before_query(candidate[field], earliest, f"{name}/{field}")
    candidate = spec["candidates"]["pool"]
    check_recipe(candidate["recipe"])
    directory = Path(candidate["remote_package"]).resolve()
    if not directory.is_relative_to(trusted_root) or directory == trusted_root:
        raise ValueError("Package outside explicitly trusted root")
    hashes = candidate.get("hashes", {})
    if not PACKAGE_FILES.issubset(hashes):
        raise ValueError("Incomplete frozen package hashes")
    for name, digest in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Package file outside trusted package")
        bind(path, digest)
    disk_manifest = read_json(directory / "manifest.json")
    if disk_manifest["package_id"] != candidate["package_id"] or disk_manifest["target_kind"] != "win_probability":
        raise ValueError("Frozen pool package identity mismatch")
    if read_json(directory / "recipe.json") != candidate["recipe"]:
        raise ValueError("Frozen pool recipe mismatch")
    replay = read_json(directory / "feature-replay.json")
    check_cutoffs(read_json(directory / "refit.json"), earliest, "Refit metadata")
    before_query(replay["training_cutoff"], earliest, "Feature replay cutoff")
    if replay.get("implementation_hashes") != implementations or any(
            environment.get(n) != v for n, v in replay.get("dependency_versions", {}).items()):
        raise ValueError("Feature replay source/environment differs from manifest")
    receipt = read_json(inference_readback)
    original = receipt["models"]["pool"]
    if receipt["input_sha256"] != bindings[str(queries.resolve())] or original["package_id"] != candidate["package_id"]:
        raise ValueError("Original pool readback input/package mismatch")
    before_query(original["training_cutoff"], earliest, "Original pool cutoff")
    if original["rows"] != len(frame) or original["races"] != keys.race_id.nunique():
        raise ValueError("Original pool readback population mismatch")
    bind(pool_predictions, original["output_sha256"])
    cached = pd.read_csv(pool_predictions, dtype={"race_id": str, "horse_id": str, "horse_name": str})
    if not identity(cached).equals(keys) or not cached["model"].eq("pool").all():
        raise ValueError("Cached original pool rows are not aligned")
    cached_values = probabilities(cached.model_probability.to_numpy(), frame)
    if any(sha256(Path(p)) != digest for p, digest in bindings.items()):
        raise ValueError("Inputs changed before guarded package load")
    package, normalize = _load_runtime(directory, source_root)
    if package.manifest().package_id != candidate["package_id"] or package.recipe.canonical_payload() != candidate["recipe"]:
        raise ValueError("Loaded package differs from trusted recipe/identity")
    graph = package.model.model
    fundamental = graph.fundamental
    if (fundamental.spec.node_id != "pool" or fundamental.spec.kind != "weighted_probability_pool"
            or fundamental.spec.output.kind != "win_probability"
            or list(fundamental.spec.parameters.get("weights", [])) != [0.6, 0.4]
            or tuple(p.spec.node_id for p in fundamental.parents) != ("benter", "boosted")):
        raise ValueError("Loaded graph is not the original two-parent WIN pool")
    for parent, kind in zip(fundamental.parents, ("benter_conditional_logit", "boosted")):
        if (parent.spec.kind != "estimator" or parent.spec.output.kind != "win_probability"
                or parent.spec.parameters.get("model_kind") != kind):
            raise ValueError("Loaded graph parent identity mismatch")
    before_query(package.feature_context.training_cutoff, earliest, "Loaded feature cutoff")
    before_query(graph.fit_report["training_cutoff"], earliest, "Loaded graph cutoff")
    check_cutoffs(graph.fit_report, earliest, "Loaded graph fits")
    features = package.model._frame(package._feature_frame(frame))
    if not identity(features).equals(keys):
        raise ValueError("Feature replay changed runner order/identity")
    graph._check_cutoff(features)
    values = [probabilities(parent.predict(features), features) for parent in fundamental.parents]
    reconstructed = probabilities(normalize(0.6 * values[0] + 0.4 * values[1], features.race_id), features)
    error = float(np.max(np.abs(reconstructed - cached_values)))
    if error > 1e-12:
        raise ValueError(f"Original pool reconstruction mismatch: {error}")
    if any(sha256(Path(p)) != digest for p, digest in bindings.items()):
        raise ValueError("Inputs changed during inference")
    result = {"inference_only": True, "environment": environment, "source_root": str(source_root),
              "source_sha256": bindings, "input_sha256": bindings[str(queries.resolve())],
              "implementation_hashes": implementations, "package_hashes": hashes,
              "package_id": candidate["package_id"], "fixed_weights": [0.6, 0.4],
              "training_cutoff": candidate["training_cutoff"], "fit_date_max": candidate["fit_date_max"],
              "original_pool_max_abs_error": error, "rows": len(frame),
              "races": int(keys.race_id.nunique()), "models": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pool-components-", dir=output.parent) as temp:
        work = Path(temp)
        for name, prediction in zip(("pool_benter", "pool_boosted"), values):
            rows = frame[KEYS].copy()
            rows["model"] = name
            rows["model_probability"] = prediction
            path = work / f"{name}.csv"
            rows.to_csv(path, index=False)
            result["models"][name] = {"output_sha256": sha256(path), "rows": len(rows),
                                      "races": result["races"], "package_id": candidate["package_id"]}
        (work / "readback.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        os.rename(work, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "queries", "output", "trusted-root", "source-root",
                 "pool-predictions", "inference-readback"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run_inference(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
