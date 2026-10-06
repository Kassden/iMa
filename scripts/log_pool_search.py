"""Publish a compact, hash-bound 588-grid paper study; never fit or load models.

prepare_bundle(root, report, bundle) validates source/output provenance and
copies portable evidence. log_bundle(bundle, client) validates it again before
creating seven descriptive runs. Use a single writer: MLflow search/create is
not an atomic uniqueness constraint. This file also runs outside the checkout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path


EXPERIMENT = "ima-pool-search-2026"
VARIANTS = (
    "baseline_pool", "candidate_pool", "baseline_pool_market_blend_hindsight",
    "candidate_pool_market_blend_hindsight", "market_final_odds_hindsight",
    "market_calibrated_hindsight",
)
LIMIT = 10 * 1024 * 1024
SPLITS = ("search", "confirmation", "calibration")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def flatten_numeric_metrics(payload, prefix=""):
    """Finite numeric leaves only; booleans are decisions, not score metrics."""
    metrics = {}
    for key, value in payload.items():
        name = re.sub(r"[^A-Za-z0-9_. /-]+", "_", f"{prefix}.{key}" if prefix else str(key))
        if isinstance(value, dict):
            nested = flatten_numeric_metrics(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            nested = {name: float(value)}
        else:
            continue
        if metrics.keys() & nested.keys():
            raise ValueError("Flattened metric key collision")
        metrics.update(nested)
    return metrics


def _resolve(path, repo):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def _hash_identity(plan):
    body = {k: v for k, v in plan.items() if k != "identity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pick(objects, paths, label):
    for obj in objects:
        for path in paths:
            value = obj
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                return value
    raise ValueError(f"Missing {label}; expected one of {paths}")


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric {label}")
    return float(value)


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _split_counts(path):
    rows = _rows(path)
    groups = {name: [] for name in SPLITS}
    for row in rows:
        name = row.get("role", row.get("split", row.get("partition", row.get("cohort"))))
        if name not in groups:
            raise ValueError("splits.csv requires search/confirmation/calibration split labels")
        groups[name].append(row)
    counts = {}
    seen = set()
    for name, group in groups.items():
        if not group:
            raise ValueError(f"Empty preseason split: {name}")
        if "race_id" in group[0]:
            ids = {row["race_id"] for row in group}
            if "" in ids or ids & seen:
                raise ValueError("Missing race identity or overlapping preseason splits")
            seen.update(ids)
            counts[name] = len(ids)
        else:
            if len(group) != 1:
                raise ValueError("Expected one count row per split")
            count = _pick([group[0]], ("races", "race_count", "n_races"), "split race count")
            counts[name] = int(count)
            if str(count) not in (str(counts[name]), str(float(counts[name]))):
                raise ValueError("Split counts must be integers")
        if counts[name] < 1:
            raise ValueError("Split counts must be positive")
    return counts


def _payloads(search, selection, summary, grid, split_counts):
    if set(summary) != set(VARIANTS):
        raise ValueError("Expected exactly six evaluation summary variants")
    if len(grid) != 588:
        raise ValueError("Expected exactly 588 grid configurations")
    declared = search.get("grid_count", search.get("grid_size", 588))
    if declared != 588 or isinstance(declared, bool):
        raise ValueError("Search grid_count must be 588")
    losses = [_number(float(row["search_race_log_loss"]), "search loss") for row in grid]
    objects = [selection, search]
    candidate = _pick(objects, ("candidate_recipe", "selected_recipe", "best_recipe", "best.recipe",
                               "candidate.recipe", "preseason_choice.recipe"), "candidate recipe")
    baseline = _pick(objects, ("baseline_recipe", "baseline.recipe"), "baseline recipe")
    if not isinstance(candidate, dict) or not candidate or not isinstance(baseline, dict) or not baseline:
        raise ValueError("Pool recipes must be nonempty objects")
    baseline_loss = _number(_pick(objects, ("baseline_confirmation_loss", "confirmation_baseline_loss",
        "confirmation_losses.baseline_pool", "confirmation.baseline_loss"), "baseline confirmation loss"), "baseline loss")
    candidate_loss = _number(_pick(objects, ("candidate_confirmation_loss", "confirmation_candidate_loss",
        "confirmation_losses.candidate_pool", "confirmation.candidate_loss"), "candidate confirmation loss"), "candidate loss")
    adopted = _pick(objects, ("adopted", "adopt_candidate", "preseason_choice.adopted"), "adoption decision")
    if not isinstance(adopted, bool):
        raise ValueError("adopted must be a boolean preseason decision")
    chosen = "candidate_pool" if adopted else "baseline_pool"
    if search.get("adopted_model", chosen) != chosen:
        raise ValueError("adopted_model disagrees with adoption decision")
    for key, expected in (("search_best_loss", min(losses)),
                          ("confirmation_delta", candidate_loss - baseline_loss)):
        if key in search and not math.isclose(_number(search[key], key), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Search {key} disagrees with underlying scores")
    if "splits" in search:
        if any(search["splits"].get(name, {}).get("races") != count for name, count in split_counts.items()):
            raise ValueError("Search split counts disagree with splits.csv")
    parent = {
        "params": {"grid_count": 588, "candidate_recipe": json.dumps(candidate, sort_keys=True),
                   "baseline_recipe": json.dumps(baseline, sort_keys=True),
                   "preseason_choice": chosen, **{f"{k}_races": v for k, v in split_counts.items()}},
        "metrics": {"search_best_loss": min(losses), "baseline_confirmation_loss": baseline_loss,
                    "candidate_confirmation_loss": candidate_loss,
                    "confirmation_delta": candidate_loss - baseline_loss, "adopted": int(adopted)},
    }
    for key in ("policy", "downstream_calibration_policy", "no_base_model_training"):
        if key in search:
            parent["params"][key] = search[key]
    children = {}
    for name in VARIANTS:
        if not isinstance(summary[name], dict):
            raise ValueError("Expected object summary metrics")
        family = name.removesuffix("_market_blend_hindsight")
        recipe = baseline if family == "baseline_pool" else candidate if family == "candidate_pool" else None
        application = "temperature_only" if name == "market_calibrated_hindsight" else (
            "market_blend_only" if name.endswith("_market_blend_hindsight") else "none")
        params = {"family": family if recipe else "market", "calibration_application": application,
                  "probability_basis": "final_win_odds_hindsight" if not recipe else "frozen_component_pool"}
        if recipe:
            params["recipe"] = json.dumps(recipe, sort_keys=True)
            params.update({f"recipe.{key}": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                           for key, value in recipe.items()})
        children[name] = {"params": params, "metrics": flatten_numeric_metrics(summary[name]),
                          "selected": name == chosen}
    return parent, children


def prepare_bundle(root, report, bundle, *, repo_root=None):
    """No MLflow writes. Provenance paths are absolute or repository-relative.

    search/selection must provide baseline_recipe, candidate_recipe (or
    selected_recipe), boolean adopted, and baseline/candidate_confirmation_loss
    (also accepts confirmation_losses.{baseline_pool,candidate_pool}). splits.csv
    has role/race_id rows (also split aliases) or split/n_races count rows. search-grid.csv contains
    588 search_race_log_loss rows. Every evidence file except report/provenance
    must be covered by the ready provenance maps.
    """
    repo = Path(repo_root or Path.cwd()).resolve()
    root, report, bundle = (_resolve(p, repo) for p in (root, report, bundle))
    proof_path = root / "result-provenance.json"
    proof = read_json(proof_path)
    if proof.get("ready_for_report") is not True:
        raise ValueError("Result provenance is not ready_for_report")
    validated = {}
    validated_maps = {}
    for key in ("source_sha256", "output_sha256"):
        validated_maps[key] = {}
        hashes = proof.get(key)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"Missing provenance map: {key}")
        for name, digest in hashes.items():
            path = _resolve(name, repo)
            if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest) or sha256(path) != digest.lower():
                raise ValueError(f"Provenance SHA256 mismatch: {path}")
            if path in validated and validated[path] != digest.lower():
                raise ValueError("Conflicting provenance SHA256 maps")
            validated[path] = digest.lower()
            validated_maps[key][path] = digest.lower()
    search, selection = read_json(root / "search.json"), read_json(root / "selection.json")
    summary = read_json(root / "evaluation/summary.json")
    parent, children = _payloads(search, selection, summary, _rows(root / "search-grid.csv"), _split_counts(root / "splits.csv"))
    try:
        name = _pick([search], ("component_receipt", "component_cache_readback", "component_cache_readback_path",
                "component_readback_path", "component_cache.readback_path", "component_cache.readback",
                "readback_path"), "component receipt")
        readback = _resolve(name, repo)
    except ValueError:
        readback = root / "component-cache/readback.json"
        if not readback.is_file():
            matches = [path for path in validated if path.name == "readback.json"]
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous component-cache readback receipt")
            readback = matches[0]
    sources = {name: root / name for name in ("search.json", "search-grid.csv", "splits.csv", "selection.json")}
    sources["component-cache/readback.json"] = readback
    if (root / "protocol.json").is_file():
        sources["protocol.json"] = root / "protocol.json"
    prediction_receipt = root / "predictions/readback.json"
    if prediction_receipt.is_file():
        if prediction_receipt.resolve() not in validated_maps["output_sha256"]:
            raise ValueError("Prediction receipt missing from output provenance")
        sources["predictions/readback.json"] = prediction_receipt
    for receipt, directory, provenance_key in (
        (readback, "component-cache", "source_sha256"),
        (prediction_receipt, "predictions", "output_sha256"),
    ):
        if not receipt.is_file():
            continue
        for model, item in read_json(receipt).get("models", {}).items():
            if not isinstance(model, str) or Path(model).name != model or model in {".", ".."}:
                raise ValueError("Unsafe prediction model filename")
            path = receipt.parent / f"{model}.csv"
            if not path.is_file():
                continue
            declared = item.get("output_sha256") if isinstance(item, dict) else None
            digest = sha256(path)
            if not isinstance(declared, str) or declared.lower() != digest:
                raise ValueError(f"Prediction declared output SHA256 mismatch: {path}")
            if validated_maps[provenance_key].get(path.resolve()) != digest:
                raise ValueError(f"Prediction CSV missing from {provenance_key} provenance: {path}")
            sources[f"{directory}/{model}.csv"] = path
    for path in sorted((root / "evaluation").rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"} and "raw" not in path.relative_to(root).parts:
            sources[path.relative_to(root).as_posix()] = path
    for name, path in sources.items():
        if path.resolve() not in validated or sha256(path) != validated[path.resolve()]:
            raise ValueError(f"Copied evidence missing from provenance: {name}")
    sources.update({"result-provenance.json": proof_path, f"report/{report.name}": report,
                    "log_pool_search.py": Path(__file__).resolve()})
    inventory = {}
    for name, path in sources.items():
        if not path.is_file() or path.stat().st_size >= LIMIT:
            raise ValueError(f"Expected compact artifact below 10MB: {path}")
        inventory[name] = sha256(path)
    plan = {"schema_version": 1, "experiment": EXPERIMENT, "artifacts": inventory,
            "parent": parent, "children": children}
    plan["identity"] = _hash_identity(plan)
    if bundle.exists() and any(bundle.iterdir()):
        existing = verify_bundle(bundle)
        if existing != plan:
            raise ValueError("Bundle already contains a different identity")
        return plan
    bundle.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        destination = bundle / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha256(destination) != inventory[name]:
            raise ValueError(f"Artifact changed during prepare: {path}")
    save_json(bundle / "plan.json", plan)
    verify_bundle(bundle)
    return plan


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    plan = read_json(bundle / "plan.json")
    if plan.get("schema_version") != 1 or plan.get("experiment") != EXPERIMENT or plan.get("identity") != _hash_identity(plan):
        raise ValueError("Bundle plan identity mismatch")
    if set(plan.get("children", {})) != set(VARIANTS):
        raise ValueError("Bundle must contain six variant payloads")
    for name, digest in plan["artifacts"].items():
        path = bundle / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(bundle):
            raise ValueError("Artifact escapes bundle")
        if path.is_symlink() or not path.is_file() or path.stat().st_size >= LIMIT or sha256(path) != digest:
            raise ValueError(f"Bundle artifact SHA256 mismatch: {name}")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    if actual - set(plan["artifacts"]) - {"plan.json", "mlflow-receipt.json"}:
        raise ValueError("Unbound files in bundle")
    return plan


def _tags(plan, role, selected, parent=None):
    tags = {"ima.pool_search_identity": plan["identity"], "ima.paper_only": "true",
            "ima.non_agentic": "true", "ima.study": "pool_search_2026",
            "ima.variant": role, "ima.selected": str(selected).lower(),
            "ima.profit_claim": "retrospective_paper_not_executable", "mlflow.runName": role}
    if parent:
        tags["mlflow.parentRunId"] = parent
    return tags


def _readback(client, run_id, tags, payload, *, finished):
    run = client.get_run(run_id)
    if finished and run.info.status != "FINISHED":
        raise ValueError("Tracking readback is not FINISHED")
    if getattr(run.info, "lifecycle_stage", "active") != "active":
        raise ValueError("Tracking identity is deleted")
    if any(run.data.tags.get(k) != v for k, v in tags.items()):
        raise ValueError("Tracking tag readback mismatch")
    if any(run.data.params.get(k) != str(v) for k, v in payload["params"].items()):
        raise ValueError("Tracking parameter readback mismatch")
    if any(k not in run.data.metrics or not math.isclose(run.data.metrics[k], v, rel_tol=1e-12, abs_tol=1e-12)
           for k, v in payload["metrics"].items()):
        raise ValueError("Tracking metric readback mismatch")


def log_bundle(bundle, client):
    """Validate first; reuse only a complete seven-run identity without writes."""
    bundle = Path(bundle).resolve()
    plan = verify_bundle(bundle)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    eid = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    found = client.search_runs([eid], filter_string=f"tags.`ima.pool_search_identity` = '{plan['identity']}'",
                               max_results=8, run_view_type=3)
    created = []
    parent = None
    children = {}
    reused = bool(found)
    if found:
        roles = {}
        for run in found:
            role = run.data.tags.get("ima.variant")
            if role in roles:
                raise ValueError("Pool-search identity collision: duplicate role")
            roles[role] = run.info.run_id
        if len(found) != 7 or set(roles) != {"parent", *VARIANTS}:
            raise ValueError("Pool-search identity collision: expected seven complete runs")
        parent = roles["parent"]
        if "mlflow.parentRunId" in client.get_run(parent).data.tags:
            raise ValueError("Pool-search identity collision: parent is nested")
        _readback(client, parent, _tags(plan, "parent", True), plan["parent"], finished=True)
        for name in VARIANTS:
            payload = plan["children"][name]
            _readback(client, roles[name], _tags(plan, name, payload["selected"], parent), payload, finished=True)
            children[name] = roles[name]
    else:
        try:
            def create(role, payload, parent_id=None):
                tags = _tags(plan, role, True if parent_id is None else payload["selected"], parent_id)
                rid = client.create_run(eid, tags=tags).info.run_id
                created.append(rid)
                for key, value in payload["params"].items():
                    client.log_param(rid, key, str(value))
                for key, value in payload["metrics"].items():
                    client.log_metric(rid, key, value, timestamp=int(time.time() * 1000))
                _readback(client, rid, tags, payload, finished=False)
                return rid

            parent = create("parent", plan["parent"])
            for name in (*plan["artifacts"], "plan.json"):
                path = bundle / name
                client.log_artifact(parent, str(path), artifact_path=str(Path(name).parent) if Path(name).parent != Path(".") else None)
            for name in VARIANTS:
                payload = plan["children"][name]
                rid = create(name, payload, parent)
                children[name] = rid
                for artifact in plan["artifacts"]:
                    if artifact.startswith(f"evaluation/{name}-"):
                        client.log_artifact(rid, str(bundle / artifact), artifact_path="ledgers")
            if verify_bundle(bundle) != plan:
                raise ValueError("Bundle changed during publication")
            for rid in created:
                client.set_terminated(rid, status="FINISHED")
            _readback(client, parent, _tags(plan, "parent", True), plan["parent"], finished=True)
            for name, rid in children.items():
                payload = plan["children"][name]
                _readback(client, rid, _tags(plan, name, payload["selected"], parent), payload, finished=True)
            receipt = {"experiment": EXPERIMENT, "identity": plan["identity"], "parent_run_id": parent,
                       "children": children, "run_count": 7, "reused": False, "readback_verified": True}
            save_json(bundle / "mlflow-receipt.json", receipt)
            client.log_artifact(parent, str(bundle / "mlflow-receipt.json"))
            return receipt
        except Exception as error:
            for rid in created:
                try:
                    client.set_terminated(rid, status="FAILED")
                except Exception as cleanup:
                    error.add_note(f"Unable to terminate newly created run {rid}: {cleanup}")
            try:
                save_json(bundle / "mlflow-receipt.json", {
                    "experiment": EXPERIMENT, "identity": plan["identity"],
                    "parent_run_id": parent, "children": children,
                    "run_count": len(created), "status": "FAILED", "readback_verified": False,
                })
            except Exception as cleanup:
                error.add_note(f"Unable to write failure receipt: {cleanup}")
            raise
    receipt = {"experiment": EXPERIMENT, "identity": plan["identity"], "parent_run_id": parent,
               "children": children, "run_count": 7, "reused": reused, "readback_verified": True}
    save_json(bundle / "mlflow-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "log"), required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/pool-search-20261007"))
    parser.add_argument("--report", type=Path, default=Path("docs/PRESEASON_POOL_SEARCH_RESULTS.md"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args()
    if args.stage == "prepare":
        plan = prepare_bundle(args.root, args.report, args.bundle)
        receipt = {"prepared": True, "identity": plan["identity"], "bundle": str(args.bundle),
                   "variants": list(plan["children"]), "artifact_count": len(plan["artifacts"])}
    else:
        if not args.tracking_uri:
            parser.error("--tracking-uri required for log")
        # Validate before even constructing a tracking client.
        verify_bundle(args.bundle)
        from mlflow.tracking import MlflowClient
        receipt = log_bundle(args.bundle, MlflowClient(tracking_uri=args.tracking_uri))
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
