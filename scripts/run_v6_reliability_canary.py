"""Offline, bounded V6 scientific canary; its small protocol is not a production ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from concurrent.futures import TimeoutError
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from ima.dataset_registry import file_sha256
from ima.research_executor import (
    RecipeExecutionRequest, execute_recipe, preparation_dependency_id, prepare_recipe_folds,
)
from ima.research_preparation import PreparationArtifact, PreparationCache
from ima.research_resources import JobMonitor
from ima.research_specs import PipelineRecipe, V6_PORTFOLIO_VERSION
from ima.research_worker_runtime import BoundedFitExecutor, cleanup_attempt_group, read_runtime


FIT_TIMEOUT_SECONDS = 600
PROTOCOL = {"min_train_races": 200, "calibration_races": 25, "score_races": 25,
            "max_folds": 1, "whole_meeting_boundaries": True}
MODELS = (
    ("benter_conditional_logit", {"l2": .1, "max_iter": 300}),
    ("boosted", {"max_iter": 40}),
    ("gaussian_probit", {"heteroscedastic": True, "quadrature_order": 32,
                         "l2": .1, "scale_l2": .1, "max_iter": 300}),
)


def _write_receipt(output, receipt):
    path = output / "receipt.json"
    temporary = output / "receipt.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _readonly_evidence(artifacts):
    evidence = {}
    for fold_id, artifact in artifacts.items():
        manifest = artifact.manifest()
        arrays = artifact.load_arrays()
        evidence[fold_id] = {"artifact_id": artifact.artifact_id, "path": str(artifact.path),
            "checksums": manifest["checksums"], "read_only": all(not a.flags.writeable for a in arrays.values()),
            "arrays": manifest["arrays"]}
        if not arrays or not evidence[fold_id]["read_only"]:
            raise ValueError("Canary requires validated read-only preparation arrays")
    return evidence


def _measured_task(request, *, prepare=False, artifacts=None):
    # Fail closed on accidental provider/tracking/network use inside scientific work.
    denied = RuntimeError("Network access is disabled in the offline reliability canary")
    with patch.object(socket.socket, "connect", side_effect=denied), \
         patch.object(socket.socket, "connect_ex", side_effect=denied), \
         patch.object(socket, "getaddrinfo", side_effect=denied), \
         threadpool_limits(limits=1), JobMonitor() as monitor:
        if prepare:
            prepared = prepare_recipe_folds(request)
            result = {"artifacts": {name: asdict(artifact) for name, artifact in prepared.items()},
                      "readonly_artifacts": _readonly_evidence(prepared)}
        else:
            evidence = _readonly_evidence(artifacts)
            result = {"result": execute_recipe(request).serializable(), "readonly_artifacts": evidence}
    resources = monitor.report()
    resources["failed"] = not prepare and result["result"]["status"] != "completed"
    return {**result, "resources": resources}


def _bounded_task(executor, request, **kwargs):
    runtime = request.output_dir / "runtime.json"
    started = time.monotonic()
    future = executor.submit(_measured_task, request, runtime_path=runtime,
                             timeout=FIT_TIMEOUT_SECONDS, **kwargs)
    try:
        payload = future.result(timeout=FIT_TIMEOUT_SECONDS + 30)
        payload.update(supervised_wall_seconds=time.monotonic() - started, runtime=read_runtime(runtime))
        return payload
    except Exception as exc:
        future.cancel()
        cleanup_attempt_group(runtime)
        state = read_runtime(runtime)
        return {"status": "timeout" if isinstance(exc, TimeoutError) else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "supervised_wall_seconds": time.monotonic() - started,
                "runtime": state, "resources": {"supported": False,
                    "source": "supervisor_heartbeat", "failed": True, "censored": True,
                    "private_peak_bytes": state.get("private_peak_bytes"),
                    "cpu_seconds": state.get("cpu_seconds"),
                    "limitation": "Last heartbeat is a lower bound; no completed JobMonitor report"}}


def _fit_receipt(request, payload, prepared, dependency_id):
    result = payload.get("result", {})
    objective = result.get("objective_value")
    finite = isinstance(objective, (int, float)) and math.isfinite(objective)
    errors = []
    cache_receipts = {}
    if result.get("status") == "completed":
        if not finite:
            errors.append("Objective is not finite")
        if result.get("metrics", {}).get("objective_source") != "model":
            errors.append("Objective is not fundamental-only")
        if result.get("lineage", {}).get("dataset_hash") != request.dataset_hash:
            errors.append("Result dataset hash mismatch")
        if result.get("lineage", {}).get("code_revision") != request.code_revision:
            errors.append("Result code revision mismatch")
        for fold_id, artifact in prepared.items():
            path = request.output_dir / f"preparation-{fold_id}.json"
            try:
                row = json.loads(path.read_text())
                cache_receipts[fold_id] = row
                if row["artifact_id"] != artifact.artifact_id or row["cache_status"] != "hit":
                    errors.append("Fit did not reuse the prewarmed artifact: " + fold_id)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"Missing/invalid preparation receipt: {fold_id}: {exc}")
    completed = result.get("status") == "completed" and not errors
    return {"attempt_id": request.attempt_id, "model_family": request.recipe.model.kind,
            "recipe": request.recipe.model_dump(mode="json"), "recipe_hash": request.recipe.recipe_hash(),
            "preparation_dependency_id": dependency_id,
            "status": "completed" if completed else payload.get("status", "failed"),
            "completed": completed, "objective_finite": finite,
            "objective_name": result.get("objective_name"), "objective_value": objective if finite else None,
            "objective_source": result.get("metrics", {}).get("objective_source"),
            "duration_seconds": result.get("duration_seconds"),
            "supervised_wall_seconds": payload["supervised_wall_seconds"],
            "resources": payload["resources"], "runtime": payload["runtime"],
            "readonly_artifacts": payload.get("readonly_artifacts", {}),
            "preparation": cache_receipts, "lineage": result.get("lineage", {}),
            "artifacts": result.get("artifacts", {}), "result_path": str(request.output_dir / "result.json"),
            "error": payload.get("error") or result.get("error"), "validation_errors": errors}


def run_canary(dataset_path, output, code_revision):
    """Reserve a fresh output directory and publish a receipt after each bounded step."""
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError("Canary output already exists; it will not be modified: " + str(output))
    dataset_path = Path(dataset_path).resolve(strict=True)
    if not dataset_path.is_file():
        raise ValueError("Dataset path must be a local file")
    if not isinstance(code_revision, str) or not code_revision.strip():
        raise ValueError("Nonempty code revision required")
    dataset_hash = file_sha256(dataset_path)
    dependencies = {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pydantic", "pebble", "threadpoolctl", "psutil")}
    environment = {"python": platform.python_version(), "dependencies": dependencies}
    environment_hash = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    # Exclusive mkdir is also the race-safe guard against overwriting another run.
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output = output.resolve()
    requests = [RecipeExecutionRequest(
        attempt_id="canary-" + kind, proposal_id="offline-v6-reliability", trial_number=index,
        recipe=PipelineRecipe(schema_version=3, target={"kind": "win_probability"},
            feature_schema="baseline-v1", blend={"kind": "none"}, model={"kind": kind, "parameters": parameters}),
        dataset_path=dataset_path, dataset_hash=dataset_hash, output_dir=output / "trials" / kind,
        protocol_parameters=dict(PROTOCOL), code_revision=code_revision,
        environment_hash=environment_hash, portfolio_version=V6_PORTFOLIO_VERSION,
    ) for index, (kind, parameters) in enumerate(MODELS)]
    receipt = {"schema_version": 1, "status": "running", "completed": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash, "code_revision": code_revision, "environment": environment,
        "environment_hash": environment_hash, "protocol_parameters": dict(PROTOCOL),
        "portfolio_version": V6_PORTFOLIO_VERSION, "max_workers": 1, "native_threads": 1,
        "per_fit_timeout_seconds": FIT_TIMEOUT_SECONDS, "tracking_enabled": False,
        "scope": "Offline scientific reliability only; this smaller protocol cannot rank production champions",
        "preparation": None, "results": []}
    _write_receipt(output, receipt)
    try:
        ids = [preparation_dependency_id(request) for request in requests]
        if ids[0] is None or len(set(ids)) != 1:
            raise ValueError("All three model-only recipes must share one preparation dependency")
        receipt["preparation_dependency_ids"] = ids
        producer = replace(requests[0], attempt_id="canary-preparation", output_dir=output / "trials" / "preparation")
        with BoundedFitExecutor(max_workers=1) as executor:
            preparation = _bounded_task(executor, producer, prepare=True)
            receipt["preparation"] = preparation
            _write_receipt(output, receipt)
            if "artifacts" not in preparation:
                raise RuntimeError("Prewarming failed: " + preparation.get("error", "unknown failure"))
            prepared = {name: PreparationArtifact(**row) for name, row in preparation["artifacts"].items()}
            if len(prepared) != 1:
                raise ValueError("Canary must prewarm exactly one whole-meeting fold")
            with ExitStack() as pins:
                for artifact in prepared.values():
                    pins.enter_context(PreparationCache(artifact.root).pin(artifact))
                for request in requests:
                    payload = _bounded_task(executor, request, artifacts=prepared)
                    receipt["results"].append(_fit_receipt(request, payload, prepared, ids[0]))
                    _write_receipt(output, receipt)
                if _readonly_evidence(prepared) != preparation["readonly_artifacts"]:
                    raise ValueError("Prewarmed immutable artifacts changed during model fitting")
        if file_sha256(dataset_path) != dataset_hash:
            raise ValueError("Dataset content changed during the canary")
        protocols = {row["lineage"].get("protocol_id") for row in receipt["results"]}
        receipt["completed"] = (len(receipt["results"]) == 3 and all(row["completed"] for row in receipt["results"])
                                and len(protocols) == 1 and None not in protocols)
        receipt["status"] = "completed" if receipt["completed"] else "failed"
    except Exception as exc:
        receipt.update(status="failed", completed=False, error=f"{type(exc).__name__}: {exc}")
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_receipt(output, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", "--datasetPath", dest="dataset_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_canary(args.dataset_path, args.output, args.code_revision)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": receipt["status"], "completed": receipt["completed"],
                      "receipt": str(args.output.absolute() / "receipt.json")}, sort_keys=True))
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
