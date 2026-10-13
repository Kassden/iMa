"""Additive campaign integration for measured probit and reusable feature work."""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import re

VERSION = 1
# Independently inspected pre-dispatch failure, sealed in the deployed release.
# Never infer exemptions for future failures from mutable campaign records alone.
QUALIFIED_PREFLIGHTS = {"D000236": {
    "decision_sha256": "a2294b248a8eb309c9a326e0ee97a7602b6188c7dc220df91dce9b8e161584af",
    "evidence_sha256": "556758ef40306fe63b0c96cb20e02c3952be8d42422acedf2ef2251160bf0ad7",
}, "D000239": {
    "decision_sha256": "0b92a53f0aaedc47b8d12bd40990810b2a101233612abf0b2385eb3fb391ec7a",
    "evidence_sha256": "9f3473e801aaed49028a8bd4719664c7302be3f6daa590477860c3a7db59fa13",
}}
_REQUEST = ContextVar("ima_lifecycle_request", default=None)
_PROFILES = ContextVar("ima_lifecycle_profiles", default=None)


def compact_optional_history(bundle):
    """Keep identities, champions, safety and full pending recipes; bound narratives."""
    view = deepcopy(bundle)
    evidence = view.get("evidence", view)
    omissions = {}
    for key, keep in (("hypothesis_memory", 4), ("recent_planner_failures", 2)):
        records = evidence.get(key)
        if isinstance(records, list) and len(records) > keep:
            evidence[key] = records[:keep]
            omissions[key] = len(records)-keep
    for mapping_key in ("champions_by_family", "champions_by_contract"):
        records = evidence.get(mapping_key, {})
        for row in records.values() if isinstance(records, dict) else ():
            if isinstance(row, dict):
                for key in ("artifacts", "worker_resources"):
                    if key in row:
                        row.pop(key)
                        omissions[mapping_key+"."+key] = "execution paths omitted; recipes, scores and contracts retained"
    evidence["lifecycle_context_omissions"] = omissions
    return view


def prove_local_preflight(directory, decision_id, *, persist=True):
    """A local budget rejection is not a provider bill; never invent call usage."""
    directory = Path(directory)
    if not re.fullmatch(r"D[0-9]{6}", decision_id):
        raise ValueError("Invalid decision identity")
    path = directory/"decisions"/(decision_id+".json")
    row = _json(path)
    error = row.get("error", "")
    pattern = r"Mandatory planner context exceeds byte budget: [0-9]+>[0-9]+; no paid call"
    if row.get("planner_status") != "failed" or not re.fullmatch(pattern, error):
        raise ValueError("Not a verified local context-budget failure")
    if row.get("transport_attempts") or row.get("planner_usage"):
        raise ValueError("Planner dispatch/billing evidence exists")
    for folder in ("planner-calls", "planner-transport"):
        if list((directory/folder).glob(decision_id+"-*.json")):
            raise ValueError("Durable transport/call evidence exists")
    response = directory/"planner-responses"/(decision_id+".json")
    if response.exists():
        payload = _json(response).get("response") or {}
        if payload.get("raw_response") or payload.get("decision") or payload.get("usage"):
            raise ValueError("Provider response evidence exists")
    evidence_path = directory/"evidence"/(decision_id+".json")
    if not evidence_path.exists():
        raise ValueError("Missing durable input evidence")
    proof = {"version": VERSION, "decision_id": decision_id,
        "decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "basis": "Exact local context-budget failure before _post_json; no provider call or transport evidence",
        "provider_cost": None, "dispatch": "verified_not_dispatched"}
    qualification = QUALIFIED_PREFLIGHTS.get(decision_id)
    if not qualification or any(proof[key] != value for key, value in qualification.items()):
        raise ValueError("No sealed independent pre-dispatch qualification")
    from .research_controller import _write_json_atomic
    if persist:
        _write_json_atomic(directory/"lifecycle-preflight-proofs"/(decision_id+".json"), proof)
    return proof


def _qualified_unsent(directory, decision_id):
    directory = Path(directory)
    path = directory/"lifecycle-preflight-proofs"/(decision_id+".json")
    if not path.exists():
        return False
    try:
        stored = _json(path)
        current = prove_local_preflight(directory, decision_id, persist=False)
        return current == stored
    except (ValueError, OSError, KeyError, TypeError):
        return False


def is_probit_recipe(recipe):
    if recipe.model.kind == "gaussian_probit":
        return True
    graph = recipe.pipeline_graph or {}
    return any(node.get("parameters", {}).get("model_kind") == "gaussian_probit"
               for node in graph.get("nodes", ()))


def _json(path):
    return json.loads(Path(path).read_text())


def _ledger_rows(directory):
    path = Path(directory)/"ledger.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT attempt_id,status,payload_json,result_json FROM attempts").fetchall()
    return [{"attempt_id": attempt, "status": status, "payload": json.loads(payload),
             "result": json.loads(result) if result else {}}
            for attempt, status, payload, result in rows]


def feature_feedback(directory, candidates):
    """Expose observed use, not a universal score across different targets or races."""
    rows = _ledger_rows(directory)
    counts = {"discovery_completed": 0, "definitions_completed": 0,
              "discovery_pending": 0, "definitions_pending": 0}
    for row in rows:
        recipe = row["payload"].get("recipe", {})
        for field, label in (("feature_discovery", "discovery"), ("feature_definitions", "definitions")):
            if recipe.get(field):
                state = "completed" if row["status"] == "completed" else "pending" if row["status"] in {"reserved", "running"} else None
                if state:
                    counts[f"{label}_{state}"] += 1
    outcomes = []
    for candidate in candidates:
        dataset_id = candidate.get("dataset_id")
        results = []
        for row in rows:
            if row["payload"].get("recipe", {}).get("dataset_ref") != dataset_id:
                continue
            result = row["result"]
            lineage = result.get("lineage", {})
            results.append({"attempt_id": row["attempt_id"], "status": row["status"],
                "objective": result.get("objective_name"), "value": result.get("objective_value"),
                "population": lineage.get("evaluation_population_id"),
                "protocol": lineage.get("protocol_hash"), "basis": lineage.get("probability_basis"),
                "error": result.get("error")})
        outcomes.append({"dataset_id": dataset_id, "latest_model_outcomes": results[-8:],
                         "status": "tested_development_only" if any(r["status"] == "completed" for r in results) else "untested"})
    return {"counts": counts, "outcomes": outcomes,
            "interpretation": "Feature builds are not model improvements. Compare matched population, protocol, target and probability basis; confirm separately."}


def lifecycle_worker(request, workload, threads):
    from . import research_expansion as expansion
    token = _REQUEST.set(request)
    profiles = []
    profile_token = _PROFILES.set(profiles)
    try:
        result = expansion._lifecycle_original_worker(request, workload, threads)
        result.metrics["research_lifecycle"] = {"version": VERSION,
            "release_sha256": getattr(expansion, "_lifecycle_release_hash", None)}
        if profiles:
            result.metrics["probit_fits"] = {f"fit_{index:03d}": profile for index, profile in enumerate(profiles)}
            result.artifacts["probit_profiles"] = str(request.output_dir/"probit-profiles.json")
        if result.status == "completed":
            try:
                link_feature_outcome(request, result)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                # Tracking is retryable; a successful scientific fit stays successful.
                expansion.core._write_json_atomic(request.output_dir/"feature-feedback-pending.json",
                    {"attempt_id": request.attempt_id, "error": str(exc), "retry": "link_feature_outcome"})
                result.artifacts["feature_feedback_pending"] = str(request.output_dir/"feature-feedback-pending.json")
        expansion.core._write_json_atomic(request.output_dir/"result.json", result.serializable())
        return result
    finally:
        _PROFILES.reset(profile_token)
        _REQUEST.reset(token)


def link_feature_outcome(request, result):
    """Attach actual model feedback to the build's comparison-scoped catalog."""
    manifest_path = Path(request.dataset_path).parent/"manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _json(manifest_path)
    build_id = manifest.get("feature_build_id")
    if not build_id:
        return None
    registry_root = manifest_path.parent.parent.parent
    from .feature_lifecycle import FeatureLifecycle, record_model_outcome
    receipt = FeatureLifecycle(registry_root).receipt(build_id)
    used = set(request.recipe.extra_numeric_features or ())
    definition_ids = sorted({row["definition_id"] for name, row in manifest["predictor_catalog"].items()
                             if name in used and row.get("definition_id")})
    result.lineage.update(feature_build_id=build_id,
        feature_build_run_id=receipt["tracking"].get("run_id") or "tracking_pending",
        feature_definition_ids=json.dumps(definition_ids))
    if not definition_ids:
        return None
    lineage = result.lineage
    contract = {"comparison_contract_id": manifest["comparison_contract_id"],
        "target_contract": request.recipe.target.kind,
        "evaluation_population_id": lineage["evaluation_population_id"],
        "declared_evaluation_population_id": lineage["declared_evaluation_population_id"],
        "protocol_id": manifest["comparison_contract"]["protocol_id"], "protocol_hash": lineage["protocol_hash"],
        "score_population_sha256": lineage["evaluation_population_hash"],
        "metric": result.objective_name, "direction": "minimize", "probability_basis": lineage["probability_basis"]}
    outcome = {"attempt_id": request.attempt_id, "model_id": request.recipe.model.kind+":"+request.recipe.recipe_hash(),
               "dataset_id": manifest["dataset_id"], "definition_ids": definition_ids,
               "score": result.objective_value, "comparison_contract": contract}
    report = record_model_outcome(registry_root, build_id, outcome)
    from .research_controller import _write_json_atomic
    _write_json_atomic(request.output_dir/"feature-build-outcome.json", report)
    result.artifacts["feature_build_outcome"] = str(request.output_dir/"feature-build-outcome.json")
    return report


def lifecycle_probit_fit(train, feature_schema, parameters=None, **kwargs):
    from . import performance_probit as probit
    from .research_executor import _execution_progress
    request = _REQUEST.get()
    if request is None:
        return probit._lifecycle_original_fit(train, feature_schema, parameters, **kwargs)
    previous = _json(request.output_dir/"progress.json") if (request.output_dir/"progress.json").exists() else {}
    fold = {key: previous.get(key) for key in ("fold_id", "fold_index", "fold_count")}

    def callback(event):
        data = {key: value for key, value in event.items()
                if key not in {"stage", "pid", "updated_at_epoch", *fold}}
        _execution_progress(request, event.get("stage", "optimizer"), **fold, **data)
        profiles = _PROFILES.get()
        if event.get("stage") == "fit_completed" and profiles is not None:
            profiles.append({**fold, **data})
            from .research_controller import _write_json_atomic
            _write_json_atomic(request.output_dir/"probit-profiles.json", {"fits": profiles, "version": VERSION})

    kwargs.setdefault("progress_callback", callback)
    kwargs.setdefault("cache_dir", request.output_dir.parent.parent/"probit-warm-starts")
    payload = request.recipe.canonical_payload()
    kwargs.setdefault("training_context", {
        "dataset_hash": request.dataset_hash, "fold_id": fold["fold_id"],
        "protocol": request.protocol_parameters, "code_revision": request.code_revision,
        "transforms": payload.get("transforms"),
        "feature_discovery": payload.get("feature_discovery"),
        "feature_definitions": payload.get("feature_definitions"),
        "train_window": payload.get("train_window"),
    })
    return probit._lifecycle_original_fit(train, feature_schema, parameters, **kwargs)


def lifecycle_prepare(dataset, recipe, digest, root):
    from . import research_expansion as expansion
    if recipe.feature_definitions:
        from .feature_lifecycle import build_feature_candidate
        config = _json(Path(root)/"config.json")
        manifest = Path(dataset).parent/"manifest.json"
        if not manifest.exists():
            raise ValueError("Feature lifecycle publication requires a verified parent manifest")
        candidate = build_feature_candidate(
            parent_features_path=dataset, parent_manifest_path=manifest,
            registry_root=config["dataset_registry_path"],
            definitions=recipe.feature_definitions,
            tracking_uri=config.get("mlflow_tracking_uri"),
            experiment_name="ima-feature-builds")
        expansion.core._write_json_atomic(Path(root)/"feature-builds"/(recipe.recipe_hash()+".json"), candidate)
    return expansion._lifecycle_original_prepare(dataset, recipe, digest, root)


def install_campaign_policy(directory, *, release_sha256=None):
    """Install only additive hooks; original paid guard and allocation remain owners."""
    from . import research_expansion as expansion
    from .research_worker_runtime import BoundedFitExecutor
    from .research_resources import JobEstimator
    from .probit_runtime_policy import resolve_probit_budget
    from . import performance_probit as probit
    directory = Path(directory)
    if getattr(expansion, "_lifecycle_installed", False):
        return {"version": VERSION, "already_installed": True}
    original_caps = expansion._capabilities
    original_spend = expansion._planner_spend
    expansion._lifecycle_original_worker = expansion._worker
    expansion._lifecycle_original_prepare = expansion._prepare_program
    expansion._lifecycle_release_hash = release_sha256
    probit._lifecycle_original_fit = probit.fit_probit
    probit.fit_probit = lifecycle_probit_fit
    from . import mlflow_tracking
    original_parameters = mlflow_tracking.research_run_parameters

    def run_parameters(package_dir, result, **kwargs):
        params = original_parameters(package_dir, result, **kwargs)
        for key in ("feature_build_id", "feature_build_run_id", "feature_definition_ids"):
            value = result.get("lineage", {}).get(key)
            if value is not None:
                params[key] = str(value)
        return params

    mlflow_tracking.research_run_parameters = run_parameters

    def spend(root):
        result = original_spend(root)
        local = [identifier for identifier in result.get("unresolved_decision_ids", [])
                 if _qualified_unsent(root, identifier)]
        if local:
            result = dict(result)
            result["unresolved_decision_ids"] = [identifier for identifier in result["unresolved_decision_ids"] if identifier not in local]
            result["not_dispatched_decision_ids"] = sorted(set(result.get("not_dispatched_decision_ids", [])) | set(local))
            result["spend_unknown"] = bool(result["unresolved_decision_ids"])
            result["total_cost_usd"] = None if result["spend_unknown"] else result["known_spend_usd"]
            result["verified_local_preflight_failures"] = local
        return result

    def capabilities(config, dataset, **kwargs):
        from .feature_lifecycle import catalog, deterministic_candidate_definitions
        result = original_caps(config, dataset, **kwargs)
        registry_root = getattr(config, "dataset_registry_path", None)
        snapshot = catalog(registry_root, limit=8) if registry_root else {"builds": []}
        candidates = snapshot.get("builds", [])
        result["feature_lifecycle"] = {
            "version": VERSION, "controller": "one research planner, shared feature/model feedback",
            "candidates": candidates, "feedback": feature_feedback(directory, candidates),
            "build_counts": snapshot.get("counts", {}),
            "definition_examples": [definition.model_dump(mode="json") for definition in
                deterministic_candidate_definitions(result.get("eligible_predictors", {}), limit=3)],
            "actions": {
                "create": "Propose schema_version=3 recipe.feature_definitions using typed expression_ast and declare feature_creation. Preparation validates and publishes a reusable successor dataset with a separate feature_build MLflow run.",
                "reuse": "Choose a verified candidate dataset_id as recipe.dataset_ref and include candidate feature_ids in extra_numeric_features. Do not repeat definitions already materialized in that dataset.",
                "evaluate": "Use a matched parent control, same scored races/protocol/target, and only training-fold fitting/selection. Isolate feature changes before claiming gains.",
                "build": "dataset_requests remains available for new sources/history/populations; feature publication does not replace the raw-data builder.",
            },
            "priority": "Consider untested validated candidates and pending controls before tuning saturated recipes. No forced adoption or fixed allocation quota.",
        }
        result["probit_runtime"] = {"initial_seconds": 14400,
            "progress": "iterations, best objective, evaluations, fold, profiling",
            "extensions": "fresh productive progress only, durable bounded hard deadline",
            "warm_start": "same transformed training matrix, labels, preprocessing and fold context only",
            "threads": "use stage measurements; never assume linear speedup"}
        return result

    class LifecycleFitExecutor(BoundedFitExecutor):
        def submit(self, function, *args, runtime_path, timeout, **kwargs):
            request = args[0] if args else None
            if function is lifecycle_worker and request is not None and is_probit_recipe(request.recipe):
                workload = args[1]
                if workload.family.removeprefix("graph:") != "gaussian_probit":
                    workload = workload.model_copy(update={"family": "gaussian_probit"})
                policy = resolve_probit_budget(workload, JobEstimator(directory/"job-estimates.json").samples, timeout)
                kwargs["progress_policy"] = policy
                timeout = policy["initial_seconds"]
            return super().submit(function, *args, runtime_path=runtime_path, timeout=timeout, **kwargs)

    expansion._capabilities = capabilities
    expansion._planner_spend = spend
    expansion._worker = lifecycle_worker
    expansion._prepare_program = lifecycle_prepare
    expansion.BoundedFitExecutor = LifecycleFitExecutor
    # Operator policies compile this loop with an isolated globals dictionary.
    # Assigning module attributes alone does not replace its captured consumers.
    hooks = {"_capabilities": capabilities, "_planner_spend": spend,
             "_worker": lifecycle_worker, "_prepare_program": lifecycle_prepare,
             "BoundedFitExecutor": LifecycleFitExecutor}
    run_namespace = expansion.run_expansion_campaign.__globals__
    if not all(key in run_namespace for key in hooks):
        raise ValueError("Campaign lifecycle hook consumers changed")
    run_namespace.update(hooks)
    expansion._lifecycle_installed = True
    return {"version": VERSION, "release_sha256": release_sha256,
            "paid_calls": 0, "fits_cancelled": 0,
            "hooks": ["fit_submission", "feature_preparation", "planner_capabilities", "trial_lineage"]}
