"""Refit fixed season candidates on pre-calibration features, without campaign execution.

Run with PYTHONPATH pointing to the parent's pinned original source tree. Inputs
are precomputed rich/speed features; this utility never reconstructs history or
uses query outcomes. Numeric missing values stay missing for existing imputers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

FAMILIES = ("benter_conditional_logit", "boosted", "pool", "gaussian_probit")
CUTOFF = "2026-01-14"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_candidates(path):
    from ima.research_specs import PipelineRecipe

    frozen = json.loads(Path(path).read_text())
    if set(frozen["candidates"]) != set(FAMILIES):
        raise ValueError("Manifest must contain exactly the four frozen families")
    for family, candidate in frozen["candidates"].items():
        recipe = PipelineRecipe.model_validate(candidate["recipe"])
        expected_kind = "benter_conditional_logit" if family == "pool" else family
        if (recipe.model.kind != expected_kind or recipe.target.kind != "win_probability"
                or recipe.calibration.kind != "none" or recipe.blend.kind != "none"
                or recipe.train_window != "all_history" or recipe.schema_version != 3
                or recipe.feature_discovery or recipe.feature_definitions):
            raise ValueError(f"Unsupported fixed final-refit contract: {family}")
        if family == "pool":
            graph = recipe.pipeline_graph
            if not graph or [node["kind"] for node in graph["nodes"]] != [
                    "estimator", "estimator", "weighted_probability_pool"]:
                raise ValueError("Pool must be the fixed two-estimator graph")
            if graph["nodes"][-1]["parameters"]["weights"] != [.6, .4]:
                raise ValueError("Pool weights must remain fixed at 0.6/0.4")
        elif recipe.pipeline_graph:
            raise ValueError("Flat frozen family cannot acquire a graph")
    return frozen


def prepare_inputs(training, queries, recipe, cutoff=CUTOFF):
    import numpy as np
    import pandas as pd
    from ima.feature_sets import FeatureSchema, FEATURE_SCHEMAS, drop_feature_families
    from ima.research_executor import _schema_with_transform_features

    base = drop_feature_families(FEATURE_SCHEMAS[recipe.feature_schema], recipe.drop_feature_families)
    base = FeatureSchema(base.name, tuple(dict.fromkeys((*base.numeric, *(recipe.extra_numeric_features or ())))),
                         base.categorical)
    prepared = []
    inputs = [("training", training)] + ([("queries", queries)] if queries is not None else [])
    for name, original in inputs:
        frame = original.copy()
        required = {"date", "race_id", "race_no", "horse_no", "field_size", *base.features}
        if name == "training":
            required.update(("target_win", "target_probability"))
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"Missing {name} columns: {missing}")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.empty or frame.date.isna().any() or frame.date.dt.tz is not None:
            raise ValueError(f"{name} requires nonempty naive calendar dates")
        if name == "training" and not frame.date.lt(pd.Timestamp(cutoff)).all():
            raise ValueError("Training must be strictly BEFORE 2026-01-14; no calibration or season rows")
        if name == "queries" and not frame.date.ge(pd.Timestamp(cutoff)).all():
            raise ValueError("Queries must follow the pre-calibration fitting boundary")
        if frame[["race_id", "horse_no", "race_no"]].isna().any().any():
            raise ValueError(f"Missing {name} runner identity")
        frame["race_id"] = frame.race_id.astype(str)
        frame["horse_no"] = frame.horse_no.astype(str)
        if frame.duplicated(["race_id", "horse_no"]).any():
            raise ValueError(f"Duplicate {name} runner keys")
        if frame.groupby("race_id").date.nunique().ne(1).any():
            raise ValueError("A race cannot span dates")
        counts = frame.groupby("race_id").race_id.transform("size")
        if not counts.eq(pd.to_numeric(frame.field_size, errors="raise")).all():
            raise ValueError(f"Incomplete {name} races")
        if name == "training":
            y = pd.to_numeric(frame.target_win, errors="raise")
            target = pd.to_numeric(frame.target_probability, errors="raise")
            if (not y.isin([0, 1]).all() or not frame.assign(target_win=y).groupby("race_id").target_win.sum().eq(1).all()
                    or not np.array_equal(y.to_numpy(float), target.to_numpy(float))):
                raise ValueError("Training requires exactly one winner and matching categorical target per race")
        # Only declared predictors, identity, and pre-cutoff fitting labels survive.
        frame = frame[list(dict.fromkeys(["date", "race_id", "race_no", "horse_no", "field_size",
                  *base.features, *(["target_win", "target_probability"] if name == "training" else []),
                  *[c for c in ("horse_id", "horse_name", "venue") if c in frame]]))].copy()
        for column in base.numeric:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if np.isinf(frame[column].to_numpy(float)).any():
                raise ValueError(f"Infinite predictor: {column}")
        for column in base.categorical:
            frame[column] = frame[column].fillna("UNKNOWN").astype(str)
        prepared.append(frame.reset_index(drop=True))
    if len(prepared) == 2 and set(prepared[0].race_id) & set(prepared[1].race_id):
        raise ValueError("Training/query races overlap")
    return prepared[0], prepared[1] if len(prepared) == 2 else None, base, _schema_with_transform_features(base, recipe)


def fit_candidate(training, queries, candidate, *, code_revision, cutoff=CUTOFF):
    from ima.modeling import RaceProbabilityModel
    from ima.performance_probit import fit_probit
    from ima.pipeline_graph import fit_graph
    from ima.research_executor import FittedWinRecipeModel, FittedGraphRecipeModel
    from ima.research_model_package import ResearchModelPackage, FeatureReplayContext
    from ima.research_specs import PipelineRecipe
    from ima.research_transforms import FittedResearchTransforms, TransformSpec

    recipe = PipelineRecipe.model_validate(candidate["recipe"])
    train, query, base, schema = prepare_inputs(training, queries, recipe, cutoff)
    transforms = FittedResearchTransforms.fit(train, tuple(TransformSpec(t.kind, dict(t.parameters))
                                                          for t in recipe.transforms))
    transformed = transforms.transform(train)
    if recipe.pipeline_graph:
        estimator = fit_graph(recipe.pipeline_graph, transformed, None, schema, recipe.seed,
                              model_spec=recipe.model)
        fitted = FittedGraphRecipeModel(estimator, transforms, schema)
        physical_fits = estimator.fit_report["physical_fits"]
    else:
        estimator = (fit_probit(transformed, schema, recipe.model.parameters)
                     if recipe.model.kind == "gaussian_probit" else
                     RaceProbabilityModel(kind=recipe.model.kind, parameters=dict(recipe.model.parameters),
                                          random_state=recipe.seed, feature_schema=schema).fit(transformed))
        fitted = FittedWinRecipeModel(estimator, transforms, None, None, schema)
        physical_fits = 1
    import ima
    source_root = Path(ima.__file__).resolve().parent
    implementations = {p.name: sha256(p) for p in source_root.glob("*.py")}
    context = FeatureReplayContext(
        required_extra_features=tuple(recipe.extra_numeric_features or ()),
        training_cutoff=str(train.date.max()),
        dependency_versions={n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn")},
        implementation_hashes=implementations)
    population_hash = hashlib.sha256(train[["date", "race_id", "horse_no", "target_win"]]
                                    .to_csv(index=False).encode()).hexdigest()
    package = ResearchModelPackage(fitted, recipe, "finalrefit-" + population_hash[:16], code_revision,
                                   feature_context=context)
    probabilities = package.predict_proba(query) if query is not None else None
    report = {"fit_date_min": str(train.date.min()), "fit_date_max": str(train.date.max()),
              "training_cutoff": context.training_cutoff, "training_rows": len(train),
              "training_races": int(train.race_id.nunique()), "fit_population_sha256": population_hash,
              "physical_fits": physical_fits, "base_schema": base.contract(), "fitted_schema": schema.contract(),
              "recipe_hash": recipe.recipe_hash(), "fit_scope": "training_only_before_2026-01-14",
              "query_rows": len(query) if query is not None else 0,
              "query_races": int(query.race_id.nunique()) if query is not None else 0,
              "calibration_fitted": False, "market_blend_fitted": False}
    return package, query, probabilities, report


def worker(args):
    import resource
    # Limits affect only this fresh worker, never pre-existing jobs or services.
    limit = int(args.memory_gb * 1024**3 / args.workers)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    import pandas as pd
    import numpy as np
    from threadpoolctl import threadpool_limits
    from ima.research_model_package import load_research_package

    fit_only = getattr(args, "fit_only", False)
    source_paths = [args.manifest, args.training, args.history, Path(__file__)]
    if not fit_only:
        source_paths.append(args.queries)
    sources = {str(p.resolve()): sha256(p) for p in source_paths}
    snapshot_path = args.training.parent / "input-snapshot.json"
    input_snapshot = None
    if snapshot_path.exists():
        sources[str(snapshot_path.resolve())] = sha256(snapshot_path)
        input_snapshot = json.loads(snapshot_path.read_text())
    if args.provenance:
        sources[str(args.provenance.resolve())] = sha256(args.provenance)
    frozen = load_candidates(args.manifest)
    history = pd.read_parquet(args.history, columns=["date"])
    if history.empty or pd.to_datetime(history.date, errors="raise").isna().any():
        raise ValueError("Normalized raw history requires valid dates")
    provenance = json.loads(args.provenance.read_text()) if args.provenance else {}
    training = pd.read_parquet(args.training)
    queries = None if fit_only else pd.read_parquet(args.queries)
    import ima
    module_hashes = {p.name: sha256(p) for p in Path(ima.__file__).resolve().parent.glob("*.py")}
    environment = {n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")}
    for path, actual in ((args.expected_environment, environment), (args.expected_code_hashes, module_hashes)):
        if path:
            sources[str(path.resolve())] = sha256(path)
            expected = json.loads(path.read_text())
            if any(actual.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Pinned environment/source mismatch: {path}")
    started = time.monotonic()
    with threadpool_limits(limits=args.threads):
        package, query, probabilities, report = fit_candidate(training, queries,
            frozen["candidates"][args.worker], code_revision=args.code_revision)
        destination = args.output / "final-fits" / args.worker
        destination.mkdir(exist_ok=False)
        package.save(destination)
        reloaded = load_research_package(destination)
        if reloaded.manifest().package_id != package.manifest().package_id:
            raise ValueError("Fresh package identity readback mismatch")
        if query is not None:
            restored = reloaded.predict_proba(query)
            if not np.allclose(probabilities, restored, atol=1e-12, rtol=0):
                raise ValueError("Fresh package prediction readback mismatch")
    prediction_metadata = {"prediction_readback_performed": False}
    if query is not None:
        rows = query[[c for c in ("date", "race_id", "race_no", "horse_id", "horse_no", "horse_name", "field_size")
                      if c in query]].copy()
        rows["model"], rows["model_probability"] = args.worker, probabilities
        prediction_path = args.output / "fresh-predictions" / f"{args.worker}.csv"
        rows.to_csv(prediction_path, index=False)
        prediction_metadata = dict(prediction_readback_performed=True,
            readback_max_absolute_error=float(np.max(np.abs(probabilities - restored))),
            input_sha256=sources[str(args.queries.resolve())], output_sha256=sha256(prediction_path),
            prediction_path=str(prediction_path.resolve()),
            probability_sum_min=float(rows.groupby("race_id").model_probability.sum().min()),
            probability_sum_max=float(rows.groupby("race_id").model_probability.sum().max()))
    for path, digest in sources.items():
        if sha256(path) != digest:
            raise ValueError(f"Refit input changed while fitting: {path}")
    for name, digest in module_hashes.items():
        if sha256(Path(ima.__file__).resolve().parent / name) != digest:
            raise ValueError(f"Pinned source changed while fitting: {name}")
    report.update(family=args.worker, elapsed_seconds=time.monotonic() - started,
                  sources_sha256=sources, source_root=str(Path(ima.__file__).resolve().parent),
                  implementation_hashes=module_hashes, environment=environment,
                  code_revision=args.code_revision, code_revision_policy="Parent-supplied pinned revision; actual module bytes are hashed",
                  source_provenance=provenance, input_attrs=training.attrs,
                  input_snapshot=input_snapshot,
                  history_date_min=str(pd.to_datetime(history.date).min()),
                  history_date_max=str(pd.to_datetime(history.date).max()),
                  history_use="Provenance only; predictors were materialized by parent before refit",
                  package_id=package.manifest().package_id,
                  **prediction_metadata,
                  artifact_sha256={str(p.relative_to(destination)): sha256(p) for p in destination.rglob("*") if p.is_file()})
    write_json(destination / "refit.json", report)


def stop_worker(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=5)
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def snapshot_inputs(args):
    """Workers use exclusive copied inputs even if parent main-ops files change."""
    import ima
    from types import SimpleNamespace

    snapshot = args.output / "refit-inputs"
    snapshot.mkdir(exist_ok=False)
    copied = SimpleNamespace(**vars(args))
    records = {}

    def freeze(path, name):
        original = Path(path).resolve()
        digest = sha256(original)
        destination = snapshot / name
        shutil.copyfile(original, destination)
        if sha256(destination) != digest or sha256(original) != digest:
            raise ValueError(f"Input changed while snapshotting: {original}")
        destination.chmod(0o444)
        records[name] = {"original_path": str(original), "snapshot_sha256": digest}
        return destination

    for option, name in (("manifest", "frozen-candidates.json"), ("training", "training.parquet"),
                         ("history", "history.parquet"), ("queries", "queries.parquet"),
                         ("provenance", "training-provenance.json"),
                         ("expected_environment", "expected-environment.json")):
        if getattr(args, option, None) is not None:
            setattr(copied, option, freeze(getattr(args, option), name))
    source_root = Path(ima.__file__).resolve().parent
    hashes = {p.name: sha256(p) for p in source_root.glob("*.py")}
    if args.expected_code_hashes:
        freeze(args.expected_code_hashes, "parent-expected-code-hashes.json")
        expected = json.loads(args.expected_code_hashes.read_text())
        if any(hashes.get(k) != v for k, v in expected.items()):
            raise ValueError("Pinned source differs from parent's expected source hashes")
    archive = snapshot / "pinned-source"
    archive.mkdir()
    for name, digest in hashes.items():
        shutil.copyfile(source_root / name, archive / name)
        if sha256(archive / name) != digest:
            raise ValueError("Pinned source changed during archive")
        (archive / name).chmod(0o444)
    copied.expected_code_hashes = snapshot / "expected-code-hashes.json"
    write_json(copied.expected_code_hashes, hashes)
    copied.expected_code_hashes.chmod(0o444)
    freeze(Path(__file__), "refit_season_2026.py")
    write_json(snapshot / "input-snapshot.json", {"sources": records, "pinned_source_root": str(source_root),
                                                "implementation_hashes": hashes})
    (snapshot / "input-snapshot.json").chmod(0o444)
    return copied


def write_readback(args, statuses):
    models = {}
    fresh = {"candidates": {}, "fit_boundary_exclusive": CUTOFF,
             "stage": "fresh_final_refit_of_prior_history_frozen_recipes"}
    environment, implementations = None, None
    for family, status in statuses.items():
        if status["status"] == "completed":
            report_path = args.output / "final-fits" / family / "refit.json"
            report = json.loads(report_path.read_text())
            if environment is not None and (report["environment"] != environment
                                           or report["implementation_hashes"] != implementations):
                raise ValueError("Fresh families used different source/environment snapshots")
            environment, implementations = report["environment"], report["implementation_hashes"]
            candidate = {
                "remote_package": str(report_path.parent.resolve()),
                "local_package": str(report_path.parent.resolve()),
                "recipe": json.loads((report_path.parent / "recipe.json").read_text()),
                "hashes": {p.name: sha256(p) for p in report_path.parent.iterdir() if p.is_file()},
                "package_id": report["package_id"], "fit_date_min": report["fit_date_min"],
                "fit_date_max": report["fit_date_max"], "training_cutoff": report["training_cutoff"],
                "selection_policy": "Original four recipes fixed using prior history; no reselection"}
            fresh["candidates"][family] = candidate
            if getattr(args, "fit_only", False):
                continue
            models[family] = {k: report[k] for k in ("package_id", "training_cutoff", "fit_date_min", "fit_date_max",
                "training_rows", "training_races", "query_rows", "query_races", "input_sha256", "output_sha256",
                "prediction_path", "probability_sum_min", "probability_sum_max", "readback_max_absolute_error")}
            models[family].update(rows=report["query_rows"], races=report["query_races"],
                                 package_path=str(report_path.parent.resolve()), refit_report_sha256=sha256(report_path))
    fresh.update(environment=environment, implementation_hashes=implementations, fit_status=statuses)
    write_json(args.output / "final-fits" / "fresh-candidates.json", fresh)
    if getattr(args, "fit_only", False):
        return
    write_json(args.output / "fresh-predictions" / "readback.json", {
        "input_sha256": sha256(args.queries), "models": models, "fit_status": statuses,
        "environment": environment, "implementation_hashes": implementations,
        "fit_boundary_exclusive": CUTOFF, "calibration_or_season_outcomes_used_for_fit": False})


def run(args):
    frozen = load_candidates(args.manifest)
    for candidate in frozen["candidates"].values():
        for key in ("local_package", "remote_package"):
            if candidate.get(key):
                old = Path(candidate[key]).resolve()
                new = (args.output / "final-fits").resolve()
                predictions = (args.output / "fresh-predictions").resolve()
                if any(p.is_relative_to(old) or old.is_relative_to(p) for p in (new, predictions)):
                    raise ValueError("Fresh outputs must be separate from every frozen package")
    models_output = args.output / "final-fits"
    predictions_output = args.output / "fresh-predictions"
    if models_output.exists() or (not getattr(args, "fit_only", False) and predictions_output.exists()):
        raise ValueError("Fresh final-fits/ and fresh-predictions/ must not already exist")
    models_output.mkdir(parents=True, exist_ok=False)
    if not getattr(args, "fit_only", False):
        predictions_output.mkdir(exist_ok=False)
    args = snapshot_inputs(args)
    pending, active, statuses = list(args.families), {}, {}
    command_args = ["--manifest", str(args.manifest.resolve()), "--training", str(args.training.resolve()),
                    "--history", str(args.history.resolve()),
                    "--output", str(args.output.resolve()), "--code-revision", args.code_revision,
                    "--workers", str(args.workers), "--threads", str(args.threads),
                    "--memory-gb", str(args.memory_gb)]
    if getattr(args, "fit_only", False):
        command_args.append("--fit-only")
    else:
        command_args.extend(["--queries", str(args.queries.resolve())])
    for option in ("provenance", "expected_environment", "expected_code_hashes"):
        if getattr(args, option):
            command_args.extend(["--" + option.replace("_", "-"), str(getattr(args, option).resolve())])
    try:
        while pending or active:
            while pending and len(active) < args.workers:
                family = pending.pop(0)
                log = (models_output / f"{family}.log").open("w")
                env = os.environ.copy()
                for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS"):
                    env[name] = str(args.threads)
                try:
                    process = subprocess.Popen([sys.executable, str((args.output / "refit-inputs/refit_season_2026.py").resolve()), *command_args,
                                                "--worker", family], stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[family] = (process, log, time.monotonic())
            for family, (process, log, started) in list(active.items()):
                elapsed = time.monotonic() - started
                timed_out = process.poll() is None and elapsed > args.time_limit
                if timed_out:
                    stop_worker(process)
                if process.poll() is not None:
                    log.close()
                    complete = process.returncode == 0 and (models_output / family / "refit.json").is_file()
                    statuses[family] = {"status": "timeout" if timed_out else "completed" if complete else "failed",
                                        "returncode": process.returncode, "elapsed_seconds": elapsed}
                    del active[family]
                    print(json.dumps({"family": family, **statuses[family]}), flush=True)
            if active:
                time.sleep(.1)
    finally:
        for family, (process, log, started) in active.items():
            stop_worker(process)
            log.close()
            statuses[family] = {"status": "cancelled", "returncode": process.returncode}
        write_json(models_output / "status.json", {"models": statuses, "time_limit_seconds_per_family": args.time_limit,
            "max_concurrent_families": args.workers, "threads_per_worker": args.threads,
            "aggregate_address_space_ceiling_gib": args.memory_gb,
            "memory_policy": "Each own worker has RLIMIT_AS = ceiling/workers; pool components fit sequentially",
            "no_out_of_sample_performance_claim": True})
        write_readback(args, statuses)
    return 0 if all(s["status"] == "completed" for s in statuses.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "training", "history", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--fit-only", action="store_true", help="Fit now; use fresh-candidates.json for later prediction")
    parser.add_argument("--code-revision", required=True)
    for name in ("provenance", "expected-environment", "expected-code-hashes"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES[:3]))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--memory-gb", type=float, default=90, help="Aggregate own-worker address-space limit in GiB")
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--worker", choices=FAMILIES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.fit_only and args.queries is None:
        parser.error("--queries is required unless --fit-only is set")
    if (not 1 <= args.workers <= 4 or not 3 <= args.threads <= 4
            or not math.isfinite(args.memory_gb) or not 0 < args.memory_gb <= 100
            or not math.isfinite(args.time_limit) or args.time_limit <= 0
            or len(set(args.families)) != len(args.families)):
        parser.error("Require 1-4 workers, 3-4 threads, <=100 GB, positive finite timeout and unique families")
    if not args.worker:
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Refit coordinator interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
    return worker(args) if args.worker else run(args)


if __name__ == "__main__":
    sys.exit(main())
