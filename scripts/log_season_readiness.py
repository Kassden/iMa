"""Hash-bound, descriptive paper tracking; no fitting, model loading or registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from ima.mlflow_tracking import flatten_numeric_metrics

EXPERIMENT = "ima-season-2026-readiness"
FAMILIES = ("benter_conditional_logit", "boosted", "gaussian_probit", "pool")
VARIANTS = (*FAMILIES, *(f"{f}_market_blend_hindsight" for f in FAMILIES),
            "market_final_odds_hindsight", "market_calibrated_hindsight")
POOLS = {"WIN", "PLACE", "QIN", "QPL", "TRI", "TIERCE", "FIRST4", "QUARTET"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_forecast_provenance(path: Path, outputs: tuple[Path, ...], input_hash: str,
                                 source_files: tuple[Path, ...] = ()) -> None:
    proof = read_json(path)
    query_hashes = [digest for name, digest in proof.get("source_sha256", {}).items()
                    if Path(name).name == "query-features.parquet"]
    for key in ("query_features_sha256", "input_sha256"):
        if key in proof:
            query_hashes.append(proof[key])
    if not query_hashes or any(digest != input_hash for digest in query_hashes):
        raise ValueError(f"Forecast provenance query hash mismatch or missing: {path}")
    for source in source_files:
        expected = [digest for name, digest in proof.get("source_sha256", {}).items()
                    if Path(name).name == source.name]
        if len(expected) != 1 or expected[0] != sha256(source):
            raise ValueError(f"Forecast provenance source hash mismatch or missing: {source}")
    hashes = proof.get("output_sha256", {})
    for output in outputs:
        expected = [digest for name, digest in hashes.items() if Path(name).name == output.name]
        if len(expected) != 1 or expected[0] != sha256(output):
            raise ValueError(f"Forecast provenance output hash mismatch or missing: {output}")


def model_identity(recipe: dict) -> dict:
    model = recipe["model"]
    identity = {"model_kind": model["kind"], "recipe_model_kind": model["kind"]}
    graph = recipe.get("pipeline_graph")
    if graph:
        head_id = graph.get("fundamental_node_id") or graph["output_node_id"]
        head = next(node for node in graph["nodes"] if node["node_id"] == head_id)
        parameters = head.get("parameters", {})
        identity.update(model_kind=parameters["model_kind"] if head["kind"] == "estimator" else head["kind"],
                        head_node_id=head_id, head_parameters=json.dumps(parameters, sort_keys=True),
                        component_model_kinds=json.dumps({node["node_id"]: node["parameters"]["model_kind"]
                            for node in graph["nodes"] if node["kind"] == "estimator"}, sort_keys=True))
    if model["kind"] == "gaussian_probit":
        identity["distribution_variance"] = ("heteroscedastic" if model.get("parameters", {}).get(
            "heteroscedastic", False) else "homoscedastic")
    return identity


def prepare_logging(root: Path, evaluation: Path, predictions: Path, manifest: Path,
                    report: Path, evaluation_code: Path, *, generation: str = "fresh",
                    include_query: bool = False, package_root: Path | None = None,
                    forecast_artifacts: tuple[Path, ...] = (),
                    forecast_provenance: Path | None = None,
                    report_artifacts: tuple[Path, ...] = ()) -> dict:
    """Validate everything before any tracking writes; allow relocation by content hash."""
    provenance = read_json(evaluation / "provenance.json")
    receipt = read_json(predictions / "readback.json")
    candidates = read_json(manifest)
    summary = read_json(evaluation / "summary.json")
    calibration = read_json(evaluation / "calibration.json")
    input_hash = sha256(root / "query-features.parquet")
    checks = ((provenance["query_features_sha256"], input_hash),
              (receipt["input_sha256"], input_hash),
              (provenance["inference_readback_sha256"], sha256(predictions / "readback.json")),
              (provenance["evaluation_code_sha256"], sha256(evaluation_code)))
    if any(expected != actual for expected, actual in checks):
        raise ValueError("Evaluation/query/readback/code hash mismatch")
    if set(receipt["models"]) != set(FAMILIES) or set(candidates["candidates"]) != set(FAMILIES):
        raise ValueError("Expected four frozen family identities")
    if set(summary) != set(VARIANTS):
        raise ValueError("Expected all ten descriptive summary variants")
    artifacts = [(evaluation / name, "evaluation") for name in
                 ("summary.json", "calibration.json", "provenance.json", "tomorrow-combinations.csv")]
    artifacts += [(report, "report"), (manifest, "source"),
                  (predictions / "readback.json", "source"),
                  (evaluation_code, "source"), (root / "preparation.json", "source"),
                  (root / "query-metadata.parquet", "cohort")]
    # Explicit compact evidence only: never recurse into official/ or raw/.
    for name in ("inference-final-verification.json", "training-provenance.json",
                 "calibration-race-census.csv", "official/data-freeze.json",
                 "official/coverage_manifest.json", "official/history-race-coverage.json",
                 "official/history_coverage_manifest.json", "official/calibration-enrichment-coverage.json",
                 "official/season-enrichment-coverage.json", "official/dividend_units.json",
                 "official/odds-unit-semantics.json", "official/odds-quotes.json",
                 "official/odds-semantics-archived-quotes.json",
                 "official/odds-availability.json", "official/odds-semantics-source-manifest.json"):
        if (root / name).is_file():
            artifacts.append((root / name, "evidence"))
    blend_outputs = tuple(evaluation / name for name in
                          ("tomorrow-market-blends.csv", "tomorrow-market-blend-combinations.csv"))
    blend_provenance = evaluation / "tomorrow-market-blend-provenance.json"
    if any(path.is_file() for path in (*blend_outputs, blend_provenance)):
        if not all(path.is_file() for path in (*blend_outputs, blend_provenance)):
            raise ValueError("Incomplete optional market-blend forecast bundle")
        if read_json(blend_provenance).get("ready_for_report") is not True:
            raise ValueError("Market-blend forecast is not ready_for_report")
        blend_sources = (evaluation / "summary.json", evaluation / "calibration.json",
                         evaluation / "provenance.json", evaluation_code, predictions / "readback.json")
        validate_forecast_provenance(blend_provenance, blend_outputs, input_hash, blend_sources)
        artifacts.extend((path, "forecast") for path in (*blend_outputs, blend_provenance))
    for path in forecast_artifacts:
        if path.suffix not in {".json", ".csv"} or not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Expected a compact forecast JSON/CSV: {path}")
        artifacts.append((path, "forecast"))
    if forecast_artifacts:
        if forecast_provenance is None:
            raise ValueError("Optional combined/ranked forecasts require --forecast-provenance")
        validate_forecast_provenance(forecast_provenance, forecast_artifacts, input_hash)
        artifacts.append((forecast_provenance, "forecast"))
    for path in report_artifacts:
        if path.suffix not in {".json", ".csv", ".md"} or not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Expected a compact report companion: {path}")
        artifacts.append((path, "report"))
    destinations = [f"{dest}/{path.name}" for path, dest in artifacts]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Duplicate artifact destination")
    if include_query:
        artifacts.append((root / "query-features.parquet", "source"))
    children = {}
    for family in FAMILIES:
        model = receipt["models"][family]
        candidate = candidates["candidates"][family]
        if model["package_id"] != candidate["package_id"]:
            raise ValueError(f"Package identity mismatch: {family}")
        if sha256(predictions / f"{family}.csv") != model["output_sha256"]:
            raise ValueError(f"Prediction hash mismatch: {family}")
        if not math.isclose(model["probability_sum_min"], 1, abs_tol=1e-6) or not math.isclose(
                model["probability_sum_max"], 1, abs_tol=1e-6):
            raise ValueError(f"Invalid probability receipt: {family}")
        artifacts.append((evaluation / f"{family}-tomorrow-runners.csv", "forecast"))
        if package_root is not None:
            for name, digest in candidate["hashes"].items():
                path = (package_root / family / name).resolve()
                if not path.is_relative_to((package_root / family).resolve()) or sha256(path) != digest:
                    raise ValueError(f"Package artifact hash mismatch: {family}/{name}")
                artifacts.append((path, f"packages/{family}"))
    for variant in VARIANTS:
        scores = summary[variant]
        if scores["stake_hkd"] != 6230 or set(scores["by_pool"]) != POOLS:
            raise ValueError(f"Expected HKD6230 across eight pools: {variant}")
        for metrics in (scores, *scores["by_pool"].values()):
            if not math.isclose(metrics["gross_hkd"] - metrics["stake_hkd"], metrics["profit_hkd"], abs_tol=1e-6):
                raise ValueError(f"Gross/net mismatch: {variant}")
            if not math.isclose(metrics["profit_hkd"] / metrics["stake_hkd"], metrics["roi"], abs_tol=1e-9):
                raise ValueError(f"ROI mismatch: {variant}")
        for key in ("stake_hkd", "gross_hkd", "profit_hkd", "nsettled", "nmissing"):
            if not math.isclose(sum(p[key] for p in scores["by_pool"].values()), scores[key], abs_tol=1e-6):
                raise ValueError(f"Pool totals mismatch: {variant}/{key}")
        if scores["nsettled"] != 623 or scores["nmissing"] != 1:
            raise ValueError(f"Settlement population mismatch: {variant}")
        family = variant.removesuffix("_market_blend_hindsight")
        market = family not in FAMILIES
        cal = calibration["market_final_odds_hindsight" if market else family]
        params = {"family": "market" if market else family,
                  "model_kind": "inverse_final_win_odds" if market else "unknown",
                  "feature_schema": "not_applicable" if market else
                                    candidates["candidates"][family]["recipe"]["feature_schema"],
                  "target_kind": "win_probability" if market else
                                 candidates["candidates"][family]["recipe"]["target"]["kind"],
                  "calibration": json.dumps(cal, sort_keys=True),
                  "calibration_application": ("temperature_only" if variant == "market_calibrated_hindsight" else
                                              "market_blend_only" if variant.endswith("_market_blend_hindsight") else
                                              "none"),
                  "price_basis": scores["price_basis"]}
        if not market:
            candidate = candidates["candidates"][family]
            params.update(model_identity(candidate["recipe"]))
            params.update(package_id=candidate["package_id"],
                          recipe=json.dumps(candidate["recipe"], sort_keys=True),
                          training_cutoff=str(candidate.get("training_cutoff", "unknown")),
                          fit_date_max=str(candidate.get("fit_date_max", "unknown")))
        metrics = flatten_numeric_metrics(scores)
        metrics.update(flatten_numeric_metrics(cal, "calibration"))
        interval = scores.get("meeting_cluster_bootstrap_roi_interval_95")
        if interval is not None:
            metrics.update(roi_ci95_lower=interval[0], roi_ci95_upper=interval[1])
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Non-finite metric: {variant}")
        ledgers = [(evaluation / f"{variant}-{suffix}.csv", "ledgers")
                   for suffix in ("tickets", "bankroll", "meetings", "runners")]
        ev = evaluation / f"{variant}-hindsight-win-ev.csv"
        if ev.is_file():
            ledgers.append((ev, "ledgers"))
        children[variant] = {"metrics": metrics, "params": params, "artifacts": ledgers}
    # Bind every uploaded file, plus the query and prediction hashes even when not uploaded.
    inventory = {f"{dest}/{path.name}": sha256(path) for path, dest in artifacts}
    for variant, child in children.items():
        inventory.update({f"{variant}/{dest}/{path.name}": sha256(path)
                          for path, dest in child["artifacts"]})
    inventory["query_features_sha256"] = input_hash
    inventory["prediction_outputs"] = {f: receipt["models"][f]["output_sha256"] for f in FAMILIES}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    validation_files = [root / "query-features.parquet", *(predictions / f"{f}.csv" for f in FAMILIES)]
    return {"input_hash": input_hash, "evaluation_hash": digest, "generation": generation,
            "children": children, "artifacts": artifacts, "inventory": inventory,
            "validation_files": [(path, sha256(path)) for path in validation_files]}


def log_evaluation(plan: dict, client, *, artifact_location: str | None = None) -> dict:
    """Only create new runs. Existing complete identities are reused, never edited.

    Run a single writer: MLflow search/create is not an atomic uniqueness constraint.
    """
    from mlflow.entities import Metric, Param

    # Reject a source that changed between planning and upload before creating an experiment.
    for path, digest in plan["validation_files"]:
        if sha256(path) != digest:
            raise ValueError("Inference source changed after validation")
    for path, dest in plan["artifacts"]:
        if sha256(path) != plan["inventory"][f"{dest}/{path.name}"]:
            raise ValueError("Artifact changed after validation")
    for name, child in plan["children"].items():
        for path, dest in child["artifacts"]:
            if sha256(path) != plan["inventory"][f"{name}/{dest}/{path.name}"]:
                raise ValueError("Ledger changed after validation")
    experiment = client.get_experiment_by_name(EXPERIMENT)
    eid = experiment.experiment_id if experiment else client.create_experiment(
        EXPERIMENT, artifact_location=artifact_location)
    tags = {"ima.input_sha256": plan["input_hash"], "ima.evaluation_sha256": plan["evaluation_hash"],
            "ima.generation": plan["generation"], "ima.paper_only": "true",
            "ima.non_agentic": "true", "ima.study": "season_readiness",
            "ima.profit_claim": "retrospective_paper_not_executable"}

    def create(name, payload, parent=None):
        identity = f"{plan['generation']}/{name}"
        query = " AND ".join(f"tags.`{k}` = '{v}'" for k, v in
                             {**{k: tags[k] for k in ("ima.input_sha256", "ima.evaluation_sha256")},
                              "ima.run_identity": identity}.items())
        matches = client.search_runs([eid], filter_string=query, max_results=2)
        if matches:
            if len(matches) != 1 or matches[0].info.status != "FINISHED":
                raise ValueError(f"Duplicate/incomplete run identity: {identity}; inspect without mutating")
            return matches[0].info.run_id
        run_tags = {**tags, "ima.run_identity": identity, "mlflow.runName": identity}
        if parent:
            run_tags["mlflow.parentRunId"] = parent
            run_tags["ima.family"] = payload["params"]["family"]
            run_tags["ima.model_kind"] = payload["params"]["model_kind"]
            run_tags["ima.variant"] = name
        run_id = client.create_run(eid, tags=run_tags).info.run_id
        try:
            now = int(time.time() * 1000)
            client.log_batch(run_id, metrics=[Metric(k, v, now, 0) for k, v in payload.get("metrics", {}).items()],
                             params=[Param(k, str(v)) for k, v in payload.get("params", {}).items()])
            for path, dest in payload["artifacts"]:
                client.log_artifact(run_id, str(path), artifact_path=dest)
            if not parent:
                client.log_dict(run_id, plan["inventory"], "source/datafreeze.json")
            client.set_terminated(run_id, status="FINISHED")
        except Exception:
            client.set_terminated(run_id, status="FAILED")
            raise
        return run_id

    parent = create("season-and-tomorrow", {"artifacts": plan["artifacts"]})
    children = {name: create(name, payload, parent) for name, payload in plan["children"].items()}
    return {"experiment": EXPERIMENT, "parent_run_id": parent, "children": children,
            "input_sha256": plan["input_hash"], "evaluation_sha256": plan["evaluation_hash"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "evaluation-dir", "predictions-dir", "manifest", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--evaluation-code", type=Path, default=Path("scripts/evaluate_season_2026.py"))
    parser.add_argument("--generation", choices=("fresh", "old-benchmark"), default="fresh")
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--artifact-location", help="Optional shared/server-accessible artifact root")
    parser.add_argument("--package-root", type=Path, help="Optional four package directories; hash-check bytes only")
    parser.add_argument("--include-query", action="store_true")
    parser.add_argument("--forecast-artifact", type=Path, action="append", default=[],
                        help="Repeat for compact market-blend/combo provenance and ranked forecast files")
    parser.add_argument("--forecast-provenance", type=Path,
                        help="Combined forecast proof: query hash and output_sha256 for each optional file")
    parser.add_argument("--report-artifact", type=Path, action="append", default=[],
                        help="Repeat for final report companions, hash-bound but not claimed in forecast provenance")
    parser.add_argument("--log", action="store_true", help="Explicitly authorize writes after final parent approval")
    args = parser.parse_args()
    plan = prepare_logging(args.root, args.evaluation_dir, args.predictions_dir, args.manifest,
                           args.report, args.evaluation_code, generation=args.generation,
                           include_query=args.include_query, package_root=args.package_root,
                           forecast_artifacts=tuple(args.forecast_artifact),
                           forecast_provenance=args.forecast_provenance,
                           report_artifacts=tuple(args.report_artifact))
    if args.log:
        from mlflow.tracking import MlflowClient
        result = log_evaluation(plan, MlflowClient(tracking_uri=args.tracking_uri),
                                artifact_location=args.artifact_location)
    else:
        result = {"dry_run": True, "experiment": EXPERIMENT, "variants": list(plan["children"]),
                  "input_sha256": plan["input_hash"], "evaluation_sha256": plan["evaluation_hash"],
                  "artifact_bytes": sum(path.stat().st_size for path, _ in plan["artifacts"]) +
                  sum(path.stat().st_size for child in plan["children"].values() for path, _ in child["artifacts"])}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
