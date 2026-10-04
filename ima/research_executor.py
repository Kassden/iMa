"""Execute validated research recipes against protected temporal folds."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .feature_sets import FEATURE_SCHEMAS, FeatureSchema, drop_feature_families
from .modeling import MarketBlend, RaceProbabilityModel, TemperatureCalibrator, normalize_by_race
from .research_evaluation import (
    ProtocolManifest,
    baseline_probabilities,
    evaluate_research_probabilities,
    make_protocol_manifest,
    select_fold,
)
from .research_model_package import FeatureReplayContext, ResearchModelPackage, speed_to_finish_seconds
from .research_models import ResearchClassifier, ResearchRegressor, secondary_target_diagnostics
from .research_specs import PerformanceDistributionSpec, PipelineRecipe, is_fundamental_first_portfolio
from .research_targets import apply_target_contract, target_contract
from .research_transforms import (
    FittedResearchTransforms,
    TransformSpec as RuntimeTransformSpec,
)


RESULT_SCHEMA_VERSION = 1
METRIC_CONTRACT_VERSION = 2

LOWER_IS_BETTER_METRICS = {
    "binary_log_loss",
    "brier",
    "ece",
    "mae",
    "invalid_prediction_rows",
    "invalid_label_rows",
    "mean_winner_rank",
    "race_binary_log_loss",
    "race_brier",
    "race_log_loss",
    "race_mae",
    "race_rmse",
    "rmse",
    "winner_rank",
    "worst_race_brier",
    "worst_race_log_loss",
}


@dataclass(frozen=True)
class RecipeExecutionRequest:
    attempt_id: str
    proposal_id: str
    trial_number: int
    recipe: PipelineRecipe
    dataset_path: Path
    output_dir: Path
    protocol_parameters: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str | None = None
    code_revision: str = "unknown"
    environment_hash: str = "unknown"
    portfolio_version: str | None = None


@dataclass(frozen=True)
class RecipeExecutionResult:
    schema_version: int
    attempt_id: str
    proposal_id: str
    trial_number: int
    recipe_hash: str
    target_kind: str
    status: str
    objective_name: str
    objective_value: float | None
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    lineage: dict[str, str]
    duration_seconds: float
    error: str | None = None

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FittedWinRecipeModel:
    model: RaceProbabilityModel
    transforms: FittedResearchTransforms
    calibrator: TemperatureCalibrator | None
    blend: MarketBlend | None
    feature_schema: FeatureSchema
    residual_history: Any = None

    def predict_fundamental_proba(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transforms.transform(self.residual_history.transform(frame) if self.residual_history else frame)
        transformed = _ensure_feature_columns(transformed, self.feature_schema)
        probabilities = self.model.predict_proba(transformed)
        if self.calibrator is not None:
            probabilities = self.calibrator.transform(probabilities, transformed["race_id"])
        return probabilities

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_fundamental_proba(frame)
        if self.blend is not None:
            probabilities = self.blend.transform(
                probabilities,
                frame["market_probability"].to_numpy(dtype=float),
                frame["race_id"],
            )
        return probabilities


@dataclass
class FittedSecondaryRecipeModel:
    model: ResearchClassifier | ResearchRegressor
    transforms: FittedResearchTransforms
    feature_schema: FeatureSchema
    calibrator: IsotonicRegression | None = None
    win_calibrator: TemperatureCalibrator | None = None
    residual_history: Any = None

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transforms.transform(self.residual_history.transform(frame) if self.residual_history else frame)
        transformed = _ensure_feature_columns(transformed, self.feature_schema)
        predictions = self.model.predict(transformed)
        if self.calibrator is not None:
            predictions = self.calibrator.predict(predictions)
        return np.asarray(predictions, dtype=float)

    def predict_win_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_distribution"):
            from .probabilistic_adapters import distribution_to_win
            return distribution_to_win(self.predict_distribution(frame), frame["race_id"])
        if self.win_calibrator is None:
            raise TypeError("This secondary model has no calibrated win-probability endpoint")
        transformed = self.transforms.transform(self.residual_history.transform(frame) if self.residual_history else frame)
        transformed = _ensure_feature_columns(transformed, self.feature_schema)
        raw = _ranking_scores_to_probabilities(
            self.model.predict(transformed), transformed["race_id"]
        )
        return self.win_calibrator.transform(raw, transformed["race_id"])

    def predict_distribution(self, frame):
        if not hasattr(self.model, "predict_distribution"):
            raise TypeError("This secondary model has no performance distribution")
        transformed = self.transforms.transform(self.residual_history.transform(frame) if self.residual_history else frame)
        return self.model.predict_distribution(_ensure_feature_columns(transformed, self.feature_schema))


@dataclass
class FittedGraphRecipeModel:
    model: Any
    transforms: FittedResearchTransforms
    feature_schema: FeatureSchema
    residual_history: Any = None

    def _frame(self, frame):
        return _ensure_feature_columns(self.transforms.transform(frame), self.feature_schema)

    def predict_fundamental_proba(self, frame):
        return self.model.predict_fundamental_proba(self._frame(frame))

    def predict_proba(self, frame):
        return self.model.predict_proba(self._frame(frame))

    def predict(self, frame):
        return self.model.predict(self._frame(frame))

    def predict_distribution(self, frame):
        return self.model.predict_distribution(self._frame(frame))

    def predict_joint(self, frame):
        return self.model.predict_joint(self._frame(frame))


def execute_recipe(request: RecipeExecutionRequest) -> RecipeExecutionResult:
    """Train/evaluate one recipe and atomically persist its accepted artifacts."""
    started = time.perf_counter()
    try:
        _write_json_atomic(request.output_dir/"stage.json",{"phase":"loading_dataset","started_at_epoch":time.time()})
        frame = _load_dataset(request.dataset_path)
        observed_hash = _hash_file(request.dataset_path)
        if request.dataset_hash is not None and request.dataset_hash != observed_hash:
            raise ValueError("dataset hash does not match execution request")
        if request.recipe.schema_version == 3:
            frame = _v6_feature_frame(frame, request)
        if request.recipe.feature_discovery:
            from .feature_program import materialize
            _write_json_atomic(request.output_dir/"stage.json",{"phase":"building_features","discovery_id":request.recipe.feature_discovery.discovery_id(),"started_at_epoch":time.time()})
            matrix, manifest = materialize(frame, request.recipe.feature_discovery, observed_hash,
                request.output_dir.parent.parent / "discovery-cache", shared=request.recipe.schema_version == 3,
                input_metadata=frame.attrs.get("synthesis_metadata",frame.attrs.get("input_metadata")))
            if request.recipe.schema_version == 3:
                frame.attrs["discovery_matrix"] = matrix
            else:
                frame = matrix
            _write_json_atomic(request.output_dir / "discovery-manifest.json", manifest)
        _write_json_atomic(request.output_dir/"stage.json",{"phase":"training","started_at_epoch":time.time()})
        with ExitStack() as pins:
            result = _execute_frame(request, frame, observed_hash, started, pins=pins)
        if request.recipe.schema_version == 3:
            result.lineage.update(_v6_result_lineage(request, frame, result, observed_hash))
        if request.recipe.feature_discovery:
            result.artifacts["discovery_manifest"] = str(request.output_dir / "discovery-manifest.json")
            result.lineage["matrix_id"] = manifest["matrix_id"]
            result.lineage["discovery_id"] = manifest["discovery_id"]
        _write_json_atomic(request.output_dir / "result.json", result.serializable())
    except Exception as exc:
        result = RecipeExecutionResult(
            schema_version=RESULT_SCHEMA_VERSION,
            attempt_id=request.attempt_id,
            proposal_id=request.proposal_id,
            trial_number=request.trial_number,
            recipe_hash=request.recipe.recipe_hash(),
            target_kind=request.recipe.target.kind,
            status="failed",
            objective_name=_objective_name(request.recipe.target.kind),
            objective_value=None,
            metrics={},
            artifacts={},
            lineage={
                "dataset_hash": request.dataset_hash or "unknown",
                "code_revision": request.code_revision,
                "environment_hash": request.environment_hash,
            },
            duration_seconds=round(time.perf_counter() - started, 6),
            error=f"{type(exc).__name__}: {exc}",
        )
        _write_json_atomic(request.output_dir / "result.json", result.serializable())
    return result


def _v6_result_lineage(request, frame, result, dataset_hash):
    manifest = frame.attrs.get("verified_dataset_manifest", {})
    comparison = manifest.get("comparison_contract", {})
    manifest_path = Path(request.dataset_path).parent / "manifest.json"
    manifest_hash = frame.attrs.get("dataset_manifest_hash")
    if (manifest_path.exists() and _hash_file(manifest_path) != manifest_hash) or (
        manifest_hash is not None and not manifest_path.exists()
    ):
        raise ValueError("Dataset manifest changed during evaluation")
    if _hash_file(request.dataset_path) != dataset_hash:
        raise ValueError("Dataset changed during evaluation")
    predictions = pd.read_csv(result.artifacts["predictions"], dtype={
        "fold_id": str, "race_id": str, "horse_no": str})
    columns = ["fold_id", "race_id", "horse_no", "date"]
    label = "target_win" if request.recipe.target.kind == "win_probability" else "label"
    population = predictions[columns + [label]].sort_values(columns, kind="stable")
    if population.duplicated(["fold_id", "race_id", "horse_no"]).any():
        raise ValueError("Evaluation population contains duplicate fold/runner keys")
    population_hash = hashlib.sha256(population.to_json(orient="records", date_format="iso").encode()).hexdigest()
    population_id = "evaluated-" + population_hash
    population_path = request.output_dir / "evaluation-population.json"
    _write_json_atomic(population_path, {
        "schema_version": 1, "evaluation_population_id": population_id,
        "declared_evaluation_population_id": manifest.get("evaluation_population_id", comparison.get("evaluation_population_id")),
        "protocol_id": result.lineage["protocol_id"], "identity_columns": columns + [label],
        "rows": json.loads(population.to_json(orient="records", date_format="iso")),
        "policy": "actual scored fold/runner keys and target labels after target eligibility exclusions",
    })
    result.artifacts["evaluation_population"] = str(population_path)
    units = {"win_probability": "1", "placing_top_k": "1", "ranking_strength": "1",
             "adjusted_finish_time_or_speed": "m/s", "market_odds_forecast": "1",
             "recorded_final_win_odds": "log_decimal_odds"}
    unit = units[request.recipe.target.kind]
    declared_unit = manifest.get("target_units", {}).get(request.recipe.target.kind, manifest.get("target_unit"))
    if declared_unit is not None and declared_unit != unit:
        raise ValueError("Dataset manifest target unit disagrees with the executed target contract")
    if request.recipe.target.kind != "win_probability":
        basis = "not_applicable"
    elif result.metrics.get("objective_source") == "model":
        basis = "fundamental"
    else:
        market_output = request.recipe.blend.kind != "none"
        if request.recipe.pipeline_graph:
            from .pipeline_graph import PipelineGraph
            graph = PipelineGraph.from_dict(request.recipe.pipeline_graph)
            market_output = next(n.output.market for n in graph.nodes if n.node_id == graph.output_node_id)
        basis = "market_blended" if market_output else "fundamental"
    declared_basis = manifest.get("probability_basis", comparison.get("probability_basis"))
    if basis == "fundamental" and declared_basis in {"fundamental", "fundamental_pre_race"}:
        basis = declared_basis
    catalog_hash = hashlib.sha256(json.dumps(frame.attrs["input_metadata"],
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    catalog_id = manifest.get("predictor_catalog_id", manifest.get("catalog_id")) or "catalog-" + catalog_hash
    lineage = {
        "dataset_id": str(manifest.get("dataset_id") or "synthetic-" + dataset_hash),
        "dataset_manifest_hash": manifest_hash or "absent",
        "predictor_catalog_id": str(catalog_id), "catalog_id": str(catalog_id),
        "predictor_catalog_hash": catalog_hash,
        "availability_policy": (frame.attrs["availability_policy"] if isinstance(frame.attrs["availability_policy"], str)
                                else json.dumps(frame.attrs["availability_policy"], sort_keys=True, separators=(",", ":"))),
        "evaluation_population_id": population_id, "evaluation_population_hash": population_hash,
        "protocol_hash": _hash_file(Path(result.artifacts["protocol"])),
        "prediction_sha256": _hash_file(Path(result.artifacts["predictions"])),
        "target_unit": unit, "probability_basis": basis,
    }
    declared_population = manifest.get("evaluation_population_id", comparison.get("evaluation_population_id"))
    if declared_population:
        lineage["declared_evaluation_population_id"] = str(declared_population)
    return lineage


def _execute_frame(
    request: RecipeExecutionRequest,
    frame: pd.DataFrame,
    dataset_hash: str,
    started: float,
    *, pins=None,
) -> RecipeExecutionResult:
    recipe = request.recipe
    contract = target_contract(recipe.target.kind, recipe.target.parameters)
    exclusions: dict[str, Any] = {}
    if recipe.target.kind == "win_probability":
        frame, exclusions = _exclude_non_single_winner_races(frame)
    labelled = apply_target_contract(frame, contract)
    if recipe.pipeline_graph is not None:
        from .pipeline_graph import PipelineGraph
        before = labelled.race_id.nunique()
        labelled = _graph_target_labels(labelled, PipelineGraph.from_dict(recipe.pipeline_graph))
        exclusions["graph_untimed_or_unlabelled_races"] = int(before-labelled.race_id.nunique())
        labelled.attrs["dataset_exclusions"] = exclusions
    schema = _effective_schema(recipe)
    if recipe.schema_version == 3:
        schema = FeatureSchema(schema.name, tuple(dict.fromkeys((*schema.numeric,
            *(recipe.extra_numeric_features or ()), *frame.attrs.get("formula_features", ())))), schema.categorical)
        ineligible = set(frame.attrs.get("ineligible_predictors", ()))
        schema = FeatureSchema(schema.name, tuple(c for c in schema.numeric if c not in ineligible),
                               tuple(c for c in schema.categorical if c not in ineligible))
    labelled = _ensure_feature_columns(labelled, schema)
    protocol_parameters = dict(request.protocol_parameters)
    if recipe.schema_version == 3:
        protocol_parameters.setdefault("whole_meeting_boundaries", True)
    protocol = make_protocol_manifest(
        labelled,
        target=contract,
        **protocol_parameters,
    )
    if recipe.pipeline_graph is not None:
        return _execute_graph_frame(request, labelled, schema, protocol, dataset_hash, started, pins=pins)
    if recipe.target.kind != "win_probability":
        return _execute_secondary_frame(
            request, labelled, schema, protocol, dataset_hash, started, pins=pins
        )

    folds: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    final_model: FittedWinRecipeModel | None = None
    effective_training: list[dict[str, Any]] = []
    for fold in protocol.folds:
        train = select_fold(labelled, fold.train_race_ids)
        calibration = select_fold(labelled, fold.calibration_race_ids)
        score = select_fold(labelled, fold.score_race_ids)
        train = _apply_training_window(train, calibration, recipe.train_window)
        transformed_train, transformed_calibration, transformed_score, fitted_transforms, fold_schema, residual_history = _prepared_fold(
            request,train,calibration,score,schema,fold.fold_id,"target_win",pins=pins)
        if recipe.model.kind == "gaussian_probit":
            from .performance_probit import fit_probit
            model = fit_probit(transformed_train,fold_schema,recipe.model.parameters)
        else:
            model = RaceProbabilityModel(
                kind=recipe.model.kind, parameters=dict(recipe.model.parameters),
                random_state=recipe.seed, feature_schema=fold_schema,
            ).fit(transformed_train)
        calibration_probabilities = model.predict_proba(transformed_calibration)
        calibrator = None
        if recipe.calibration.kind == "temperature":
            calibrator = TemperatureCalibrator.fit(
                calibration_probabilities, transformed_calibration
            )
            calibration_probabilities = calibrator.transform(
                calibration_probabilities, transformed_calibration["race_id"]
            )
        blend = None
        if recipe.blend.kind == "market_softmax":
            blend = MarketBlend.fit(
                calibration_probabilities,
                transformed_calibration["market_probability"].to_numpy(dtype=float),
                transformed_calibration,
            )
        score_probabilities = model.predict_proba(transformed_score)
        if calibrator is not None:
            score_probabilities = calibrator.transform(
                score_probabilities, transformed_score["race_id"]
            )
        standalone = evaluate_research_probabilities(
            score_probabilities, transformed_score, label="model"
        )
        selected_probabilities = score_probabilities
        if blend is not None:
            selected_probabilities = blend.transform(
                score_probabilities,
                transformed_score["market_probability"].to_numpy(dtype=float),
                transformed_score["race_id"],
            )
        selected = evaluate_research_probabilities(
            selected_probabilities, transformed_score, label="selected"
        )
        market = evaluate_research_probabilities(
            baseline_probabilities(transformed_score, "raw_market"),
            transformed_score,
            label="raw_market",
        )
        market_calibrator = TemperatureCalibrator.fit(
            transformed_calibration["market_probability"].to_numpy(dtype=float),
            transformed_calibration,
        )
        calibrated_market_probabilities = market_calibrator.transform(
            transformed_score["market_probability"].to_numpy(dtype=float),
            transformed_score["race_id"],
        )
        calibrated_market = evaluate_research_probabilities(
            calibrated_market_probabilities, transformed_score, label="calibrated_market"
        )
        uniform = evaluate_research_probabilities(
            baseline_probabilities(transformed_score, "uniform"),
            transformed_score,
            label="uniform",
        )
        folds.append({
            "fold_id": fold.fold_id,
            "model": standalone["metrics"],
            "selected": selected["metrics"],
            "raw_market": market["metrics"],
            "calibrated_market": calibrated_market["metrics"],
            "uniform": uniform["metrics"],
            "selected_minus_market": (
                selected["race_weighted_log_loss"] - market["race_weighted_log_loss"]
            ),
            "selected_minus_calibrated_market": (
                selected["race_weighted_log_loss"] - calibrated_market["race_weighted_log_loss"]
            ),
            "market_temperature": market_calibrator.temperature,
            "blend_fundamental_weight": blend.fundamental_weight if blend else None,
            "blend_market_weight": blend.market_weight if blend else None,
        })
        predictions = transformed_score[["race_id", "date", "race_no", "horse_no", "target_win"]].copy()
        predictions["fold_id"] = fold.fold_id
        predictions["model_probability"] = score_probabilities
        predictions["selected_probability"] = selected_probabilities
        predictions["market_probability"] = transformed_score["market_probability"].to_numpy()
        predictions["calibrated_market_probability"] = calibrated_market_probabilities
        prediction_rows.append(predictions)
        effective_training.append({
            "fold_id": fold.fold_id,
            "rows": int(len(transformed_train)),
            "races": int(transformed_train["race_id"].nunique()),
            "first_date": str(pd.to_datetime(transformed_train["date"]).min().date()),
            "last_date": str(pd.to_datetime(transformed_train["date"]).max().date()),
            "features": list(fold_schema.features),
            "available_features": [
                column for column in fold_schema.features
                if transformed_train[column].notna().any()
            ],
            "unavailable_features": [
                column for column in fold_schema.features
                if not transformed_train[column].notna().any()
            ],
            "clip_bounds": {
                key: list(value) for key, value in fitted_transforms.clip_bounds.items()
            },
        })
        final_model = FittedWinRecipeModel(
            model, fitted_transforms, calibrator, blend, fold_schema, residual_history
        )
        _discovery_diagnostics(final_model,transformed_train,
            transformed_calibration if recipe.schema_version == 3 else calibration,
            transformed_score if recipe.schema_version == 3 else score, recipe,request.output_dir,fold.fold_id,
            prepared=recipe.schema_version == 3)

    if not folds or final_model is None:
        raise ValueError("protocol produced no executable folds")
    fundamental_first = is_fundamental_first_portfolio(request.portfolio_version)
    objective_source = "model" if fundamental_first else "selected"
    objective = float(np.mean([row[objective_source]["race_log_loss"] for row in folds]))
    metrics = {
        "objective": objective,
        "objective_source": objective_source,
        "folds": folds,
        "summary": _aggregate_fold_metrics(folds),
        "metric_contract_version": 3 if fundamental_first else METRIC_CONTRACT_VERSION,
        "metric_directions": _metric_directions(folds),
        "effective_training": effective_training,
        "dataset_exclusions": exclusions,
        "mean_selected_minus_market": float(np.mean([
            row["selected_minus_market"] for row in folds
        ])),
        "mean_selected_minus_calibrated_market": float(np.mean([
            row["selected_minus_calibrated_market"] for row in folds
        ])),
        "zero_fundamental_weight_fold_fraction": float(np.mean([
            row["blend_fundamental_weight"] is not None
            and row["blend_fundamental_weight"] <= 1e-6 for row in folds
        ])),
    }
    request.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = request.output_dir / "protocol.json"
    predictions_path = request.output_dir / "predictions.csv"
    package_path = request.output_dir / "package"
    _write_json_atomic(protocol_path, protocol.to_dict())
    _write_csv_atomic(predictions_path, pd.concat(prediction_rows, ignore_index=True))
    package = ResearchModelPackage(
        final_model,
        recipe,
        protocol_id=protocol.protocol_id,
        code_revision=request.code_revision,
        portfolio_version=request.portfolio_version,
        feature_context=_feature_replay_context(request, fold_schema, score, dataset_hash=dataset_hash) if recipe.schema_version == 3 else None,
    )
    package.save(package_path)
    conditional = getattr(final_model.model,"conditional",None)
    if conditional is not None and conditional.preprocessor is not None:
        weights = pd.DataFrame({
            "feature": conditional.preprocessor.get_feature_names_out(),
            "coefficient": conditional.coefficients,
        })
        weights.to_csv(package_path / "feature-weights.csv", index=False)
    feature_program_id = recipe.feature_program_id(dataset_hash)
    feature_program_path = request.output_dir / "feature-program.json"
    _write_json_atomic(feature_program_path, {
        "feature_program_id": feature_program_id,
        "dataset_hash": dataset_hash,
        "feature_schema": recipe.feature_schema,
        "drop_feature_families": list(recipe.drop_feature_families),
        "transforms": [spec.model_dump(mode="json") for spec in recipe.transforms],
        "train_window": recipe.train_window,
        "effective_training": effective_training,
    })
    result = RecipeExecutionResult(
        schema_version=RESULT_SCHEMA_VERSION,
        attempt_id=request.attempt_id,
        proposal_id=request.proposal_id,
        trial_number=request.trial_number,
        recipe_hash=recipe.recipe_hash(),
        target_kind=recipe.target.kind,
        status="completed",
        objective_name=(
            "development_fundamental_race_log_loss"
            if fundamental_first else _objective_name(recipe.target.kind)
        ),
        objective_value=objective,
        metrics=metrics,
        artifacts={
            "protocol": str(protocol_path),
            "predictions": str(predictions_path),
            "package": str(package_path),
            "feature_program": str(feature_program_path),
        },
        lineage={
            "dataset_hash": dataset_hash,
            "feature_program_id": feature_program_id,
            "protocol_id": protocol.protocol_id,
            "code_revision": request.code_revision,
            "environment_hash": request.environment_hash,
        },
        duration_seconds=round(time.perf_counter() - started, 6),
    )
    _write_json_atomic(request.output_dir / "result.json", result.serializable())
    return result


def _exclude_non_single_winner_races(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "race_id" not in frame or "target_win" not in frame:
        return frame, {}
    winners = pd.to_numeric(frame["target_win"], errors="coerce").groupby(
        frame["race_id"]
    ).sum(min_count=1)
    excluded = winners[~np.isclose(winners.to_numpy(dtype=float), 1.0)].index
    if len(excluded) == 0:
        return frame, {"non_single_winner_races": 0, "excluded_rows": 0}
    keep = ~frame["race_id"].isin(excluded)
    return frame.loc[keep].copy(), {
        "non_single_winner_races": int(len(excluded)),
        "excluded_rows": int((~keep).sum()),
        "policy": "exclude_dead_heats_and_missing_winners_for_win_probability_v1",
    }


def _fit_distribution_recipe(recipe, train, calibration, schema, label):
    from .probabilistic_adapters import fit_performance_distribution
    specification = PerformanceDistributionSpec.model_validate(recipe.performance_distribution).runtime_payload()
    if specification["kind"] == "catboost_uncertainty":
        specification["parameters"] = dict(specification.get("parameters", {})) | dict(recipe.model.parameters)
    return fit_performance_distribution(specification, train, calibration, schema,
        label_column=label, seed=recipe.seed, model_spec=recipe.model)


def _graph_target_labels(frame, graph):
    labels = {node.output.target for node in graph.nodes if node.kind == "estimator"
              and node.output.kind in {"point_prediction", "ranking_score", "performance_distribution"}}
    unknown = labels-{"target_rank_score", "target_speed", "target_finish_seconds"}
    if unknown:
        raise ValueError(f"Graph ancestor labels lack a registered target contract: {sorted(unknown)}")
    labelled = frame
    if "target_rank_score" in labels:
        labelled = apply_target_contract(labelled, target_contract("ranking_strength"))
    if labels & {"target_speed", "target_finish_seconds"}:
        labelled = apply_target_contract(labelled, target_contract("adjusted_finish_time_or_speed"))
        if "target_finish_seconds" in labels:
            from .data import _finish_time_seconds
            seconds = pd.to_numeric(labelled.finish_seconds,errors="coerce") if "finish_seconds" in labelled else _finish_time_seconds(labelled.finish_time)
            labelled["target_finish_seconds"] = seconds
    return labelled


def _attach_discovery(frame, columns=None):
    descriptor = frame.attrs.get("discovery_matrix")
    if descriptor is None:
        return frame
    names = descriptor.feature_names if columns is None else tuple(c for c in columns if c in descriptor.feature_names)
    missing = tuple(c for c in names if c not in frame)
    if not missing:
        return frame
    keys = descriptor.manifest()["runner_keys"]
    index = {(str(row["race_id"]), str(row["horse_no"])): i for i, row in enumerate(keys)}
    try:
        positions = [index[(str(r), str(h))] for r, h in zip(frame.race_id, frame.horse_no)]
    except KeyError as exc:
        raise ValueError("Shared discovery artifact does not cover these runner keys") from exc
    generated = descriptor.frame(columns=missing, positions=positions)
    output = pd.concat([frame.reset_index(drop=True), generated], axis=1)
    output.attrs = dict(frame.attrs)
    return output


@dataclass
class GraphComponentRecipeModel:
    """A graph ancestor owns selection/transforms fitted only on its own past rows."""

    recipe: PipelineRecipe
    schema: FeatureSchema
    model_kind: str
    parameters: dict
    output_kind: str
    label: str
    output_dir: Path
    component_id: str
    model: Any = None
    transforms: Any = None
    feature_schema: Any = None
    selected_schema: Any = None
    residual_history: Any = None

    def fit(self, frame):
        left = frame
        if self.recipe.feature_discovery and self.recipe.feature_discovery.adjusted_speed_residuals:
            from .feature_residuals import AdjustedSpeedHistory
            self.residual_history = AdjustedSpeedHistory.fit(frame, self.recipe.feature_discovery,
                pd.to_datetime(frame.date).max().normalize()+pd.Timedelta(days=1))
            left = self.residual_history.transform(left)
        left = _attach_discovery(left)
        identity = hashlib.sha256(pd.util.hash_pandas_object(left[["race_id", "horse_no"]], index=False).to_numpy().tobytes()).hexdigest()[:16]
        selected = _discovery_schema(left, self.schema, self.recipe, self.label,
                                    self.output_dir, f"graph-{self.component_id}-{identity}")
        self.selected_schema = selected
        self.transforms = FittedResearchTransforms.fit(left,
            tuple(RuntimeTransformSpec(s.kind, dict(s.parameters)) for s in self.recipe.transforms))
        self.feature_schema = _schema_with_transform_features(selected, self.recipe)
        left = _ensure_feature_columns(self.transforms.transform(left), self.feature_schema)
        if self.output_kind == "win_probability":
            if self.model_kind == "gaussian_probit":
                from .performance_probit import fit_probit
                self.model = fit_probit(left, self.feature_schema, self.parameters)
            else:
                self.model = RaceProbabilityModel(kind=self.model_kind, parameters=self.parameters,
                    random_state=self.recipe.seed, feature_schema=self.feature_schema).fit(left)
        elif self.output_kind in {"point_prediction", "ranking_score"}:
            self.model = ResearchRegressor(self.model_kind, self.parameters).fit(left, self.feature_schema, self.label)
        else:
            raise ValueError("Fold-local graph feature pipelines currently support win/rank/point ancestors; distribution ancestors must declare their own preprocessing")
        return self

    def _frame(self, frame):
        transformed = self.residual_history.transform(frame) if self.residual_history else frame
        transformed = _attach_discovery(transformed, self.feature_schema.numeric)
        return _ensure_feature_columns(self.transforms.transform(transformed), self.feature_schema)

    def predict_proba(self, frame):
        return self.model.predict_proba(self._frame(frame))

    def predict(self, frame):
        return self.model.predict(self._frame(frame))


def _execute_graph_frame(request, labelled, schema, protocol, dataset_hash, started, *, pins=None):
    from functools import partial
    from .pipeline_graph import PipelineGraph, fit_graph
    from .prediction_store import PredictionStore
    recipe = request.recipe
    contract = target_contract(recipe.target.kind, recipe.target.parameters)
    graph_spec = PipelineGraph.from_dict(recipe.pipeline_graph).validate()
    if recipe.feature_discovery and recipe.feature_discovery.diagnostics:
        raise ValueError("Graph discovery diagnostics require constituent controls; this combination is unsupported")
    if recipe.feature_discovery and recipe.feature_discovery.adjusted_speed_residuals:
        raise ValueError("Graph adjusted-speed histories require a nested residual fit contract; this combination is unsupported")
    learned_features = bool(recipe.transforms or recipe.feature_discovery)
    folds, prediction_rows, reports, effective_training = [], [], [], []
    final_model = None
    for fold in protocol.folds:
        calibration = select_fold(labelled, fold.calibration_race_ids)
        train = _apply_training_window(select_fold(labelled, fold.train_race_ids), calibration, recipe.train_window)
        score = select_fold(labelled, fold.score_race_ids)
        factories = {}
        if learned_features:
            nodes = {n.node_id: n for n in graph_spec.nodes}
            for node in graph_spec.nodes:
                if node.kind != "estimator":
                    continue
                node_schema = schema
                if node.inputs:
                    from .pipeline_graph import _schema_subset
                    node_schema = _schema_subset(schema, nodes[node.inputs[0]].parameters.get("features", schema.features))
                options = dict(node.parameters)
                kind = options.pop("model_kind", recipe.model.kind if node.node_id == graph_spec.primary_node_id else "benter_conditional_logit")
                parameters = dict(options.pop("model_parameters", options.pop("parameters", {})))
                if node.node_id == graph_spec.primary_node_id:
                    if kind != recipe.model.kind:
                        raise ValueError("Outer model family differs from primary graph estimator")
                    parameters.update(recipe.model.parameters)
                if options:
                    raise ValueError(f"Unsupported fold-local graph estimator options: {sorted(options)}")
                factories[node.node_id] = partial(GraphComponentRecipeModel, recipe, node_schema, kind, parameters,
                    node.output.kind, "target_win" if node.output.kind == "win_probability" else node.output.target,
                    request.output_dir, node.node_id)
        fitted = fit_graph(graph_spec, train, calibration, schema, recipe.seed,
            model_spec=recipe.model, estimator_factories=factories or None,
            prediction_store=PredictionStore(request.output_dir.parent.parent/"graph-prediction-cache"))
        component_features = []
        visited = set()
        def collect(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            feature_schema = getattr(node.state, "feature_schema", None)
            if isinstance(node.state, GraphComponentRecipeModel):
                component_features.extend(node.state.selected_schema.numeric)
            for parent in node.parents:
                collect(parent)
        collect(fitted.output)
        collect(fitted.fundamental)
        if fitted.joint is not None:
            collect(fitted.joint)
        replay_schema = FeatureSchema(schema.name, tuple(dict.fromkeys((*schema.numeric, *component_features))), schema.categorical)
        # Components own their fitted transforms; the outer wrapper is deliberately stateless.
        final_model = FittedGraphRecipeModel(fitted, FittedResearchTransforms.fit(train, ()),
                                             replay_schema)
        if contract.kind == "win_probability":
            fundamental = fitted.predict_fundamental_proba(score)
            selected = fitted.predict_proba(score)
            market = baseline_probabilities(score, "raw_market")
            market_calibrator = TemperatureCalibrator.fit(calibration.market_probability.to_numpy(float), calibration)
            calibrated_market = market_calibrator.transform(market, score.race_id)
            row = {"fold_id": fold.fold_id}
            for name, p in (("model", fundamental), ("selected", selected), ("raw_market", market),
                            ("calibrated_market", calibrated_market), ("uniform", baseline_probabilities(score, "uniform"))):
                row[name] = evaluate_research_probabilities(p, score, label=name)["metrics"]
            row["selected_minus_market"] = row["selected"]["race_log_loss"]-row["raw_market"]["race_log_loss"]
            row["selected_minus_calibrated_market"] = row["selected"]["race_log_loss"]-row["calibrated_market"]["race_log_loss"]
            output = score[["race_id", "date", "race_no", "horse_no", "target_win"]].copy()
            output["model_probability"], output["selected_probability"] = fundamental, selected
            output["market_probability"], output["calibrated_market_probability"] = market, calibrated_market
        else:
            predictions = fitted.predict(score)
            model_metrics = secondary_target_diagnostics(score, contract, predictions)
            baseline = _secondary_baseline(train, score, contract.kind, contract.parameters)
            row = {"fold_id": fold.fold_id, "model": model_metrics,
                "baseline": secondary_target_diagnostics(score, contract, baseline),
                "objective": _secondary_objective(contract.kind, model_metrics)}
            output = score[["race_id", "date", "race_no", "horse_no"]].copy()
            output["label"], output["prediction"], output["baseline_prediction"] = score[contract.label_column].to_numpy(), predictions, baseline
        output["fold_id"] = fold.fold_id
        folds.append(row)
        prediction_rows.append(output)
        reports.append({"fold_id": fold.fold_id, **fitted.fit_report,
            "prediction_cache_policy": "disabled_for_fold_local_feature_pipelines" if learned_features else "content_addressed"})
        effective_training.append({"fold_id": fold.fold_id, "rows": len(train), "races": train.race_id.nunique(),
            "features": list(replay_schema.features), "fit_scope": "recursive_forward_oof",
            "training_cutoff": fitted.fit_report["training_cutoff"]})
    if final_model is None:
        raise ValueError("Graph protocol produced no executable folds")
    fundamental_first = is_fundamental_first_portfolio(request.portfolio_version)
    source = "model" if fundamental_first else "selected"
    objective = float(np.mean([f[source]["race_log_loss"] if contract.kind == "win_probability" else f["objective"] for f in folds]))
    protocol_path, predictions_path = request.output_dir/"protocol.json", request.output_dir/"predictions.csv"
    package_path, graph_report_path = request.output_dir/"package", request.output_dir/"graph-fits.json"
    _write_json_atomic(protocol_path, protocol.to_dict())
    _write_csv_atomic(predictions_path, pd.concat(prediction_rows, ignore_index=True))
    _write_json_atomic(graph_report_path, {"folds": reports, "physical_fits": sum(r["physical_fits"] for r in reports),
        "component_execution": "sequential; no nested worker pool"})
    ResearchModelPackage(final_model, recipe, protocol.protocol_id, request.code_revision, request.portfolio_version,
        _feature_replay_context(request, replay_schema, score, dataset_hash=dataset_hash)).save(package_path)
    result = RecipeExecutionResult(RESULT_SCHEMA_VERSION, request.attempt_id, request.proposal_id,
        request.trial_number, recipe.recipe_hash(), recipe.target.kind, "completed",
        "development_fundamental_race_log_loss" if fundamental_first and contract.kind == "win_probability" else _objective_name(contract.kind),
        objective, {"objective": objective, "objective_source": source if contract.kind == "win_probability" else "model",
            "folds": folds, "summary": _aggregate_fold_metrics(folds), "metric_contract_version": 3 if fundamental_first else METRIC_CONTRACT_VERSION,
            "metric_directions": _metric_directions(folds), "effective_training": effective_training,
            "physical_component_fits": sum(r["physical_fits"] for r in reports),
            "dataset_exclusions": labelled.attrs.get("dataset_exclusions",{})},
        {"protocol": str(protocol_path), "predictions": str(predictions_path), "package": str(package_path), "graph_fits": str(graph_report_path)},
        {"dataset_hash": dataset_hash, "protocol_id": protocol.protocol_id,
            "feature_program_id": recipe.feature_program_id(dataset_hash), "code_revision": request.code_revision,
            "environment_hash": request.environment_hash}, round(time.perf_counter()-started, 6))
    _write_json_atomic(request.output_dir/"result.json", result.serializable())
    return result


def _execute_secondary_frame(
    request: RecipeExecutionRequest,
    labelled: pd.DataFrame,
    schema: FeatureSchema,
    protocol: ProtocolManifest,
    dataset_hash: str,
    started: float,
    *, pins=None,
) -> RecipeExecutionResult:
    recipe = request.recipe
    contract = target_contract(recipe.target.kind, recipe.target.parameters)
    folds: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    final_model: FittedSecondaryRecipeModel | None = None
    effective_training: list[dict[str, Any]] = []
    for fold in protocol.folds:
        calibration = select_fold(labelled, fold.calibration_race_ids)
        train = _apply_training_window(
            select_fold(labelled, fold.train_race_ids),
            calibration,
            recipe.train_window,
        )
        score = select_fold(labelled, fold.score_race_ids)
        transformed_train, transformed_calibration, transformed_score, fitted_transforms, fold_schema, residual_history = _prepared_fold(
            request,train,calibration,score,schema,fold.fold_id,contract.label_column,pins=pins)

        distribution_model = recipe.performance_distribution is not None
        if distribution_model:
            model = _fit_distribution_recipe(recipe, transformed_train, transformed_calibration, fold_schema, contract.label_column)
            shuffled_model = None
        elif contract.model_task == "classifier":
            model: ResearchClassifier | ResearchRegressor = ResearchClassifier(
                recipe.model.kind, dict(recipe.model.parameters)
            ).fit(transformed_train, fold_schema, contract.label_column)
            shuffled_model: ResearchClassifier | ResearchRegressor = ResearchClassifier(
                recipe.model.kind, dict(recipe.model.parameters)
            )
        else:
            model_kind = recipe.model.kind
            model = ResearchRegressor(model_kind, dict(recipe.model.parameters)).fit(
                transformed_train, fold_schema, contract.label_column
            )
            shuffled_model = ResearchRegressor(model_kind, dict(recipe.model.parameters))
        calibrator = None
        if contract.kind == "placing_top_k":
            calibration_labels = transformed_calibration[contract.label_column].to_numpy()
            if len(np.unique(calibration_labels)) == 2:
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    model.predict(transformed_calibration), calibration_labels,
                )
        predictions = model.predict(transformed_score)
        if calibrator is not None:
            predictions = calibrator.predict(predictions)
        metrics = secondary_target_diagnostics(transformed_score, contract, predictions)
        distribution_metrics = None
        shared_metrics = None
        performance_win_metrics = None
        if distribution_model:
            from .probabilistic_adapters import distribution_to_win
            distribution = model.predict_distribution(transformed_score)
            distribution_metrics = distribution.diagnostics(transformed_score[contract.label_column].to_numpy())
            performance_win_metrics = evaluate_research_probabilities(
                distribution_to_win(distribution, transformed_score.race_id), transformed_score,
                label="performance_win")["metrics"]
            if hasattr(model, "scale_shrinkage"):
                shared_metrics = model.predict_distribution(transformed_score, shared_scale=True).diagnostics(
                    transformed_score[contract.label_column].to_numpy())
        win_calibrator = None
        rank_win_metrics = None
        market_win_metrics = None
        rank_win_probabilities = None
        if contract.kind == "ranking_strength":
            calibration_win = _ranking_scores_to_probabilities(
                model.predict(transformed_calibration), transformed_calibration["race_id"]
            )
            win_calibrator = TemperatureCalibrator.fit(
                calibration_win, transformed_calibration
            )
            rank_win_probabilities = win_calibrator.transform(
                _ranking_scores_to_probabilities(predictions, transformed_score["race_id"]),
                transformed_score["race_id"],
            )
            rank_win_metrics = evaluate_research_probabilities(
                rank_win_probabilities, transformed_score, label="ranker_calibrated_win"
            )["metrics"]
            market_win_calibrator = TemperatureCalibrator.fit(
                transformed_calibration["market_probability"].to_numpy(dtype=float),
                transformed_calibration,
            )
            market_win_metrics = evaluate_research_probabilities(
                market_win_calibrator.transform(
                    transformed_score["market_probability"].to_numpy(dtype=float),
                    transformed_score["race_id"],
                ), transformed_score, label="calibrated_market_win"
            )["metrics"]
        baseline = _secondary_baseline(transformed_train, transformed_score, contract.kind, contract.parameters)
        baseline_metrics = secondary_target_diagnostics(transformed_score, contract, baseline)

        shuffled = transformed_train.copy()
        shuffled[contract.label_column] = _shuffle_labels_by_race(
            shuffled, contract.label_column, recipe.seed + len(folds)
        )
        if distribution_model:
            shuffled_model = _fit_distribution_recipe(recipe, shuffled, transformed_calibration, fold_schema, contract.label_column)
        else:
            shuffled_model.fit(shuffled, fold_schema, contract.label_column)
        shuffled_predictions = shuffled_model.predict(transformed_score)
        shuffled_metrics = secondary_target_diagnostics(
            transformed_score, contract, shuffled_predictions
        )
        objective = _secondary_objective(contract.kind, metrics)
        finish_time_metrics = None
        if recipe.schema_version == 3 and contract.kind == "adjusted_finish_time_or_speed":
            distance_m = pd.to_numeric(score.distance, errors="coerce").to_numpy(float)
            prediction_finish_seconds = speed_to_finish_seconds(distance_m, predictions)
            if "finish_seconds" in score:
                label_finish_seconds = pd.to_numeric(score.finish_seconds, errors="coerce").to_numpy(float)
            else:
                from .data import _finish_time_seconds
                label_finish_seconds = _finish_time_seconds(score.finish_time).to_numpy(float)
            valid_prediction = np.isfinite(prediction_finish_seconds) & (prediction_finish_seconds > 0)
            valid_time = valid_prediction & np.isfinite(label_finish_seconds) & (label_finish_seconds > 0)
            errors = prediction_finish_seconds[valid_time] - label_finish_seconds[valid_time]
            finish_time_metrics = {
                "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                "unit": "seconds", "valid_rows": int(valid_time.sum()),
                "invalid_prediction_rows": int((~valid_prediction).sum()),
                "invalid_label_rows": int((~(np.isfinite(label_finish_seconds) & (label_finish_seconds > 0))).sum()),
                "prediction_basis": "distance_over_mean_speed_not_mean_finish_time" if distribution_model else "distance_over_predicted_speed",
            }
        folds.append({
            "fold_id": fold.fold_id,
            "model": metrics,
            "baseline": baseline_metrics,
            "shuffled_control": shuffled_metrics,
            "objective": objective,
            **({"finish_time": finish_time_metrics} if finish_time_metrics is not None else {}),
            **({"distribution": distribution_metrics, "performance_win": performance_win_metrics,
                "distribution_fit": model.metadata,
                **({"shared_scale_control": shared_metrics} if shared_metrics is not None else {})}
               if distribution_model else {}),
            **({
                "calibrated_win": rank_win_metrics,
                "calibrated_market_win": market_win_metrics,
                "win_log_loss_delta_vs_calibrated_market": (
                    rank_win_metrics["race_log_loss"] - market_win_metrics["race_log_loss"]
                ),
            } if rank_win_metrics is not None and market_win_metrics is not None else {}),
        })
        output = transformed_score[["race_id", "date", "race_no", "horse_no"]].copy()
        output["fold_id"] = fold.fold_id
        output["label"] = transformed_score[contract.label_column].to_numpy()
        output["prediction"] = predictions
        output["baseline_prediction"] = baseline
        if finish_time_metrics is not None:
            output["distance_m"] = distance_m
            output["prediction_speed_mps"] = predictions
            output["label_finish_seconds"] = label_finish_seconds
            output["prediction_finish_seconds"] = prediction_finish_seconds
            output["finish_time_prediction_valid"] = valid_prediction
            output["prediction_finish_seconds_basis"] = finish_time_metrics["prediction_basis"]
        if distribution_model:
            output["performance_location"] = distribution.location
            output["performance_scale"] = distribution.scale
            output["performance_win_probability"] = distribution_to_win(distribution, transformed_score.race_id)
        if rank_win_probabilities is not None:
            output["calibrated_win_probability"] = rank_win_probabilities
        prediction_rows.append(output)
        effective_training.append({
            "fold_id": fold.fold_id,
            "rows": int(len(transformed_train)),
            "races": int(transformed_train["race_id"].nunique()),
            "features": list(fold_schema.features),
            "available_features": [
                column for column in fold_schema.features
                if transformed_train[column].notna().any()
            ],
            "unavailable_features": [
                column for column in fold_schema.features
                if not transformed_train[column].notna().any()
            ],
            "clip_bounds": {
                key: list(value) for key, value in fitted_transforms.clip_bounds.items()
            },
        })
        final_model = FittedSecondaryRecipeModel(
            model, fitted_transforms, fold_schema, calibrator, win_calibrator, residual_history
        )
        if distribution_model and recipe.feature_discovery and recipe.feature_discovery.diagnostics:
            raise ValueError("Distribution discovery diagnostics require separate point-model controls; this combination is unsupported")
        _discovery_diagnostics(final_model,transformed_train,
            transformed_calibration if recipe.schema_version == 3 else calibration,
            transformed_score if recipe.schema_version == 3 else score,recipe,request.output_dir,fold.fold_id,
            prepared=recipe.schema_version == 3)

    if not folds or final_model is None:
        raise ValueError("protocol produced no executable folds")
    objective = float(np.mean([row["objective"] for row in folds]))
    request.output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = request.output_dir / "protocol.json"
    predictions_path = request.output_dir / "predictions.csv"
    package_path = request.output_dir / "package"
    _write_json_atomic(protocol_path, protocol.to_dict())
    _write_csv_atomic(predictions_path, pd.concat(prediction_rows, ignore_index=True))
    ResearchModelPackage(
        final_model,
        recipe,
        protocol_id=protocol.protocol_id,
        code_revision=request.code_revision,
        portfolio_version=request.portfolio_version,
        feature_context=_feature_replay_context(request, fold_schema, score, dataset_hash=dataset_hash) if recipe.schema_version == 3 else None,
    ).save(package_path)
    result = RecipeExecutionResult(
        schema_version=RESULT_SCHEMA_VERSION,
        attempt_id=request.attempt_id,
        proposal_id=request.proposal_id,
        trial_number=request.trial_number,
        recipe_hash=recipe.recipe_hash(),
        target_kind=recipe.target.kind,
        status="completed",
        objective_name=_objective_name(recipe.target.kind),
        objective_value=objective,
        metrics={
            "objective": objective,
            "folds": folds,
            "summary": _aggregate_fold_metrics(folds),
            "metric_contract_version": METRIC_CONTRACT_VERSION,
            "metric_directions": _metric_directions(folds),
            "effective_training": effective_training,
        },
        artifacts={
            "protocol": str(protocol_path),
            "predictions": str(predictions_path),
            "package": str(package_path),
        },
        lineage={
            "dataset_hash": dataset_hash,
            "protocol_id": protocol.protocol_id,
            "code_revision": request.code_revision,
            "environment_hash": request.environment_hash,
        },
        duration_seconds=round(time.perf_counter() - started, 6),
    )
    _write_json_atomic(request.output_dir / "result.json", result.serializable())
    return result


def _secondary_baseline(
    train: pd.DataFrame,
    score: pd.DataFrame,
    target_kind: str,
    parameters: dict[str, Any],
) -> np.ndarray:
    if target_kind == "placing_top_k":
        top_k = int(parameters.get("top_k", 3))
        field_size = score.groupby("race_id")["race_id"].transform("size").to_numpy(dtype=float)
        return np.minimum(1.0, top_k / field_size)
    if target_kind == "ranking_strength":
        if "market_probability" in score:
            return score["market_probability"].to_numpy(dtype=float)
        return np.full(len(score), 0.5)
    if target_kind == "recorded_final_win_odds":
        return np.full(
            len(score),
            float(pd.to_numeric(train["target_log_final_win_odds"], errors="raise").median()),
        )
    label = target_contract(target_kind, parameters).label_column
    return np.full(len(score), float(pd.to_numeric(train[label], errors="coerce").mean()))


def _ranking_scores_to_probabilities(
    scores: np.ndarray, race_ids: pd.Series | np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    maxima = pd.Series(values).groupby(np.asarray(race_ids)).transform("max").to_numpy()
    return normalize_by_race(np.exp(values - maxima), race_ids)


def _shuffle_labels_by_race(frame: pd.DataFrame, column: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = frame[column].to_numpy(copy=True)
    for indices in frame.groupby("race_id", sort=False).indices.values():
        output[np.asarray(indices)] = rng.permutation(output[np.asarray(indices)])
    return output


def _secondary_objective(target_kind: str, metrics: dict[str, float]) -> float:
    if target_kind == "ranking_strength":
        return -float(metrics["race_ndcg_at_3"])
    if target_kind == "placing_top_k":
        return float(metrics["race_brier"])
    if target_kind == "recorded_final_win_odds":
        return float(metrics["race_mae"])
    return float(metrics["race_mae"])


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    sources = sorted({
        source
        for fold in folds
        for source, payload in fold.items()
        if source not in {"fold_id", "objective", "selected_minus_market"}
        and isinstance(payload, dict)
    })
    summary: dict[str, Any] = {}
    for source in sources:
        metric_names = sorted({
            metric
            for fold in folds
            for metric, value in (fold.get(source) or {}).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        })
        source_summary: dict[str, dict[str, float]] = {}
        for metric in metric_names:
            values = np.asarray([
                float(fold[source][metric])
                for fold in folds
                if isinstance(fold.get(source), dict)
                and isinstance(fold[source].get(metric), int | float)
            ])
            if not len(values):
                continue
            source_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "worst": float(
                    values.max() if _lower_is_better(metric) else values.min()
                ),
            }
        summary[source] = source_summary
    return summary


def _metric_directions(folds: list[dict[str, Any]]) -> dict[str, str]:
    metrics = {
        metric
        for fold in folds
        for source in fold.values()
        if isinstance(source, dict)
        for metric, value in source.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    return {
        metric: "minimize" if _lower_is_better(metric) else "maximize"
        for metric in sorted(metrics)
    }


def _lower_is_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER_METRICS or metric.endswith(("_loss", "_error"))


def _effective_schema(recipe: PipelineRecipe) -> FeatureSchema:
    return drop_feature_families(
        FEATURE_SCHEMAS[recipe.feature_schema], recipe.drop_feature_families
    )


def _discovery_diagnostics(fitted, train, calibration, score, recipe, output, fold_id, *, prepared=False):
    if not recipe.feature_discovery or not recipe.feature_discovery.diagnostics:
        return
    import copy
    contract=target_contract(recipe.target.kind,recipe.target.parameters)
    def objective(estimator, frame, calibrator):
        transformed=_ensure_feature_columns(frame if prepared else fitted.transforms.transform(frame),fitted.feature_schema)
        prediction=estimator.predict_proba(transformed) if contract.kind=="win_probability" else estimator.predict(transformed)
        if calibrator is not None:
            prediction=calibrator.transform(prediction,transformed.race_id) if contract.kind=="win_probability" else calibrator.predict(prediction)
        return evaluate_research_probabilities(prediction,transformed,label="diagnostic")["race_weighted_log_loss"] if contract.kind=="win_probability" else _secondary_objective(contract.kind,secondary_target_diagnostics(transformed,contract,prediction))
    baseline=objective(fitted.model,score,fitted.calibrator)
    report={"target_kind":contract.kind,"baseline":baseline,"scope":"Exploratory outer-development diagnostics; not used to select features for this scored trial"}
    if "race_permutation" in recipe.feature_discovery.diagnostics:
        columns=[c for c in fitted.feature_schema.numeric if c.startswith("dfs_") and c in score]
        rng=np.random.default_rng(recipe.seed)
        deltas=[]
        for _ in range(2):
            changed=score.copy()
            if columns:
                values=score[columns].to_numpy().copy()
                for positions in score.groupby("race_id",sort=True).indices.values():
                    values[positions]=values[rng.permutation(positions)]
                changed[columns]=values
            deltas.append(float(objective(fitted.model,changed,fitted.calibrator)-baseline))
        report["race_permutation"]={"columns":columns,"policy":"Synchronized generated-feature family swaps within each race, after past-only construction","repeats":2,"objective_deltas":deltas}
    if "learning_curve" in recipe.feature_discovery.diagnostics:
        races=train[["race_id","date","race_no"]].drop_duplicates("race_id").sort_values(["date","race_no"])
        smaller=train[train.race_id.isin(races.race_id.iloc[len(races)//2:])]
        model=copy.deepcopy(fitted.model)
        if contract.kind=="win_probability":
            model.fit(smaller)
            calframe=_ensure_feature_columns(calibration if prepared else fitted.transforms.transform(calibration),fitted.feature_schema)
            calibrator=TemperatureCalibrator.fit(model.predict_proba(calframe),calframe) if recipe.calibration.kind=="temperature" else None
        else:
            model.fit(smaller,fitted.feature_schema,contract.label_column)
            calibrator=None
        report["learning_curve"]={"training_fraction":.5,"training_races":smaller.race_id.nunique(),"objective":objective(model,score,calibrator),"caveat":"Model-only diagnostic with frozen full-training feature mask and transforms; placing recalibration disabled"}
    _write_json_atomic(output/f"discovery-diagnostics-{fold_id}.json",report)


def _residual_fold(train, calibration, score, recipe, output, fold_id):
    if not recipe.feature_discovery or not recipe.feature_discovery.adjusted_speed_residuals:
        return train, calibration, score, None
    from .feature_residuals import AdjustedSpeedHistory, covariates, records
    residual = AdjustedSpeedHistory.fit(train, recipe.feature_discovery, pd.to_datetime(calibration.date).min().normalize())
    transformed_train = residual.transform(train)
    transformed_calibration = residual.transform(calibration)
    speed = pd.to_numeric(calibration.distance) / pd.to_numeric(calibration.finish_seconds).where(lambda x: x > 0)
    residual.history = pd.concat([residual.history, records(calibration, speed.to_numpy() - residual.model.predict(covariates(calibration)))], ignore_index=True)
    _write_json_atomic(output / f"discovery-residual-{fold_id}.json", residual.report)
    return transformed_train, transformed_calibration, residual.transform(score), residual


def _discovery_schema(train, schema, recipe, label, output, fold_id):
    if recipe.feature_discovery is None:
        return schema
    from .feature_screening import DiscoverySelection
    selection = DiscoverySelection.fit(train, recipe.feature_discovery, recipe.target.kind, label)
    required = sorted({c for transform in recipe.transforms for c in transform.parameters["columns"] if c.startswith("dfs_")})
    if any(c not in train or not train[c].notna().any() for c in required):
        raise ValueError("Discovery transform references an absent or unavailable generated column")
    columns = tuple(dict.fromkeys((*selection.columns,*required)))
    if recipe.schema_version == 2 and len(columns)>32:
        raise ValueError("Explicit discovery transforms exceed the selected-feature ceiling")
    selection.report["explicit_transform_inputs"] = required
    selection.report["effective_columns"] = columns
    _write_json_atomic(output / f"discovery-selection-{fold_id}.json", selection.report)
    import joblib
    joblib.dump(selection,output / f"discovery-selector-{fold_id}.joblib")
    return FeatureSchema(schema.name, tuple(dict.fromkeys((*schema.numeric, *columns))), schema.categorical)


def _v6_feature_frame(frame, request):
    from .feature_definitions import evaluate_formulas
    manifest_path = Path(request.dataset_path).parent/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be an object")
    dataset_id = manifest.get("dataset_id", "")
    if dataset_id.startswith("dataset-") and len(dataset_id) == 72:
        from .dataset_registry import _digest, file_sha256
        checksum = manifest_path.parent / "manifest.sha256"
        if "dataset-" + _digest(manifest.get("identity")) != dataset_id:
            raise ValueError("Dataset identity payload changed")
        if manifest["identity"].get("predictor_catalog_sha256") != _digest(manifest.get("predictor_catalog")):
            raise ValueError("Dataset predictor catalog identity changed")
        if manifest["identity"].get("eligible_categorical_predictors_sha256") != _digest(manifest.get("eligible_categorical_predictors")):
            raise ValueError("Dataset categorical predictor identity changed")
        if not checksum.is_file() or checksum.read_text().strip() != file_sha256(manifest_path):
            raise ValueError("Dataset manifest checksum mismatch")
    hashes = [manifest[key] for key in ("dataset_hash", "dataset_sha256") if manifest.get(key)]
    if "files" in manifest:
        files = manifest["files"]
        if not isinstance(files, dict) or Path(request.dataset_path).name not in files:
            raise ValueError("Dataset manifest does not bind the executed file")
        for name, digest in files.items():
            if Path(name).name != name or _hash_file(manifest_path.parent / name) != digest:
                raise ValueError(f"Dataset manifest checksum mismatch: {name}")
        hashes.append(files[Path(request.dataset_path).name])
    if hashes and any(value != _hash_file(request.dataset_path) for value in hashes):
        raise ValueError("Dataset manifest checksum does not match dataset")
    registry_catalog = "feature_catalog" in manifest and bool(hashes)
    catalog = manifest.get("predictor_catalog", manifest.get("eligible_predictors", manifest.get("feature_catalog", {})))
    allowed = set().union(*(s.features for s in FEATURE_SCHEMAS.values()))
    metadata = {name:{"unit":"1", "temporal_scope":"pre_race"} for name in allowed}
    for name, unit in {"distance": "m", "actual_weight": "kg", "declared_weight": "kg", "horse_rating": "rating"}.items():
        if name in metadata:
            metadata[name]["unit"] = unit
    if isinstance(catalog,list):
        if any(not isinstance(row,dict) or "name" not in row for row in catalog):
            raise ValueError("Predictor catalog must contain named metadata objects")
        catalog = {row["name"]:row for row in catalog}
    if not isinstance(catalog,dict):
        raise ValueError("Unsupported dataset predictor catalog shape")
    ineligible = set(metadata) - set(catalog) - set(manifest.get("eligible_categorical_predictors", ())) if registry_catalog else set()
    for name in ineligible:
        metadata.pop(name, None)
        allowed.discard(name)
    from .research_targets import validate_no_forbidden_label_features
    from .feature_expressions import InputMetadata
    for name,row in catalog.items():
        if not isinstance(row,dict):
            raise ValueError(f"Predictor catalog metadata missing for {name}")
        if registry_catalog and "temporal_scope" not in row:
            if row.get("cutoff_rule") not in {"occurrence_and_availability_before_cutoff",
                    "occurrence_and_availability_strictly_before_cutoff",
                    "pre_race_or_previous_race_registered_formula", "available_at_strictly_before_race_cutoff"}:
                raise ValueError(f"Registry predictor lacks a supported cutoff rule: {name}")
            row = dict(row, temporal_scope="pre_race", source_family=row.get("family", "registered"))
        if not row.get("eligible",True) or row.get("target_tainted") or row.get("temporal_scope") in {"current_outcome", "future"}:
            ineligible.add(name)
            metadata.pop(name,None)
            allowed.discard(name)
            continue
        try:
            validate_no_forbidden_label_features(target_contract(request.recipe.target.kind),[name])
        except ValueError:
            ineligible.add(name)
            metadata.pop(name,None)
            allowed.discard(name)
            continue
        if row.get("eligible",True):
            values = {k:v for k,v in row.items() if k in {
                "unit","dtype","temporal_scope","target_tainted","available_at_column","source_family"}}
            if name not in allowed and ("unit" not in values or "temporal_scope" not in values):
                raise ValueError(f"Custom predictor requires units and temporal metadata: {name}")
            metadata[name] = InputMetadata.model_validate(values).model_dump(mode="json")
            allowed.add(name)
    if set(request.recipe.extra_numeric_features or ())-allowed:
        raise ValueError("Extra features must be in the verified dataset predictor catalog")
    validate_no_forbidden_label_features(target_contract(request.recipe.target.kind),list(request.recipe.extra_numeric_features or ()))
    for name in request.recipe.extra_numeric_features or ():
        if name not in frame:
            raise ValueError(f"Declared dataset predictor is absent: {name}")
    # Apply the same source availability masks to direct predictors and formulas.
    for name, row in metadata.items():
        available = row.get("available_at_column")
        if available and name in frame:
            if available not in frame:
                raise ValueError(f"Predictor availability binding is missing: {available}")
            observed = pd.to_datetime(frame[available],utc=True,errors="coerce")
            valid = observed.notna() & observed.lt(pd.to_datetime(frame.date,utc=True))
            frame = frame.copy()
            frame.loc[~valid,name] = np.nan
    formulas = ()
    if request.recipe.feature_definitions:
        from .feature_definitions import FeatureDefinition
        derived = set()
        for value in request.recipe.feature_definitions:
            definition = FeatureDefinition.model_validate(value)
            for name in (definition.name, definition.feature_id):
                if name in frame or name in metadata:
                    row = catalog.get(name, {})
                    if not (hashes and isinstance(row, dict) and row.get("definition_id") == definition.content_id()
                            and row.get("eligible", True) and not row.get("target_tainted")
                            and row.get("temporal_scope") == "pre_race" and row.get("unit") == definition.unit):
                        raise ValueError(f"Preexisting formula output lacks identical verified untainted definition: {name}")
                derived.add(name)
        frame = frame.drop(columns=list(derived & set(frame)))
        metadata = {name: row for name, row in metadata.items() if name not in derived}
        frame,formulas,report = evaluate_formulas(frame, request.recipe.feature_definitions,metadata,
                                                cutoff=pd.to_datetime(frame["date"],utc=True))
        _write_json_atomic(request.output_dir/"formula-report.json",report)
    if "finish_seconds" not in frame and "finish_time" in frame:
        from .data import _finish_time_seconds
        frame = frame.copy()
        frame["finish_seconds"] = _finish_time_seconds(frame.finish_time)
    synthesis_metadata = dict(metadata)
    if request.recipe.feature_definitions:
        from .feature_definitions import FeatureDefinition
        for definition in request.recipe.feature_definitions:
            parsed = FeatureDefinition.model_validate(definition)
            entry = {"unit": parsed.unit, "dtype": parsed.dtype, "temporal_scope": "pre_race"}
            synthesis_metadata[parsed.name] = entry
            synthesis_metadata[parsed.feature_id] = entry
    required = set(_effective_schema(request.recipe).features) | set(request.recipe.extra_numeric_features or ()) | set(formulas)
    required |= {"date","race_id","race_no","horse_no","horse_id","field_size","target_win",
                 "target_probability","result","finishing_status","market_probability","win_odds",
                 "finish_seconds","finish_time","distance","actual_weight","lengths_raw","source",
                 "jockey_id","jockey_key","trainer_id","trainer_key"}
    if request.recipe.feature_definitions:
        required.update(d["name"] for d in request.recipe.feature_definitions)
        required.update(ref for d in request.recipe.feature_definitions for ref in d["input_refs"])
    if request.recipe.feature_discovery:
        required.update(request.recipe.feature_discovery.measurements)
    required.update(row["available_at_column"] for row in metadata.values() if row.get("available_at_column") and row.get("available_at_column") in frame)
    required.update({"observed_at", "venue", "course", "surface", "horse_rating", "odds_snapshot_at", "forecast_at", "future_market_probability"})
    frame = frame[[c for c in frame if c in required]].copy()
    comparison = manifest.get("comparison_contract", {})
    policy = manifest.get("availability_policy", manifest.get("event_policy_id", comparison.get("availability_tier", "pre_race_synthetic")))
    if comparison.get("retrospective_lags"):
        policy = {"tier": policy, "retrospective_lags": comparison["retrospective_lags"]}
    frame.attrs.update(input_metadata=metadata,synthesis_metadata=synthesis_metadata,formula_features=formulas,
                       ineligible_predictors=tuple(sorted(ineligible)),
                       dataset_manifest_hash=_hash_file(manifest_path) if manifest_path.exists() else None,
                       verified_dataset_manifest=manifest,
                       availability_policy=policy if hashes else "pre_race_synthetic")
    return frame


def _feature_replay_context(request, schema, score, *, dataset_hash=None):
    from importlib.metadata import version
    if dataset_hash is not None and _hash_file(request.dataset_path) != dataset_hash:
        raise ValueError("Dataset changed during trial; inference package cannot be published")
    source = _v6_feature_frame(_load_dataset(request.dataset_path), request)
    if source.attrs.get("dataset_manifest_hash") != score.attrs.get("dataset_manifest_hash"):
        raise ValueError("Dataset predictor manifest changed during trial")
    metadata = source.attrs["input_metadata"]
    discovery_features = ()
    history = None
    implementations = ["feature_definitions.py", "feature_expressions.py"] if request.recipe.feature_definitions else []
    dependencies = {name:version(name) for name in ("numpy", "pandas")}
    if request.recipe.feature_discovery:
        manifest = json.loads((request.output_dir/"discovery-manifest.json").read_text())
        generated = set(manifest.get("feature_names", ())) or {row["feature_id"] for row in manifest["catalog"]}
        discovery_features = tuple(c for c in schema.numeric if c in generated)
        spec = request.recipe.feature_discovery
        history_columns = {"date", "race_id", "horse_no", "source", "observed_at"}
        history_columns.update(f"{entity}_{suffix}" for entity in spec.entities for suffix in ("id", "key"))
        if "speed_mps" in spec.measurements or spec.domain_history:
            history_columns.update({"finish_seconds", "distance"})
        if "carried_weight" in spec.measurements or spec.domain_history:
            history_columns.add("actual_weight")
        if "beaten_lengths" in spec.measurements:
            history_columns.add("lengths_raw")
        if spec.domain_history:
            history_columns.update({"surface", "venue", "horse_rating"})
        custom = set(spec.measurements)-{"speed_mps", "carried_weight", "beaten_lengths"}
        history_columns.update(custom)
        synthesis_metadata = source.attrs["synthesis_metadata"]
        history_columns.update(synthesis_metadata[name]["available_at_column"] for name in custom
            if name in synthesis_metadata and synthesis_metadata[name].get("available_at_column"))
        history = source.loc[pd.to_datetime(source.date) < pd.to_datetime(score.date).min().normalize(),
                            [c for c in source if c in history_columns]].copy()
        history.attrs = {}
        implementations.append("feature_program.py")
        dependencies.update({name:version(name) for name in ("featuretools", "woodwork")})
    cutoff = pd.to_datetime(score.date).min().normalize()-pd.Timedelta(nanoseconds=1)
    derived = set()
    if request.recipe.feature_definitions:
        from .feature_definitions import FeatureDefinition
        for value in request.recipe.feature_definitions:
            definition = FeatureDefinition.model_validate(value)
            derived.update((definition.name, definition.feature_id))
    return FeatureReplayContext(tuple(request.recipe.feature_definitions or ()), metadata,
        tuple(name for name in request.recipe.extra_numeric_features or () if name not in derived), request.recipe.feature_discovery,
        discovery_features, history, str(cutoff), dependencies,
        {name:_hash_file(Path(__file__).parent/name) for name in implementations},
        source.attrs.get("dataset_manifest_hash"))


def _prepared_fold(request, train, calibration, score, schema, fold_id, label, *, pins=None):
    """Cache pure fold preparation, independent of downstream model-only parameters."""
    recipe = request.recipe
    def build_frames():
        left,middle,right,residual = _residual_fold(train,calibration,score,recipe,request.output_dir,fold_id)
        descriptor = train.attrs.get("discovery_matrix")
        if descriptor is not None:
            left = _attach_discovery(left)
        selected = _discovery_schema(left,schema,recipe,label,request.output_dir,fold_id)
        if descriptor is not None:
            generated = tuple(c for c in selected.numeric if c in descriptor.feature_names)
            middle,right = _attach_discovery(middle,generated),_attach_discovery(right,generated)
            left = left[[c for c in left if not c.startswith("dfs_") or c in selected.numeric]]
        runtime = tuple(RuntimeTransformSpec(s.kind,dict(s.parameters)) for s in recipe.transforms)
        transforms = FittedResearchTransforms.fit(left,runtime)
        effective = _schema_with_transform_features(selected,recipe)
        prepared = tuple(_ensure_feature_columns(transforms.transform(f),effective) for f in (left,middle,right))
        return (*prepared,transforms,effective,residual)
    if recipe.schema_version == 2:
        return build_frames()
    from .research_preparation import PreparationCache, PreparationKey, PreparedFold
    from importlib.metadata import version
    from .feature_discovery_specs import content_id
    def row_keys(frame):
        return tuple(json.dumps([str(r),str(h)],separators=(",",":")) for r,h in zip(frame.race_id,frame.horse_no))
    keys = {"train":row_keys(train),"calibration":row_keys(calibration),"score":row_keys(score)}
    key = PreparationKey(source_content_hash=request.dataset_hash or _hash_file(request.dataset_path),
        training_row_keys=keys["train"],calibration_row_keys=keys["calibration"],score_row_keys=keys["score"],
        target_labels_hash=content_id({"target":recipe.target.model_dump(mode="json"),"labels":train[label].tolist()}),
        availability_policy={"rule":train.attrs.get("availability_policy","validated_pre_race_schema"),
            "metadata":train.attrs.get("input_metadata",{}), "manifest_hash":train.attrs.get("dataset_manifest_hash"),
            "matrix_id":getattr(train.attrs.get("discovery_matrix"),"matrix_id",None)},
        feature_definitions=tuple(recipe.feature_definitions or ())+( {"schema":schema.name,"features":list(schema.features)},),
        selection_settings=recipe.feature_discovery.model_dump(mode="json") if recipe.feature_discovery else {},
        fitted_transform_settings={"transforms":[s.model_dump(mode="json") for s in recipe.transforms],"window":recipe.train_window},
        fold_dates=tuple(str(f.date.min())+"/"+str(f.date.max()) for f in (train,calibration,score)),
        seed=recipe.seed, implementation_revision=request.code_revision+":"+_hash_file(Path(__file__)),
        dependency_versions={n:version(n) for n in ("numpy","pandas","scikit-learn","joblib",*( ("feature-engine", "featuretools") if recipe.feature_discovery else ()))})
    cache = PreparationCache(request.output_dir.parent.parent/"fold-preparation-cache")
    def builder():
        left,middle,right,transforms,effective,residual = build_frames()
        arrays = {name+"_x":f[list(effective.numeric)].to_numpy(dtype=float)
                  for name,f in zip(("train","calibration","score"),(left,middle,right))}
        reports = {}
        for suffix in ("selection", "residual"):
            path = request.output_dir/f"discovery-{suffix}-{fold_id}.json"
            if path.exists():
                reports[suffix] = json.loads(path.read_text())
        selector_path = request.output_dir/f"discovery-selector-{fold_id}.joblib"
        import joblib
        return PreparedFold(arrays,tuple(effective.numeric),keys,
            fitted_state={"transforms":transforms,"schema":effective,"residual":residual,
                          "reports":reports, "selector":joblib.load(selector_path) if selector_path.exists() else None})
    artifact = cache.prepare_fold(key,builder)
    if pins is not None:
        pins.enter_context(cache.pin(artifact))
    state,arrays = artifact.load_fitted_state(),artifact.load_arrays()
    for suffix, report in state.get("reports",{}).items():
        _write_json_atomic(request.output_dir/f"discovery-{suffix}-{fold_id}.json",report)
    if state.get("selector") is not None:
        import joblib
        joblib.dump(state["selector"],request.output_dir/f"discovery-selector-{fold_id}.joblib")
    effective = state["schema"]
    reconstructed = []
    for name,frame in zip(("train","calibration","score"),(train,calibration,score)):
        numeric = pd.DataFrame(arrays[name+"_x"],columns=effective.numeric,copy=False)
        rest = frame[[c for c in frame if c not in effective.numeric]].reset_index(drop=True)
        rebuilt = pd.concat([rest,numeric],axis=1,copy=False)
        for column in effective.categorical:
            rebuilt[column] = rebuilt[column].fillna("UNKNOWN").astype(str)
        rebuilt.attrs = dict(frame.attrs)
        reconstructed.append(rebuilt)
    _write_json_atomic(request.output_dir/f"preparation-{fold_id}.json",
        {"artifact_id":artifact.artifact_id,"path":str(artifact.path),"cache_status":artifact.cache_status,
         "wait_seconds":artifact.wait_seconds,"features":list(effective.features),
         "pin_scope":"trial_fit_evaluation_and_package_save" if pins is not None else "caller_unpinned"})
    return (*reconstructed,state["transforms"],effective,state["residual"])


def _schema_with_transform_features(
    schema: FeatureSchema,
    recipe: PipelineRecipe,
) -> FeatureSchema:
    added: list[str] = []
    for spec in recipe.transforms:
        if spec.kind == "race_relative_rank":
            suffix = str(spec.parameters.get("suffix", "_race_rank"))
            added.extend(f"{column}{suffix}" for column in spec.parameters["columns"])
        elif spec.kind == "race_relative_center":
            added.extend(f"{column}_race_centered" for column in spec.parameters["columns"])
        elif spec.kind == "signed_log1p":
            added.extend(f"{column}_signed_log1p" for column in spec.parameters["columns"])
        elif spec.kind == "numeric_interaction":
            left, right = spec.parameters["columns"]
            added.append(f"{left}_x_{right}")
    if len(added) != len(set(added)) or set(added) & set(schema.features):
        raise ValueError("Generated feature names must be unique and must not replace source features")
    return FeatureSchema(schema.name, (*schema.numeric, *added), schema.categorical)


def _ensure_feature_columns(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    output = frame.copy()
    missing_discovery = [c for c in schema.numeric if c.startswith("dfs_") and c not in frame]
    if missing_discovery:
        raise ValueError("Inference requires materialized discovery features; missing: " + str(missing_discovery))
    missing_numeric = {
        column: pd.Series(np.nan, index=output.index, dtype=float)
        for column in schema.numeric if column not in output
    }
    missing_categorical = {
        column: pd.Series("UNKNOWN", index=output.index, dtype=object)
        for column in schema.categorical if column not in output
    }
    if missing_numeric or missing_categorical:
        output = pd.concat(
            [output, pd.DataFrame(missing_numeric | missing_categorical, index=output.index)],
            axis=1,
        )
    for column in schema.numeric:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in schema.categorical:
        output[column] = output[column].fillna("UNKNOWN").astype(str)
    output.attrs = dict(frame.attrs)
    return output


def _apply_training_window(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    window: str,
) -> pd.DataFrame:
    if window == "all_history":
        return train
    if window != "trailing_3_years":
        raise ValueError(f"Unknown training window: {window}")
    cutoff = pd.to_datetime(calibration["date"]).min() - pd.DateOffset(years=3)
    selected = train[pd.to_datetime(train["date"]).ge(cutoff)].copy()
    if selected.empty:
        raise ValueError("trailing_3_years produced an empty training window")
    return selected


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame.sort_values(["date", "race_no", "race_id", "horse_no"], kind="stable").reset_index(drop=True)


def _objective_name(target_kind: str) -> str:
    return {
        "win_probability": "development_race_log_loss",
        "ranking_strength": "negative_development_race_ndcg_at_3",
        "placing_top_k": "development_race_brier",
        "adjusted_finish_time_or_speed": "development_race_mae",
        "market_odds_forecast": "development_race_mae",
        "recorded_final_win_odds": "development_log_final_odds_mae",
    }[target_kind]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
