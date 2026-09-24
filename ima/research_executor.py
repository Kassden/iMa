"""Execute validated research recipes against protected temporal folds."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_sets import FEATURE_SCHEMAS, FeatureSchema, drop_feature_families
from .modeling import MarketBlend, RaceProbabilityModel, TemperatureCalibrator
from .research_evaluation import (
    ProtocolManifest,
    baseline_probabilities,
    evaluate_research_probabilities,
    make_protocol_manifest,
    select_fold,
)
from .research_model_package import ResearchModelPackage
from .research_models import ResearchClassifier, ResearchRegressor, secondary_target_diagnostics
from .research_specs import PipelineRecipe
from .research_targets import apply_target_contract, target_contract
from .research_transforms import (
    FittedResearchTransforms,
    TransformSpec as RuntimeTransformSpec,
)


RESULT_SCHEMA_VERSION = 1


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

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transforms.transform(frame)
        transformed = _ensure_feature_columns(transformed, self.feature_schema)
        probabilities = self.model.predict_proba(transformed)
        if self.calibrator is not None:
            probabilities = self.calibrator.transform(probabilities, transformed["race_id"])
        if self.blend is not None:
            probabilities = self.blend.transform(
                probabilities,
                transformed["market_probability"].to_numpy(dtype=float),
                transformed["race_id"],
            )
        return probabilities


@dataclass
class FittedSecondaryRecipeModel:
    model: ResearchClassifier | ResearchRegressor
    transforms: FittedResearchTransforms
    feature_schema: FeatureSchema

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.transforms.transform(frame)
        transformed = _ensure_feature_columns(transformed, self.feature_schema)
        return self.model.predict(transformed)


def execute_recipe(request: RecipeExecutionRequest) -> RecipeExecutionResult:
    """Train/evaluate one recipe and atomically persist its accepted artifacts."""
    started = time.perf_counter()
    try:
        frame = _load_dataset(request.dataset_path)
        observed_hash = _hash_file(request.dataset_path)
        if request.dataset_hash is not None and request.dataset_hash != observed_hash:
            raise ValueError("dataset hash does not match execution request")
        result = _execute_frame(request, frame, observed_hash, started)
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


def _execute_frame(
    request: RecipeExecutionRequest,
    frame: pd.DataFrame,
    dataset_hash: str,
    started: float,
) -> RecipeExecutionResult:
    recipe = request.recipe
    contract = target_contract(recipe.target.kind, recipe.target.parameters)
    labelled = apply_target_contract(frame, contract)
    schema = _effective_schema(recipe)
    labelled = _ensure_feature_columns(labelled, schema)
    protocol = make_protocol_manifest(
        labelled,
        target=contract,
        **request.protocol_parameters,
    )
    if recipe.target.kind != "win_probability":
        return _execute_secondary_frame(
            request, labelled, schema, protocol, dataset_hash, started
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
        runtime_specs = tuple(
            RuntimeTransformSpec(spec.kind, dict(spec.parameters)) for spec in recipe.transforms
        )
        fitted_transforms = FittedResearchTransforms.fit(train, runtime_specs)
        transformed_train = fitted_transforms.transform(train)
        transformed_calibration = fitted_transforms.transform(calibration)
        transformed_score = fitted_transforms.transform(score)
        fold_schema = _schema_with_transform_features(schema, recipe)
        transformed_train = _ensure_feature_columns(transformed_train, fold_schema)
        transformed_calibration = _ensure_feature_columns(transformed_calibration, fold_schema)
        transformed_score = _ensure_feature_columns(transformed_score, fold_schema)

        model = RaceProbabilityModel(
            kind=recipe.model.kind,
            parameters=dict(recipe.model.parameters),
            random_state=recipe.seed,
            feature_schema=fold_schema,
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
            "uniform": uniform["metrics"],
            "selected_minus_market": (
                selected["race_weighted_log_loss"] - market["race_weighted_log_loss"]
            ),
        })
        predictions = transformed_score[["race_id", "date", "race_no", "horse_no", "target_win"]].copy()
        predictions["fold_id"] = fold.fold_id
        predictions["model_probability"] = score_probabilities
        predictions["selected_probability"] = selected_probabilities
        predictions["market_probability"] = transformed_score["market_probability"].to_numpy()
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
            model, fitted_transforms, calibrator, blend, fold_schema
        )

    if not folds or final_model is None:
        raise ValueError("protocol produced no executable folds")
    objective = float(np.mean([row["selected"]["race_log_loss"] for row in folds]))
    metrics = {
        "objective": objective,
        "folds": folds,
        "effective_training": effective_training,
        "mean_selected_minus_market": float(np.mean([
            row["selected_minus_market"] for row in folds
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
    )
    package.save(package_path)
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
        metrics=metrics,
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


def _execute_secondary_frame(
    request: RecipeExecutionRequest,
    labelled: pd.DataFrame,
    schema: FeatureSchema,
    protocol: ProtocolManifest,
    dataset_hash: str,
    started: float,
) -> RecipeExecutionResult:
    recipe = request.recipe
    contract = target_contract(recipe.target.kind, recipe.target.parameters)
    folds: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    final_model: FittedSecondaryRecipeModel | None = None
    effective_training: list[dict[str, Any]] = []
    for fold in protocol.folds:
        train = _apply_training_window(
            select_fold(labelled, fold.train_race_ids),
            select_fold(labelled, fold.calibration_race_ids),
            recipe.train_window,
        )
        score = select_fold(labelled, fold.score_race_ids)
        runtime_specs = tuple(
            RuntimeTransformSpec(spec.kind, dict(spec.parameters)) for spec in recipe.transforms
        )
        fitted_transforms = FittedResearchTransforms.fit(train, runtime_specs)
        transformed_train = fitted_transforms.transform(train)
        transformed_score = fitted_transforms.transform(score)
        fold_schema = _schema_with_transform_features(schema, recipe)
        transformed_train = _ensure_feature_columns(transformed_train, fold_schema)
        transformed_score = _ensure_feature_columns(transformed_score, fold_schema)

        if contract.model_task == "classifier":
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
        predictions = model.predict(transformed_score)
        metrics = secondary_target_diagnostics(transformed_score, contract, predictions)
        baseline = _secondary_baseline(transformed_train, transformed_score, contract.kind, contract.parameters)
        baseline_metrics = secondary_target_diagnostics(transformed_score, contract, baseline)

        shuffled = transformed_train.copy()
        shuffled[contract.label_column] = _shuffle_labels_by_race(
            shuffled, contract.label_column, recipe.seed + len(folds)
        )
        shuffled_model.fit(shuffled, fold_schema, contract.label_column)
        shuffled_predictions = shuffled_model.predict(transformed_score)
        shuffled_metrics = secondary_target_diagnostics(
            transformed_score, contract, shuffled_predictions
        )
        objective = _secondary_objective(contract.kind, metrics)
        folds.append({
            "fold_id": fold.fold_id,
            "model": metrics,
            "baseline": baseline_metrics,
            "shuffled_control": shuffled_metrics,
            "objective": objective,
        })
        output = transformed_score[["race_id", "date", "race_no", "horse_no"]].copy()
        output["fold_id"] = fold.fold_id
        output["label"] = transformed_score[contract.label_column].to_numpy()
        output["prediction"] = predictions
        output["baseline_prediction"] = baseline
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
        final_model = FittedSecondaryRecipeModel(model, fitted_transforms, fold_schema)

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
        metrics={"objective": objective, "folds": folds, "effective_training": effective_training},
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
    label = target_contract(target_kind, parameters).label_column
    return np.full(len(score), float(pd.to_numeric(train[label], errors="coerce").mean()))


def _shuffle_labels_by_race(frame: pd.DataFrame, column: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = frame[column].to_numpy(copy=True)
    for indices in frame.groupby("race_id", sort=False).indices.values():
        output[np.asarray(indices)] = rng.permutation(output[np.asarray(indices)])
    return output


def _secondary_objective(target_kind: str, metrics: dict[str, float]) -> float:
    if target_kind == "ranking_strength":
        return -float(metrics["spearman"])
    if target_kind == "placing_top_k":
        return float(metrics["brier"])
    return float(metrics["mae"])


def _effective_schema(recipe: PipelineRecipe) -> FeatureSchema:
    return drop_feature_families(
        FEATURE_SCHEMAS[recipe.feature_schema], recipe.drop_feature_families
    )


def _schema_with_transform_features(
    schema: FeatureSchema,
    recipe: PipelineRecipe,
) -> FeatureSchema:
    added: list[str] = []
    for spec in recipe.transforms:
        if spec.kind == "race_relative_rank":
            suffix = str(spec.parameters.get("suffix", "_race_rank"))
            added.extend(f"{column}{suffix}" for column in spec.parameters["columns"])
    return FeatureSchema(schema.name, (*schema.numeric, *added), schema.categorical)


def _ensure_feature_columns(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    output = frame.copy()
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
        "ranking_strength": "negative_development_spearman",
        "placing_top_k": "development_brier",
        "adjusted_finish_time_or_speed": "development_mae",
        "market_odds_forecast": "development_mae",
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
