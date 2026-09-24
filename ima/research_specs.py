"""Strict pipeline recipe contracts for the agentic optimizer."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .feature_sets import (
    FEATURE_FAMILIES,
    FEATURE_SCHEMAS,
    NOTEBOOK_FEATURE_FAMILIES,
    drop_feature_families,
)
from .research_targets import TargetContractError, target_contract, validate_no_forbidden_label_features
from .research_transforms import TransformSpec as RuntimeTransformSpec
from .research_transforms import validate_transform_spec


class RecipeValidationError(ValueError):
    """Raised when a recipe is invalid for the registered pipeline capabilities."""


FORBIDDEN_RESEARCH_TERMS = {
    "final_odds",
    "dividend",
    "dividends",
    "result",
    "results",
    "target_win",
    "target_probability",
    "holdout",
    "live_execution",
    "promotion",
    "promote",
    "hkjc_credentials",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetSpec(StrictModel):
    kind: Literal[
        "win_probability",
        "ranking_strength",
        "placing_top_k",
        "adjusted_finish_time_or_speed",
        "market_odds_forecast",
    ] = "win_probability"
    parameters: dict[str, Any] = Field(default_factory=dict)


class TransformSpec(StrictModel):
    kind: Literal["clip_numeric_quantiles", "race_relative_rank"]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_transform(self) -> "TransformSpec":
        validate_transform_spec(RuntimeTransformSpec(self.kind, self.parameters))
        return self


class ModelSpec(StrictModel):
    kind: Literal[
        "logit",
        "boosted",
        "pairwise_ranker",
        "hist_gradient_regressor",
        "ridge_regressor",
    ] = "logit"
    parameters: dict[str, Any] = Field(default_factory=dict)


class CalibrationSpec(StrictModel):
    kind: Literal["temperature", "none"] = "temperature"
    parameters: dict[str, Any] = Field(default_factory=dict)


class BlendSpec(StrictModel):
    kind: Literal["market_softmax", "none"] = "market_softmax"
    parameters: dict[str, Any] = Field(default_factory=dict)


class PipelineRecipe(StrictModel):
    schema_version: Literal[2] = 2
    target: TargetSpec = Field(default_factory=TargetSpec)
    feature_schema: Literal["baseline-v1", "benter-rich-v1", "notebook-rich-v2"] = "baseline-v1"
    drop_feature_families: tuple[str, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
    train_window: Literal["all_history", "trailing_3_years"] = "all_history"
    model: ModelSpec = Field(default_factory=ModelSpec)
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    blend: BlendSpec = Field(default_factory=BlendSpec)
    seed: int = 42

    @field_validator("drop_feature_families")
    @classmethod
    def _sort_and_validate_families(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        known_families = set(FEATURE_FAMILIES) | set(NOTEBOOK_FEATURE_FAMILIES)
        unknown = sorted(set(value) - known_families)
        if unknown:
            raise ValueError(f"Unknown feature families: {unknown}")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "PipelineRecipe":
        validate_recipe(self)
        return self

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def recipe_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ResearchProposal(StrictModel):
    proposal_id: str
    parent_trial_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    hypothesis: str
    changed_axes: tuple[
        Literal[
            "hyperparameters",
            "feature_schema",
            "feature_family",
            "transform",
            "dataset_window",
            "model_family",
            "calibration",
            "market_blend",
            "target",
        ],
        ...,
    ]
    recipe: PipelineRecipe
    expected_observation: str
    falsification_rule: str
    max_trials: int = 1
    max_wall_seconds: int = 1200

    @model_validator(mode="after")
    def _validate_safe_text(self) -> "ResearchProposal":
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).lower()
        matches = sorted(term for term in FORBIDDEN_RESEARCH_TERMS if term in payload)
        if matches:
            raise ValueError(f"Research proposal contains forbidden terms: {matches}")
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")
        return self


def validate_recipe(recipe: PipelineRecipe) -> None:
    try:
        contract = target_contract(recipe.target.kind, recipe.target.parameters)
    except TargetContractError as exc:
        raise RecipeValidationError(str(exc)) from exc
    schema = FEATURE_SCHEMAS[recipe.feature_schema]
    schema = drop_feature_families(schema, recipe.drop_feature_families)
    validate_no_forbidden_label_features(contract, list(schema.features))
    _validate_model_target(recipe.model.kind, contract.model_task, recipe.target.kind)
    if recipe.target.kind != "win_probability" and recipe.blend.kind != "none":
        raise RecipeValidationError("Only win_probability recipes may use market blend")
    if recipe.target.kind != "win_probability" and recipe.calibration.kind == "temperature":
        raise RecipeValidationError("Temperature calibration is only registered for win_probability")


def experiment_spec_from_recipe(recipe: PipelineRecipe):
    """Convert currently executable recipes into legacy ExperimentSpec objects."""
    validate_recipe(recipe)
    if recipe.target.kind != "win_probability":
        raise RecipeValidationError("Legacy ExperimentSpec conversion only supports win_probability")
    if recipe.transforms:
        raise RecipeValidationError("Legacy ExperimentSpec conversion does not apply transforms")
    if recipe.train_window != "all_history":
        raise RecipeValidationError("Legacy ExperimentSpec conversion only supports all_history")
    if recipe.drop_feature_families:
        raise RecipeValidationError("Legacy ExperimentSpec conversion does not apply ablations")
    if recipe.model.kind not in {"logit", "boosted"}:
        raise RecipeValidationError(f"Unsupported legacy model kind: {recipe.model.kind}")
    from .experiments import ExperimentSpec

    run_id = f"recipe-{recipe.recipe_hash()}-{recipe.model.kind}"
    return ExperimentSpec(run_id, recipe.model.kind, dict(recipe.model.parameters))


def _validate_model_target(model_kind: str, task: str, target_kind: str) -> None:
    classifiers = {"logit", "boosted"}
    rankers = {"pairwise_ranker"}
    regressors = {"hist_gradient_regressor", "ridge_regressor"}
    if task == "classifier" and model_kind not in classifiers:
        raise RecipeValidationError(f"{target_kind} requires a classifier model")
    if task == "ranker" and model_kind not in rankers:
        raise RecipeValidationError(f"{target_kind} requires a ranker model")
    if task == "regressor" and model_kind not in regressors:
        raise RecipeValidationError(f"{target_kind} requires a regressor model")


def validate_research_proposal_batch(proposals: list[ResearchProposal]) -> None:
    proposal_ids: set[str] = set()
    recipe_hashes: set[str] = set()
    for proposal in proposals:
        if proposal.proposal_id in proposal_ids:
            raise RecipeValidationError(f"duplicate proposal_id: {proposal.proposal_id}")
        recipe_hash = proposal.recipe.recipe_hash()
        if recipe_hash in recipe_hashes:
            raise RecipeValidationError(f"duplicate recipe hash: {recipe_hash}")
        proposal_ids.add(proposal.proposal_id)
        recipe_hashes.add(recipe_hash)
