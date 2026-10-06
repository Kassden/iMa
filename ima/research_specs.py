"""Strict pipeline recipe contracts for the agentic optimizer."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from .feature_discovery_specs import DiscoverySpec, DiscoverySpecV2, parse_discovery_spec

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


FUNDAMENTAL_FIRST_PORTFOLIO_VERSION = "benter-portfolio-v3-2-fundamental"
FEATURE_DISCOVERY_PORTFOLIO_VERSION = "feature-discovery-v4-fundamental"
V5_PORTFOLIO_VERSION = "feature-discovery-v5-fundamental"
V6_PORTFOLIO_VERSION = "research-expansion-v6-fundamental"


def is_fundamental_first_portfolio(version: str | None) -> bool:
    return version in {
        FUNDAMENTAL_FIRST_PORTFOLIO_VERSION,
        FEATURE_DISCOVERY_PORTFOLIO_VERSION,
        V5_PORTFOLIO_VERSION,
        V6_PORTFOLIO_VERSION,
    }


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


MODEL_PARAMETER_CONTRACTS: dict[str, dict[str, str]] = {
    "gaussian_probit": {
        "l2": "number greater than 0 and at most 100",
        "scale_l2": "number greater than 0 and at most 100",
        "max_iter": "integer from 1 through 5000",
        "quadrature_order": "integer from 16 through 256",
        "heteroscedastic": "boolean",
    },
    "benter_conditional_logit": {
        "l2": "number from 0 through 100",
        "max_iter": "integer from 1 through 5000",
    },
    "lightgbm_lambdarank": {
        "learning_rate": "number greater than 0 and at most 1",
        "n_estimators": "integer from 1 through 2000",
        "num_leaves": "integer from 2 through 255",
        "min_child_samples": "integer from 1 through 1000",
    },
    "catboost_classifier": {
        "learning_rate": "number greater than 0 and at most 1",
        "iterations": "integer from 1 through 2000",
        "depth": "integer from 1 through 10",
        "l2_leaf_reg": "number from 0 through 100",
    },
    "catboost_regressor": {
        "learning_rate": "number greater than 0 and at most 1",
        "iterations": "integer from 1 through 2000",
        "depth": "integer from 1 through 10",
        "l2_leaf_reg": "number from 0 through 100",
    },
    "logit": {
        "C": "number greater than 0 and at most 100",
        "class_weight": "balanced or null",
        "max_iter": "integer from 1 through 5000",
    },
    "boosted": {
        "learning_rate": "number greater than 0 and at most 1",
        "max_iter": "integer from 1 through 2000; do not use n_estimators",
        "max_leaf_nodes": "integer from 2 through 255",
        "max_depth": "integer from 1 through 64 or null",
        "min_samples_leaf": "integer from 1 through 1000",
        "l2_regularization": "number from 0 through 100",
    },
    "pairwise_ranker": {
        "learning_rate": "number greater than 0 and at most 1",
        "max_iter": "integer from 1 through 2000; do not use n_estimators",
        "max_leaf_nodes": "integer from 2 through 255",
        "max_depth": "integer from 1 through 64 or null",
        "min_samples_leaf": "integer from 1 through 1000",
        "l2_regularization": "number from 0 through 100",
    },
    "hist_gradient_regressor": {
        "learning_rate": "number greater than 0 and at most 1",
        "max_iter": "integer from 1 through 2000; do not use n_estimators",
        "max_leaf_nodes": "integer from 2 through 255",
        "max_depth": "integer from 1 through 64 or null",
        "min_samples_leaf": "integer from 1 through 1000",
        "l2_regularization": "number from 0 through 100",
    },
    "ridge_regressor": {
        "alpha": "number from 0 through 1000000",
        "fit_intercept": "boolean",
    },
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
        "recorded_final_win_odds",
    ] = "win_probability"
    parameters: dict[str, Any] = Field(default_factory=dict)


class TransformSpec(StrictModel):
    kind: Literal[
        "clip_numeric_quantiles", "race_relative_rank", "race_relative_center",
        "signed_log1p", "numeric_interaction",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_transform(self) -> "TransformSpec":
        validate_transform_spec(RuntimeTransformSpec(self.kind, self.parameters))
        return self


class ModelSpec(StrictModel):
    kind: Literal[
        "logit",
        "benter_conditional_logit",
        "boosted",
        "pairwise_ranker",
        "hist_gradient_regressor",
        "ridge_regressor",
        "lightgbm_lambdarank",
        "catboost_classifier",
        "catboost_regressor",
        "gaussian_probit",
    ] = "logit"
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_parameters(self) -> "ModelSpec":
        _validate_model_parameters(self.kind, self.parameters)
        return self


class CalibrationSpec(StrictModel):
    kind: Literal["temperature", "none"] = "temperature"
    parameters: dict[str, Any] = Field(default_factory=dict)


class BlendSpec(StrictModel):
    kind: Literal["market_softmax", "none"] = "market_softmax"
    parameters: dict[str, Any] = Field(default_factory=dict)


class PerformanceDistributionSpec(StrictModel):
    kind: Literal["shared_residual", "catboost_uncertainty"] = "shared_residual"
    coordinate: Literal["log_speed_mps"] = "log_speed_mps"
    date_column: Literal["date"] = "date"
    scale_floor: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    scale_shrinkage: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_distribution(self):
        if self.kind == "shared_residual" and (self.parameters or self.scale_shrinkage != 0.1):
            raise ValueError("Shared residual scale does not take CatBoost parameters or shrinkage")
        if self.kind == "catboost_uncertainty":
            _validate_model_parameters("catboost_regressor", self.parameters)
        return self

    def runtime_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if self.kind == "shared_residual":
            payload.pop("scale_shrinkage")
            payload.pop("parameters")
        return payload


class PipelineRecipe(StrictModel):
    schema_version: Literal[2, 3] = 2
    target: TargetSpec = Field(default_factory=TargetSpec)
    feature_schema: Literal["baseline-v1", "benter-rich-v1", "notebook-rich-v2"] = "baseline-v1"
    drop_feature_families: tuple[str, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
    train_window: Literal["all_history", "trailing_3_years"] = "all_history"
    model: ModelSpec = Field(default_factory=ModelSpec)
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    blend: BlendSpec = Field(default_factory=BlendSpec)
    seed: int = 42
    feature_discovery: DiscoverySpecV2 | DiscoverySpec | None = None
    feature_definitions: tuple[dict[str, Any], ...] | None = None
    extra_numeric_features: tuple[str, ...] | None = None
    pipeline_graph: dict[str, Any] | None = None
    performance_distribution: dict[str, Any] | None = None
    dataset_ref: str | None = None

    @field_validator("feature_discovery", mode="before")
    @classmethod
    def _parse_discovery(cls, value):
        return parse_discovery_spec(value) if isinstance(value, dict) else value

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
        expansion_fields = (
            self.feature_definitions, self.extra_numeric_features, self.pipeline_graph,
            self.performance_distribution, self.dataset_ref,
        )
        if self.schema_version == 2 and any(v is not None for v in expansion_fields):
            raise ValueError("Expansion fields require recipe schema_version=3")
        if self.model.kind == "gaussian_probit" and self.schema_version != 3:
            raise ValueError("Gaussian probit requires recipe schema_version=3")
        if self.feature_discovery is not None and self.feature_discovery.schema_version == 2 and self.schema_version != 3:
            raise ValueError("V2 discovery requires recipe schema_version=3")
        if self.feature_definitions:
            from .feature_definitions import FeatureDefinition
            for definition in self.feature_definitions:
                FeatureDefinition.model_validate(definition)
        if self.performance_distribution is not None:
            distribution = PerformanceDistributionSpec.model_validate(self.performance_distribution)
            if self.target.kind != "adjusted_finish_time_or_speed":
                raise ValueError("Observed performance distributions require the speed target")
            if self.pipeline_graph is not None:
                raise ValueError("Specify distributions inside the graph or as an adapter, not both")
            if distribution.kind == "shared_residual" and self.model.kind not in {
                "ridge_regressor", "hist_gradient_regressor",
            }:
                raise ValueError("Shared residual adapter requires a supported sklearn mean regressor")
            if distribution.kind == "catboost_uncertainty" and self.model.kind != "catboost_regressor":
                raise ValueError("CatBoost uncertainty requires a CatBoost regressor mean contract")
        if self.pipeline_graph is not None:
            from .pipeline_graph import PipelineGraph
            graph = PipelineGraph.from_dict(self.pipeline_graph).validate()
            if graph.primary_node_id is None:
                raise ValueError("Tunable graphs require primary_node_id")
            for field, disabled in (("calibration", CalibrationSpec(kind="none")),
                                    ("blend", BlendSpec(kind="none"))):
                if field not in self.model_fields_set:
                    object.__setattr__(self, field, disabled)
                elif getattr(self, field).kind != "none":
                    raise ValueError(f"Graph recipes define {field} inside graph nodes, not an outer adapter")
            if self.target.kind != "win_probability":
                raise ValueError("Composed graph evaluation currently requires win_probability")
            for node in graph.nodes:
                if node.kind == "estimator":
                    kind = node.parameters.get("model_kind", "benter_conditional_logit")
                    parameters = node.parameters.get("model_parameters", node.parameters.get("parameters", {}))
                    ModelSpec(kind=kind, parameters=parameters)
                    if node.node_id == graph.primary_node_id and kind != self.model.kind:
                        raise ValueError("Outer model must match the primary graph estimator family")
            nodes = {node.node_id: node for node in graph.nodes}
            fundamental = nodes[graph.fundamental_node_id or graph.output_node_id]
            if fundamental.output.kind != "win_probability":
                raise ValueError("A win graph needs an explicit fundamental probability output")
        if self.extra_numeric_features:
            validate_no_forbidden_label_features(
                target_contract(self.target.kind, self.target.parameters),
                list(self.extra_numeric_features),
            )
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

    def feature_program_id(self, dataset_hash: str) -> str:
        payload = {
            "dataset_hash": dataset_hash,
            "feature_schema": self.feature_schema,
            "drop_feature_families": self.drop_feature_families,
            "transforms": [spec.model_dump(mode="json") for spec in self.transforms],
            "train_window": self.train_window,
            "feature_discovery": self.feature_discovery.model_dump(mode="json") if self.feature_discovery else None,
        }
        if self.schema_version == 3:
            payload.update(feature_definitions=self.feature_definitions,
                           extra_numeric_features=self.extra_numeric_features,
                           dataset_ref=self.dataset_ref)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


class SearchDimension(StrictModel):
    kind: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: tuple[str | int | float | bool | None, ...] = ()

    @model_validator(mode="after")
    def _validate_dimension(self) -> "SearchDimension":
        if self.kind == "categorical":
            if len(self.choices) < 2 or self.low is not None or self.high is not None or self.log:
                raise ValueError("categorical search requires at least two choices and no bounds")
        elif (
            self.low is None or self.high is None
            or isinstance(self.low, bool) or isinstance(self.high, bool)
            or self.low >= self.high or self.choices
        ):
            raise ValueError("numeric search requires ordered low/high bounds and no choices")
        elif self.kind == "int" and (
            not isinstance(self.low, int) or not isinstance(self.high, int)
        ):
            raise ValueError("integer search requires integer bounds")
        elif self.log and self.low <= 0:
            raise ValueError("logarithmic search requires positive bounds")
        return self


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
            "feature_creation",
            "dataset_build",
            "pipeline_composition",
            "performance_distribution",
        ],
        ...,
    ]
    recipe: PipelineRecipe
    search_space: dict[str, SearchDimension] = Field(default_factory=dict)
    fixed_parameters: bool = False
    expected_observation: str
    falsification_rule: str
    max_trials: int = 1
    max_wall_seconds: int = Field(default=1200,ge=1,strict=True)

    @model_validator(mode="after")
    def _validate_safe_text(self) -> "ResearchProposal":
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).lower()
        # Legacy contracts keep their text policy; V6 validates actual inputs instead.
        matches = sorted(term for term in FORBIDDEN_RESEARCH_TERMS if term in payload) if self.recipe.schema_version == 2 else []
        if matches:
            raise ValueError(f"Research proposal contains forbidden terms: {matches}")
        if self.recipe.schema_version == 2 and set(self.changed_axes) & {
            "feature_creation", "dataset_build", "pipeline_composition", "performance_distribution",
        }:
            raise ValueError("Expansion research axes require recipe schema_version=3")
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")
        if self.fixed_parameters and (self.max_trials != 1 or self.search_space):
            raise ValueError("A fixed-parameter control must use one trial and no search space")
        unknown = set(self.search_space) - set(MODEL_PARAMETER_CONTRACTS[self.recipe.model.kind])
        if unknown:
            raise ValueError(f"Search space contains unsupported model parameters: {sorted(unknown)}")
        for name, dimension in self.search_space.items():
            expected = MODEL_PARAMETER_CONTRACTS[self.recipe.model.kind][name]
            if ("integer" in expected and dimension.kind != "int") or (
                name in {"class_weight", "fit_intercept", "heteroscedastic"} and dimension.kind != "categorical"
            ):
                raise ValueError(f"Search dimension has wrong type for {name}")
            values = dimension.choices if dimension.kind == "categorical" else (
                dimension.low, dimension.high
            )
            for value in values:
                _validate_model_parameters(self.recipe.model.kind, {name: value})
        return self


def validate_recipe(recipe: PipelineRecipe) -> None:
    try:
        contract = target_contract(recipe.target.kind, recipe.target.parameters)
    except TargetContractError as exc:
        raise RecipeValidationError(str(exc)) from exc
    schema = FEATURE_SCHEMAS[recipe.feature_schema]
    schema = drop_feature_families(schema, recipe.drop_feature_families)
    validate_no_forbidden_label_features(contract, list(schema.features))
    if recipe.pipeline_graph is None:
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
    classifiers = {"logit", "boosted", "benter_conditional_logit", "catboost_classifier", "gaussian_probit"}
    rankers = {"pairwise_ranker", "lightgbm_lambdarank"}
    regressors = {"hist_gradient_regressor", "ridge_regressor", "catboost_regressor"}
    if model_kind in {"benter_conditional_logit", "gaussian_probit"} and target_kind != "win_probability":
        raise RecipeValidationError("benter_conditional_logit requires win_probability")
    if task == "classifier" and model_kind not in classifiers:
        raise RecipeValidationError(f"{target_kind} requires a classifier model")
    if task == "ranker" and model_kind not in rankers:
        raise RecipeValidationError(f"{target_kind} requires a ranker model")
    if task == "regressor" and model_kind not in regressors:
        raise RecipeValidationError(f"{target_kind} requires a regressor model")


def _validate_model_parameters(model_kind: str, parameters: dict[str, Any]) -> None:
    allowed = MODEL_PARAMETER_CONTRACTS[model_kind]
    unknown = sorted(set(parameters) - set(allowed))
    if unknown:
        raise RecipeValidationError(
            f"Unsupported {model_kind} parameters: {unknown}; allowed: {sorted(allowed)}"
        )

    def number(name: str, lower: float, upper: float, *, inclusive_lower: bool) -> None:
        if name not in parameters:
            return
        value = parameters[name]
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_type:
            raise RecipeValidationError(f"{model_kind}.{name} must be numeric")
        lower_ok = value >= lower if inclusive_lower else value > lower
        if not lower_ok or value > upper:
            bracket = "[" if inclusive_lower else "("
            raise RecipeValidationError(
                f"{model_kind}.{name} must be in {bracket}{lower}, {upper}]"
            )

    def integer(name: str, lower: int, upper: int, *, nullable: bool = False) -> None:
        if name not in parameters:
            return
        value = parameters[name]
        if nullable and value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= upper:
            raise RecipeValidationError(
                f"{model_kind}.{name} must be an integer from {lower} through {upper}"
            )

    if model_kind == "gaussian_probit":
        number("l2", 0, 100, inclusive_lower=False)
        number("scale_l2", 0, 100, inclusive_lower=False)
        integer("max_iter", 1, 5000)
        integer("quadrature_order", 16, 256)
        if "heteroscedastic" in parameters and not isinstance(parameters["heteroscedastic"], bool):
            raise RecipeValidationError("heteroscedastic must be boolean")
    elif model_kind == "lightgbm_lambdarank":
        number("learning_rate", 0.0, 1.0, inclusive_lower=False)
        integer("n_estimators", 1, 2000)
        integer("num_leaves", 2, 255)
        integer("min_child_samples", 1, 1000)
    elif model_kind in {"catboost_classifier", "catboost_regressor"}:
        number("learning_rate", 0.0, 1.0, inclusive_lower=False)
        number("l2_leaf_reg", 0.0, 100.0, inclusive_lower=True)
        integer("iterations", 1, 2000)
        integer("depth", 1, 10)
    elif model_kind == "benter_conditional_logit":
        number("l2", 0.0, 100.0, inclusive_lower=True)
        integer("max_iter", 1, 5000)
    elif model_kind == "logit":
        number("C", 0.0, 100.0, inclusive_lower=False)
        integer("max_iter", 1, 5000)
        if parameters.get("class_weight") not in {None, "balanced"}:
            raise RecipeValidationError("logit.class_weight must be balanced or null")
    elif model_kind in {"boosted", "pairwise_ranker", "hist_gradient_regressor"}:
        number("learning_rate", 0.0, 1.0, inclusive_lower=False)
        number("l2_regularization", 0.0, 100.0, inclusive_lower=True)
        integer("max_iter", 1, 2000)
        integer("max_leaf_nodes", 2, 255)
        integer("max_depth", 1, 64, nullable=True)
        integer("min_samples_leaf", 1, 1000)
    elif model_kind == "ridge_regressor":
        number("alpha", 0.0, 1_000_000.0, inclusive_lower=True)
        if "fit_intercept" in parameters and not isinstance(parameters["fit_intercept"], bool):
            raise RecipeValidationError("ridge_regressor.fit_intercept must be boolean")


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
