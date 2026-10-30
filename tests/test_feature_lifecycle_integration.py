"""Readback through the real dataset and model consumers, not just builders."""
import json

from ima.dataset_registry import DatasetRegistry
from ima.feature_definitions import FeatureDefinition
from ima.feature_lifecycle import build_feature_candidate, catalog
from ima.research_controller import protocol_spec_parameters
from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_specs import PipelineRecipe
from ima.research_lifecycle import link_feature_outcome
from tests.test_dataset_registry import DatasetRegistryTests


def test_successor_features_enter_training_and_keep_evaluation_protocol(tmp_path):
    registry, parent = DatasetRegistryTests()._build_row_hash_fixture(tmp_path)
    parent_dir = registry.dataset_path(parent["dataset_id"])
    definition = FeatureDefinition(name="carried_body_fraction", unit="1", input_refs=("actual_weight", "declared_weight"),
        expression_ast={"op": "divide", "args": [{"op": "ref", "ref": "actual_weight"},
                                                  {"op": "ref", "ref": "declared_weight"}]})
    successor = build_feature_candidate(parent_dir/"features.parquet", parent_dir/"manifest.json", registry.root,
                                        definitions=[definition])
    assert registry.verify(successor["dataset_id"])["ordered_row_keys_sha256"] == parent["ordered_row_keys_sha256"]
    assert successor["files"]["protocol.json"] == parent["files"]["protocol.json"]
    assert successor["files"]["confirmation_keys.json"] == parent["files"]["confirmation_keys.json"]
    recipe = PipelineRecipe.model_validate({"schema_version": 3, "feature_schema": "baseline-v1",
        "dataset_ref": successor["dataset_id"], "extra_numeric_features": [definition.feature_id],
        "model": {"kind": "benter_conditional_logit", "parameters": {"l2": 1., "max_iter": 1000}},
        "blend": {"kind": "none"}})
    request = RecipeExecutionRequest("successor-canary", "feature-build-consumer", 0, recipe,
        registry.dataset_path(successor["dataset_id"])/"features.parquet", tmp_path/"trials/successor",
        protocol_spec_parameters(json.loads((parent_dir/"protocol.json").read_text())))
    result = execute_recipe(request)
    assert result.status == "completed", result.error
    assert result.objective_value is not None
    assert all(definition.feature_id in fold["features"] for fold in result.metrics["effective_training"])
    saved = json.loads((tmp_path/"trials/successor/package/recipe.json").read_text())
    assert saved["dataset_ref"] == successor["dataset_id"]
    assert definition.feature_id in saved["extra_numeric_features"]
    assert catalog(registry.root)["counts"]["ready"] == 1
    linked = link_feature_outcome(request, result)
    assert linked["outcome"]["definition_ids"] == [definition.content_id()]
    assert result.lineage["feature_build_id"] == successor["feature_build_id"]
    assert catalog(registry.root)["builds"][0]["model_outcome_count"] == 1
