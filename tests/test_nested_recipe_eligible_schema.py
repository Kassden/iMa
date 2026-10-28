"""Declaration checks precede exact dataset-aware nested-selector validation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ima.research_executor import RecipeExecutionRequest, execute_recipe
from ima.research_model_package import load_research_package
from ima.research_specs import FUNDAMENTAL_FIRST_PORTFOLIO_VERSION, PipelineRecipe
from tests.test_nested_pool_executor import nested_graph
from tests.test_research_executor_v6 import dataset


FEATURES = ["horse_rating", "prior_win_rate"]


def recipe(registered=FEATURES):
    graph = nested_graph()
    graph["nodes"][-1]["parameters"]["selection_protocol"]["registered_features"] = registered
    return PipelineRecipe(schema_version=3, feature_schema="baseline-v1",
                          model={"kind": "benter_conditional_logit", "parameters": {}},
                          pipeline_graph=graph)


def request(root, eligible):
    source, _ = dataset(root, races=40)
    (root / "manifest.json").write_text(json.dumps({
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "feature_catalog": {name: {"eligible": True, "unit": "1", "temporal_scope": "pre_race"}
                            for name in eligible},
        "eligible_categorical_predictors": [],
    }))
    return RecipeExecutionRequest(
        "eligible-nested", "eligible-nested", 0, recipe(), source,
        root / "trials/eligible-nested",
        {"min_train_races": 25, "calibration_races": 3, "score_races": 3,
         "max_folds": 1, "whole_meeting_boundaries": True},
        code_revision="eligible-schema-test", environment_hash="fixture",
        portfolio_version=FUNDAMENTAL_FIRST_PORTFOLIO_VERSION,
    )


def test_eligible_subset_executes_and_package_replays(tmp_path):
    execution = request(tmp_path, FEATURES)
    result = execute_recipe(execution)
    assert result.status == "completed", result.error
    fits = json.loads(Path(result.artifacts["graph_fits"]).read_text())
    selection = json.loads(Path(fits["folds"][0]["selections"][0]["artifact_path"]).read_text())
    assert selection["feature_schema"]["numeric"] == FEATURES
    assert selection["feature_schema"]["categorical"] == []
    package = load_research_package(result.artifacts["package"])
    predictions = pd.read_csv(result.artifacts["predictions"])
    frame = pd.read_csv(execution.dataset_path)
    frame["date"] = pd.to_datetime(frame.date)
    frame = frame.loc[frame.race_id.isin(predictions.race_id)]
    np.testing.assert_allclose(package.predict_proba(frame), predictions.model_probability,
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("registered", [[], ["prior_win_rate", "horse_rating"],
                                      ["horse_rating", "target_win"],
                                      ["horse_rating", "horse_rating"], "horse_rating"])
def test_declaration_cannot_add_duplicate_or_reorder_features(registered):
    with pytest.raises(ValueError, match="ordered subset"):
        recipe(registered)


def test_runtime_still_requires_exact_actual_eligible_schema(tmp_path):
    execution = request(tmp_path, ["horse_age", *FEATURES])
    result = execute_recipe(execution)
    assert result.status == "failed"
    assert "registered_features must exactly match" in result.error
    assert not list(execution.output_dir.glob("nested-selection/*/selection-*.json"))
