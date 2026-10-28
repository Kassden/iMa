import copy
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ima.pipeline_graph import fit_pipeline_graph
from ima.prediction_store import PredictionArtifact, PredictionStore, frame_fingerprint, prediction_key
from ima.research_pooling import ForwardSimplexPool
from tests.test_pipeline_graph import SCHEMA, graph_spec, race_frame


def mixture_data(weight=.35):
    frame = race_frame(6, 10)
    rng = np.random.default_rng(13)
    matrix = np.column_stack([rng.dirichlet(np.ones(4), 6).ravel() for _ in range(2)])
    frame["target_probability"] = matrix @ np.array([weight, 1-weight])
    blocks = []
    for indices in np.array_split(np.arange(len(frame)), 3):
        scoring = frame.iloc[indices]
        keys = tuple(zip(scoring.race_id.astype(str), scoring.horse_id.astype(str)))
        artifacts = []
        for component in range(2):
            contract = {"kind": "estimator", "parameters": {"component": component},
                        "output": {"kind": "win_probability", "target": "target_win",
                                   "unit": "probability", "market": False}}
            metadata = {"fit_scope": "forward_oof", "component_id": prediction_key(contract),
                        "component_contract": contract, "training_fingerprint": f"past-{indices[0]}",
                        "prediction_fingerprint": frame_fingerprint(scoring),
                        "selection_mode": "development_selected", "selection_cutoff": str(frame.date.max()),
                        "training_cutoff": str(scoring.date.min()-pd.Timedelta(days=1)),
                        "prediction_start": str(scoring.date.min())}
            artifacts.append(PredictionArtifact(matrix[indices, component], keys, metadata))
        blocks.append(artifacts)
    return frame, blocks


def fit_pool(frame, blocks, **options):
    return ForwardSimplexPool.fit(frame, blocks, outer_training_cutoff=frame.date.max(), **options)


@pytest.mark.parametrize("weight", [0., .35, 1.])
def test_known_mixture_and_valid_zero_endpoints(weight):
    frame, blocks = mixture_data(weight)
    pool = fit_pool(frame, blocks)
    np.testing.assert_allclose(pool.weights, [weight, 1-weight], atol=2e-6)
    if weight in (0, 1):
        assert pool.weights[1 if weight else 0] == 0
    assert pool.report["selection_mode"] == "development_selected"
    assert pool.report["race_log_loss"] <= min(pool.report["controls"].values())+1e-10


@pytest.mark.parametrize("fault", ["missing_component", "missing_block", "reorder", "duplicate", "nan",
                                   "same_meeting", "contract", "target_contract", "fingerprint", "population"])
def test_reject_corrupt_or_missing_forward_lineage(fault):
    frame, blocks = mixture_data()
    if fault == "missing_component":
        blocks[1].pop()
    elif fault == "missing_block":
        blocks.pop()
    elif fault == "reorder":
        blocks[0][1].row_keys = tuple(reversed(blocks[0][1].row_keys))
    elif fault == "duplicate":
        blocks[0][0].row_keys = (blocks[0][0].row_keys[0],)*len(blocks[0][0].row_keys)
    elif fault == "nan":
        blocks[0][0].values[0] = np.nan
    elif fault == "same_meeting":
        blocks[0][0].metadata["training_cutoff"] = str(frame.date.min()+pd.Timedelta(hours=1))
    elif fault == "contract":
        blocks[0][0].metadata["component_contract"]["parameters"] = {"changed": True}
    elif fault == "target_contract":
        blocks[0][1].metadata["component_contract"]["output"]["target"] = "place_probability"
        blocks[0][1].metadata["component_id"] = prediction_key(blocks[0][1].metadata["component_contract"])
    elif fault == "fingerprint":
        blocks[0][0].metadata["prediction_fingerprint"] = "fake-oof"
    elif fault == "population":
        blocks[0][1].metadata["training_fingerprint"] = "different-training-rows"
    with pytest.raises(ValueError):
        fit_pool(frame, blocks)


def test_selection_cutoff_is_separate_from_fit_cutoff():
    frame, blocks = mixture_data()
    with pytest.raises(ValueError, match="explicit selection_cutoff"):
        fit_pool(frame, blocks, selection_mode="nested_selection")
    with pytest.raises(ValueError, match="source-linked"):
        fit_pool(frame, blocks, selection_mode="nested_selection", selection_cutoff=frame.date.min())
    with pytest.raises(ValueError, match="outer training"):
        fit_pool(frame, blocks, selection_cutoff=frame.date.max()+pd.Timedelta(days=1))
    for block in blocks:
        for artifact in block:
            artifact.metadata.update(selection_mode="nested_selection", selection_cutoff="2020-01-01")
    with pytest.raises(ValueError, match='source-linked'):
        fit_pool(frame, blocks, selection_mode="nested_selection", selection_cutoff="2020-01-01")


def learned_spec(**parameters):
    spec = graph_spec()
    spec["nodes"][-1]["kind"] = "learned_probability_pool"
    spec["nodes"][-1]["parameters"] = parameters
    return spec


def test_actual_graph_cache_reuse_serialization_and_meeting_blocks():
    train, score = race_frame(15), race_frame(2, 15)
    # Two races in the same meeting with different timestamps cannot straddle a split.
    extra = train.copy()
    extra["race_id"] += "-later"
    extra["date"] += pd.Timedelta(hours=2)
    train = pd.concat([train, extra], ignore_index=True)
    train = train.sort_values(["date", "horse_id"]).reset_index(drop=True)
    spec = learned_spec(selection_mode="development_selected", selection_cutoff="2019-12-31")
    with tempfile.TemporaryDirectory() as path:
        store = PredictionStore(path)
        fitted = fit_pipeline_graph(spec, train, feature_schema=SCHEMA, prediction_store=store)
        warm = fit_pipeline_graph(spec, train, feature_schema=SCHEMA, prediction_store=store)
        assert warm.fit_report["cache_hits"] == 6
        assert warm.fit_report["physical_fits"] < fitted.fit_report["physical_fits"]
        p = fitted.predict_proba(score.drop(columns=["target_win", "target_probability"]))
        np.testing.assert_allclose(warm.predict_proba(score), p)
        np.testing.assert_allclose(pickle.loads(pickle.dumps(fitted)).predict_proba(score), p)
        assert len(fitted.fit_report["learned_pools"]) == 1
        assert fitted.fit_report["learned_pools"][0]["weights"] == fitted.output.state.weights.tolist()
        for block in fitted.output.state.report["forward_blocks"]:
            assert pd.Timestamp(block["training_cutoff"]).normalize() < pd.Timestamp(block["prediction_start"])
        json.dumps(fitted.fit_report, allow_nan=False)


def test_cache_with_valid_but_wrong_metadata_is_rejected():
    train = race_frame(12)
    with tempfile.TemporaryDirectory() as path:
        store = PredictionStore(path)
        fit_pipeline_graph(learned_spec(), train, feature_schema=SCHEMA, prediction_store=store)
        cache_path = next(Path(path).glob("*.npz"))
        with np.load(cache_path, allow_pickle=False) as stored:
            payload = {k: stored[k].copy() for k in stored.files}
        metadata = json.loads(str(payload["metadata"]))
        metadata["seed"] = 999
        payload["metadata"] = json.dumps(metadata)
        np.savez(cache_path, **payload)
        with pytest.raises(ValueError, match="lineage"):
            fit_pipeline_graph(learned_spec(), train, feature_schema=SCHEMA, prediction_store=store)


def test_executor_pool_package_reload_and_outer_label_isolation(tmp_path):
    from ima.research_executor import RecipeExecutionRequest, execute_recipe
    from ima.research_model_package import load_research_package
    from ima.research_specs import PipelineRecipe

    frame = pd.read_csv("tests/fixtures/research_races.csv")
    frame["horse_id"] = frame.horse_no.astype(str)
    frame["field_size"] = frame.groupby("race_id").race_id.transform("size")
    dataset = tmp_path / "races.csv"
    frame.to_csv(dataset, index=False)
    spec = learned_spec()
    spec["n_splits"] = 2
    recipe = PipelineRecipe(schema_version=3, pipeline_graph=spec,
                            model={"kind": "benter_conditional_logit", "parameters": {"max_iter": 20}})
    request = RecipeExecutionRequest(attempt_id="forward-pool", proposal_id="pool-canary", trial_number=0,
        recipe=recipe, dataset_path=dataset, output_dir=tmp_path / "fit",
        protocol_parameters={"min_train_races": 4, "calibration_races": 1, "score_races": 1, "max_folds": 1})
    result = execute_recipe(request)
    assert result.status == "completed", result.error
    package = load_research_package(Path(result.artifacts["package"]))
    predictions = pd.read_csv(result.artifacts["predictions"])
    score = frame[frame.race_id.isin(predictions.race_id)].copy()
    score["date"] = pd.to_datetime(score.date)
    expected = package.predict_fundamental_proba(score)
    np.testing.assert_allclose(expected, predictions.model_probability, atol=1e-12)
    assert result.metrics["component_attribution"][0]["learned_pools"][0]["selection_mode"] == "development_selected"
    # Change only outer-score winners. Fitted pool and all inference probabilities must survive unchanged.
    scored = frame.race_id.isin(predictions.race_id)
    for column in ("result", "target_win", "target_probability"):
        frame.loc[scored, column] = frame.loc[scored, column].iloc[::-1].to_numpy()
    frame.to_csv(dataset, index=False)
    from dataclasses import replace
    repeated = execute_recipe(replace(request, attempt_id="changed-labels", output_dir=tmp_path / "changed"))
    assert repeated.status == "completed", repeated.error
    np.testing.assert_allclose(pd.read_csv(repeated.artifacts["predictions"]).model_probability, expected)
