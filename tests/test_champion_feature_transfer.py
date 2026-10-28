"""Real registry/trainer transfer path plus scientific mismatch attacks."""
from copy import deepcopy
import json
from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest

from ima.feature_lifecycle import FeatureLifecycle, build_feature_candidate
from ima.feature_promotion import paired_transfer_report, promotion_view, transfer_plan, validate_successor
from tests.test_feature_exploration_integration import explorer_fixture


def setup_transfer(tmp_path, **config):
    explorer = explorer_fixture(tmp_path, **config)
    explorer.refill()
    pending = explorer.queue.rows()[0]
    child = build_feature_candidate(explorer.config.parent_features,
        explorer.config.parent_features.parent/"manifest.json", explorer.config.registry,
        definitions=pending["definitions"])
    baseline = explorer.baseline()
    observation = explorer._observation(baseline, explorer.parent, [])
    ref = {"attempt_id": baseline["attempt_id"], "recipe": explorer.recipe(explorer.parent["dataset_id"], ()).canonical_payload(),
           "parent_dataset_id": explorer.parent["dataset_id"], "comparison_contract": observation["comparison_contract"],
           "code_revision": explorer.config.code_revision, "environment_hash": explorer.config.environment_hash}
    return explorer, child, ref


def test_real_complete_bundle_transfer_persists_matched_control_and_negative_evidence(tmp_path):
    explorer, child, ref = setup_transfer(tmp_path)
    row = explorer.transfer(child["feature_build_id"], ref)
    assert row["status"] == "completed"
    assert row["plan"]["feature_ids"] == sorted(child["feature_build"]["feature_names"])
    report = row["result"]["paired"]
    assert report["races"] > 0 and report["race_deltas"] and report["fold_deltas"]
    assert row["result"]["control"]["recipe_hash"] != row["result"]["candidate"]["recipe_hash"]
    assert explorer.transfer(child["feature_build_id"], ref) == row
    view = promotion_view(explorer.config.registry, ref)
    assert view["transfer_evidence"][0]["transfers"][0]["result"]["paired"] == report

def test_fixed_transformed_graph_champion_accepts_destination_specific_bundle(tmp_path):
    from tests.test_pipeline_graph import graph_spec
    explorer, child, ref = setup_transfer(tmp_path)
    recipe = deepcopy(ref["recipe"])
    recipe["model"] = {"kind": "benter_conditional_logit", "parameters": {"l2": .2, "max_iter": 40}}
    recipe["pipeline_graph"] = graph_spec()
    recipe["calibration"] = {"kind": "none", "parameters": {}}
    recipe["transforms"] = [
        {"kind": "signed_log1p", "parameters": {"columns": ["actual_weight"]}},
        {"kind": "clip_numeric_quantiles", "parameters": {"columns": ["actual_weight"], "lower": .1, "upper": .9}},
    ]
    from ima.research_specs import PipelineRecipe
    frozen_recipe = PipelineRecipe.model_validate(recipe)
    request = replace(explorer._request("transformed-graph-reference", explorer.parent), recipe=frozen_recipe)
    reference_result = explorer._execute(request)
    observation = explorer._observation(reference_result, explorer.parent, [])
    ref = dict(ref, recipe=frozen_recipe.canonical_payload(), attempt_id=reference_result["attempt_id"],
               comparison_contract=observation["comparison_contract"])
    plan = transfer_plan(explorer.config.registry, child["feature_build_id"], ref, destination="boosted")
    assert plan["control_recipe"]["transforms"] == recipe["transforms"]
    assert plan["candidate_recipe"]["transforms"] == recipe["transforms"]
    controls = {n["node_id"]: n for n in plan["control_recipe"]["pipeline_graph"]["nodes"]}
    candidates = {n["node_id"]: n for n in plan["candidate_recipe"]["pipeline_graph"]["nodes"]}
    assert candidates["benter"] == controls["benter"]
    assert candidates["pool"] == controls["pool"]
    assert candidates["boosted"]["parameters"] == controls["boosted"]["parameters"]
    control_view = controls[controls["boosted"]["inputs"][0]]
    candidate_view = candidates[candidates["boosted"]["inputs"][0]]
    assert candidate_view["parameters"]["features"] == (
        control_view["parameters"]["features"] + plan["feature_ids"])
    row = explorer.transfer(child["feature_build_id"], ref, destination="boosted", attribution=True)
    assert row["status"] == "completed"
    assert row["result"]["paired"]["races"] > 0
    original_predictions = pd.read_csv(reference_result["artifacts"]["predictions"])
    control_predictions = pd.read_csv(row["result"]["control"]["artifacts"]["predictions"])
    pd.testing.assert_series_equal(original_predictions.model_probability, control_predictions.model_probability,
                                   check_exact=True)
    from ima.feature_promotion import attribution_recipes
    variants = attribution_recipes(plan)
    for feature in plan["feature_ids"]:
        ablated = variants["without-" + feature]
        nodes = {node["node_id"]: node for node in ablated["pipeline_graph"]["nodes"]}
        assert feature not in nodes[nodes["boosted"]["inputs"][0]]["parameters"]["features"]


def test_managed_fit_ignores_interrupted_result_and_reuses_only_matching_acceptance(tmp_path):
    explorer = explorer_fixture(tmp_path)
    request = explorer._request("feature-interrupted-control", explorer.parent)
    assert request.supervised_fit is True
    from ima.feature_exploration import feature_screen_worker
    legacy = feature_screen_worker(replace(request, supervised_fit=False), explorer.config.native_threads)
    assert legacy["status"] == "completed"
    (request.output_dir/"runtime.json").write_text(json.dumps({"status": "timed_out", "task_nonce": "stale"}))

    class RejectSubmit:
        def submit(self, *args, **kwargs):
            raise AssertionError("A matching accepted episode should be reused")

    recovered = explorer._execute(request)
    assert recovered["status"] == "completed"
    assert recovered["attempt_id"].startswith(request.attempt_id + "--execution-")
    accepted_path = explorer.root/"trials"/recovered["attempt_id"]
    stored = json.loads((accepted_path/"result.json").read_text())
    receipt = stored["_execution_acceptance"]
    runtime = json.loads((accepted_path/"runtime.json").read_text())
    assert receipt["attempt_id"] == recovered["attempt_id"]
    assert receipt["task_nonce"] == runtime["task_nonce"]
    assert runtime["execution_acceptance"] == receipt
    assert explorer._execute(request, RejectSubmit()) == recovered

    changed_environment = replace(request, environment_hash="different-environment")
    with pytest.raises(ValueError, match="execution input changed: environment_hash"):
        explorer._execute(changed_environment, RejectSubmit())
    changed_dataset = replace(request, dataset_hash="different-dataset")
    with pytest.raises(ValueError, match="dataset hash differs"):
        explorer._execute(changed_dataset, RejectSubmit())

    expected = {"protocol_id": recovered["lineage"]["protocol_id"],
                "protocol_hash": recovered["lineage"]["protocol_hash"]}
    assert explorer._execute(request, RejectSubmit(), expected_protocol=expected) == recovered
    with pytest.raises(ValueError, match="protocol differs"):
        explorer._execute(request, RejectSubmit(), expected_protocol=expected | {"protocol_hash": "wrong"})


def test_transfer_mlflow_readback_binds_run_identity_and_rejects_tag_mismatch(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = "sqlite:///" + str(tmp_path/"transfer-tracking.sqlite")
    explorer, child, ref = setup_transfer(tmp_path, tracking_uri=uri)
    transfer = explorer.transfer(child["feature_build_id"], ref)
    tracking_path = FeatureLifecycle(explorer.config.registry).root/"transfers"/(transfer["transfer_id"]+"-tracking.json")
    tracking = json.loads(tracking_path.read_text())
    assert tracking["status"] == "published", tracking
    client = MlflowClient(tracking_uri=uri)
    plan = transfer["plan"]
    for role in ("control", "candidate"):
        run = client.get_run(tracking["runs"][role])
        result = transfer["result"][role]
        expected_dataset = plan["parent_dataset_id"] if role == "control" else plan["dataset_id"]
        assert run.info.status == "FINISHED"
        assert run.data.tags["ima.transfer_id"] == plan["transfer_id"]
        assert run.data.tags["ima.reference_id"] == plan["reference_id"]
        assert run.data.tags["ima.role"] == role
        assert run.data.tags["ima.attempt_id"] == result["attempt_id"]
        assert run.data.params["dataset_id"] == expected_dataset
        assert run.data.params["dataset_hash"] == result["lineage"]["dataset_hash"]
        assert run.data.params["protocol_hash"] == result["lineage"]["protocol_hash"]
        assert run.data.params["environment_hash"] == result["lineage"]["environment_hash"]

    client.set_tag(tracking["runs"]["control"], "ima.role", "candidate")
    explorer.transfer(child["feature_build_id"], ref)
    readback = json.loads(tracking_path.read_text())
    assert readback["status"] == "pending"
    assert "Transfer MLflow readback failed" in readback["error"]


@pytest.mark.parametrize("mutation", ["label", "value", "order", "drop", "schema"])
def test_parent_value_population_and_schema_changes_fail_closed(tmp_path, mutation):
    explorer, child, _ = setup_transfer(tmp_path)
    frame = pd.read_parquet(child["features_path"])
    if mutation == "label":
        frame.loc[0, "target_win"] = 999
    elif mutation == "value":
        frame.loc[0, "actual_weight"] += 1
    elif mutation == "order":
        frame = frame.iloc[::-1]
    elif mutation == "drop":
        frame = frame.iloc[:-1]
    else:
        frame = frame.drop(columns=["target_win"])
    path = tmp_path/"tampered.parquet"
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="parent"):
        validate_successor(explorer.parent, dict(child, features_path=str(path)))


def test_fixed_recipe_and_reference_population_guards(tmp_path):
    explorer, child, ref = setup_transfer(tmp_path)
    plan = transfer_plan(explorer.config.registry, child["feature_build_id"], ref)
    control, candidate = deepcopy(plan["control_recipe"]), deepcopy(plan["candidate_recipe"])
    for key in ("dataset_ref", "extra_numeric_features"):
        control.pop(key, None)
        candidate.pop(key, None)
    assert control == candidate
    changed = deepcopy(ref)
    changed["comparison_contract"]["protocol_id"] = "wrong"
    with pytest.raises(ValueError, match="comparison contract"):
        transfer_plan(explorer.config.registry, child["feature_build_id"], changed)
    with pytest.raises(ValueError, match="runtime"):
        explorer.transfer(child["feature_build_id"], dict(ref, code_revision="other"))


def test_paired_report_rejects_changed_labels_and_duplicate_runners(tmp_path):
    explorer, _, _ = setup_transfer(tmp_path)
    control = Path(explorer.baseline()["artifacts"]["predictions"])
    frame = pd.read_csv(control)
    frame.loc[0, "target_win"] = 999
    candidate = tmp_path/"candidate.csv"
    frame.to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="labels/population"):
        paired_transfer_report(candidate, control, "win_probability")
    pd.concat([frame, frame.iloc[:1]]).to_csv(candidate, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_transfer_report(candidate, control, "win_probability")
