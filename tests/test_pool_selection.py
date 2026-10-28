from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ima.feature_sets import FeatureSchema
from ima.pool_selection import (
    SelectionContractError,
    load_selection_artifact,
    select_prefix_graph,
    validate_selection_artifact,
    validate_selection_protocol,
    write_selection_artifact,
)
from ima.prediction_store import frame_fingerprint
from tests.test_pipeline_graph import race_frame


SCHEMA = FeatureSchema("selection-test", ("ability", "context"), ())
SOURCE = {
    "dataset_hash": "1" * 64,
    "dataset_manifest_hash": "2" * 64,
    "source_revision": "test-revision-70b87a4",
    "source_code_hash": "3" * 64,
    "protocol_hash": "4" * 64,
    "target_contract": {"kind": "win_probability", "label": "target_probability"},
    "evaluation_population_id": "selection-test-population",
    "evaluation_population_hash": "5" * 64,
}


def graph(*, features=("ability",), model_parameters=None, final_kind="estimator"):
    return {
        "graph_id": "prefix-candidate",
        "output_node_id": "model",
        "date_column": "date",
        "horse_column": "horse_id",
        "nodes": [
            {"node_id": "view", "kind": "feature_view",
             "parameters": {"features": list(features)},
             "output": {"kind": "feature_frame", "target": "features", "unit": "features"}},
            {"node_id": "model", "kind": final_kind, "inputs": ["view"],
             "parameters": {"model_kind": "benter_conditional_logit",
                            "model_parameters": model_parameters or {"max_iter": 80}}},
        ],
    }


def protocol(*, candidates=None, **updates):
    candidate_graphs = candidates or [
        {"candidate_id": "ability-only", "graph": graph(features=("ability",))},
        {"candidate_id": "context-only", "graph": graph(features=("context",))},
    ]
    value = {
        "schema_version": 1,
        "selection_cutoff": "2020-01-20",
        "validation_start": "2020-01-16",
        "min_prefix_train_meetings": 10,
        "min_prefix_validation_meetings": 5,
        "registered_features": ["ability", "context"],
        "candidate_graphs": candidate_graphs,
        "selection_trial_budget": len(candidate_graphs),
        "seed": 17,
        "objective": "mean_meeting_log_loss",
    }
    value.update(updates)
    return value


def fixture_frame():
    frame = race_frame(25)
    frame["field_size"] = frame.groupby("race_id")["race_id"].transform("size")
    return frame.sort_values(["date", "race_id", "horse_id"]).reset_index(drop=True)


def run_selection(tmp_path: Path, *, data=None, selection_protocol=None):
    data = fixture_frame() if data is None else data
    selection_protocol = protocol() if selection_protocol is None else selection_protocol
    return select_prefix_graph(selection_protocol, data=data, feature_schema=SCHEMA,
        source_context=SOURCE, future_oof_start="2020-01-21",
        artifact_path=tmp_path / "selection.json")


def _rehash(payload):
    value = {key: item for key, item in payload.items()
             if key not in {"selection_id", "selection_sha256"}}
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    payload["selection_sha256"] = digest
    payload["selection_id"] = "selection-" + digest[:24]
    return payload


def test_actual_shared_fitter_selects_and_writes_source_pinned_artifact(tmp_path):
    frame = fixture_frame()
    result = run_selection(tmp_path, data=frame)
    assert result.candidate_id in {"ability-only", "context-only"}
    assert result.artifact_path.read_bytes() == json.dumps(
        result.artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    assert result.artifact["selection_mode"] == "nested_selection"
    assert result.artifact["scope"] == {
        "selection_scope": "outer_training_prefix_development",
        "independent_confirmation": False,
        "promotion_allowed": False,
    }
    assert result.artifact["optuna"]["sampler"] == "GridSampler"
    assert {r["candidate_id"] for r in result.artifact["candidate_results"]} == {
        "ability-only", "context-only"}
    assert all(r["training_cutoff"] < r["validation_start"]
               for r in result.artifact["candidate_results"])
    assert all(r["training_fingerprint"] == frame_fingerprint(
        frame[pd.to_datetime(frame.date).dt.normalize() < pd.Timestamp("2020-01-16")])
        for r in result.artifact["candidate_results"])
    loaded = load_selection_artifact(result.artifact_path, protocol=protocol(),
        feature_schema=SCHEMA, source_context=SOURCE, source_frame=frame,
        future_oof_start="2020-01-21")
    assert loaded["selection_sha256"] == result.artifact["selection_sha256"]


def test_future_label_perturbation_cannot_change_prefix_selection(tmp_path):
    frame = fixture_frame()
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first = run_selection(first_dir, data=frame)

    changed = frame.copy()
    future = pd.to_datetime(changed.date).dt.normalize() >= pd.Timestamp("2020-01-21")
    changed.loc[future, "target_probability"] = float("nan")
    changed.loc[future, "target_win"] = 1 - changed.loc[future, "target_win"]
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = run_selection(second_dir, data=changed)
    assert second.candidate_id == first.candidate_id
    assert second.artifact["selection_sha256"] == first.artifact["selection_sha256"]
    assert second.artifact["prefix"] == first.artifact["prefix"]


@pytest.mark.parametrize("fault", ["stale_hash", "changed_recipe", "changed_prefix",
                                   "changed_cutoff", "changed_source", "changed_schema"])
def test_artifact_readback_rejects_corrupt_or_mismatched_evidence(tmp_path, fault):
    frame = fixture_frame()
    result = run_selection(tmp_path, data=frame)
    payload = copy.deepcopy(result.artifact)
    active_protocol = protocol()
    source_frame = frame
    active_source = SOURCE
    feature_schema = SCHEMA
    if fault == "stale_hash":
        payload["selected_candidate_id"] = "forged-candidate"
    elif fault == "changed_recipe":
        active_protocol["candidate_graphs"][0]["graph"]["nodes"][1]["parameters"]["model_parameters"]["max_iter"] = 81
    elif fault == "changed_prefix":
        source_frame = frame.copy()
        mask = pd.to_datetime(source_frame.date).dt.normalize() == pd.Timestamp("2020-01-01")
        idx = source_frame.loc[mask].index
        source_frame.loc[idx, "target_probability"] = source_frame.loc[idx, "target_probability"].to_numpy()[::-1]
    elif fault == "changed_cutoff":
        active_protocol["selection_cutoff"] = "2020-01-19"
    elif fault == "changed_source":
        active_source = {**SOURCE, "source_revision": "different-revision"}
    elif fault == "changed_schema":
        feature_schema = FeatureSchema("selection-test-changed", ("ability",), ())
    with pytest.raises(SelectionContractError):
        validate_selection_artifact(payload, protocol=active_protocol, feature_schema=feature_schema,
            source_context=active_source, source_frame=source_frame, future_oof_start="2020-01-21")


def test_protocol_rejects_small_or_missing_warmup_and_same_date_split(tmp_path):
    too_small = protocol(min_prefix_train_meetings=2)
    with pytest.raises(SelectionContractError, match=">= 3"):
        validate_selection_protocol(too_small, feature_schema=SCHEMA)
    same_day = protocol(validation_start="2020-01-15")
    frame = fixture_frame()
    same_date_race = frame.loc[pd.to_datetime(frame.date).dt.normalize() == pd.Timestamp("2020-01-16")].copy()
    same_date_race["race_id"] = same_date_race["race_id"].astype(str) + "-same-meeting"
    same_date_race["date"] = pd.Timestamp("2020-01-15")
    frame = pd.concat([frame, same_date_race], ignore_index=True).sort_values(
        ["date", "race_id", "horse_id"]).reset_index(drop=True)
    # The selector treats every runner/race on a date as one indivisible meeting.
    result = run_selection(tmp_path, data=frame, selection_protocol=same_day)
    assert result.artifact["prefix"]["training_end"] == "2020-01-14"
    assert result.artifact["prefix"]["validation_start"] == "2020-01-15"
    assert result.artifact["prefix"]["validation_races"] > result.artifact["prefix"]["validation_meetings"]


@pytest.mark.parametrize("kind", ["learned_probability_pool", "meta_estimator", "calibrate",
                                   "market_blend", "race_normalize", "rank_distribution"])
def test_protocol_rejects_recursive_or_unqualified_candidate_nodes(kind):
    candidates = [
        {"candidate_id": "bad", "graph": graph(final_kind=kind)},
        {"candidate_id": "good", "graph": graph(features=("context",))},
    ]
    with pytest.raises(SelectionContractError, match="unsupported candidate graph node kind"):
        validate_selection_protocol(protocol(candidates=candidates), feature_schema=SCHEMA)


def test_protocol_rejects_unregistered_features_custom_fields_and_unstable_endpoints():
    unregistered = protocol()
    unregistered["candidate_graphs"][0]["graph"]["nodes"][0]["parameters"]["features"] = ["secret"]
    with pytest.raises(SelectionContractError, match="registered feature"):
        validate_selection_protocol(unregistered, feature_schema=SCHEMA)

    custom = protocol()
    custom["factory"] = "arbitrary.callable"
    with pytest.raises(SelectionContractError, match="extra"):
        validate_selection_protocol(custom, feature_schema=SCHEMA)

    changed_endpoint = protocol()
    changed_endpoint["candidate_graphs"][1]["graph"]["output_node_id"] = "model-v2"
    changed_endpoint["candidate_graphs"][1]["graph"]["nodes"][1]["node_id"] = "model-v2"
    with pytest.raises(SelectionContractError, match="stable endpoint"):
        validate_selection_protocol(changed_endpoint, feature_schema=SCHEMA)


def test_prefix_cutoff_must_precede_future_oof_and_artifact_path_is_immutable(tmp_path):
    with pytest.raises(SelectionContractError, match="strictly after"):
        select_prefix_graph(protocol(), data=fixture_frame(), feature_schema=SCHEMA,
            source_context=SOURCE, future_oof_start="2020-01-20",
            artifact_path=tmp_path / "bad.json")

    first_dir = tmp_path / "immutable"
    first_dir.mkdir()
    result = run_selection(first_dir, data=fixture_frame())
    different_valid_payload = copy.deepcopy(result.artifact)
    different_valid_payload["selected_candidate_id"] = "different-but-self-hashed"
    _rehash(different_valid_payload)
    with pytest.raises(SelectionContractError, match="different bytes"):
        write_selection_artifact(result.artifact_path, different_valid_payload)


def test_empty_data_and_inadequate_validation_fail_without_fallback(tmp_path):
    with pytest.raises(SelectionContractError, match="cannot be empty"):
        run_selection(tmp_path, data=fixture_frame().iloc[:0])
    protocol_with_one_validation_day = protocol(
        validation_start="2020-01-20", min_prefix_validation_meetings=2)
    with pytest.raises(SelectionContractError, match="below its explicit meeting minimum"):
        run_selection(tmp_path, data=fixture_frame(), selection_protocol=protocol_with_one_validation_day)


def test_incomplete_race_and_nonforward_fit_scope_are_rejected(tmp_path):
    incomplete = fixture_frame().drop(index=0).reset_index(drop=True)
    with pytest.raises(SelectionContractError, match="complete race fields"):
        run_selection(tmp_path, data=incomplete)

    bad_graph = graph()
    bad_graph["nodes"][1]["fit_scope"] = "training_only"
    with pytest.raises(SelectionContractError, match="forward_oof fit_scope"):
        validate_selection_protocol(protocol(candidates=[
            {"candidate_id": "bad-scope", "graph": bad_graph},
        ]), feature_schema=SCHEMA)
