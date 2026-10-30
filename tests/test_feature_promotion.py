"""Adversarial scheduling receipts; scores never cross comparison contracts."""
from copy import deepcopy
import json

import pytest
from unittest.mock import patch

from ima.dataset_registry import _write_json
from ima.feature_lifecycle import FeatureLifecycle
from ima.feature_promotion import promotion_view, reference_identity, save_transfer


def reference():
    return {"attempt_id": "champion", "recipe": {"model": {"kind": "boosted"}},
            "parent_dataset_id": "parent", "code_revision": "sealed", "environment_hash": "env",
            "comparison_contract": {"comparison_contract_id": "contract", "target_contract": "win_probability",
                "evaluation_population_id": "population", "protocol_id": "protocol", "metric": "race_log_loss",
                "score_population_sha256": "rows", "direction": "minimize", "probability_basis": "fundamental"}}


def evidence(root, identifier, merit, *, family="fitness", contract=None, partial=False):
    lifecycle = FeatureLifecycle(root)
    definitions = [{"definition_id": identifier+"-a", "name": family+"_a"},
                   {"definition_id": identifier+"-b", "name": family+"_b"}]
    _write_json(lifecycle.root/"builds"/(identifier+".json"), {
        "build_id": identifier, "status": "ready", "dataset_id": identifier+"-dataset",
        "spec": {"parent_dataset_id": "parent", "definitions": definitions}})
    row = {"build_id": identifier, "paired": True, "improvement": merit,
           "outcome_id": identifier+"-outcome", "outcome": {
               "definition_ids": [d["definition_id"] for d in definitions][:1 if partial else 2],
               "comparison_contract": contract or reference()["comparison_contract"]}}
    _write_json(lifecycle.root/"outcomes"/(identifier+".json"), row)


def test_older_strong_bundle_survives_more_than_eight_recent_weak_and_restart(tmp_path):
    evidence(tmp_path, "old-best", .8)
    for number in range(30):
        evidence(tmp_path, f"new-{number:02}", .01)
    first = promotion_view(tmp_path, reference(), limit=2)
    assert first["candidates"][0]["build_id"] == "old-best"
    assert first["omitted_candidates"] == 29
    assert promotion_view(tmp_path, reference(), limit=2) == first


def test_compatible_full_bundle_only_with_family_diversity(tmp_path):
    evidence(tmp_path, "best", .8)
    evidence(tmp_path, "same-family", .7)
    evidence(tmp_path, "complement", .1, family="pace")
    evidence(tmp_path, "partial", 100, partial=True)
    other = deepcopy(reference()["comparison_contract"])
    other["probability_basis"] = "market"
    evidence(tmp_path, "incompatible", 500, contract=other)
    view = promotion_view(tmp_path, reference(), limit=2)
    assert [r["build_id"] for r in view["candidates"]] == ["best", "complement"]
    assert {r["build_id"] for r in view["pending"]} == {"partial", "incompatible"}


def test_negative_transfer_is_durable_deduplicated_and_reference_specific(tmp_path):
    evidence(tmp_path, "best", .8)
    ref = reference()
    plan = {"transfer_id": "transfer", "reference_id": reference_identity(ref), "build_id": "best"}
    save_transfer(tmp_path, plan)
    assert not promotion_view(tmp_path, ref)["candidates"]
    result = {"paired": {"paired_delta": .02, "verdict": "reject"}}
    save_transfer(tmp_path, plan, status="completed", result=result)
    assert save_transfer(tmp_path, plan)["status"] == "completed"
    with pytest.raises(ValueError, match="immutable"):
        save_transfer(tmp_path, plan, status="completed", result={"fake": True})
    view = promotion_view(tmp_path, ref)
    assert view["transfer_evidence"][0]["transfers"][0]["result"] == result
    changed = dict(ref, attempt_id="new-champion")
    assert promotion_view(tmp_path, changed)["candidates"][0]["build_id"] == "best"
    assert promotion_view(tmp_path, ref)["evidence_id"] == view["evidence_id"]


def test_repeated_screen_merit_uses_conservative_result_not_best_seed(tmp_path):
    evidence(tmp_path, "a", .8)
    lifecycle = FeatureLifecycle(tmp_path)
    path = lifecycle.root/"outcomes"/"a.json"
    repeat = json.loads(path.read_text()) | {"improvement": -.1, "outcome_id": "a-repeat"}
    _write_json(path.with_name("a-repeat.json"), repeat)
    assert promotion_view(tmp_path, reference())["candidates"][0]["screen_improvement"] == -.1


def test_cross_runtime_history_is_visible_without_becoming_current_reference_evidence(tmp_path):
    evidence(tmp_path, 'best', .8)
    prior = reference()
    plan = {'transfer_id': 'historical-transfer', 'reference_id': reference_identity(prior),
            'reference': prior, 'build_id': 'best', 'dataset_id': 'best-dataset'}
    result = {'paired': {'paired_delta': -.0015, 'race_bootstrap_ci95': [-.002, -.0001],
                         'race_deltas': [{'race_id': 'r', 'delta': -.0015}]},
              'control': {'attempt_id': 'actual-control', 'objective_value': 2.15},
              'candidate': {'attempt_id': 'actual-candidate', 'objective_value': 2.1485}}
    save_transfer(tmp_path, plan, status='completed', result=result)
    current = dict(prior, code_revision='successor-code')
    view = promotion_view(tmp_path, current)
    assert view['candidates'][0]['build_id'] == 'best'
    assert view['transfer_evidence'] == []
    related = view['related_reference_transfer_evidence']
    assert len(related) == 1 and related[0]['reference_match'] is False
    assert related[0]['candidate']['attempt_id'] == 'actual-candidate'
    assert related[0]['paired']['paired_delta'] == -.0015
    assert 'race_deltas' not in related[0]['paired']
    assert related[0]['receipt_sha256'] and related[0]['receipt_path']
    incompatible = deepcopy(current)
    incompatible['comparison_contract']['probability_basis'] = 'market'
    assert promotion_view(tmp_path, incompatible)['related_reference_transfer_evidence'] == []
    assert promotion_view(tmp_path, prior)['related_reference_transfer_evidence'] == []
    assert promotion_view(tmp_path, prior)['transfer_evidence'][0]['transfers'][0]['result'] == result


def test_unavailable_historical_dataset_is_reported_without_becoming_a_reference(tmp_path):
    from ima.feature_promotion import campaign_references
    champion = {'attempt_id': 'missing-history', 'recipe': {'dataset_ref': 'dataset-' + 'a' * 64}}
    rows = [{'attempt_id': 'missing-history', 'result': {}}]
    unavailable = []
    with patch('ima.experiment_discipline.attempt_rows', return_value=rows), patch(
            'ima.research_telemetry.champions', return_value={'champions_by_contract': {'contract': champion}}):
        assert campaign_references(tmp_path/'registry', tmp_path/'campaign', unavailable=unavailable) == []
    assert unavailable[0]['dataset_id'] == 'dataset-' + 'a' * 64
    assert unavailable[0]['eligible_as_reference'] is False
    assert unavailable[0]['reason'] == 'reference_dataset_unavailable'


def test_corrupt_historical_dataset_does_not_get_silently_skipped(tmp_path):
    from ima.feature_promotion import campaign_references
    champion = {'attempt_id': 'bad-history', 'recipe': {'dataset_ref': 'dataset-' + 'b' * 64}}
    with patch('ima.experiment_discipline.attempt_rows', return_value=[{'attempt_id': 'bad-history', 'result': {}}]), patch(
            'ima.research_telemetry.champions', return_value={'champions_by_contract': {'contract': champion}}), patch(
            'ima.dataset_registry.DatasetRegistry.verify', side_effect=ValueError('manifest checksum mismatch')):
        with pytest.raises(ValueError, match='checksum mismatch'):
            campaign_references(tmp_path/'registry', tmp_path/'campaign', unavailable=[])


def test_missing_payload_of_existing_history_does_not_get_skipped(tmp_path):
    from ima.feature_promotion import campaign_references
    dataset_id = 'dataset-' + 'c' * 64
    champion = {'attempt_id': 'broken-history', 'recipe': {'dataset_ref': dataset_id}}
    missing_payload = tmp_path/'registry'/'datasets'/dataset_id/'features.parquet'
    error = FileNotFoundError(2, 'missing payload', str(missing_payload))
    with patch('ima.experiment_discipline.attempt_rows', return_value=[{'attempt_id': 'broken-history', 'result': {}}]), patch(
            'ima.research_telemetry.champions', return_value={'champions_by_contract': {'contract': champion}}), patch(
            'ima.dataset_registry.DatasetRegistry.verify', side_effect=error):
        with pytest.raises(FileNotFoundError):
            campaign_references(tmp_path/'registry', tmp_path/'campaign', unavailable=[])
