import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ima.research_lifecycle import feature_feedback, is_probit_recipe, lifecycle_probit_fit, _REQUEST, compact_optional_history, prove_local_preflight, _qualified_unsent
import pytest


def test_probit_graph_detection():
    assert is_probit_recipe(SimpleNamespace(model=SimpleNamespace(kind="gaussian_probit"), pipeline_graph=None))
    assert is_probit_recipe(SimpleNamespace(model=SimpleNamespace(kind="benter_conditional_logit"), pipeline_graph={
        "nodes": [{"parameters": {"model_kind": "gaussian_probit"}}]}))
    assert not is_probit_recipe(SimpleNamespace(model=SimpleNamespace(kind="boosted"), pipeline_graph=None))


def test_feature_feedback_does_not_claim_untested_improvement(tmp_path):
    report = feature_feedback(tmp_path, [{"dataset_id": "dataset-candidate"}])
    assert report["counts"]["definitions_completed"] == 0
    assert report["outcomes"][0]["status"] == "untested"
    assert "matched" in report["interpretation"]


def test_probit_callback_preserves_fold_and_training_context(tmp_path):
    from ima import performance_probit as probit
    recipe = SimpleNamespace(canonical_payload=lambda: {"transforms": [{"kind": "signed_log1p"}], "train_window": "all_history"})
    request = SimpleNamespace(output_dir=tmp_path, attempt_id="a", dataset_hash="d", recipe=recipe,
                              protocol_parameters={"max_folds": 3}, code_revision="r")
    (tmp_path/"progress.json").write_text(json.dumps({"fold_id": "fold-001", "fold_index": 1, "fold_count": 3}))
    seen = {}
    def fit(train, schema, params, **kwargs):
        seen.update(kwargs)
        kwargs["progress_callback"]({"stage": "optimizing", "objective": 1., "iteration": 2,
                                    "objective_evaluations": 3})
        return "model"
    token = _REQUEST.set(request)
    try:
        with patch.object(probit, "_lifecycle_original_fit", fit, create=True):
            assert lifecycle_probit_fit(None, None, {}) == "model"
    finally:
        _REQUEST.reset(token)
    progress = json.loads((tmp_path/"progress.json").read_text())
    assert progress["fold_id"] == "fold-001"
    assert progress["objective"] == 1.
    assert seen["training_context"]["dataset_hash"] == "d"
    assert seen["training_context"]["protocol"] == {"max_folds": 3}
    assert "evaluation" not in seen["training_context"]


def test_direct_fit_without_campaign_context_unchanged():
    from ima import performance_probit as probit
    with patch.object(probit, "_lifecycle_original_fit", return_value="direct", create=True) as fit:
        assert lifecycle_probit_fit("train", "schema", {"l2": 1}) == "direct"
    fit.assert_called_once_with("train", "schema", {"l2": 1})


def test_context_keeps_parent_ids_safety_and_pending_recipes():
    evidence = {"completed_trial_index": [{"attempt_id": "a"}], "hypothesis_memory": [{}]*8,
                "capabilities": {"eligible_predictors": {"x": {"target_tainted": True}}},
                "pending_programs": [{"recipe": {"transforms": [1]}}]}
    result = compact_optional_history(evidence)
    assert len(result["hypothesis_memory"]) == 4
    assert len(evidence["hypothesis_memory"]) == 8
    for key in ("completed_trial_index", "capabilities", "pending_programs"):
        assert result[key] == evidence[key]


def test_preflight_proof_never_settles_uncertain_provider_cost(tmp_path):
    (tmp_path/"decisions").mkdir()
    (tmp_path/"evidence").mkdir()
    decision = {"planner_status": "failed", "error": "Mandatory planner context exceeds byte budget: 300011>300000; no paid call",
                "transport_attempts": [], "planner_usage": {}}
    path = tmp_path/"decisions/D000236.json"
    path.write_text(json.dumps(decision))
    (tmp_path/"evidence/D000236.json").write_text("{}")
    import hashlib
    qualification = {"D000236": {"decision_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256((tmp_path/"evidence/D000236.json").read_bytes()).hexdigest()}}
    with patch("ima.research_lifecycle.QUALIFIED_PREFLIGHTS", qualification):
        proof = prove_local_preflight(tmp_path, "D000236")
        assert _qualified_unsent(tmp_path, "D000236")
    assert proof["provider_cost"] is None
    decision["error"] = "TimeoutError: unknown provider billing"
    path.write_text(json.dumps(decision))
    assert not _qualified_unsent(tmp_path, "D000236")
    assert json.loads((tmp_path/"lifecycle-preflight-proofs/D000236.json").read_text()) == proof


def test_preflight_proof_rejects_any_call_evidence(tmp_path):
    (tmp_path/"decisions").mkdir()
    (tmp_path/"evidence").mkdir()
    (tmp_path/"planner-calls").mkdir()
    (tmp_path/"decisions/D000236.json").write_text(json.dumps({"planner_status": "failed",
        "error": "Mandatory planner context exceeds byte budget: 300011>300000; no paid call"}))
    (tmp_path/"evidence/D000236.json").write_text("{}")
    (tmp_path/"planner-calls/D000236-01.json").write_text("{}")
    with pytest.raises(ValueError, match="transport/call"):
        prove_local_preflight(tmp_path, "D000236")


def test_matching_error_alone_cannot_waive_unknown_billing(tmp_path):
    (tmp_path/"decisions").mkdir()
    (tmp_path/"evidence").mkdir()
    (tmp_path/"decisions/D000236.json").write_text(json.dumps({"planner_status": "failed",
        "error": "Mandatory planner context exceeds byte budget: 300011>300000; no paid call"}))
    (tmp_path/"evidence/D000236.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed independent"):
        prove_local_preflight(tmp_path, "D000236")


def test_missing_proof_input_leaves_billing_unresolved(tmp_path):
    (tmp_path/"lifecycle-preflight-proofs").mkdir()
    (tmp_path/"lifecycle-preflight-proofs/D000999.json").write_text("{}")
    assert not _qualified_unsent(tmp_path, "D000999")


def test_feedback_failure_does_not_discard_successful_fit(tmp_path):
    from ima import research_expansion as expansion
    from ima.research_lifecycle import lifecycle_worker
    request = SimpleNamespace(output_dir=tmp_path, attempt_id="a")
    result = SimpleNamespace(status="completed", metrics={}, artifacts={})
    result.serializable = lambda: {"status": result.status, "metrics": result.metrics, "artifacts": result.artifacts}
    with patch.object(expansion, "_lifecycle_original_worker", return_value=result, create=True), \
         patch("ima.research_lifecycle.link_feature_outcome", side_effect=OSError("tracking unavailable")):
        assert lifecycle_worker(request, None, 1).status == "completed"
    assert (tmp_path/"feature-feedback-pending.json").exists()
    assert json.loads((tmp_path/"result.json").read_text())["status"] == "completed"


def test_hooks_reach_compiled_campaign_namespace(tmp_path):
    from ima import research_expansion as expansion, performance_probit, mlflow_tracking
    from ima.research_lifecycle import install_campaign_policy, lifecycle_worker, lifecycle_prepare
    names = ("_capabilities", "_planner_spend", "_worker", "_prepare_program", "BoundedFitExecutor")
    isolated = {name: getattr(expansion, name) for name in names}
    exec("def run(): return _worker, _prepare_program, _capabilities, _planner_spend, BoundedFitExecutor", isolated)
    patches = {name: getattr(expansion, name) for name in names}
    patches.update(run_expansion_campaign=isolated["run"], _lifecycle_installed=False)
    with patch.multiple(expansion, **patches, create=True), \
         patch.object(performance_probit, "fit_probit", performance_probit.fit_probit), \
         patch.object(mlflow_tracking, "research_run_parameters", mlflow_tracking.research_run_parameters):
        install_campaign_policy(tmp_path, release_sha256="release")
        actual = isolated["run"]()
        assert actual[0] is lifecycle_worker and actual[1] is lifecycle_prepare
        assert actual[2] is expansion._capabilities and actual[3] is expansion._planner_spend
        assert actual[4] is expansion.BoundedFitExecutor
