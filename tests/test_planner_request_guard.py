import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import FunctionType

from ima.planner_request_guard import bind_response, bounded_context, encoded, PaidCallCircuit, effective_dataset_audit


class PlannerRequestGuardTests(unittest.TestCase):
    def fixture(self):
        from ima.research_specs import ResearchProposal, PipelineRecipe, ModelSpec, TargetSpec
        proposal = ResearchProposal(proposal_id="p", hypothesis="Test regularization against reference",
            changed_axes=("hyperparameters",), expected_observation="Matched log loss improves",
            falsification_rule="Reject if matched log loss worsens",
            recipe=PipelineRecipe(schema_version=3, model=ModelSpec(kind="benter_conditional_logit"),
                                    target=TargetSpec(kind="win_probability")), max_trials=1)
        evidence = {"decision_id": "D000001", "evidence_id": "e", "completed_trial_index": [],
                    "terminal_watermark": 10}
        return evidence, {"decision": {"decision_id": "D000001", "evidence_id": "e",
            "trial_budget": 1, "programs": [proposal.model_dump(mode="json")]},
            "raw_response": {"provider": "unchanged"}, "usage": {"cost": .02}}

    def test_empty_binding_preserves_raw_and_input(self):
        evidence, response = self.fixture()
        before = copy.deepcopy(response)
        result = bind_response(response, evidence)
        self.assertEqual(response, before)
        self.assertEqual(result["decision"]["programs"][0]["evidence_ids"], ["e"])
        self.assertEqual(result["raw_response"], response["raw_response"])
        self.assertEqual(result["usage"], response["usage"])
        self.assertEqual(result["proposal_identity_binding"]["bound_proposal_ids"], ["p"])

    def test_explicit_stale_and_unknown_parent_fail(self):
        for key, value in (("evidence_ids", ["wrong"]), ("evidence_ids", None),
                           ("parent_trial_ids", ["unknown"])):
            evidence, response = self.fixture()
            response["decision"]["programs"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bind_response(response, evidence)
        evidence, response = self.fixture()
        response["decision"]["evidence_id"] = "wrong"
        with self.assertRaises(ValueError):
            bind_response(response, evidence)

    def test_context_is_bounded_and_keeps_critical_metadata(self):
        protected = {"completed_trial_index": [{"attempt_id": "a"}],
                     "capabilities": {"research_memo": {"content": "Entire memo"}},
                     "champions_by_contract": {"c": {"score": 2.1, "recipe": {"x": 1}}}}
        bundle = {"evidence": protected | {"remaining_program_capacity": {"old": 0, "live": 12},
            "paper_research": {"eligible_attempts": ["a"], "by_comparison": {"c": [
                {"ex_post_market_tainted": True, "assumptions": "Not executable",
                 "metrics": {"search": [{"score": i} for i in range(10000)]}}]}}}}
        before = copy.deepcopy(bundle)
        result = bounded_context(bundle)
        self.assertEqual(bundle, before)
        for key, value in protected.items():
            self.assertEqual(result["evidence"][key], value)
        self.assertEqual(result["evidence"]["remaining_program_capacity"], {"live": 12})
        self.assertLess(len(encoded(result)), 5000)
        self.assertEqual(result, bounded_context(result))
        with self.assertRaisesRegex(ValueError, "no paid call"):
            bounded_context({"evidence": {"capabilities": {"safety": "x"*400000}}})

    def test_persistent_circuit_and_new_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "decisions").mkdir()
            for number in (1, 2):
                evidence = {"decision_id": f"D{number:06}", "evidence_id": "e", "terminal_watermark": 10}
                PaidCallCircuit(campaign).check_and_record(evidence)
                (campaign / "decisions" / (evidence["decision_id"]+".json")).write_text(json.dumps(
                    {"planner_status": "failed", "error": "Proposal has explicit stale evidence"}))
            evidence["decision_id"] = "D000003"
            with self.assertRaisesRegex(ValueError, "circuit open"):
                PaidCallCircuit(campaign).check_and_record(evidence)
            self.assertFalse((campaign/"planner-request-guard/D000003.json").exists())
            evidence["terminal_watermark"] = 11
            PaidCallCircuit(campaign).check_and_record(evidence)
            with self.assertRaisesRegex(ValueError, "already dispatched"):
                PaidCallCircuit(campaign).check_and_record(evidence)

    def test_real_planner_path_with_isolated_deployment_globals(self):
        from ima import openrouter_orchestrator as planner, research_planner_context as context
        from ima.planner_request_guard import install_guard
        evidence, response = self.fixture()
        raw = {"choices": [{"message": {"content": json.dumps(response["decision"])}}],
               "usage": {"cost": .001}}
        for byte_budget in (380000, 1):
            original = planner.choose_research_decision
            clone = FunctionType(original.__code__, dict(original.__globals__),
                                 original.__name__, original.__defaults__, original.__closure__)
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(planner, "choose_research_decision", clone), \
                    patch.object(context, "decision_planner_evidence", context.decision_planner_evidence), \
                    patch.object(planner, "_post_json", return_value=raw) as post:
                install_guard(directory, request_bytes=byte_budget)
                if byte_budget == 1:
                    with self.assertRaisesRegex(ValueError, "byte budget"):
                        planner.choose_research_decision(evidence, {"trial_ceiling":260,"max_new_programs":12},
                                                        planner.OpenRouterConfig("unused","fixture/model"))
                    self.assertEqual(post.call_count, 0)
                else:
                    result = planner.choose_research_decision(evidence, {"trial_ceiling":260,"max_new_programs":12},
                                                             planner.OpenRouterConfig("unused","fixture/model"))
                    self.assertEqual(post.call_count, 1)
                    self.assertEqual(result["decision"]["programs"][0]["evidence_ids"], ["e"])
                    self.assertEqual(result["raw_response"], raw)

    def test_effective_default_dataset_is_comparison_only(self):
        from ima.proposal_audit import audit_decisions
        evidence, response = self.fixture()
        proposal = response["decision"]["programs"][0]
        proposal.update(parent_trial_ids=["parent"], fixed_parameters=True)
        parent = copy.deepcopy(proposal["recipe"])
        parent["dataset_ref"] = "dataset-current"
        parent["model"]["parameters"] = {"l2": 1.0}
        evidence["champions_by_contract"] = {"c": {"attempt_id":"parent","recipe":parent}}
        before = copy.deepcopy(response)
        result = effective_dataset_audit(audit_decisions, [response["decision"]], evidence, "dataset-current")
        self.assertFalse(any(f["code"] == "undeclared_recipe_changes" for f in result["findings"]))
        self.assertEqual(response, before)
        proposal["recipe"]["dataset_ref"] = "dataset-other"
        result = effective_dataset_audit(audit_decisions, [response["decision"]], evidence, "dataset-current")
        self.assertTrue(any(f["code"] == "undeclared_recipe_changes" for f in result["findings"]))

    def test_semantic_error_uses_existing_two_call_correction_loop(self):
        from ima import openrouter_orchestrator as planner, research_planner_context as context
        from ima.planner_request_guard import install_guard
        evidence, response = self.fixture()
        evidence["completed_trial_index"] = [{"attempt_id":"parent"}]
        evidence["champions_by_contract"] = {"c":{"attempt_id":"parent", "recipe":copy.deepcopy(
            response["decision"]["programs"][0]["recipe"])}}
        proposal = response["decision"]["programs"][0]
        proposal.update(parent_trial_ids=["parent"], fixed_parameters=True)
        proposal["recipe"]["transforms"] = [{"kind":"signed_log1p","parameters":{"columns":["prior_win_rate"]}}]
        corrected = copy.deepcopy(response["decision"])
        corrected["programs"][0]["changed_axes"] = ["transform"]
        raw = [{"choices":[{"message":{"content":json.dumps(decision)}}], "usage":{"cost":.001}}
               for decision in (response["decision"], corrected)]
        original = planner.choose_research_decision
        clone = FunctionType(original.__code__, dict(original.__globals__), original.__name__,
                             original.__defaults__, original.__closure__)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(planner, "choose_research_decision", clone), \
                patch.object(context, "decision_planner_evidence", context.decision_planner_evidence), \
                patch.object(planner, "_post_json", side_effect=raw) as post:
            install_guard(directory)
            result = planner.choose_research_decision(evidence, {"trial_ceiling":260,"max_new_programs":12},
                                                     planner.OpenRouterConfig("unused","fixture/model"))
            self.assertEqual(post.call_count, 2)
            self.assertEqual(result["repair_count"], 1)
            self.assertEqual(result["usage"]["total_cost_usd"], .002)
            self.assertEqual(result["proposal_identity_binding"]["bound_proposal_ids"], ["p"])
            repair = json.loads(post.call_args.args[1]["messages"][-1]["content"])
            self.assertIn("transforms", repair["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
