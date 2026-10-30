import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ima.proposal_audit import audit_decisions, planner_proposal_feedback, render_markdown


def proposal(pid="P1", **overrides):
    return dict(proposal_id=pid, parent_trial_ids=[], evidence_ids=[],
                hypothesis="Test matched feature behavior.", changed_axes=["transform"],
                recipe={"schema_version": 3, "blend": {"kind": "none"}},
                fixed_parameters=True, expected_observation="Lower paired loss.",
                falsification_rule="Reject if paired loss does not decrease.",
                max_trials=1, max_wall_seconds=1200) | overrides


def decision(*programs, **fields):
    return dict(decision_id="D1", programs=list(programs), **fields)


def codes(result):
    return {f["code"] for f in result["findings"]}


def graph_recipe():
    return dict(schema_version=3, model={"kind": "benter_conditional_logit", "parameters": {"l2": 1}},
                pipeline_graph=dict(graph_id="one", primary_node_id="a", output_node_id="pool",
                    nodes=[dict(node_id="a", kind="estimator", parameters={"model_kind": "benter_conditional_logit"}),
                           dict(node_id="b", kind="estimator", parameters={"model_kind": "boosted"}),
                           dict(node_id="pool", kind="weighted_probability_pool", inputs=["a", "b"],
                                parameters={"weights": [.6, .4]})]))


class ProposalAuditTests(unittest.TestCase):
    def test_planner_feedback_full_memory_bounded_recent_and_unresolved(self):
        memory = [{"kind": "planned", "hypothesis_id": f"D{i:06}",
                   "created_at": f"2026-10-08T00:{i:02}:00+00:00",
                   "payload": decision(proposal(f"P{i}"))} for i in range(20)]
        for i, row in enumerate(memory):
            row["payload"]["decision_id"] = row["hypothesis_id"]
        evidence = {"hypothesis_memory": memory}
        before = copy.deepcopy(evidence)
        feedback = planner_proposal_feedback(evidence)
        self.assertEqual(feedback, planner_proposal_feedback(evidence))
        self.assertEqual(before, evidence)
        self.assertEqual(feedback["proposal_count"], 20)
        self.assertEqual(feedback["counts"]["decisions"], 20)
        self.assertEqual(feedback["audited_decision_ids"][0], "D000019")
        self.assertEqual(feedback["audited_decision_ids_omitted"], 8)
        self.assertEqual(feedback["counts"]["gain_evidence"], {"unresolved": 20})
        self.assertEqual(len(feedback["warnings"]), 12)
        self.assertEqual(feedback["warnings"][0]["proposal_id"], "P19")
        self.assertGreater(feedback["warnings_omitted"], 0)
        self.assertTrue(all(w["classification"] == "warning" for w in feedback["warnings"]))

    def test_planner_feedback_missing_malformed_payloads_and_truncation(self):
        for evidence in ({}, {"hypothesis_memory": None}, {"hypothesis_memory": [None,
                {"kind": "planned"}, {"kind": "planned", "payload": []},
                {"kind": "planned", "payload": {}},
                {"kind": "planned", "payload": {"programs": None}}]}):
            feedback = planner_proposal_feedback(evidence)
            self.assertEqual(feedback["proposal_count"], 0)
            json.dumps(feedback, allow_nan=False)
        malformed = proposal() | {"invalid" * 300: True}
        feedback = planner_proposal_feedback({"hypothesis_memory": [{"kind": "planned",
            "payload": decision(malformed)}]})
        warning = next(w for w in feedback["warnings"] if w["code"] == "invalid_proposal")
        self.assertLessEqual(len(warning["message"]), 800)
        self.assertTrue(warning["message_truncated"])
        self.assertTrue(warning["message"].endswith(" [truncated]"))
        self.assertEqual(warning["classification"], "warning")
        self.assertEqual(warning["source_classification"], "factual_failure")

    def test_planner_feedback_negatives_survive_planned_dominance(self):
        memory = [{"kind": "planned", "payload": decision(proposal(f"P{i}"))} for i in range(15)]
        memory += [{"kind": "preparation_rejected", "hypothesis_id": f"failed-{i}",
                    "created_at": f"2026-10-08T00:{i:02}:00Z", "payload": {"error": "failure"}}
                   for i in range(7)]
        feedback = planner_proposal_feedback({"hypothesis_memory": memory})
        self.assertEqual(feedback["counts"]["negative_events"], 7)
        self.assertEqual(len(feedback["recent_negatives"]), 5)
        self.assertEqual(feedback["recent_negatives"][0]["hypothesis_id"], "failed-6")
        self.assertEqual(feedback["negative_events_omitted"], 2)

    def test_actual_raw_memory_feedback_when_available(self):
        path = Path(__file__).resolve().parents[1] / ".tmp/context-baseline-D000126.json"
        if not path.exists():
            self.skipTest("Frozen local raw evidence is not part of the distribution")
        feedback = planner_proposal_feedback(json.loads(path.read_text()))
        self.assertEqual(feedback["proposal_count"], 23)
        self.assertEqual(feedback["audited_decision_ids"], [f"D{i:06}" for i in range(125, 120, -1)])
        self.assertEqual(feedback["counts"]["gain_evidence"], {"unresolved": 23})
        self.assertEqual(feedback["counts"]["negative_events"], 9)
        self.assertTrue(any(w["proposal_id"].startswith("P-D000125") for w in feedback["warnings"]))

    def test_empty_and_invalid(self):
        self.assertEqual(audit_decisions([], {})["counts"]["proposals"], 0)
        result = audit_decisions([None, decision({}), decision(proposal(recipe={"model": {"kind": "unknown"}}))], None)
        self.assertIn("invalid_proposal", codes(result))
        self.assertIn("invalid_evidence", codes(result))
        json.dumps(result, allow_nan=False)

    def test_invalid_target_model_and_search_contract(self):
        invalid = proposal(recipe={"schema_version": 3, "target": {"kind": "adjusted_finish_time_or_speed"},
                                   "model": {"kind": "logit"}})
        result = audit_decisions([decision(invalid)], {})
        self.assertFalse(result["proposals"][0]["valid"])
        result = audit_decisions([decision(proposal(search_space={"unknown": {"kind": "int", "low": 1, "high": 2}}))], {})
        self.assertIn("invalid_proposal", codes(result))

    def test_repeated_fixed_same_seed_and_different_seed(self):
        a, b, c = proposal(), proposal("P2"), proposal("P3")
        c["recipe"]["seed"] = 43
        result = audit_decisions([decision(a, b, c)], {})
        self.assertEqual(result["counts"]["findings_by_code"]["repeated_fixed_same_seed_control"], 1)
        self.assertNotIn("repeat_of", result["proposals"][2])

    def test_default_and_explicit_effective_search_spaces_repeat(self):
        from ima.research_search import _default_space
        from ima.research_specs import ResearchProposal

        omitted = proposal("omitted", fixed_parameters=False)
        explicit = copy.deepcopy(omitted)
        explicit["proposal_id"] = "explicit"
        model_kind = ResearchProposal.model_validate(omitted).recipe.model.kind
        explicit["search_space"] = {
            name: dimension.model_dump(mode="json")
            for name, dimension in _default_space(model_kind).items()
        }
        result = audit_decisions([decision(omitted, explicit)], {})
        self.assertEqual(result["proposals"][0]["search_signature_sha256"],
                         result["proposals"][1]["search_signature_sha256"])
        self.assertIn("repeated_recipe_search", codes(result))

    def test_fixed_and_effectively_searched_proposals_remain_distinct(self):
        fixed = proposal("fixed", fixed_parameters=True)
        searched = proposal("searched", fixed_parameters=False)
        result = audit_decisions([decision(fixed, searched)], {})
        self.assertNotEqual(result["proposals"][0]["search_signature_sha256"],
                            result["proposals"][1]["search_signature_sha256"])
        self.assertNotIn("repeated_recipe_search", codes(result))
        self.assertNotIn("repeated_fixed_same_seed_control", codes(result))

    def test_search_signature_ignores_only_the_tuned_graph_parameter_value(self):
        recipe = graph_recipe()
        recipe["pipeline_graph"]["tune_node_id"] = "b"
        recipe["model"] = {"kind": "boosted", "parameters": {"max_iter": 20}}
        recipe["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"] = {"l2": 1.0}
        recipe["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"] = {"max_iter": 5}
        first = proposal("first", recipe=recipe, fixed_parameters=False,
                         search_space={"max_iter": {"kind": "int", "low": 10, "high": 30}})
        tuned_initial_differs = copy.deepcopy(first)
        tuned_initial_differs["proposal_id"] = "tuned-initial-differs"
        tuned_initial_differs["recipe"]["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"]["max_iter"] = 99
        untuned_differs = copy.deepcopy(first)
        untuned_differs["proposal_id"] = "untuned-differs"
        untuned_differs["recipe"]["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"]["l2"] = 2.0

        result = audit_decisions([decision(first, tuned_initial_differs, untuned_differs)], {})
        signatures = [row["search_signature_sha256"] for row in result["proposals"]]
        self.assertEqual(signatures[0], signatures[1])
        self.assertNotEqual(signatures[0], signatures[2])
        self.assertIn("repeated_recipe_search", codes(result))

    def test_unseen_full_parent_and_champion(self):
        parent = "attempt-" + "a" * 64
        p = proposal(parent_trial_ids=[parent, parent + "x"])
        e = {"champions_by_contract": {"contract": {"attempt_id": parent, "recipe": p["recipe"]}}}
        result = audit_decisions([decision(p)], e)
        self.assertEqual(result["proposals"][0]["champion_parent_ids"], [parent])
        self.assertIn("unseen_parent", codes(result))
        self.assertIn(parent + "x", str(result))

    def test_meaningful_ablation_and_172800_second_deadline(self):
        parent = {"schema_version": 3, "blend": {"kind": "none"}}
        recipe = copy.deepcopy(parent)
        recipe["train_window"] = "trailing_3_years"
        p = proposal(recipe=recipe, parent_trial_ids=["attempt-parent"],
                     changed_axes=["dataset_window"], max_wall_seconds=172800)
        result = audit_decisions([decision(p)], {"champions_by_contract": {"c": {"attempt_id": "attempt-parent", "recipe": parent}}})
        self.assertEqual(result["findings"], [])
        row = result["proposals"][0]
        self.assertEqual(row["max_wall_seconds"], 172800)
        self.assertEqual(row["parent_comparisons"][0]["changed_recipe_fields"], ["train_window"])
        self.assertEqual(row["gain_evidence"], "unresolved")

    def test_graph_names_preserve_edges_weights_roles_and_primary_binding(self):
        a = proposal(recipe=graph_recipe())
        b = copy.deepcopy(a)
        b["proposal_id"] = "P2"
        g = b["recipe"]["pipeline_graph"]
        g["graph_id"] = "cosmetic"
        rename = {"a": "x", "b": "y", "pool": "z"}
        for node in g["nodes"]:
            node["node_id"] = rename[node["node_id"]]
            node["inputs"] = [rename[i] for i in node.get("inputs", [])]
        g["primary_node_id"], g["output_node_id"] = "x", "z"
        g["nodes"][0]["parameters"]["model_parameters"] = {"l2": 1}
        g["nodes"].reverse()
        result = audit_decisions([decision(a, b)], {})
        self.assertIn("repeated_fixed_same_seed_control", codes(result))
        self.assertNotEqual(result["proposals"][0]["recipe_sha256"], result["proposals"][1]["recipe_sha256"])
        g["nodes"][0]["inputs"].reverse()
        self.assertNotIn("repeated_fixed_same_seed_control", codes(audit_decisions([decision(a, b)], {})))
        g["nodes"][0]["inputs"].reverse()
        g["nodes"][0]["parameters"]["weights"] = [.5, .5]
        self.assertNotIn("repeated_fixed_same_seed_control", codes(audit_decisions([decision(a, b)], {})))

    def test_tuned_estimator_binding_does_not_normalize_untuned_blend_component(self):
        from ima.proposal_audit import _semantic_recipe

        recipe = graph_recipe()
        recipe["pipeline_graph"]["tune_node_id"] = "b"
        recipe["model"] = {"kind": "boosted", "parameters": {"max_iter": 20}}
        recipe["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"] = {"l2": 1.0}
        recipe["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"] = {"max_iter": 5}
        first = _semantic_recipe(recipe)
        changed_tuned_initial = copy.deepcopy(recipe)
        changed_tuned_initial["pipeline_graph"]["nodes"][1]["parameters"]["model_parameters"]["max_iter"] = 99
        self.assertEqual(first, _semantic_recipe(changed_tuned_initial))
        changed_untuned = copy.deepcopy(recipe)
        changed_untuned["pipeline_graph"]["nodes"][0]["parameters"]["model_parameters"]["l2"] = 2.0
        self.assertNotEqual(first, _semantic_recipe(changed_untuned))
    def test_search_signature_and_budgets(self):
        a = proposal(fixed_parameters=False, search_space={"C": {"kind": "float", "low": .1, "high": 1}})
        b = copy.deepcopy(a)
        b["proposal_id"] = "P2"
        b["search_space"]["C"]["high"] = 2
        result = audit_decisions([decision(a, b, trial_budget=1)], {})
        self.assertIn("program_budget_exceeds_decision", codes(result))
        self.assertNotIn("repeated_recipe_search", codes(result))
        b["search_space"]["C"]["high"] = 1
        self.assertIn("repeated_recipe_search", codes(audit_decisions([decision(a, b)], {})))

    def test_execution_claims_failures_and_unfinished(self):
        e = {"hypothesis_memory": [
            {"kind": "evaluated", "hypothesis_id": "prog", "payload": {"attempt_id": "a", "status": "completed", "objective_value": 1}},
            {"kind": "preparation_rejected", "hypothesis_id": "prog2", "payload": {"error": "deadline expired"}}]}
        result = audit_decisions([decision(proposal(), proposal("P2"), program_ids=["prog", "prog2"])], e)
        self.assertEqual(result["proposals"][0]["gain_evidence"], "completed_without_verified_matched_gain")
        self.assertIn("observed_execution_failure", codes(result))
        self.assertEqual(result["proposals"][1]["gain_evidence"], "unresolved")

    def test_contract_matching_is_required_for_development_delta(self):
        p = proposal(parent_trial_ids=["parent"])
        e = {"champions_by_contract": {"c": {"attempt_id": "parent", "recipe": p["recipe"],
              "comparison_key": "contract-a", "objective_value": 2}},
             "active_trials": [{"attempt_id": "child", "program_id": "prog", "status": "completed",
                                "comparison_key": "contract-b", "objective_value": 1}]}
        d = [decision(p, program_ids=["prog"])]
        self.assertEqual(audit_decisions(d, e)["proposals"][0]["gain_evidence"], "completed_without_verified_matched_gain")
        e["active_trials"][0]["comparison_key"] = "contract-a"
        row = audit_decisions(d, e)["proposals"][0]
        self.assertEqual(row["gain_evidence"], "matched_development_delta_only")
        self.assertEqual(row["parent_comparisons"][0]["observed_objective_delta"], -1)

    def test_graph_settings_and_shared_node_identity_are_preserved(self):
        a, b = proposal(recipe=graph_recipe()), proposal("P2", recipe=graph_recipe())
        b["recipe"]["pipeline_graph"]["n_splits"] = 4
        self.assertNotIn("repeated_fixed_same_seed_control", codes(audit_decisions([decision(a, b)], {})))
        b["recipe"]["pipeline_graph"]["n_splits"] = 3
        b["recipe"]["pipeline_graph"]["nodes"][-1]["inputs"] = ["a", "a"]
        self.assertNotIn("repeated_fixed_same_seed_control", codes(audit_decisions([decision(a, b)], {})))

    def test_read_only_deterministic_and_missing_declarations(self):
        d, e = [decision(proposal(expected_observation="", changed_axes=[]))], {}
        before = copy.deepcopy((d, e))
        a = audit_decisions(d, e)
        self.assertEqual(a, audit_decisions(d, e))
        self.assertEqual((d, e), before)
        self.assertIn("missing_declaration", codes(a))
        self.assertIn("missing_axes", codes(a))
        self.assertIn("No", render_markdown(a))

    def test_cli_empty_malformed_and_input_protection(self):
        script = Path(__file__).resolve().parents[1] / "scripts/audit_research_proposals.py"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            evidence = root / "evidence.json"
            evidence.write_text("{}")
            output = root / "audit.json"
            cmd = [sys.executable, str(script), "--decisions", str(root), "--evidence", str(evidence), "--output", str(output)]
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            first = output.read_bytes()
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(subprocess.run(cmd[:-1] + [str(evidence)], capture_output=True).returncode, 2)
            (root / "D1.json").write_text("{")
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)
            self.assertEqual(first, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
