import copy
import json
from pathlib import Path
import unittest

from ima.research_planner_context import compact_planner_evidence, expand_planner_evidence, decision_planner_evidence
from ima.proposal_audit import planner_proposal_feedback


class PlannerContextTests(unittest.TestCase):
    def test_malformed_bundle_parent_and_runtime_shapes_fail_closed(self):
        for bundle in (None, [], {'evidence':None}, {'paper_research':{},'completed_trial_index':[None]},
                       {'invocation_history':[None]}):
            with self.subTest(bundle=bundle), self.assertRaises(ValueError):
                decision_planner_evidence(bundle)

    def test_feedback_audits_full_memory_before_curation(self):
        from tests.test_proposal_audit import proposal
        memory = [{"kind": "planned", "hypothesis_id": f"D{i:06}",
                   "payload": {"decision_id": f"D{i:06}", "programs": [proposal(f"P{i}")]}}
                  for i in range(20, 0, -1)]
        memory.append({"kind": "preparation_rejected", "hypothesis_id": "failed-old",
                       "payload": {"error": "preparation failed"}})
        bundle = {"evidence": {"hypothesis_memory": memory}}
        before = copy.deepcopy(bundle)
        view = decision_planner_evidence(bundle)
        feedback = view["evidence"]["proposal_audit_feedback"]
        self.assertEqual(feedback, planner_proposal_feedback(before["evidence"]))
        self.assertEqual(feedback["proposal_count"], 20)
        self.assertEqual(len(view["evidence"]["hypothesis_memory"]), 8)
        self.assertNotIn("recipe", view["evidence"]["hypothesis_memory"][0]["payload"]["programs"][0])
        self.assertEqual(feedback["counts"]["gain_evidence"], {"unresolved": 20})
        self.assertEqual(feedback["recent_negatives"][0]["hypothesis_id"], "failed-old")
        self.assertEqual(view, decision_planner_evidence(view))
        self.assertEqual(bundle, before)

    def test_missing_and_malformed_memory_payloads_are_explicit(self):
        for memory in (None, [None, {"kind": "planned"}, {"kind": "planned", "payload": None},
                       {"kind": "planned", "payload": {"programs": [None, "bad"]}},
                       {"kind": "planned", "payload": {"programs": "bad"}}]):
            with self.subTest(memory=memory):
                view = decision_planner_evidence({"hypothesis_memory": memory})
                feedback = view["proposal_audit_feedback"]
                self.assertTrue(feedback["warnings"])
                self.assertTrue(feedback["advisory_only"])
                self.assertTrue(all(w["classification"] == "warning" for w in feedback["warnings"]))
        self.assertEqual(decision_planner_evidence({})["proposal_audit_feedback"]["proposal_count"], 0)

    def test_paper_projection_keeps_actual_provenance_and_declares_bulk_omissions(self):
        paper_row = dict(assumptions="PL assumptions", comparison={"protocol_hash": "full-hash"},
            coverage={"evaluated_races": 100}, evaluation_key="eval-id", ex_post_market_tainted=False,
            executable_evidence=False, model_ids=[{"attempt_id": "attempt-full", "prediction_sha256": "f" * 64}],
            paper_only=True, probability_basis="fundamental", quote_mode="fair_price",
            race_results_preview=[{"bulk": "x" * 10000}], report_path="private/path",
            request={"attempt_ids": ["attempt-full"], "seed": 42}, request_id="request-id",
            unsupported_requirements=["No pre-race quote"])
        bundle = {"paper_research": {"by_comparison": {"c": [paper_row]}}}
        before = copy.deepcopy(bundle)
        view = decision_planner_evidence(bundle)
        retained = view["paper_research"]["by_comparison"]["c"][0]
        for field in set(paper_row) - {"race_results_preview", "report_path"}:
            self.assertEqual(retained[field], paper_row[field])
        self.assertNotIn("race_results_preview", retained)
        self.assertNotIn("report_path", retained)
        self.assertIn("race_results_preview", view["planner_decision_view"]["omissions"]["paper_bulk_fields"])
        self.assertEqual(bundle, before)
        for bad in ({"by_comparison": {"c": [None]}}, {"eligible_attempts": [None]},
                    {"by_comparison": []}, {"eligible_attempts": [{"comparison": None}]}):
            with self.assertRaises(ValueError):
                decision_planner_evidence({"paper_research": bad})

    def test_actual_feedback_is_available_in_projected_request(self):
        path = Path(__file__).resolve().parents[1] / ".tmp/context-baseline-D000126.json"
        if not path.exists():
            self.skipTest("Frozen local evidence is not part of the distribution")
        evidence = json.loads(path.read_text())
        before = copy.deepcopy(evidence)
        view = decision_planner_evidence({"evidence": evidence})
        feedback = view["evidence"]["proposal_audit_feedback"]
        self.assertEqual(feedback, planner_proposal_feedback(before))
        self.assertEqual(feedback["proposal_count"], 23)
        self.assertEqual(feedback["counts"]["negative_events"], 9)
        self.assertEqual(len(feedback["recent_negatives"]), 5)
        self.assertLessEqual(len(feedback["warnings"]), 12)
        self.assertEqual(evidence, before)
        self.assertEqual(view["evidence"]["completed_trial_index"], before["completed_trial_index"])
        restored = expand_planner_evidence(view)["evidence"]
        self.assertEqual(restored["capabilities"], before["capabilities"])
        for key in ("champions_by_contract", "champions_by_family"):
            for identity, champion in before[key].items():
                self.assertEqual(restored[key][identity], {k: v for k, v in champion.items() if k != "artifacts"})
        for key, records in before["paper_research"]["by_comparison"].items():
            projected = view["evidence"]["paper_research"]["by_comparison"][key]
            for original, retained in zip(records[-2:], projected):
                for field in ("model_ids", "request", "comparison", "unsupported_requirements", "paper_only"):
                    if field in original:
                        self.assertEqual(original[field], retained[field])

    def test_decision_projection_keeps_champions_memo_safety_and_parent_identity(self):
        bundle = self.bundle()
        evidence = bundle['evidence']
        evidence.update(completed_trial_index=[{'attempt_id':'a'}, {'attempt_id':'b'}],
            champions_by_contract={'contract':{'attempt_id':'b', 'objective_value':2.152076,
                'recipe':{'model':'probit'}, 'artifacts':{'file':'private-path'}}},
            paper_research={'eligible_attempts':[{'attempt_id':'b','comparison':{'population':'p'}}],
                'by_comparison':{'p':[{'status':'completed','executable_evidence':False,
                    'ex_post_market_tainted':True,'assumptions':'historical only'}]*8}},
            hypothesis_memory=[{'kind':'planned','payload':{'programs':[{'proposal_id':'p',
                'hypothesis':'h'*2000,'falsification_rule':'reject if loss worsens',
                'recipe':{'large':'repeated '*1000}}]}}]*30)
        evidence['capabilities']['research_memo']={'sha256':'memo-id','content':'Read all of this.'}
        original = copy.deepcopy(bundle)
        view = decision_planner_evidence(bundle)
        restored = expand_planner_evidence(view)['evidence']
        self.assertEqual(original, bundle)
        self.assertEqual(evidence['capabilities'], restored['capabilities'])
        self.assertEqual(evidence['completed_trial_index'], restored['completed_trial_index'])
        self.assertEqual(2.152076, restored['champions_by_contract']['contract']['objective_value'])
        self.assertEqual({'model':'probit'}, restored['champions_by_contract']['contract']['recipe'])
        table = restored['paper_research']['eligible_attempts']
        self.assertEqual([[1,0]], table['rows'])
        self.assertEqual([{'population':'p'}], table['comparisons'])
        self.assertTrue(restored['paper_research']['by_comparison']['p'][0]['ex_post_market_tainted'])
        self.assertFalse(restored['paper_research']['by_comparison']['p'][0]['executable_evidence'])
        self.assertEqual(8, len(restored['hypothesis_memory']))
        self.assertTrue(restored['hypothesis_memory'][0]['payload']['programs'][0]['hypothesis_truncated'])
        self.assertEqual(view, decision_planner_evidence(view))
        self.assertLess(len(json.dumps(view)), len(json.dumps(original))*.4)

    def bundle(self):
        catalog = {f"predictor_{i}":{"safe":True,"unit":"seconds","availability":"pre_race",
            "eligible":True,"description":"A repeated safe numeric measurement"} for i in range(600)}
        catalog["predictor_3"]["safe"] = False
        catalog["predictor_4"]["unit"] = "metres"
        catalog["predictor_5"]["eligible"] = None
        catalog["predictor_6"].pop("description")
        catalog["predictor_7"]["extra"] = {"target_taint":True,"temporal_lag":2}
        return {"evidence":{"evidence_id":"frozen-id","capabilities":{"eligible_predictors":catalog}}}

    def test_lossless_metadata_projection_and_input_immutability(self):
        bundle = self.bundle()
        before = copy.deepcopy(bundle)
        projected = compact_planner_evidence(bundle)
        self.assertEqual(before,bundle)
        self.assertEqual(bundle,expand_planner_evidence(projected))
        self.assertEqual(bundle,expand_planner_evidence(json.loads(json.dumps(projected))))
        self.assertLess(len(json.dumps(projected)),len(json.dumps(bundle))*.5)
        self.assertEqual("frozen-id",projected["evidence"]["evidence_id"])
        projected["evidence"]["capabilities"]["eligible_predictors"]["defaults"]["safe"] = "changed"
        self.assertEqual(before,bundle)

    def test_missing_fields_are_not_inherited_and_null_is_explicit(self):
        projected = compact_planner_evidence(self.bundle())
        catalog = projected["evidence"]["capabilities"]["eligible_predictors"]
        self.assertNotIn("description",catalog["defaults"])
        restored = expand_planner_evidence(projected)["evidence"]["capabilities"]["eligible_predictors"]
        self.assertIsNone(restored["predictor_5"]["eligible"])
        self.assertFalse(restored["predictor_3"]["safe"])
        self.assertEqual("metres",restored["predictor_4"]["unit"])

    def test_boolean_and_integer_metadata_types_are_preserved(self):
        bundle = {"capabilities":{"eligible_predictors":{str(i):{"safe":True,"value":1 if i==2 else True,
                  "description":"long repeated field "*10} for i in range(4)}}}
        restored = expand_planner_evidence(compact_planner_evidence(bundle))
        self.assertIs(type(restored["capabilities"]["eligible_predictors"]["2"]["value"]),int)
        self.assertIs(type(restored["capabilities"]["eligible_predictors"]["1"]["value"]),bool)

    def test_real_decision_schema_contains_both_exact_standalone_schemas(self):
        from ima.research_expansion import PlannerDecision
        from ima.research_specs import PipelineRecipe
        from ima.research_betting import PaperResearchRequest
        bundle = self.bundle()
        bundle["decision_schema"] = PlannerDecision.model_json_schema()
        capabilities = bundle["evidence"]["capabilities"]
        capabilities["recipe_schema"] = PipelineRecipe.model_json_schema()
        capabilities["paper_research"] = {"request_schema":PaperResearchRequest.model_json_schema(),"active_ceiling":1}
        projected = compact_planner_evidence(bundle)
        actual = projected["evidence"]["capabilities"]
        self.assertIn("$ref",actual["recipe_schema"])
        self.assertIn("$ref",actual["paper_research"]["request_schema"])
        self.assertEqual(bundle,expand_planner_evidence(projected))

    def test_missing_unreachable_or_different_top_schema_never_deduplicates(self):
        schema = {"title":"Recipe","type":"object","properties":{"safe":{"type":"boolean"}}}
        for decision in (None,{"$defs":{"Recipe":schema}},
                         {"$ref":"#/$defs/Recipe","$defs":{"Recipe":dict(schema,title="Different")}}):
            with self.subTest(decision=decision):
                bundle = self.bundle()
                bundle["evidence"]["capabilities"]["recipe_schema"] = schema
                if decision is not None:
                    bundle["decision_schema"] = decision
                projected = compact_planner_evidence(bundle)
                self.assertEqual(schema,projected["evidence"]["capabilities"]["recipe_schema"])
                self.assertEqual(bundle,expand_planner_evidence(projected))

    def test_changed_dependency_keeps_standalone_schema(self):
        schema = {"title":"Recipe","type":"object","properties":{"child":{"$ref":"#/$defs/Child"}},
                  "$defs":{"Child":{"type":"number"}}}
        bundle = self.bundle()
        bundle["evidence"]["capabilities"]["recipe_schema"] = schema
        bundle["decision_schema"] = {"$ref":"#/$defs/Recipe","$defs":{
            "Recipe":{key:value for key,value in schema.items() if key!="$defs"},"Child":{"type":"string"}}}
        self.assertEqual(schema,compact_planner_evidence(bundle)["evidence"]["capabilities"]["recipe_schema"])

    def test_list_catalog_empty_and_nonbeneficial_catalog_remain_unchanged(self):
        for catalog in ([],{},["a","b"],{"a":{"unit":"s"}},{"a":"metadata"}):
            bundle = {"capabilities":{"eligible_predictors":catalog}}
            self.assertEqual(bundle,compact_planner_evidence(bundle))

    def test_compaction_is_idempotent_and_does_not_share_nested_objects(self):
        compacted = compact_planner_evidence(self.bundle())
        again = compact_planner_evidence(compacted)
        self.assertEqual(compacted,again)
        self.assertIsNot(compacted,again)
        self.assertIsNot(compacted["evidence"],again["evidence"])

    def test_captured_catalog_round_trip_when_local_audit_artifact_is_available(self):
        path = Path(__file__).resolve().parents[1]/".tmp"/"openrouter-evidence-inspect.json"
        if not path.is_file():
            self.skipTest("Local read-only audit artifact is not part of the test distribution")
        from ima.research_expansion import PlannerDecision
        original = {"decision_schema":PlannerDecision.model_json_schema(),"evidence":json.loads(path.read_text())}
        projected = compact_planner_evidence(original)
        self.assertEqual(original,expand_planner_evidence(json.loads(json.dumps(projected))))
        before = original["evidence"]["capabilities"]["eligible_predictors"]
        after = projected["evidence"]["capabilities"]["eligible_predictors"]
        self.assertLess(len(json.dumps(after)),len(json.dumps(before))*.15)


if __name__ == "__main__":
    unittest.main()
