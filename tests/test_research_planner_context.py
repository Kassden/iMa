import copy
import json
from pathlib import Path
import unittest

from ima.research_planner_context import compact_planner_evidence, expand_planner_evidence


class PlannerContextTests(unittest.TestCase):
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
