import json
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd

from ima.feature_definitions import FeatureDefinition, FeatureRegistry, evaluate_formulas
from ima.feature_expressions import ExpressionError, NumericExpression, unit_dimensions, unit_scale


def formula(name="prior_speed", **extra):
    return {"name":name,"expression_ast":{"op":"divide","args":[{"op":"ref","ref":"distance"},{"op":"ref","ref":"seconds"}]},"input_refs":["distance","seconds"],"unit":"m/s","hypothesis":"Historical result speed may improve prediction",**extra}


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"distance":[1200.,1400.,1000.],"seconds":[60.,0.,50.],"available":["2020-01-01","2020-01-03","2020-01-04"]})
        self.meta = {"distance":{"unit":"m"},"seconds":{"unit":"s","temporal_scope":"historical","available_at_column":"available"}}

    def test_materialization_cutoff_and_pickle_replay(self):
        definitions = [formula()]
        output,names,report = evaluate_formulas(self.frame,definitions,self.meta,cutoff="2020-01-04")
        self.assertEqual(20.,output[names[0]][0])
        self.assertTrue(output[names[0]].iloc[1:].isna().all())
        self.assertNotIn(names[0],self.frame)
        replay = evaluate_formulas(self.frame,pickle.loads(pickle.dumps(definitions)),self.meta,cutoff="2020-01-04")
        np.testing.assert_equal(output[names[0]].to_numpy(),replay[0][names[0]].to_numpy())
        self.assertEqual(report,replay[2])

    def test_registry_content_identity(self):
        with tempfile.TemporaryDirectory() as root:
            registry = FeatureRegistry(root)
            identity = registry.register(formula(),self.meta)
            self.assertEqual(identity,registry.get(identity).content_id())
            self.assertEqual(identity,registry.register(formula(hypothesis="Different narrative result"),self.meta))
            path = registry.root/(identity+".json")
            payload = json.loads(path.read_text())
            payload["unit"] = "kg"
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                registry.get(identity)

    def test_unit_target_and_temporal_taint(self):
        for field,value in (("target_tainted",True),("temporal_scope","current_outcome"),("temporal_scope","future")):
            meta = {**self.meta,"seconds":{**self.meta["seconds"],field:value}}
            with self.assertRaises(ExpressionError):
                evaluate_formulas(self.frame,[formula()],meta,cutoff="2020-01-04")
        with self.assertRaises(ExpressionError):
            evaluate_formulas(self.frame,[formula(unit="kg")],self.meta,cutoff="2020-01-04")
        with self.assertRaises(ExpressionError):
            evaluate_formulas(self.frame,[formula()],self.meta)

    def test_ast_attacks_and_cycles(self):
        for payload in ({"op":"eval","value":1},{"op":"ref","ref":"x.__class__"},{"op":"constant","value":float("inf")},{"op":"constant","value":1,"source":"import os"}):
            with self.assertRaises(ValueError):
                NumericExpression.model_validate(payload)
        definition = {"name":"a","expression_ast":{"op":"ref","ref":"b"},"input_refs":["b"]}
        other = {"name":"b","expression_ast":{"op":"ref","ref":"a"},"input_refs":["a"]}
        with self.assertRaises(ExpressionError):
            evaluate_formulas(self.frame,[definition,other],{})

    def test_reject_learned_formula_and_numeric_strings(self):
        with self.assertRaises(ValueError):
            FeatureDefinition.model_validate(formula(fit_required=True))
        frame = self.frame.assign(seconds=self.frame.seconds.astype(str))
        with self.assertRaises(ExpressionError):
            evaluate_formulas(frame,[formula()],self.meta,cutoff="2020-01-04")

    def test_compound_units_and_scale_rejection(self):
        self.assertEqual({"length":2,"time":-2},unit_dimensions("m^2/s^2"))
        for unit in ("os.system('id')","m.__class__","m^1000"):
            with self.assertRaises(ExpressionError):
                unit_dimensions(unit)
        with self.assertRaises(ExpressionError):
            evaluate_formulas(self.frame,[formula()],{**self.meta,"distance":{"unit":"km"}},cutoff="2020-01-04")

    def test_pounds_are_mass_but_not_silently_kilograms(self):
        self.assertEqual(unit_dimensions("kg"), unit_dimensions("lb"))
        self.assertEqual(.45359237, unit_scale("lb"))
        definition = {"name": "carried_weight_copy", "expression_ast": {"op": "ref", "ref": "weight"},
                      "input_refs": ["weight"], "unit": "lb"}
        frame = pd.DataFrame({"weight": [120., 125.]})
        metadata = {"weight": {"unit": "lb"}}
        values, names, _ = evaluate_formulas(frame, [definition], metadata)
        np.testing.assert_equal(values[names[0]].to_numpy(), [120., 125.])
        with self.assertRaises(ExpressionError):
            evaluate_formulas(frame, [dict(definition, unit="kg")], metadata)

    def test_one_definition_two_estimator_replays(self):
        from sklearn.linear_model import Ridge, LogisticRegression
        from threadpoolctl import threadpool_limits
        frame = pd.DataFrame({"distance":np.arange(40,dtype=float)+1000,"seconds":np.arange(40,dtype=float)+50.,"available":["2020-01-01"]*40})
        matrix,names,report = evaluate_formulas(frame,[formula()],self.meta,cutoff="2020-01-02")
        x = matrix[list(names)].to_numpy()
        y = np.arange(40)%2
        with threadpool_limits(limits=1):
            for estimator in (Ridge(),LogisticRegression()):
                fitted = estimator.fit(x,y)
                np.testing.assert_allclose(fitted.predict(x),pickle.loads(pickle.dumps(fitted)).predict(x),rtol=1e-10,atol=1e-12)
        self.assertEqual(FeatureDefinition.model_validate(formula()).content_id(),report["definitions"][0]["definition_id"])
