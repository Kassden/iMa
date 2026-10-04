import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.feature_sets import FeatureSchema
from ima.modeling import RaceConditionalLogitModel, conditional_loss_gradient
from ima.research_models import (
    RaceSoftmaxOffsetModel,
    ResearchClassifier,
    ResearchRegressor,
    offset_gradient_check,
    secondary_target_diagnostics,
)
from ima.research_targets import apply_target_contract, target_contract


FIXTURE = Path(__file__).parent / "fixtures" / "research_races.csv"


class ResearchModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(FIXTURE, parse_dates=["date"])
        self.frame["horse_rating"] = [60, 55, 50, 58, 62, 54, 50, 52, 66, 51, 57, 63, 65, 60, 55, 56, 67, 53]

    def test_offset_model_zero_coefficients_recover_market(self):
        model = RaceSoftmaxOffsetModel(l2=1000.0).fit(self.frame, ["horse_rating"])
        model.coefficients = np.zeros_like(model.coefficients)
        probability = model.predict_proba(self.frame)
        market_loss = secondary_target_diagnostics(
            self.frame,
            target_contract("win_probability"),
            self.frame["market_probability"].to_numpy(),
        )["race_log_loss"]
        self.assertAlmostEqual(
            market_loss,
            secondary_target_diagnostics(
                self.frame, target_contract("win_probability"), probability,
            )["race_log_loss"],
            places=8,
        )

    def test_offset_gradient_matches_finite_difference(self):
        self.assertLess(offset_gradient_check(self.frame, ["horse_rating"], l2=0.5), 1e-4)

    def test_offset_probabilities_sum_to_one_by_race(self):
        model = RaceSoftmaxOffsetModel(l2=0.1).fit(self.frame, ["horse_rating"])
        probability = model.predict_proba(self.frame)
        totals = pd.Series(probability).groupby(self.frame["race_id"]).sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0)

    def test_finish_time_regressor_and_diagnostics(self):
        contract = target_contract("adjusted_finish_time_or_speed", {"min_coverage": 1.0})
        labelled = apply_target_contract(self.frame, contract)
        schema = FeatureSchema("speed-test", ("horse_rating", "distance"), ())
        model = ResearchRegressor("ridge_regressor").fit(labelled, schema, contract.label_column)
        diagnostics = secondary_target_diagnostics(labelled, contract, model.predict(labelled))
        self.assertIn("mae", diagnostics)
        self.assertIn("spearman", diagnostics)

    def test_ranking_and_placing_diagnostics_are_target_specific(self):
        ranking = secondary_target_diagnostics(
            self.frame,
            target_contract("ranking_strength"),
            -self.frame["result"].to_numpy(dtype=float),
        )
        placing = secondary_target_diagnostics(
            self.frame,
            target_contract("placing_top_k", {"top_k": 2}),
            np.full(len(self.frame), 2 / 3),
        )
        self.assertGreater(ranking["spearman"], 0.9)
        self.assertIn("brier", placing)

    def test_conditional_logit_gradient_and_race_probabilities(self):
        x = np.array([[1.0], [0.0], [-1.0], [0.0], [1.0], [-1.0]])
        y = np.array([1.0, 0, 0, 0, 1.0, 0])
        codes = np.array([0, 0, 0, 1, 1, 1])
        coefficients = np.array([0.3])
        loss, gradient = conditional_loss_gradient(coefficients, x, y, codes, 2, 0.1)
        delta = 1e-6
        plus = conditional_loss_gradient(coefficients + delta, x, y, codes, 2, 0.1)[0]
        minus = conditional_loss_gradient(coefficients - delta, x, y, codes, 2, 0.1)[0]
        self.assertAlmostEqual(float(gradient[0]), (plus - minus) / (2 * delta), places=6)
        schema = FeatureSchema("conditional-test", ("horse_rating",), ())
        model = RaceConditionalLogitModel(l2=0.1, feature_schema=schema).fit(self.frame)
        probabilities = model.predict_proba(self.frame)
        totals = pd.Series(probabilities).groupby(self.frame["race_id"]).sum()
        np.testing.assert_allclose(totals.to_numpy(), 1.0)
        shuffled = self.frame.sample(frac=1, random_state=5)
        shuffled_probabilities = pd.Series(model.predict_proba(shuffled), index=shuffled.index)
        np.testing.assert_allclose(shuffled_probabilities.loc[self.frame.index], probabilities)
        self.assertGreater(loss, 0)

    def test_recorded_final_odds_are_label_only(self):
        from ima.research_targets import TargetContractError, validate_no_forbidden_label_features

        contract = target_contract("recorded_final_win_odds")
        self.frame["win_odds"] = 1.0 / self.frame["market_probability"]
        labelled = apply_target_contract(self.frame, contract)
        np.testing.assert_allclose(
            labelled[contract.label_column], np.log(self.frame["win_odds"])
        )
        with self.assertRaises(TargetContractError):
            validate_no_forbidden_label_features(contract, ["win_odds"])

    def test_catboost_place_and_odds_models(self):
        self.frame["win_odds"] = 1.0 / self.frame["market_probability"]
        schema = FeatureSchema("native-test", ("horse_rating",), ())
        place = apply_target_contract(self.frame, target_contract("placing_top_k", {"top_k": 2}))
        classifier = ResearchClassifier(
            "catboost_classifier", {"iterations": 20, "depth": 2}
        ).fit(place, schema, "target_top_2")
        values = classifier.predict(place)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        odds = apply_target_contract(self.frame, target_contract("recorded_final_win_odds"))
        regressor = ResearchRegressor(
            "catboost_regressor", {"iterations": 20, "depth": 2}
        ).fit(odds, schema, "target_log_final_win_odds")
        self.assertTrue(np.isfinite(regressor.predict(odds)).all())


class NativeThreadBudgetTests(unittest.TestCase):
    def test_worker_budget_is_explicit_and_legacy_default_unchanged(self):
        import os
        from unittest.mock import patch
        from ima.research_models import _thread_override
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual({}, _thread_override("thread_count"))
        with patch.dict(os.environ, {"IMA_RESEARCH_THREADS": "1"}):
            self.assertEqual({"thread_count": 1}, _thread_override("thread_count"))
        for value in ("0", "-1", "auto", "100000"):
            with self.subTest(value=value), patch.dict(os.environ, {"IMA_RESEARCH_THREADS": value}):
                with self.assertRaises(ValueError):
                    _thread_override("n_jobs")


if __name__ == "__main__":
    unittest.main()
