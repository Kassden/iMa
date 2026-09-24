import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.feature_sets import FeatureSchema
from ima.research_models import (
    RaceSoftmaxOffsetModel,
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


if __name__ == "__main__":
    unittest.main()
