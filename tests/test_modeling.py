import unittest

import numpy as np
import pandas as pd

from ima.feature_sets import FeatureSchema
from ima.modeling import (
    MarketBlend, TemperatureCalibrator, apply_public_fallback,
    evaluate_probabilities, incremental_pseudo_r2, normalize_by_race, RaceProbabilityModel,
)
from ima.pools import OrderExponents, canonical_pool_name, rank_combinations


class ModelingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "race_id": [1, 1, 1, 2, 2],
            "target_win": [1, 0, 0, 0, 1],
            "target_probability": [1.0, 0.0, 0.0, 0.0, 1.0],
        })

    def test_probabilities_are_normalized_per_race(self):
        probability = normalize_by_race(np.array([3, 2, 1, 1, 4]), self.frame["race_id"])
        totals = pd.Series(probability).groupby(self.frame["race_id"]).sum()
        np.testing.assert_allclose(totals, 1.0)

    def test_calibration_and_market_blend_remain_coherent(self):
        fundamental = np.array([0.6, 0.25, 0.15, 0.4, 0.6])
        market = np.array([0.5, 0.3, 0.2, 0.55, 0.45])
        calibrator = TemperatureCalibrator.fit(fundamental, self.frame)
        calibrated = calibrator.transform(fundamental, self.frame["race_id"])
        blend = MarketBlend.fit(calibrated, market, self.frame)
        combined = blend.transform(calibrated, market, self.frame["race_id"])
        self.assertGreater(evaluate_probabilities(combined, self.frame)["top_pick_win_rate"], 0)
        np.testing.assert_allclose(pd.Series(combined).groupby(self.frame["race_id"]).sum(), 1.0)

    def test_pool_probabilities_are_coherent(self):
        runners = ["1", "2", "3"]
        strengths = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "WIN")), 1.0)
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "QIN")), 1.0)
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "TIERCE")), 1.0)

    def test_quinella_place_means_both_horses_finish_in_top_three(self):
        runners = ["1", "2", "3", "4"]
        strengths = np.array([0.4, 0.3, 0.2, 0.1])
        quinella = {
            item.runners: item.probability
            for item in rank_combinations(runners, strengths, "QIN")
        }
        qpl = {
            item.runners: item.probability
            for item in rank_combinations(runners, strengths, "QUINELLA PLACE")
        }
        self.assertEqual("QPL", canonical_pool_name("quinella-place"))
        self.assertGreater(qpl[("1", "2")], quinella[("1", "2")])
        self.assertAlmostEqual(sum(qpl.values()), 3.0)

    def test_unratable_runner_inherits_public_probability(self):
        result = apply_public_fallback(
            np.array([0.7, 0.3, 0.0]), np.array([0.4, 0.35, 0.25]),
            np.array([1, 1, 1]), np.array([True, True, False]),
        )
        self.assertAlmostEqual(0.25, result[2])
        self.assertAlmostEqual(1.0, result.sum())

    def test_fitted_order_exponents_change_exotic_distribution(self):
        normal = rank_combinations(["1", "2", "3"], np.array([0.7, 0.2, 0.1]), "TIERCE")
        corrected = rank_combinations(
            ["1", "2", "3"], np.array([0.7, 0.2, 0.1]), "TIERCE",
            OrderExponents(second=0.8, third=0.6),
        )
        self.assertNotEqual(normal[0].probability, corrected[0].probability)
        self.assertAlmostEqual(sum(item.probability for item in corrected), 1.0)

    def test_incremental_pseudo_r2_is_zero_for_identical_models(self):
        probability = np.array([0.6, 0.25, 0.15, 0.4, 0.6])
        self.assertAlmostEqual(0.0, incremental_pseudo_r2(probability, probability, self.frame))

    def test_probability_model_accepts_explicit_feature_schema(self):
        frame = pd.DataFrame({
            "race_id": [1, 1, 2, 2, 3, 3],
            "target_win": [1, 0, 0, 1, 1, 0],
            "signal": [1.0, 0.0, 0.1, 0.9, 0.8, 0.2],
            "venue": ["ST", "ST", "HV", "HV", "ST", "ST"],
        })
        schema = FeatureSchema("test", ("signal",), ("venue",))
        model = RaceProbabilityModel(feature_schema=schema).fit(frame)
        probability = model.predict_proba(frame)
        np.testing.assert_allclose(pd.Series(probability).groupby(frame["race_id"]).sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
