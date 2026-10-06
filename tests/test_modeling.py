import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp

from ima.feature_sets import FeatureSchema
from ima.modeling import (
    MarketBlend, MultiMarketBlend, TemperatureCalibrator, apply_public_fallback,
    evaluate_probabilities, incremental_pseudo_r2, normalize_by_race, RaceProbabilityModel,
    blend_probabilities, blend_multi_market_probabilities, temperature_scale,
    _blend_loss_gradient, _probability_log_features, _calibration_target,
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

    def test_place_market_can_be_fitted_as_separate_market_source(self):
        fundamental = np.array([0.6, 0.25, 0.15, 0.4, 0.6])
        win_market = np.array([0.5, 0.3, 0.2, 0.55, 0.45])
        place_market = np.array([0.45, 0.35, 0.20, 0.35, 0.65])
        blend = MultiMarketBlend.fit(fundamental, win_market, place_market, self.frame)
        combined = blend.transform(fundamental, win_market, self.frame["race_id"], place_market)
        np.testing.assert_allclose(pd.Series(combined).groupby(self.frame["race_id"]).sum(), 1.0)
        self.assertGreaterEqual(blend.place_market_weight, 0.0)

    def test_log_domain_transforms_match_reference_without_product_floor(self):
        fundamental = np.array([.999, .001, .02, .01])
        market = np.array([.001, .999, .01, .02])
        ids = np.array([1, 1, 2, 2])
        for weights in ([1, 1], [4, 4], [0, 1.1]):
            logits = weights[0] * np.log(fundamental) + weights[1] * np.log(market)
            expected = np.concatenate([np.exp(z - logsumexp(z)) for z in (logits[:2], logits[2:])])
            np.testing.assert_allclose(blend_probabilities(fundamental, market, ids, *weights), expected)
        expected = np.array([.02**4, .01**4]); expected /= expected.sum()
        np.testing.assert_allclose(temperature_scale(fundamental[2:], [2, 2], .25), expected)
        three_way = blend_multi_market_probabilities(fundamental, market, market, ids, 4, 4, 4)
        win = normalize_by_race(market, ids)
        logits = 4 * np.log(fundamental) + 8 * np.log(win)
        expected = np.concatenate([np.exp(z - logsumexp(z)) for z in (logits[:2], logits[2:])])
        np.testing.assert_allclose(three_way, expected)

    def test_plateau_positive_control_recovers_from_both_starts(self):
        frame = self.frame.iloc[:2].copy()
        fundamental, market = np.array([.999, .001]), np.array([.001, .999])
        features, codes, count = _probability_log_features([fundamental, market], frame.race_id)
        target = _calibration_target(frame, codes, count)
        objective = lambda w: _blend_loss_gradient(w, features, target, codes, count)
        loss, gradient = objective(np.array([4., 4.]))
        self.assertAlmostEqual(loss, np.log(2))
        np.testing.assert_allclose(gradient, [-3.4533773893242765, 3.4533773893242765])
        for start in ([1., 1.], [4., 4.]):
            result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(0, 4)] * 2,
                              options={"gtol": 1e-8, "ftol": 1e-12})
            self.assertTrue(result.success)
            self.assertGreater(result.x[0], result.x[1])
            self.assertLess(result.fun, 1e-8)
        blend = MarketBlend.fit(fundamental, market, frame)
        self.assertGreater(blend.fundamental_weight, blend.market_weight)
        self.assertGreater(blend.transform(fundamental, market, frame.race_id)[0], 1 - 1e-8)
        multi = MultiMarketBlend.fit(fundamental, market, market, frame)
        self.assertGreater(multi.fundamental_weight, multi.win_market_weight + multi.place_market_weight)
        self.assertGreater(multi.transform(fundamental, market, frame.race_id, market)[0], 1 - 1e-8)

    def test_boundary_optimum_keeps_zero_fundamental_weight(self):
        frame = self.frame.iloc[:2].copy()
        fundamental, market = np.array([.001, .999]), np.array([.999, .001])
        blend = MarketBlend.fit(fundamental, market, frame)
        self.assertEqual(blend.fundamental_weight, 0)
        self.assertGreater(blend.market_weight, 0)

    def test_analytical_gradient_matches_finite_difference_for_two_and_three_sources(self):
        columns = [np.array([.6, .25, .15, .4, .6]), np.array([.5, .3, .2, .55, .45]),
                   np.array([.2, .3, .5, .7, .3])]
        for number in (2, 3):
            features, codes, count = _probability_log_features(columns[:number], self.frame.race_id)
            target = _calibration_target(self.frame, codes, count)
            for weights in (np.ones(number), np.full(number, 4.0), np.zeros(number)):
                objective = lambda w: _blend_loss_gradient(w, features, target, codes, count)
                gradient = objective(weights)[1]
                step = np.eye(number) * 1e-5
                numerical = [(objective(weights + d)[0] - objective(weights - d)[0]) / 2e-5 for d in step]
                np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-7)

    def test_blend_rejects_failed_nonfinite_out_of_bounds_and_false_convergence(self):
        frame = self.frame.iloc[:2].copy()
        f, m = np.array([.999, .001]), np.array([.001, .999])
        cases = [dict(success=False, x=[0, 1], fun=1.0),
                 dict(success=True, x=[np.nan, 1], fun=1.0),
                 dict(success=True, x=[0, 1], fun=np.nan),
                 dict(success=True, x=[5, 0], fun=0.0),
                 dict(success=True, x=[4, 4], fun=np.log(2)),
                 dict(success=True, x=[4, 0], fun=123.0)]
        for case in cases:
            with self.subTest(case=case), patch("ima.modeling.minimize", return_value=OptimizeResult(
                    status=2, message="injected", **case)):
                with self.assertRaises(RuntimeError):
                    MarketBlend.fit(f, m, frame)
        with patch("ima.modeling.minimize", return_value=OptimizeResult(
                success=False, x=[0, 1, 1], fun=1., message="injected")):
            with self.assertRaises(RuntimeError):
                MultiMarketBlend.fit(f, m, m, frame)
        features, codes, count = _probability_log_features([f, m, m], frame.race_id)
        target = _calibration_target(frame, codes, count)
        false_loss = _blend_loss_gradient(np.array([4., 4., 4.]), features, target, codes, count)[0]
        with patch("ima.modeling.minimize", return_value=OptimizeResult(
                success=True, x=[4, 4, 4], fun=false_loss, message="injected")):
            with self.assertRaises(RuntimeError):
                MultiMarketBlend.fit(f, m, m, frame)

    def test_temperature_rejects_failed_nonfinite_and_wrong_scalar_results(self):
        p = np.array([.6, .25, .15, .4, .6])
        cases = [dict(success=False, x=0., fun=1.), dict(success=True, x=np.nan, fun=1.),
                 dict(success=True, x=0., fun=np.inf), dict(success=True, x=3., fun=1.),
                 dict(success=True, x=0., fun=123.)]
        for case in cases:
            with self.subTest(case=case), patch("ima.modeling.minimize_scalar", return_value=OptimizeResult(
                    message="injected", **case)):
                with self.assertRaises(RuntimeError):
                    TemperatureCalibrator.fit(p, self.frame)

    def test_invalid_inputs_rejected_and_positional_permutation_preserved(self):
        p = np.array([.6, .25, .15, .4, .6])
        for bad in (p[:-1], p[:, None], np.array([np.nan, .25, .15, .4, .6]),
                    np.array([np.inf, .25, .15, .4, .6]), -p):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MarketBlend.fit(bad, p, self.frame)
        for temperature in (0, -1, np.nan, np.inf):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                temperature_scale(p, self.frame.race_id, temperature)
        for target in ([0, 0, 0, 0, 1], [1, 1, 0, 0, 1], [np.nan, 0, 0, 0, 1]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                MarketBlend.fit(p, p, self.frame.assign(target_probability=target))
        original = blend_probabilities(p, p, self.frame.race_id, 2, 3)
        permutation = np.array([4, 2, 0, 3, 1])
        shuffled = blend_probabilities(p[permutation], p[permutation],
                                       self.frame.race_id.to_numpy()[permutation], 2, 3)
        np.testing.assert_allclose(shuffled, original[permutation])

    def test_missing_place_probability_fallback_is_preserved(self):
        p = np.array([.6, .25, .15, .4, .6])
        missing = np.array([np.nan, 0, -1, np.inf, .6])
        actual = blend_multi_market_probabilities(p, p, missing, self.frame.race_id, 1, 1, 1)
        expected = blend_multi_market_probabilities(p, p, p, self.frame.race_id, 1, 1, 1)
        np.testing.assert_allclose(actual, expected)
        with self.assertRaises(ValueError):
            blend_multi_market_probabilities(p, p, missing[:-1], self.frame.race_id, 1, 1, 1)

    def test_pool_probabilities_are_coherent(self):
        runners = ["1", "2", "3"]
        strengths = np.array([0.5, 0.3, 0.2])
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "WIN")), 1.0)
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "QIN")), 1.0)
        self.assertAlmostEqual(sum(x.probability for x in rank_combinations(runners, strengths, "TIERCE")), 1.0)

    def test_quinella_place_means_both_horses_finish_in_top_three(self):
        runners = ["1", "2", "3", "4", "5", "6", "7"]
        strengths = np.array([0.28, 0.22, 0.18, 0.12, 0.09, 0.06, 0.05])
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
