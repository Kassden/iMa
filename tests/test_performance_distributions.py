import pickle
import unittest
import numpy as np
import pandas as pd
from ima.feature_sets import FeatureSchema
from ima.performance_distributions import PerformanceDistribution, SharedResidualPerformanceModel, CatBoostUncertaintyModel
from ima.probabilistic_adapters import fit_performance_distribution


def performance_frame(n=160, start=0):
    rng = np.random.default_rng(123+start)
    x = rng.uniform(-1, 1, n)
    scale = 0.02 + 0.12*(x > 0)
    return pd.DataFrame({"x": x, "speed": np.exp(2.7+0.08*x+rng.normal(size=n)*scale),
        "date": pd.date_range("2020-01-01", periods=n)[0:] + pd.Timedelta(days=start)})


class PerformanceDistributionTests(unittest.TestCase):
    def test_shared_scale_past_support_pickle(self):
        train, calibration, score = performance_frame(), performance_frame(40, 160), performance_frame(20, 200)
        model = SharedResidualPerformanceModel(("x",)).fit(train, calibration, label_column="speed")
        dist = model.predict_distribution(score)
        self.assertEqual(dist.support, "positive_physical_via_log")
        self.assertTrue((dist.physical_samples(np.full((20, 20), -10)) > 0).all())
        np.testing.assert_allclose(dist.scale, dist.scale[0])
        np.testing.assert_allclose(model.predict(score), pickle.loads(pickle.dumps(model)).predict(score), rtol=1e-10, atol=1e-12)
        self.assertTrue(np.isfinite(list(dist.diagnostics(score.speed).values())).all())
        with self.assertRaisesRegex(ValueError, "must follow"):
            SharedResidualPerformanceModel(("x",)).fit(train, train, label_column="speed")

    def test_catboost_real_conditional_fit_raw_scale_parity(self):
        train = performance_frame(240)
        calibration, score = performance_frame(50, 240), performance_frame(50, 290)
        model = CatBoostUncertaintyModel(("x",), parameters={"iterations": 70, "depth": 3}, scale_shrinkage=0).fit(train, label_column="speed", calibration=calibration)
        dist = model.predict_distribution(score)
        raw = model.model.predict(score[["x"]], prediction_type="RawFormulaVal")
        np.testing.assert_allclose(dist.scale, np.maximum(model.scale_floor, np.exp(raw[:, 1])*model.scale_multiplier))
        self.assertGreater(dist.scale[score.x > 0].mean(), dist.scale[score.x < 0].mean())
        shared = model.predict_distribution(score, shared_scale=True)
        np.testing.assert_array_equal(dist.location, shared.location)
        np.testing.assert_allclose(shared.scale, shared.scale[0])
        np.testing.assert_allclose(model.predict(score), pickle.loads(pickle.dumps(model)).predict(score), rtol=1e-10, atol=1e-12)
        fitted = fit_performance_distribution({"kind": "shared_residual"}, train, calibration,
            FeatureSchema("physical", ("x",), ()), label_column="speed")
        self.assertTrue((fitted.predict(score) > 0).all())

    def test_physical_inverse_mean_is_not_mean_inverse(self):
        dist = PerformanceDistribution(np.array([2.0]), np.array([0.7]), "log_speed_mps")
        inverse_mean = 1200/dist.physical_mean()[0]
        mean_inverse = 1200*np.exp(-dist.location[0]+dist.variance[0]/2)
        self.assertGreater(mean_inverse, inverse_mean)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            PerformanceDistribution(np.array([16.]), np.array([2.]), "speed_mps")
        with self.assertRaisesRegex(ValueError, "positive"):
            PerformanceDistribution(np.array([1.]), np.array([0.]))
