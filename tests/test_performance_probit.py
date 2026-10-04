import pickle
import unittest
import numpy as np
from scipy.special import ndtr
from ima.performance_probit import GaussianRaceProbit, gaussian_win_probabilities, gaussian_log_win_probabilities
from tests.test_pipeline_graph import race_frame


class PerformanceProbitTests(unittest.TestCase):
    def test_symmetric_analytic_shift_scale_controls(self):
        for n in (2, 3, 6):
            p = gaussian_win_probabilities(np.zeros(n), np.ones(n))
            np.testing.assert_allclose(p, np.full(n, 1/n), atol=1e-8)
        p = gaussian_win_probabilities([1., 0.], [2., 1.])
        self.assertAlmostEqual(p[0], ndtr(1/np.sqrt(5)), places=14)
        mu, sigma = np.array([0.2, -0.3, 0.7]), np.array([0.8, 1.1, 1.2])
        p, diag = gaussian_win_probabilities(mu, sigma, return_diagnostics=True)
        self.assertTrue(diag["converged"])
        np.testing.assert_allclose(p, gaussian_win_probabilities(7+mu*3, sigma*3), atol=1e-10)
        self.assertTrue(np.isfinite(gaussian_log_win_probabilities([-1000, 1000], [1, 1])).all())
        rare, diagnostics = gaussian_win_probabilities([-1000, 1000], [1, 1], return_diagnostics=True)
        self.assertGreater(rare[0], 0)
        self.assertEqual(diagnostics["float64_underflow_count"], 1)

    def test_real_shared_and_heteroscedastic_likelihood_fit(self):
        train, score = race_frame(30), race_frame(4, 30)
        for conditional in (False, True):
            model = GaussianRaceProbit(("ability", "context"), heteroscedastic=conditional, max_iter=120).fit(train)
            p = model.predict_proba(score)
            self.assertLess(-np.log(p[score.target_win == 1]).mean(), np.log(4))
            np.testing.assert_allclose(p.reshape(-1, 4).sum(axis=1), 1)
            dist = model.predict_distribution(score)
            np.testing.assert_allclose(dist.location.reshape(-1, 4).sum(axis=1), 0, atol=1e-12)
            np.testing.assert_allclose(np.log(dist.scale).reshape(-1, 4).sum(axis=1), 0, atol=1e-12)
            np.testing.assert_allclose(p, pickle.loads(pickle.dumps(model)).predict(score), rtol=1e-10, atol=1e-12)
            self.assertEqual(model.fit_diagnostics["lane"], "experimental")

    def test_invalid_races_scales_and_unidentified_regularization(self):
        with self.assertRaises(ValueError):
            gaussian_win_probabilities([0, 1], [1, 0])
        with self.assertRaises(ValueError):
            GaussianRaceProbit(("ability",), l2=0).fit(race_frame(5))
        frame = race_frame(5)
        frame["target_win"] = 0
        with self.assertRaisesRegex(ValueError, "one winner"):
            GaussianRaceProbit(("ability",)).fit(frame)
