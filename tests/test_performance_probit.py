import pickle
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.special import ndtr
from ima.performance_probit import GaussianRaceProbit, fit_probit, gaussian_win_probabilities, gaussian_log_win_probabilities
from tests.test_pipeline_graph import SCHEMA, race_frame


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
        frame = race_frame(5)
        frame['field_size'] = 5
        with self.assertRaisesRegex(ValueError, "complete pre-race fields"):
            GaussianRaceProbit(("ability",)).fit(frame)

    def test_optimizer_and_quadrature_gates_are_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            GaussianRaceProbit(("ability", "context"), max_iter=1).fit(race_frame(10))
        failure = OptimizeResult(success=False, message='iteration limit')
        with patch('ima.performance_probit.minimize', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "did not converge: iteration limit"):
                GaussianRaceProbit(("ability",)).fit(race_frame(5))
        model = GaussianRaceProbit(("ability", "context"), max_iter=120)
        with patch('ima.performance_probit.gaussian_win_probabilities', return_value=np.full(4, 0.25)):
            with self.assertRaisesRegex(RuntimeError, "higher quadrature_order"):
                model.fit(race_frame(10))
        self.assertIsNone(model.coefficients)
        self.assertIsNone(model.scale_coefficients)
        with patch('ima.performance_probit.gaussian_win_probabilities', side_effect=RuntimeError('Gaussian quadrature did not converge')):
            with self.assertRaisesRegex(RuntimeError, "Gaussian quadrature did not converge"):
                GaussianRaceProbit(("ability",), max_iter=120).fit(race_frame(5))

    def test_parent_dispatch_prediction_contract(self):
        model = fit_probit(race_frame(30), SCHEMA, {'heteroscedastic': True, 'max_iter': 120})
        probabilities = model.predict_proba(race_frame(3, 30))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities > 0).all())
        np.testing.assert_allclose(probabilities.reshape(-1, 4).sum(axis=1), 1)
        self.assertTrue(model.fit_diagnostics['converged'])
