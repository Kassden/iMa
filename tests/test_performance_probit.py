import pickle
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.special import ndtr
from ima.performance_probit import GaussianRaceProbit, fit_probit, gaussian_win_probabilities, gaussian_log_win_probabilities, _quadrature
from tests.test_pipeline_graph import SCHEMA, race_frame


class PerformanceProbitTests(unittest.TestCase):
    def test_sharp_unequal_scales_use_accurate_adaptive_fallback(self):
        for sigma in ([.02, .8, 2.5], [.001, .8, 2.5], [.1, 1., 3.]):
            sigma = np.asarray(sigma)
            # A zero-mean three-runner win is a bivariate Gaussian orthant.
            expected = np.array([
                .25 + np.arcsin(np.prod(sigma[i]/np.hypot(sigma[i], np.delete(sigma, i))))/(2*np.pi)
                for i in range(3)
            ])
            for tolerance in (1e-6, 1e-9):
                with self.subTest(sigma=sigma, tolerance=tolerance):
                    p, diag = gaussian_win_probabilities(np.zeros(3), sigma,
                        tolerance=tolerance, return_diagnostics=True)
                    self.assertEqual(diag['integration_method'], 'adaptive_quad')
                    self.assertGreater(diag['hermite_refinement_error'], tolerance)
                    self.assertTrue(diag['converged'])
                    for key in ('normalization_error', 'refinement_error', 'adaptive_error_bound'):
                        self.assertLessEqual(diag[key], tolerance)
                    np.testing.assert_allclose(p, expected, rtol=0, atol=2e-10)
                    np.testing.assert_allclose(p, gaussian_win_probabilities(
                        np.full(3, 7.), sigma*3, tolerance=tolerance), rtol=0, atol=2e-10)
        sigma = np.array([.02, .4, .8, 1.2, 2., 3.])
        p, diag = gaussian_win_probabilities(np.zeros(6), sigma, return_diagnostics=True)
        self.assertTrue(diag['converged'])
        self.assertEqual(diag['integration_method'], 'adaptive_quad')
        self.assertAlmostEqual(p.sum(), 1., places=14)
        self.assertTrue((p > 0).all())
        np.testing.assert_allclose(p[::-1], gaussian_win_probabilities(
            np.zeros(6), sigma[::-1]), rtol=0, atol=2e-10)

    def test_adaptive_failure_and_error_estimates_remain_fail_closed(self):
        failures = (
            (0.1, 1e-10, {}, 'forced subdivision failure'),
            (np.nan, 0., {}), (0.1, np.inf, {}), (-0.1, 0., {}),
            (0.1, -1., {}), (0., 0., {}),
            # Perfect normalization and refinement cannot excuse a large error bound.
            (1/9, 1e-3, {}), (1/10, 0., {}),
        )
        for result in failures:
            with self.subTest(result=result), patch('ima.performance_probit.quad', return_value=result):
                _, diag = gaussian_win_probabilities([0., 0., 0.], [.02, .8, 2.5],
                    return_diagnostics=True)
                self.assertFalse(diag['converged'])
                self.assertEqual(diag['integration_method'], 'adaptive_quad')
                with self.assertRaisesRegex(RuntimeError, 'Gaussian quadrature did not converge'):
                    gaussian_win_probabilities([0., 0., 0.], [.02, .8, 2.5])
        with patch('ima.performance_probit.quad', side_effect=RuntimeError('forced failure')):
            _, diag = gaussian_win_probabilities([0., 0., 0.], [.02, .8, 2.5],
                return_diagnostics=True)
            self.assertIn('forced failure', diag['adaptive_error'])
            self.assertFalse(diag['converged'])

    def test_converged_hermite_does_not_call_adaptive(self):
        with patch('ima.performance_probit.quad', side_effect=AssertionError('unexpected fallback')):
            for n in (1, 2, 6):
                np.testing.assert_allclose(gaussian_win_probabilities(np.zeros(n), np.ones(n)),
                    np.full(n, 1/n), atol=1e-10)

    def test_zero_quadrature_weights_ignore_only_divide_warning(self):
        _quadrature.cache_clear()
        with np.errstate(divide='raise', invalid='raise'):
            nodes, logs = _quadrature(512)
        self.assertTrue(np.isfinite(nodes).all())
        self.assertTrue(np.isneginf(logs).any())
        self.assertAlmostEqual(np.exp(logs).sum(), 1., places=14)
        with patch('ima.performance_probit.roots_hermitenorm', return_value=(np.zeros(2), np.array([-1., 1.]))):
            with np.errstate(invalid='raise'), self.assertRaises(FloatingPointError):
                _quadrature(513)

    def test_training_coarse_gate_refines_32_without_weakening_accuracy(self):
        frame = race_frame(1).iloc[:3].copy()
        frame['target_win'] = [1, 0, 0]
        logged = np.log([.25, 1., 1.5])
        logged -= logged.mean()
        matrix = (3*np.arctanh(logged/3))[:, None]
        result = OptimizeResult(success=True, x=np.array([0., 1.]), fun=1., nit=1)
        for order in (32, 128, 256):
            model = GaussianRaceProbit(('ability',), heteroscedastic=True, quadrature_order=order)
            with self.subTest(order=order), patch.object(model, '_matrix',
                    return_value=(matrix, np.zeros(3, dtype=int))), patch(
                    'ima.performance_probit.minimize', return_value=result):
                model.fit(frame)
                self.assertEqual(model.fit_diagnostics['configured_quadrature_order'],order)
                self.assertLessEqual(model.fit_diagnostics['training_quadrature_max_refinement_error'], 1e-4)
                if order == 32:
                    self.assertGreater(model.quadrature_order,32)
                    self.assertGreater(len(model.fit_diagnostics['quadrature_rounds']),1)
                    self.assertGreater(model.fit_diagnostics['quadrature_rounds'][0]['max_refinement_error'],1e-4)

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
        uniform = lambda mu,sigma,**kwargs: np.full_like(mu,1/mu.shape[1])
        with patch('ima.performance_probit._batched_win_probabilities', side_effect=uniform):
            with self.assertRaisesRegex(RuntimeError, "higher quadrature_order"):
                model.fit(race_frame(10))
        self.assertIsNone(model.coefficients)
        self.assertIsNone(model.scale_coefficients)
        with patch('ima.performance_probit._batched_win_probabilities', side_effect=RuntimeError('Gaussian quadrature did not converge')):
            with self.assertRaisesRegex(RuntimeError, "Gaussian quadrature did not converge"):
                GaussianRaceProbit(("ability",), max_iter=120).fit(race_frame(5))

    def test_refinement_does_not_multiply_the_iteration_budget(self):
        frame = race_frame(1).iloc[:3].copy()
        frame['target_win'] = [1,0,0]
        logged = np.log([.25,1.,1.5])
        logged -= logged.mean()
        matrix = (3*np.arctanh(logged/3))[:,None]
        result = OptimizeResult(success=True,x=np.array([0.,1.]),fun=1.,nit=5)
        model = GaussianRaceProbit(('ability',),heteroscedastic=True,quadrature_order=32,max_iter=5)
        with patch.object(model,'_matrix',return_value=(matrix,np.zeros(3,dtype=int))), \
                patch('ima.performance_probit.minimize',return_value=result) as optimizer:
            with self.assertRaisesRegex(RuntimeError,'higher quadrature_order'):
                model.fit(frame)
        optimizer.assert_called_once()

    def test_parent_dispatch_prediction_contract(self):
        model = fit_probit(race_frame(30), SCHEMA, {'heteroscedastic': True, 'max_iter': 120})
        probabilities = model.predict_proba(race_frame(3, 30))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities > 0).all())
        np.testing.assert_allclose(probabilities.reshape(-1, 4).sum(axis=1), 1)
        self.assertTrue(model.fit_diagnostics['converged'])
