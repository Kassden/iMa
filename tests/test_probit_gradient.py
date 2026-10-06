from functools import lru_cache
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp, roots_hermitenorm

from ima.performance_probit import (
    GaussianRaceProbit,
    _gaussian_log_win_probabilities,
    _race_batches,
    _race_center,
    _race_groups,
    gaussian_log_win_probabilities,
)
from tests.test_pipeline_graph import race_frame


@lru_cache(maxsize=16)
def reference_quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return nodes, np.log(weights / np.sqrt(2*np.pi))


def reference_logs(means, scales, order):
    """The pre-gradient likelihood, kept independent of the new evaluator."""
    if len(means) == 1:
        return np.zeros(1)
    if len(means) == 2:
        difference = (means[0]-means[1]) / np.hypot(*scales)
        return np.array([log_ndtr(difference), log_ndtr(-difference)])
    nodes, log_weights = reference_quadrature(order)
    logs = []
    for i in range(len(means)):
        mask = np.arange(len(means)) != i
        thresholds = (means[i]+scales[i]*nodes[:, None]-means[mask]) / scales[mask]
        logs.append(logsumexp(log_weights+log_ndtr(thresholds).sum(axis=1)))
    return np.array(logs)


def reference_objective(model, parameters, x, groups, winners):
    dimension = x.shape[1]
    means = x @ parameters[:dimension]
    scales = np.ones(len(x))
    if model.heteroscedastic:
        logged = 3*np.tanh((x @ parameters[dimension:])/3)
        for group in groups:
            logged[group] -= logged[group].mean()
        scales = np.exp(logged)
    losses = []
    for group, winner in zip(groups, winners):
        logs = reference_logs(means[group], scales[group], model.quadrature_order)
        losses.append(-(logs-logsumexp(logs))[winner])
    return float(np.mean(losses) + model.l2*np.sum(parameters[:dimension]**2)/2
                 + model.scale_l2*np.sum(parameters[dimension:]**2)/2)


def finite_difference(function, point, step=1e-5):
    columns = []
    for i in range(len(point)):
        offset = np.zeros_like(point)
        offset[i] = step
        columns.append((function(point+offset)-function(point-offset))/(2*step))
    return np.asarray(columns).T


class ProbitGradientTests(unittest.TestCase):
    def test_group_reductions_preserve_shuffled_rows_and_singletons(self):
        codes = np.array([2, 0, 1, 2, 0, 2])
        values = np.arange(18, dtype=float).reshape(6, 3)
        expected = values.copy()
        for group in _race_groups(codes):
            expected[group] -= expected[group].mean(axis=0)
        np.testing.assert_allclose(_race_center(values, codes), expected, atol=1e-14)
        np.testing.assert_allclose(_race_center(values[:, 0], codes), expected[:, 0], atol=1e-14)

    def test_mixed_size_batched_objective_matches_independent_scalar_oracle(self):
        rng = np.random.default_rng(91)
        sizes = [1, 2, 3, 4, 7, 14]*3
        codes = np.repeat(np.arange(len(sizes)), sizes)
        rng.shuffle(codes)
        x = _race_center(rng.normal(size=(len(codes), 4)), codes)
        groups = _race_groups(codes)
        winners = [int(rng.integers(len(group))) for group in groups]
        for heterogeneous in (False, True):
            for order in (8, 32, 96):
                with self.subTest(heterogeneous=heterogeneous, order=order):
                    model = GaussianRaceProbit(tuple('abcd'), heteroscedastic=heterogeneous,
                                               l2=.7, scale_l2=.9, quadrature_order=order)
                    parameters = rng.normal(scale=.3, size=8 if heterogeneous else 4)
                    actual, gradient = model._objective_and_gradient(parameters,x,codes,groups,winners)
                    reference = lambda p: reference_objective(model,p,x,groups,winners)
                    self.assertAlmostEqual(actual,reference(parameters),places=12)
                    np.testing.assert_allclose(gradient,finite_difference(reference,parameters),
                                               rtol=2e-6,atol=2e-7)

    def test_quadrature_chunks_are_bounded_independently_of_corpus_size(self):
        groups = np.arange(14000).reshape(1000,14)
        batches = _race_batches(groups,np.zeros(1000,dtype=int),512)
        self.assertEqual(sum(len(indices) for indices,_ in batches),1000)
        for indices,winners in batches:
            self.assertLessEqual(indices.shape[0]*14*14*512,262144)
            self.assertEqual(len(indices),len(winners))

    def test_log_jacobians_equal_and_heterogeneous_scales(self):
        for count in (1, 2, 4, 7):
            for heterogeneous in (False, True):
                with self.subTest(count=count, heterogeneous=heterogeneous):
                    means = np.linspace(-0.7, 0.9, count)
                    scales = np.linspace(0.6, 1.7, count) if heterogeneous else np.ones(count)
                    logs, mean_jac, scale_jac = _gaussian_log_win_probabilities(
                        means, scales, order=48, with_gradient=True)
                    np.testing.assert_array_equal(logs, reference_logs(means, scales, 48))
                    np.testing.assert_array_equal(logs, gaussian_log_win_probabilities(means, scales, order=48))
                    numerical_mean = finite_difference(lambda mu: reference_logs(mu, scales, 48), means)
                    numerical_scale = finite_difference(
                        lambda logged: reference_logs(means, np.exp(logged), 48), np.log(scales))
                    np.testing.assert_allclose(mean_jac, numerical_mean, rtol=2e-7, atol=2e-8)
                    np.testing.assert_allclose(scale_jac, numerical_scale, rtol=2e-7, atol=2e-8)
                    np.testing.assert_allclose(mean_jac.sum(axis=1), 0, atol=1e-12)
                    np.testing.assert_allclose(mean_jac @ means + scale_jac.sum(axis=1), 0, atol=1e-12)

    def test_normalized_logs_include_quadrature_normalizer_gradient(self):
        means = np.array([0.3, -0.6, 1., 0.1])
        logged = np.log([0.4, 1.4, 0.9, 2.])
        logs, mean_jac, scale_jac = _gaussian_log_win_probabilities(
            means, np.exp(logged), order=8, with_gradient=True)
        self.assertGreater(abs(np.exp(logsumexp(logs))-1), 1e-3)
        weights = np.exp(logs-logsumexp(logs))
        weights[1] -= 1

        def loss(point):
            values = reference_logs(point[:4], np.exp(point[4:]), 8)
            return -(values-logsumexp(values))[1]

        expected = finite_difference(loss, np.r_[means, logged])
        np.testing.assert_allclose(np.r_[weights @ mean_jac, weights @ scale_jac],
                                   expected, rtol=2e-7, atol=2e-8)

    def test_quadrature_order_parity(self):
        means = np.linspace(-0.6, 0.8, 14)
        scales = np.linspace(0.8, 1.2, 14)
        for order in (8, 24, 48, 96, 192):
            with self.subTest(order=order):
                logs, mean_jac, scale_jac = _gaussian_log_win_probabilities(
                    means, scales, order=order, with_gradient=True)
                np.testing.assert_array_equal(logs, reference_logs(means, scales, order))
                np.testing.assert_allclose(mean_jac, finite_difference(
                    lambda mu: reference_logs(mu, scales, order), means), rtol=2e-7, atol=2e-8)
                np.testing.assert_allclose(scale_jac, finite_difference(
                    lambda logged: reference_logs(means, np.exp(logged), order), np.log(scales)),
                    rtol=2e-7, atol=2e-8)

    def test_parameter_gradient_centering_bounded_link_and_regularization(self):
        x = np.random.default_rng(41).normal(size=(9, 3))
        codes = np.array([0, 1, 2, 0, 1, 2, 1, 2, 2])
        groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
        winners = [0, 2, 1]
        for heterogeneous in (False, True):
            model = GaussianRaceProbit(('a', 'b', 'c'), heteroscedastic=heterogeneous,
                                       l2=0.17, scale_l2=0.23, quadrature_order=24)
            for scale_coefficients in (np.zeros(3), np.array([0.4, -0.6, 0.2]),
                                       np.array([6., -9., 12.]), np.array([60., -90., 120.])):
                with self.subTest(heterogeneous=heterogeneous, scale_coefficients=scale_coefficients):
                    parameters = np.array([0.3, -0.2, 0.1])
                    if heterogeneous:
                        parameters = np.r_[parameters, scale_coefficients]
                    value, gradient = model._objective_and_gradient(parameters, x, codes, groups, winners)
                    reference = lambda point: reference_objective(model, point, x, groups, winners)
                    self.assertAlmostEqual(value, reference(parameters), places=13)
                    numerical = finite_difference(reference, parameters)
                    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)
                    if heterogeneous:
                        scales = model._scales(x, scale_coefficients, codes)
                        for group in groups:
                            self.assertAlmostEqual(np.log(scales[group]).sum(), 0, places=12)
                    stronger = GaussianRaceProbit(model.feature_columns, heteroscedastic=heterogeneous,
                                                   l2=model.l2+0.4, scale_l2=model.scale_l2+0.7,
                                                   quadrature_order=model.quadrature_order)
                    strong_value, strong_gradient = stronger._objective_and_gradient(
                        parameters, x, codes, groups, winners)
                    penalty = np.r_[np.full(3, 0.4), np.full(3, 0.7)] if heterogeneous else np.full(3, 0.4)
                    np.testing.assert_allclose(strong_gradient-gradient, penalty*parameters, atol=1e-12)
                    self.assertAlmostEqual(strong_value-value, np.sum(penalty*parameters**2)/2, places=10)

    def test_rare_outcome_gradient_stays_finite(self):
        for means, scales in ((np.array([-1000., 1000.]), np.ones(2)),
                              (np.array([-100., 0., 100.]), np.array([0.8, 1., 1.2]))):
            logs, mean_jac, scale_jac = _gaussian_log_win_probabilities(
                means, scales, order=48, with_gradient=True)
            self.assertTrue(np.isfinite(logs).all())
            self.assertTrue(np.isfinite(mean_jac).all())
            self.assertTrue(np.isfinite(scale_jac).all())
            np.testing.assert_allclose(mean_jac, finite_difference(
                lambda mu: reference_logs(mu, scales, 48), means, step=1e-3), rtol=2e-6, atol=1e-5)
            np.testing.assert_allclose(scale_jac, finite_difference(
                lambda logged: reference_logs(means, np.exp(logged), 48), np.log(scales)),
                rtol=2e-6, atol=1e-5)

    def test_model_parity_and_small_timed_benchmark(self):
        frame, score = race_frame(16), race_frame(4, 30)
        rng = np.random.default_rng(27)
        for k in range(6):
            frame[f'extra_{k}'] = rng.normal(size=len(frame))
            score[f'extra_{k}'] = rng.normal(size=len(score))
        columns = ('ability', 'context') + tuple(f'extra_{k}' for k in range(6))
        for heterogeneous in (False, True):
            options = dict(heteroscedastic=heterogeneous, l2=0.1, scale_l2=1., max_iter=120)
            reference_model = GaussianRaceProbit(columns, **options)
            x, codes = reference_model._matrix(frame, fit=True)
            groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
            winners = [int(np.argmax(frame.target_win.to_numpy()[group])) for group in groups]
            reference = lambda parameters: reference_objective(reference_model, parameters, x, groups, winners)
            # Warm both quadrature caches before measuring either fit.
            initial = np.zeros(len(columns)*(2 if heterogeneous else 1))
            reference(initial)
            reference_model._objective_and_gradient(initial, x, codes, groups, winners)
            results = []

            def numerical_fit(objective, initial, **kwargs):
                self.assertIs(kwargs.pop('jac'), True)
                result = minimize(reference, initial, **kwargs)
                results.append(result)
                return result

            start = perf_counter()
            with patch('ima.performance_probit.minimize', side_effect=numerical_fit):
                reference_model.fit(frame)
            numerical_seconds = perf_counter()-start

            def analytic_fit(*args, **kwargs):
                self.assertIs(kwargs['jac'], True)
                result = minimize(*args, **kwargs)
                results.append(result)
                return result

            start = perf_counter()
            with patch('ima.performance_probit.minimize', side_effect=analytic_fit):
                model = GaussianRaceProbit(columns, **options).fit(frame)
            analytic_seconds = perf_counter()-start
            np.testing.assert_allclose(results[1].fun, results[0].fun, atol=1e-9, rtol=0)
            np.testing.assert_allclose(results[1].x, results[0].x, atol=2e-5, rtol=2e-4)
            np.testing.assert_allclose(model.predict_proba(score), reference_model.predict_proba(score), atol=2e-5, rtol=2e-4)
            self.assertEqual(model.fit_diagnostics['objective_engine'],'bounded_numpy_batches_v1')
            self.assertGreater(model.fit_diagnostics['objective_evaluations'],0)
            self.assertGreater(model.fit_diagnostics['optimizer_seconds'],0)
            self.assertLess(results[1].nfev*4, results[0].nfev)
            print(f'probit benchmark heteroscedastic={heterogeneous}: '
                  f'finite_difference={numerical_seconds:.4f}s/{results[0].nfev} evaluations; '
                  f'analytic={analytic_seconds:.4f}s/{results[1].nfev} evaluations', flush=True)
