"""Independent accuracy and fail-closed checks for batched race probabilities."""

import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from ima import performance_probit as probit


def two_runner_probabilities(location, scale):
    difference = (location[:, 0] - location[:, 1]) / np.hypot(
        scale[:, 0], scale[:, 1])
    # Evaluate both tails directly, avoiding subtraction from one.
    return np.column_stack((ndtr(difference), ndtr(-difference)))


def centered_three_runner_probabilities(scale):
    """P(X_i-X_j > 0, X_i-X_k > 0) for independent centered normals."""
    expected = np.empty_like(scale, dtype=float)
    for i in range(3):
        j, k = [index for index in range(3) if index != i]
        correlation = (scale[:, i] / np.hypot(scale[:, i], scale[:, j])) * (
            scale[:, i] / np.hypot(scale[:, i], scale[:, k]))
        expected[:, i] = 0.25 + np.arcsin(correlation) / (2 * np.pi)
    return expected


class BatchedProbitRefinementTests(unittest.TestCase):
    def assert_probability_rows(self, actual, shape):
        self.assertEqual(actual.shape, shape)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual > 0).all())
        self.assertTrue((actual <= 1).all())
        np.testing.assert_allclose(actual.sum(axis=1), 1., rtol=0, atol=2e-14)

    def scalar_oracle(self, location, scale, **kwargs):
        values = []
        for means, sigmas in zip(location, scale):
            probabilities, diagnostics = probit.gaussian_win_probabilities(
                means, sigmas, return_diagnostics=True, **kwargs)
            self.assertTrue(diagnostics['converged'], diagnostics)
            values.append(probabilities)
        return np.stack(values)

    def test_symmetric_fields_and_singletons(self):
        for size in (1, 2, 3, 6, 14):
            location = np.broadcast_to(np.array([-4., 0., 9.])[:, None],
                                       (3, size)).copy()
            scale = np.broadcast_to(np.array([.02, 1., 3.])[:, None],
                                    (3, size)).copy()
            with self.subTest(size=size):
                actual = probit._batched_win_probabilities(location, scale)
                self.assert_probability_rows(actual, location.shape)
                np.testing.assert_allclose(actual, np.full(location.shape, 1 / size),
                                           rtol=0, atol=1e-9)
                np.testing.assert_allclose(actual, self.scalar_oracle(location, scale),
                                           rtol=0, atol=1e-9)

    def test_two_runner_unequal_scales_and_representable_rare_tails(self):
        location = np.array([[1., 0.], [-12., 0.], [12., 0.], [0., 0.],
                             [-7., 2.], [30., -30.]])
        scale = np.array([[2., 1.], [1., 1.], [1., 1.], [.001, 4.],
                          [.2, 2.], [4., 3.]])
        expected = two_runner_probabilities(location, scale)
        self.assertGreater(expected[1, 0], 0)
        self.assertLess(expected[1, 0], 1e-16)
        actual = probit._batched_win_probabilities(location, scale)
        self.assert_probability_rows(actual, location.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0)
        np.testing.assert_allclose(actual, self.scalar_oracle(location, scale),
                                   rtol=2e-13, atol=0)

    def test_underflow_floor_matches_scalar_without_zero_probabilities(self):
        location = np.array([[-1000., 1000.], [1000., -1000.]])
        scale = np.ones_like(location)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            actual = probit._batched_win_probabilities(location, scale)
        self.assert_probability_rows(actual, location.shape)
        np.testing.assert_array_equal(actual, self.scalar_oracle(location, scale))
        self.assertEqual(actual[0, 0], np.finfo(float).tiny)
        self.assertEqual(actual[1, 1], np.finfo(float).tiny)

    def test_three_runner_orthant_formula_after_hermite_nonconvergence(self):
        scale = np.array([[.02, .8, 2.5], [.001, .8, 2.5], [.1, 1., 3.],
                          [2.5, .02, .8]])
        location = np.zeros_like(scale)
        expected = centered_three_runner_probabilities(scale)
        for tolerance in (1e-6, 1e-9):
            with self.subTest(tolerance=tolerance):
                for means, sigmas in zip(location, scale):
                    _, diagnostics = probit.gaussian_win_probabilities(
                        means, sigmas, tolerance=tolerance, return_diagnostics=True)
                    self.assertEqual(diagnostics['integration_method'], 'adaptive_quad')
                    self.assertTrue(diagnostics['converged'], diagnostics)
                actual = probit._batched_win_probabilities(
                    location, scale, tolerance=tolerance)
                self.assert_probability_rows(actual, location.shape)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
                np.testing.assert_allclose(actual, self.scalar_oracle(
                    location, scale, tolerance=tolerance), rtol=0, atol=2e-10)

    def test_mixed_convergence_rows_match_scalar_adaptive_oracle(self):
        location = np.array([[0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0.],
                             [-9., -2., 0., .1, 1., 2.],
                             [.2, -.3, .7, -.9, .1, .4]])
        scale = np.array([[1., 1., 1., 1., 1., 1.],
                          [.02, .4, .8, 1.2, 2., 3.],
                          [.5, .8, 1., 1.2, 1.4, 2.],
                          [.8, 1.1, 1.2, .6, 1.5, .9]])
        original_location, original_scale = location.copy(), scale.copy()
        for order, max_order in ((8, 16), (32, 128), (64, 512)):
            with self.subTest(order=order, max_order=max_order):
                options = dict(order=order, max_order=max_order, tolerance=1e-9)
                expected = self.scalar_oracle(location, scale, **options)
                actual = probit._batched_win_probabilities(location, scale, **options)
                self.assert_probability_rows(actual, location.shape)
                np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)
                # Small probabilities need a relative check in addition to absolute accuracy.
                np.testing.assert_allclose(actual[2, 0], expected[2, 0],
                                           rtol=2e-6, atol=0)
        np.testing.assert_array_equal(location, original_location)
        np.testing.assert_array_equal(scale, original_scale)

    def test_permutation_affine_invariance_and_noncontiguous_inputs(self):
        location = np.array([[.2, -.3, .7], [-.8, .1, .5], [0., 0., 0.]])
        scale = np.array([[.8, 1.1, 1.2], [.4, 1.4, .9], [.02, .8, 2.5]])
        baseline = self.scalar_oracle(location, scale, tolerance=1e-9)
        actual = probit._batched_win_probabilities(
            (7. + 3. * location)[::-1, ::-1], (3. * scale)[::-1, ::-1],
            tolerance=1e-9)
        self.assert_probability_rows(actual, location.shape)
        np.testing.assert_allclose(actual, baseline[::-1, ::-1], rtol=0, atol=2e-9)

    def test_invalid_shapes_and_scales_raise(self):
        invalid = (
            ([0., 1.], [1., 1.]),
            (0., 1.),
            (np.zeros((2, 3, 1)), np.ones((2, 3, 1))),
            (np.zeros((2, 3)), np.ones((3, 2))),
            (np.zeros((2, 3)), np.ones((1, 3))),
            (np.zeros((2, 3)), np.ones(3)),
            (np.empty((2, 0)), np.empty((2, 0))),
            ([[0., 1.]], [[1., 0.]]),
            ([[0., 1.]], [[1., -1.]]),
            ([[0., 1.]], [[1., np.nan]]),
            ([[0., 1.]], [[1., np.inf]]),
            ([[0., np.nan]], [[1., 1.]]),
            ([[0., np.inf]], [[1., 1.]]),
        )
        for location, scale in invalid:
            with self.subTest(location=location, scale=scale):
                with self.assertRaises(ValueError):
                    probit._batched_win_probabilities(location, scale)

    def test_invalid_tolerance_and_quadrature_order_raise(self):
        location, scale = np.zeros((2, 3)), np.ones((2, 3))
        for tolerance in (0., -1., np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                probit._batched_win_probabilities(location, scale, tolerance=tolerance)
        with self.assertRaises(ValueError):
            probit._batched_win_probabilities(location, scale, order=7)

    def test_forced_adaptive_failure_remains_raised_for_entire_batch(self):
        location = np.zeros((3, 3))
        scale = np.array([[1., 1., 1.], [.02, .8, 2.5], [1., 1., 1.]])
        with patch.object(probit, 'quad', side_effect=RuntimeError('forced failure')) as quad:
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                probit._batched_win_probabilities(location, scale)
            self.assertGreater(quad.call_count, 0)

    def test_adaptive_error_bounds_and_invalid_integrals_remain_fail_closed(self):
        location, scale = np.zeros((1, 3)), np.array([[.02, .8, 2.5]])
        results = ((.1, 1e-10, {}, 'forced subdivision failure'),
                   (np.nan, 0., {}), (.1, np.inf, {}), (-.1, 0., {}),
                   (.1, -1., {}), (0., 0., {}), (1 / 9, 1e-3, {}),
                   (1 / 10, 0., {}))
        for result in results:
            with self.subTest(result=result), patch.object(probit, 'quad', return_value=result):
                with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                    probit._batched_win_probabilities(location, scale)

    def test_quadrature_scratch_chunks_do_not_grow_with_row_count(self):
        original_log_ndtr = probit.log_ndtr
        largest_scratch = []
        for rows in (128, 513):
            observed_sizes = []

            def checked_log_ndtr(values, *args, **kwargs):
                observed_sizes.append(np.size(values))
                self.assertLessEqual(np.size(values), 262144)
                return original_log_ndtr(values, *args, **kwargs)

            with self.subTest(rows=rows), patch.object(probit, 'log_ndtr',
                                                       side_effect=checked_log_ndtr):
                actual = probit._batched_win_probabilities(
                    np.zeros((rows, 7)), np.ones((rows, 7)), order=64, max_order=128)
                self.assert_probability_rows(actual, (rows, 7))
                np.testing.assert_allclose(actual, 1 / 7, rtol=0, atol=1e-9)
            self.assertTrue(observed_sizes, 'Quadrature scratch was not observed')
            largest_scratch.append(max(observed_sizes))
        self.assertEqual(largest_scratch[0], largest_scratch[1])


if __name__ == '__main__':
    unittest.main()
