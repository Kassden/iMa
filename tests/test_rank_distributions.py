import pickle
import unittest
import numpy as np
from ima.performance_distributions import PerformanceDistribution
from ima.performance_probit import gaussian_win_probabilities
from ima.rank_distributions import simulate_rank_distribution, converged_rank_distribution, plackett_luce_distribution, sample_plackett_luce_distribution


class RankDistributionTests(unittest.TestCase):
    def test_fixed_seed_chunk_invariance_gaussian_control_and_reload(self):
        dist = PerformanceDistribution(np.array([0., 0.4, 0.9]), np.array([1., 0.8, 1.2]))
        runners = ["a", "b", "c"]
        joint = simulate_rank_distribution(dist, runners, seed=81, draws=40000, chunk_size=1024)
        other = simulate_rank_distribution(dist, runners, seed=81, draws=40000, chunk_size=397)
        self.assertEqual(joint.order_probabilities, other.order_probabilities)
        np.testing.assert_allclose(joint.win_probabilities(), gaussian_win_probabilities(dist.location, dist.scale), atol=0.01)
        self.assertAlmostEqual(sum(joint.order_probabilities.values()), 1)
        self.assertAlmostEqual(sum(joint.place_probability(h, 2) for h in runners), 2)
        self.assertEqual(joint.order_probabilities, pickle.loads(pickle.dumps(joint)).order_probabilities)
        shifted = PerformanceDistribution(dist.location+1000, dist.scale)
        self.assertEqual(joint.order_probabilities, simulate_rank_distribution(shifted, runners, seed=81, draws=40000).order_probabilities)

    def test_exact_small_field_and_rare_outcome_interval(self):
        exact = plackett_luce_distribution([1/3]*3, ["a", "b", "c"])
        np.testing.assert_allclose(list(exact.order_probabilities.values()), 1/6)
        self.assertAlmostEqual(exact.quinella_probability(["a", "c"]), 1/3)
        self.assertAlmostEqual(exact.trio_probability(["a", "b", "c"]), 1)
        joint = simulate_rank_distribution(PerformanceDistribution(np.array([100., 0.]), np.ones(2)), ["a", "b"], draws=1000)
        self.assertEqual(joint.order_probability(["b", "a"]), 0)
        self.assertGreater(joint.probability_interval(0)[1], 0)

    def test_convergence_budget_and_covariance(self):
        dist = PerformanceDistribution(np.zeros(3), np.ones(3))
        converged = converged_rank_distribution(dist, ["a", "b", "c"], tolerance=0.04, min_draws=4096, max_draws=8192)
        self.assertTrue(converged.metadata["converged"])
        limited = converged_rank_distribution(dist, ["a", "b", "c"], tolerance=1e-5, min_draws=100, max_draws=100)
        self.assertFalse(limited.metadata["converged"])
        with self.assertRaisesRegex(ValueError, "semidefinite"):
            simulate_rank_distribution(dist, ["a", "b", "c"], covariance=[[1, 2, 0], [2, 1, 0], [0, 0, 1]])

    def test_sampled_plackett_luce_against_exact_and_large_field(self):
        probabilities = [0.2, 0.3, 0.5]
        exact = plackett_luce_distribution(probabilities, ["a", "b", "c"])
        sampled = sample_plackett_luce_distribution(probabilities, ["a", "b", "c"], draws=30000)
        for order, value in exact.order_probabilities.items():
            self.assertAlmostEqual(sampled.order_probability(order), value, delta=0.01)
        repeated = sample_plackett_luce_distribution(probabilities, ["a", "b", "c"], draws=30000, chunk_size=113)
        self.assertEqual(sampled.order_probabilities, repeated.order_probabilities)
        large = sample_plackett_luce_distribution(np.ones(14), [str(i) for i in range(14)], draws=2000)
        self.assertAlmostEqual(large.win_probabilities().sum(), 1)
        self.assertFalse(large.metadata["converged"])
