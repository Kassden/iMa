import pickle
import unittest
import numpy as np
from ima.probabilistic_adapters import RankingWinAdapter, ranking_scores_to_probabilities, distribution_to_win, win_to_joint
from ima.performance_distributions import PerformanceDistribution
from ima.research_executor import _ranking_scores_to_probabilities
from tests.test_pipeline_graph import race_frame


class ProbabilisticAdapterTests(unittest.TestCase):
    def test_legacy_ranker_softmax_and_temperature_parity(self):
        frame = race_frame(10)
        scores = frame.ability.to_numpy()
        np.testing.assert_array_equal(ranking_scores_to_probabilities(scores, frame.race_id), _ranking_scores_to_probabilities(scores, frame.race_id))
        adapter = RankingWinAdapter().fit(scores, frame)
        np.testing.assert_array_equal(adapter.transform(scores, frame.race_id), adapter.calibrator.transform(_ranking_scores_to_probabilities(scores, frame.race_id), frame.race_id))
        np.testing.assert_array_equal(adapter.transform(scores, frame.race_id), pickle.loads(pickle.dumps(adapter)).transform(scores, frame.race_id))

    def test_time_direction_and_joint_assumption(self):
        speed = PerformanceDistribution(np.array([2., 3.]), np.array([0.2, 0.2]), "log_speed_mps")
        time = PerformanceDistribution(-speed.location, speed.scale, "log_time_seconds", "lower")
        np.testing.assert_array_equal(distribution_to_win(speed, ["r", "r"]), distribution_to_win(time, ["r", "r"]))
        joint = win_to_joint([0.2, 0.3, 0.5], ["a", "b", "c"])
        np.testing.assert_allclose(joint.win_probabilities(), [0.2, 0.3, 0.5])
        self.assertIn("not identified", joint.metadata["assumption"])
