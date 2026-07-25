import unittest

import numpy as np
import pandas as pd

from ima.pools import paid_place_count, rank_combinations, evaluate_top_pool_selections


class PoolTests(unittest.TestCase):
    def test_hkjc_paid_place_count_depends_on_field_size(self):
        self.assertEqual(2, paid_place_count(4))
        self.assertEqual(2, paid_place_count(6))
        self.assertEqual(3, paid_place_count(7))
        self.assertEqual(3, paid_place_count(14))

    def test_first4_is_unordered_and_quartet_is_ordered(self):
        runners = ["1", "2", "3", "4"]
        probabilities = np.array([0.4, 0.3, 0.2, 0.1])
        first4 = rank_combinations(runners, probabilities, "FIRST4")
        quartet = rank_combinations(runners, probabilities, "QUARTET")
        self.assertEqual(1, len(first4))
        self.assertEqual(24, len(quartet))
        self.assertAlmostEqual(1.0, first4[0].probability)

    def test_every_pool_reports_top_selection_probability_and_hit_rate(self):
        frame = pd.DataFrame({
            "race_id": ["r1"] * 7,
            "horse_no": ["1", "2", "3", "4", "5", "6", "7"],
            "result": [1, 2, 3, 4, 5, 6, 7],
            "probability": [0.30, 0.24, 0.18, 0.12, 0.08, 0.05, 0.03],
        })
        report = evaluate_top_pool_selections(frame, "probability")
        self.assertEqual(
            {"WIN", "PLACE", "QIN", "QPL", "TRI", "TIERCE", "FIRST4", "QUARTET"},
            set(report),
        )
        for metrics in report.values():
            self.assertEqual(1, metrics["races"])
            self.assertEqual(1, metrics["hits"])
            self.assertGreater(metrics["mean_top_probability"], 0)
            self.assertLessEqual(metrics["mean_top_probability"], 1)

    def test_first4_hit_ignores_order_but_quartet_does_not(self):
        frame = pd.DataFrame({
            "race_id": ["r1"] * 7,
            "horse_no": ["1", "2", "3", "4", "5", "6", "7"],
            "result": [2, 1, 4, 3, 5, 6, 7],
            "probability": [0.30, 0.24, 0.18, 0.12, 0.08, 0.05, 0.03],
        })
        report = evaluate_top_pool_selections(frame, "probability")
        self.assertEqual(1, report["FIRST4"]["hits"])
        self.assertEqual(0, report["QUARTET"]["hits"])


if __name__ == "__main__":
    unittest.main()
