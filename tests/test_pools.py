import itertools
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.pools import (
    SUPPORTED_POOLS, CombinationProbability, OrderExponents, benter_order_probability,
    evaluate_top_pool_selections, paid_place_count, rank_combinations, rank_pool_combinations,
)
from scripts.enrich_pool_metrics import pool_metrics_for_artifact


class PoolTests(unittest.TestCase):
    def test_paid_position_rankings_match_original_benter_marginals_exactly(self):
        for size in [0, 1, 2, 6, 7]:
            for exponents in [OrderExponents(), OrderExponents(second=0.5, third=1.7)]:
                with self.subTest(size=size, exponents=exponents):
                    runners = [str(i + 1) for i in range(size)]
                    probabilities = np.arange(size, dtype=float)
                    strengths = np.clip(probabilities, 1e-12, None)
                    places = paid_place_count(size)
                    # Reference the original per-ticket sums, preserving order of addition.
                    orders = [(order, benter_order_probability(order, strengths, exponents))
                              for order in itertools.permutations(range(size), places)]
                    expected_place = [CombinationProbability(
                        "PLACE", (runner,), sum(p for order, p in orders if index in order),
                    ) for index, runner in enumerate(runners)]
                    expected_qpl = [CombinationProbability(
                        "QPL", tuple(sorted(runners[i] for i in pair)),
                        sum(p for order, p in orders if set(pair).issubset(order)),
                    ) for pair in itertools.combinations(range(size), 2)]
                    expected = {
                        "PLACE": sorted(expected_place, key=lambda item: (
                            -round(item.probability, 12), tuple(map(int, item.runners)))),
                        "QPL": sorted(expected_qpl, key=lambda item: (
                            -round(item.probability, 12), tuple(sorted(map(int, item.runners))))),
                    }
                    actual = rank_pool_combinations(runners, probabilities, ["PLACE", "QPL"], exponents)
                    self.assertEqual(expected, actual)
                    for pool in ["PLACE", "QPL"]:
                        self.assertEqual(expected[pool], rank_combinations(runners, probabilities, pool, exponents))
                    if size:
                        self.assertAlmostEqual(places, sum(item.probability for item in actual["PLACE"]))
                        self.assertAlmostEqual(places * (places - 1) / 2,
                                               sum(item.probability for item in actual["QPL"]))

    def test_all_pool_batch_preserves_rankings_and_aliases(self):
        runners = ["1", "2", "3", "4"]
        for probabilities in [np.ones(4), np.array([0.4, 0.3, 0.2, 0.1])]:
            exponents = OrderExponents(second=0.7, third=1.4)
            batch = rank_pool_combinations(runners, probabilities, exponents=exponents)
            self.assertEqual(set(SUPPORTED_POOLS), set(batch))
            for pool in SUPPORTED_POOLS:
                self.assertEqual(rank_combinations(runners, probabilities, pool, exponents), batch[pool])
        aliases = rank_pool_combinations(runners, np.ones(4), ["quinella place", "PLACE"])
        self.assertEqual({"QPL", "PLACE"}, set(aliases))
        with self.assertRaises(ValueError):
            rank_pool_combinations(runners, np.ones(3))

    def test_fourteen_runner_paid_pools_share_exactly_one_enumeration(self):
        runners = [str(i + 1) for i in range(14)]
        probabilities = np.arange(14, 0, -1, dtype=float)
        with patch("ima.pools.benter_order_probability", wraps=benter_order_probability) as order:
            batch = rank_pool_combinations(runners, probabilities, ["PLACE", "QPL"])
            self.assertEqual(14 * 13 * 12, order.call_count)
        self.assertEqual(14, len(batch["PLACE"]))
        self.assertEqual(91, len(batch["QPL"]))
        for pool in ["PLACE", "QPL"]:
            with patch("ima.pools.benter_order_probability", wraps=benter_order_probability) as order:
                self.assertEqual(batch[pool], rank_combinations(runners, probabilities, pool))
                self.assertEqual(14 * 13 * 12, order.call_count)

    def test_equal_strength_rankings_are_numeric_and_independent_of_input_order(self):
        runners = ["13", "10", "2", "1", "4", "3", "7"]
        exponents = OrderExponents(second=0.7, third=1.4)
        baseline = rank_pool_combinations(runners, np.ones(7), exponents=exponents)
        for pool, tickets in baseline.items():
            count = len(tickets[0].runners)
            expected_top = tuple(map(str, [1, 2, 3, 4][:count]))
            self.assertEqual(expected_top, tickets[0].runners, pool)
        for permutation in [list(range(6, -1, -1)), [3, 0, 5, 2, 6, 1, 4]]:
            reordered = [runners[index] for index in permutation]
            replay = rank_pool_combinations(reordered, np.ones(7), exponents=exponents)
            for pool in SUPPORTED_POOLS:
                self.assertEqual([ticket.runners for ticket in baseline[pool]],
                                 [ticket.runners for ticket in replay[pool]], pool)
                for ticket, repeated in zip(baseline[pool], replay[pool]):
                    self.assertAlmostEqual(ticket.probability, repeated.probability, places=14)

    def test_negligible_strength_noise_preserves_all_rankings_and_exact_probabilities(self):
        runners = ["13", "1", "10", "2", "3", "4", "7"]
        probabilities = np.array([0.2, 0.4, 0.2, 0.05, 0.05, 0.05, 0.05])
        baseline = rank_pool_combinations(runners, probabilities)
        self.assertEqual(("1", "10"), baseline["QIN"][0].runners)
        noisy = probabilities.copy()
        noisy[0] = np.nextafter(noisy[0], np.inf)
        noisy[2] = np.nextafter(noisy[2], 0)
        permutation = [6, 0, 4, 2, 1, 5, 3]
        replay = rank_pool_combinations([runners[i] for i in permutation], noisy[permutation])
        for pool in SUPPORTED_POOLS:
            self.assertEqual([ticket.runners for ticket in baseline[pool]],
                             [ticket.runners for ticket in replay[pool]], pool)
            self.assertEqual(replay[pool], rank_combinations(
                [runners[i] for i in permutation], noisy[permutation], pool,
            ))
        for ticket in replay["WIN"]:
            index = runners.index(ticket.runners[0])
            expected = benter_order_probability((permutation.index(index),), noisy[permutation])
            self.assertEqual(expected, ticket.probability)
        self.assertTrue(any(ticket.probability != round(ticket.probability, 12)
                            for ticket in replay["WIN"]))

    def test_ranking_precision_does_not_erase_larger_probability_differences(self):
        runners = ["13", "10"]
        probabilities = np.array([0.5 + 1e-10, 0.5 - 1e-10])
        self.assertEqual(("13",), rank_combinations(runners, probabilities, "WIN")[0].runners)
        noisy = np.array([np.nextafter(0.5, 1), 0.5])
        tickets = rank_combinations(runners, noisy, "WIN")
        self.assertEqual(("10",), tickets[0].runners)
        self.assertGreater(tickets[1].probability, tickets[0].probability)

    def test_unordered_tie_comparison_sorts_numeric_tuple_without_changing_output(self):
        tickets = rank_combinations(["10", "2", "1"], np.ones(3), "QIN")
        self.assertEqual([("1", "2"), ("1", "10"), ("10", "2")],
                         [ticket.runners for ticket in tickets])

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

    def test_artifact_enrichment_reports_fundamental_and_combined(self):
        class Model:
            def predict_proba(self, frame):
                return frame["signal"].to_numpy()

        class Calibrator:
            def transform(self, values, race_ids):
                return values

        class Blend:
            def transform(self, fundamental, market, race_ids):
                return market

        frame = pd.DataFrame({
            "race_id": [f"r{i}" for i in range(10) for _ in range(4)],
            "race_no": [i for i in range(10) for _ in range(4)],
            "date": pd.to_datetime([f"2026-01-{i + 1:02d}" for i in range(10) for _ in range(4)]),
            "horse_no": [1, 2, 3, 4] * 10,
            "result": [1, 2, 3, 4] * 10,
            "target_win": [1, 0, 0, 0] * 10,
            "target_probability": [1.0, 0.0, 0.0, 0.0] * 10,
            "market_probability": [0.4, 0.3, 0.2, 0.1] * 10,
            "signal": [0.35, 0.30, 0.20, 0.15] * 10,
        })
        artifact = {
            "model": Model(), "calibrator": Calibrator(), "blend": Blend(),
            "order_exponents": OrderExponents(),
        }
        report = pool_metrics_for_artifact(frame, artifact)
        self.assertEqual({"fundamental", "combined"}, set(report))
        self.assertEqual(8, len(report["fundamental"]))


if __name__ == "__main__":
    unittest.main()
