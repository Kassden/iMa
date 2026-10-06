"""Probability-pool search contracts: no season labels or model fitting."""

import itertools
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

from ima.modeling import race_log_loss
from ima.probability_pool_search import (
    PoolRecipe, chronological_split, pool_probabilities, search_pool,
)


def calibration_frame(n_dates=10):
    rows = []
    for day, date in enumerate(pd.date_range("2026-01-14", periods=n_dates)):
        for race_no in (1, 2):
            winner = 1 + (day + race_no) % 3
            for horse_no in (1, 2, 3):
                rows.append({
                    "cohort": "calibration", "date": date,
                    "race_id": f"HKJC:{date.date()}:ST:R{race_no}",
                    "horse_no": horse_no, "target_win": int(horse_no == winner),
                })
    return pd.DataFrame(rows)


def component_matrix(frame):
    return np.array([[0.6, 0.1], [0.3, 0.2], [0.1, 0.7]])[
        frame.horse_no.to_numpy() - 1
    ]


def manual_probabilities(matrix, race_ids, recipe):
    if recipe.method == "arithmetic":
        pooled = recipe.benter_weight * matrix[:, 0] + (1 - recipe.benter_weight) * matrix[:, 1]
    else:
        pooled = np.exp(recipe.benter_weight * np.log(matrix[:, 0])
                        + (1 - recipe.benter_weight) * np.log(matrix[:, 1]))
    tempered = pooled ** (1 / recipe.temperature)
    output = np.empty(len(tempered))
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        output[positions] = tempered[positions] / tempered[positions].sum()
    return output


class PoolProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array(["r1", "r2", "r1", "r2", "r2"])
        self.matrix = np.array([[0.7, 0.2], [0.1, 0.5], [0.3, 0.8],
                                [0.2, 0.3], [0.7, 0.2]])

    def test_arithmetic_and_geometric_match_independent_race_normalization(self):
        original = self.matrix.copy()
        for method, temperature in itertools.product(("arithmetic", "geometric"), (0.7, 1, 2)):
            with self.subTest(method=method, temperature=temperature):
                recipe = PoolRecipe("original_pair", method, 0.6, temperature)
                actual = pool_probabilities(self.matrix, self.ids, recipe)
                np.testing.assert_allclose(actual, manual_probabilities(self.matrix, self.ids, recipe),
                                           rtol=1e-13, atol=1e-14)
                self.assertEqual((5,), np.asarray(actual).shape)
                for race_id in ("r1", "r2"):
                    self.assertAlmostEqual(1, actual[self.ids == race_id].sum(), places=14)
        np.testing.assert_array_equal(original, self.matrix)

    def test_temperature_one_endpoints_preserve_component_including_zeroes(self):
        matrix = np.array([[1, 0], [0, 1], [0.2, 0], [0.8, 1]], dtype=float)
        ids = np.array(["a", "a", "b", "b"])
        for method, weight in itertools.product(("arithmetic", "geometric"), (0, 1)):
            component = 0 if weight == 1 else 1
            with self.subTest(method=method, weight=weight):
                actual = pool_probabilities(matrix, ids, PoolRecipe("original_pair", method, weight, 1))
                np.testing.assert_allclose(actual, matrix[:, component], rtol=0, atol=1e-14)
                zeroes = matrix[:, component] == 0
                np.testing.assert_array_equal(np.asarray(actual)[zeroes], np.zeros(zeroes.sum()))

    def test_zero_support_is_stable_and_floating_sums_are_accepted(self):
        matrix = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        ids = np.array(["r"] * 3)
        for method, weight, temperature in itertools.product(("arithmetic", "geometric"),
                                                               (0.25, 0.5, 0.75), (0.7, 2)):
            with self.subTest(method=method, weight=weight, temperature=temperature):
                actual = np.asarray(pool_probabilities(matrix, ids, PoolRecipe(
                    "original_pair", method, weight, temperature)))
                self.assertTrue(np.isfinite(actual).all())
                self.assertTrue((actual >= 0).all())
                self.assertAlmostEqual(1, actual.sum(), places=14)
        floating = np.array([[0.1, 0.7], [0.2, 0.2], [0.7, 0.1]])
        floating[0, 0] = np.nextafter(floating[0, 0], np.inf)
        actual = pool_probabilities(floating, ids, PoolRecipe("original_pair", "arithmetic", 0.6, 1))
        self.assertAlmostEqual(1, np.sum(actual), places=14)

    def test_invalid_matrices_ids_and_component_totals_are_rejected(self):
        recipe = PoolRecipe("original_pair", "arithmetic", 0.6, 1)
        cases = [(np.zeros((0, 2)), np.array([])), (self.matrix[:, 0], self.ids),
                 (np.ones((5, 3)), self.ids), (self.matrix, self.ids[:-1]),
                 (self.matrix, np.array(["r1", None, "r1", "r2", "r2"], dtype=object))]
        for value in (-0.1, np.nan, np.inf):
            changed = self.matrix.copy()
            changed[0, 0] = value
            cases.append((changed, self.ids))
        for component in (0, 1):
            changed = self.matrix.copy()
            changed[self.ids == "r1", component] *= 0.9
            cases.append((changed, self.ids))
        for index, (matrix, ids) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                pool_probabilities(matrix, ids, recipe)

    def test_invalid_method_weight_and_temperature_are_rejected(self):
        parameters = [("unknown", 0.6, 1)]
        parameters += [("arithmetic", weight, 1) for weight in (-0.1, 1.1, np.nan, np.inf)]
        parameters += [("geometric", 0.6, temperature) for temperature in (0, -1, np.nan, np.inf)]
        for method, weight, temperature in parameters:
            with self.subTest(method=method, weight=weight, temperature=temperature), self.assertRaises(ValueError):
                recipe = PoolRecipe("original_pair", method, weight, temperature)
                pool_probabilities(self.matrix, self.ids, recipe)


class ChronologicalPoolSplitTests(unittest.TestCase):
    def test_whole_unique_dates_split_60_20_20_without_reordering_leakage(self):
        frame = calibration_frame().sample(frac=1, random_state=19)
        original = frame.copy(deep=True)
        split = chronological_split(frame)
        self.assertEqual({"search", "confirmation", "calibration"}, set(split))
        dates = sorted(frame.date.unique())
        expected = {"search": set(dates[:6]), "confirmation": set(dates[6:8]),
                    "calibration": set(dates[8:])}
        seen = set()
        for name, part in split.items():
            self.assertEqual(expected[name], set(part.date.unique()))
            self.assertEqual(6 * len(expected[name]), len(part))
            self.assertTrue(seen.isdisjoint(set(part.race_id)))
            seen.update(part.race_id)
            self.assertTrue(part.groupby("race_id").size().eq(3).all())
        self.assertEqual(set(frame.race_id), seen)
        pd.testing.assert_frame_equal(frame, original)
        self.assertLess(split["search"].date.max(), split["confirmation"].date.min())
        self.assertLess(split["confirmation"].date.max(), split["calibration"].date.min())

    def test_five_dates_minimum_preserves_whole_races(self):
        frame = calibration_frame(5)
        split = chronological_split(frame)
        self.assertEqual({"search": 3, "confirmation": 1, "calibration": 1},
                         {name: part.date.dt.normalize().nunique() for name, part in split.items()})
        with self.assertRaises(ValueError):
            chronological_split(calibration_frame(4))

    def test_duplicate_missing_inconsistent_and_holdout_rows_are_rejected(self):
        frame = calibration_frame()
        cases = [frame.drop(columns=[name]) for name in ("cohort", "date", "race_id")]
        cases.append(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
        for column, value in (("race_id", None), ("date", pd.NaT),
                              ("date", pd.Timestamp("2026-09-01")),
                              ("date", pd.Timestamp("2026-10-07")),
                              ("cohort", "season"), ("cohort", "confirmation")):
            changed = frame.copy()
            changed.loc[0, column] = value
            cases.append(changed)
        inconsistent = frame.copy()
        inconsistent.loc[0, "date"] += pd.Timedelta(days=1)
        cases.append(inconsistent)
        for index, changed in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                chronological_split(changed)


class ProbabilityPoolSearchTests(unittest.TestCase):
    def test_one_pair_small_grid_matches_manual_exhaustive_best(self):
        frame = calibration_frame()
        matrix = component_matrix(frame)
        original = frame.copy(deep=True)
        weights, temperatures = [0, 0.6, 1], [0.85, 1, 1.3]
        scored = []
        for method, weight, temperature in itertools.product(("arithmetic", "geometric"), weights, temperatures):
            recipe = PoolRecipe("original_pair", method, weight, temperature)
            probabilities = manual_probabilities(matrix, frame.race_id.to_numpy(), recipe)
            loss = race_log_loss(probabilities, frame.assign(target_probability=frame.target_win))
            scored.append((loss, recipe))
        expected = min(scored, key=lambda item: (round(item[0], 12), item[1].method != "arithmetic",
                       abs(item[1].benter_weight - 0.6), abs(item[1].temperature - 1),
                       item[1].benter_weight, item[1].temperature))[1]
        table, best = search_pool(frame, {"original_pair": matrix}, weights=weights, temperatures=temperatures)
        self.assertEqual(expected, best)
        self.assertEqual(18, len(table))
        columns = ["pair", "method", "benter_weight", "temperature"]
        self.assertFalse(table.duplicated(columns).any())
        self.assertEqual({tuple(asdict(recipe).values()) for _, recipe in scored},
                         set(table[columns].itertuples(index=False, name=None)))
        expected_losses = {tuple(asdict(recipe).values()): loss for loss, recipe in scored}
        for row in table.itertuples():
            key = (row.pair, row.method, row.benter_weight, row.temperature)
            self.assertAlmostEqual(expected_losses[key], row.search_race_log_loss, places=14)
        pd.testing.assert_frame_equal(frame, original)
        np.testing.assert_array_equal(matrix, component_matrix(frame))

    def test_default_two_pair_grid_has_588_candidates_and_original_baseline(self):
        frame = calibration_frame(5)
        pairs = {"original_pair": component_matrix(frame),
                 "standalone_pair": component_matrix(frame)[:, ::-1]}
        table, best = search_pool(frame, pairs)
        self.assertIsInstance(best, PoolRecipe)
        self.assertEqual(588, len(table))
        self.assertEqual({"original_pair", "standalone_pair"}, set(table.pair))
        self.assertEqual({"arithmetic", "geometric"}, set(table.method))
        self.assertEqual(21, table.benter_weight.nunique())
        self.assertEqual({0.7, 0.85, 1, 1.15, 1.3, 1.5, 2}, set(table.temperature))
        baseline = table[table.pair.eq("original_pair") & table.method.eq("arithmetic") &
                         np.isclose(table.benter_weight, 0.6) & table.temperature.eq(1)]
        self.assertEqual(1, len(baseline))

    def test_exact_loss_ties_prefer_original_arithmetic_point6_temperature1(self):
        frame = calibration_frame(5)
        frame = frame[frame.horse_no.le(2)].copy()
        frame["target_win"] = frame.horse_no.eq(1).astype(int)
        uniform = np.full((len(frame), 2), 0.5)
        for pairs in ({"standalone_pair": uniform, "original_pair": uniform},
                      {"original_pair": uniform, "standalone_pair": uniform}):
            _, best = search_pool(frame, pairs, weights=[0.8, 0.4, 0.6], temperatures=[1.15, 0.85, 1])
            self.assertEqual(PoolRecipe("original_pair", "arithmetic", 0.6, 1), best)

    def test_final_tie_parameters_are_deterministic_under_reversed_grids(self):
        frame = calibration_frame(5)
        frame = frame[frame.horse_no.le(2)].copy()
        frame["target_win"] = frame.horse_no.eq(1).astype(int)
        uniform = np.full((len(frame), 2), 0.5)
        for weights, temperatures in (([0.725, 0.475], [1.25, 0.75]),
                                      ([0.475, 0.725], [0.75, 1.25])):
            _, best = search_pool(frame, {"original_pair": uniform},
                                  weights=weights, temperatures=temperatures)
            self.assertEqual(PoolRecipe("original_pair", "arithmetic", 0.475, 0.75), best)

    def test_invalid_alignment_winners_and_noncalibration_labels_are_rejected(self):
        frame = calibration_frame(5)
        matrix = component_matrix(frame)
        for changed in (frame.assign(cohort="season"), frame.drop(columns="cohort"),
                        frame.drop(columns="target_win"),
                        frame.drop(columns="race_id"), frame.assign(race_id=None),
                        frame.assign(date=pd.NaT),
                        frame.assign(date=pd.Timestamp("2026-09-01")),
                        frame.assign(target_win=0), frame.assign(target_win=1)):
            with self.subTest(columns=changed.columns.tolist()), self.assertRaises(ValueError):
                search_pool(changed, {"original_pair": matrix}, weights=[0.6], temperatures=[1])
        invalid_target = frame.copy()
        invalid_target.loc[:2, "target_win"] = [2, -1, 0]
        with self.assertRaises(ValueError):
            search_pool(invalid_target, {"original_pair": matrix}, weights=[0.6], temperatures=[1])
        with self.assertRaises(ValueError):
            search_pool(frame, {"original_pair": matrix[:-1]}, weights=[0.6], temperatures=[1])
        with self.assertRaises(ValueError):
            search_pool(frame, {}, weights=[0.6], temperatures=[1])

    def test_invalid_or_duplicate_custom_grids_are_rejected(self):
        frame = calibration_frame(5)
        pairs = {"original_pair": component_matrix(frame)}
        for weights, temperatures in (([], [1]), ([0.6], []), ([0.6, 0.6], [1]),
                                      ([0.6], [1, 1]), ([-0.1], [1]), ([1.1], [1]),
                                      ([0.6], [0]), ([0.6], [np.inf])):
            with self.subTest(weights=weights, temperatures=temperatures), self.assertRaises(ValueError):
                search_pool(frame, pairs, weights=weights, temperatures=temperatures)


if __name__ == "__main__":
    unittest.main()
