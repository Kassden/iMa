import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.experiments import research_protocol_summary
from ima.research_evaluation import (
    ResearchEvaluationError,
    baseline_probabilities,
    build_expanding_folds,
    evaluate_research_probabilities,
    make_protocol_manifest,
    per_race_log_losses,
    select_fold,
)


FIXTURE = Path(__file__).parent / "fixtures" / "research_races.csv"


class ResearchEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(FIXTURE, parse_dates=["date"])

    def test_expanding_folds_keep_whole_races_and_time_order(self):
        folds = build_expanding_folds(
            self.frame, min_train_races=2, calibration_races=1, score_races=1,
        )
        self.assertEqual(3, len(folds))
        first = folds[0]
        self.assertEqual(("R1", "R2"), first.train_race_ids)
        self.assertEqual(("R3",), first.calibration_race_ids)
        self.assertEqual(("R4",), first.score_race_ids)
        self.assertTrue(set(first.train_race_ids).isdisjoint(first.score_race_ids))

    def test_protocol_manifest_is_stable_and_exposed_from_experiments(self):
        direct = make_protocol_manifest(
            self.frame, min_train_races=2, calibration_races=1, score_races=1,
        ).to_dict()
        via_experiments = research_protocol_summary(
            self.frame, min_train_races=2, calibration_races=1, score_races=1,
        )
        self.assertEqual(direct, via_experiments)
        self.assertEqual("win_probability", direct["target_kind"])
        self.assertEqual(6, direct["race_count"])
        self.assertEqual(16, len(direct["protocol_id"]))

    def test_select_fold_rejects_missing_race_ids(self):
        with self.assertRaisesRegex(ResearchEvaluationError, "Missing fold race rows"):
            select_fold(self.frame, ["R1", "NOPE"])

    def test_baselines_sum_to_one_by_race(self):
        for kind in ("uniform", "raw_market", "calibrated_market"):
            probabilities = baseline_probabilities(self.frame, kind)
            totals = pd.Series(probabilities).groupby(self.frame["race_id"]).sum()
            self.assertTrue(np.allclose(totals.to_numpy(), 1.0))

    def test_per_race_losses_match_race_weighted_metric(self):
        probabilities = baseline_probabilities(self.frame, "raw_market")
        losses = per_race_log_losses(probabilities, self.frame, label="raw_market")
        report = evaluate_research_probabilities(probabilities, self.frame, label="raw_market")
        self.assertAlmostEqual(
            losses["race_log_loss"].mean(),
            report["race_weighted_log_loss"],
        )
        self.assertEqual("raw_market", report["label"])
        self.assertEqual(len(losses), len(report["per_race_losses"]))

    def test_probability_shape_and_race_totals_are_guarded(self):
        bad = baseline_probabilities(self.frame, "uniform").copy()
        bad[0] = 0.99
        with self.assertRaisesRegex(ResearchEvaluationError, "sum to one by race"):
            evaluate_research_probabilities(bad, self.frame, label="bad")
        with self.assertRaisesRegex(ResearchEvaluationError, "length"):
            evaluate_research_probabilities(bad[:-1], self.frame, label="bad")


if __name__ == "__main__":
    unittest.main()
