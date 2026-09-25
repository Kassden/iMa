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
    placing_metrics,
    ranking_metrics,
    regression_metrics,
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

    def test_probability_metrics_include_race_dispersion_and_winner_mrr(self):
        probabilities = baseline_probabilities(self.frame, "uniform")
        metrics = evaluate_research_probabilities(
            probabilities, self.frame, label="uniform"
        )["metrics"]
        self.assertIn("race_log_loss_std", metrics)
        self.assertIn("worst_race_brier", metrics)
        self.assertGreater(metrics["winner_mrr"], 0.0)
        self.assertLessEqual(metrics["winner_mrr"], 1.0)

    def test_placing_metrics_weight_races_equally_and_score_exact_top_k(self):
        frame = pd.DataFrame({
            "race_id": ["small"] * 3 + ["large"] * 5,
            "place": [1, 1, 0, 1, 1, 0, 0, 0],
        })
        perfect = np.array([0.9, 0.8, 0.1, 0.9, 0.8, 0.3, 0.2, 0.1])
        metrics = placing_metrics(perfect, frame, label_column="place", top_k=2)
        self.assertAlmostEqual(1.0, metrics["race_precision_at_k"])
        self.assertAlmostEqual(1.0, metrics["race_recall_at_k"])
        self.assertAlmostEqual(1.0, metrics["race_f1_at_k"])
        self.assertAlmostEqual(1.0, metrics["race_top_pick_place_rate"])
        self.assertGreater(metrics["race_binary_log_loss"], 0.0)

    def test_ranking_metrics_are_grouped_by_race_and_handle_ties(self):
        frame = pd.DataFrame({
            "race_id": ["R1"] * 4 + ["R2"] * 2,
            "relevance": [1.0, 0.75, 0.5, 0.25, 1.0, 0.5],
        })
        perfect = frame["relevance"].to_numpy()
        metrics = ranking_metrics(perfect, frame, label_column="relevance")
        self.assertAlmostEqual(1.0, metrics["race_ndcg_at_3"])
        self.assertAlmostEqual(1.0, metrics["race_ndcg"])
        self.assertAlmostEqual(1.0, metrics["race_pairwise_accuracy"])
        self.assertAlmostEqual(1.0, metrics["winner_mrr"])

        tied = ranking_metrics(
            np.ones(len(frame)), frame, label_column="relevance"
        )
        self.assertAlmostEqual(0.5, tied["race_pairwise_accuracy"])
        self.assertEqual(0.0, tied["race_spearman"])

    def test_regression_metrics_report_race_mae_rmse_and_rank_quality(self):
        frame = pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2", "R2"],
            "speed": [10.0, 9.0, 8.0, 7.0, 6.0],
        })
        predictions = np.array([11.0, 9.0, 8.0, 5.0, 6.0])
        metrics = regression_metrics(
            predictions, frame, label_column="speed"
        )
        self.assertAlmostEqual((0.5 + (2.0 / 3.0)) / 2.0, metrics["race_mae"])
        self.assertGreaterEqual(metrics["race_rmse"], metrics["race_mae"])
        self.assertLessEqual(metrics["race_spearman"], 1.0)

    def test_target_metrics_reject_non_finite_and_wrong_length_predictions(self):
        frame = pd.DataFrame({"race_id": ["R1", "R1"], "label": [1, 0]})
        with self.assertRaisesRegex(ResearchEvaluationError, "finite"):
            placing_metrics(
                np.array([np.nan, 0.2]), frame, label_column="label", top_k=1
            )
        with self.assertRaisesRegex(ResearchEvaluationError, "length"):
            regression_metrics(np.array([1.0]), frame, label_column="label")


if __name__ == "__main__":
    unittest.main()
