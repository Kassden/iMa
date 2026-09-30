import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.report_v4_programs import _winner_losses, paired_bootstrap_interval


class V4ProgressReportTests(unittest.TestCase):
    def test_paired_interval_uses_identical_races(self):
        baseline = pd.Series([2.0, 1.0], index=["r1", "r2"])
        candidate = pd.Series([1.5, 0.5], index=["r1", "r2"])
        delta, low, high = paired_bootstrap_interval(candidate, baseline)
        self.assertEqual((-0.5, -0.5, -0.5), (delta, low, high))
        with self.assertRaises(ValueError):
            paired_bootstrap_interval(candidate.iloc[::-1], baseline)

    def test_winner_loss_rejects_missing_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.DataFrame({
                "race_id": ["r1", "r1", "r2"],
                "target_win": [1, 0, 0],
                "model_probability": [0.5, 0.5, 1.0],
            }).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one winner"):
                _winner_losses(path)
