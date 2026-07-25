import unittest

import numpy as np
import pandas as pd

from ima.feature_analysis import (
    association_report, benjamini_hochberg, correlation_report, permutation_importance_report,
)
from ima.feature_sets import FeatureSchema
from ima.modeling import normalize_by_race


class SignalModel:
    def predict_proba(self, frame):
        return normalize_by_race(np.exp(frame["signal"].to_numpy()), frame["race_id"])


class FeatureAnalysisTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for race in range(100):
            rows.extend([
                {"race_id": race, "target_win": 1, "target_probability": 1.0, "signal": 2.0, "copy": 2.0, "noise": race % 7, "venue": "ST"},
                {"race_id": race, "target_win": 0, "target_probability": 0.0, "signal": 0.0, "copy": 0.0, "noise": (race + 3) % 7, "venue": "HV"},
            ])
        self.frame = pd.DataFrame(rows)
        self.schema = FeatureSchema("test", ("signal", "copy", "noise"), ("venue",))

    def test_benjamini_hochberg_is_monotone_and_bounded(self):
        adjusted = benjamini_hochberg([0.04, 0.001, 0.02])
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))
        self.assertLess(adjusted[1], adjusted[0])

    def test_association_and_redundancy_reports_find_signal(self):
        association = association_report(self.frame, self.schema).set_index("feature")
        self.assertTrue(association.at["signal", "significant_fdr_05"])
        matrix, redundant = correlation_report(self.frame, self.schema.numeric, sample_size=1000)
        self.assertAlmostEqual(1.0, matrix.at["signal", "copy"])
        self.assertIn(("signal", "copy"), set(zip(redundant.feature_a, redundant.feature_b)))

    def test_permutation_importance_ranks_predictive_signal(self):
        report = permutation_importance_report(
            SignalModel(), self.frame, {"signal": ("signal",), "noise": ("noise",)}, repeats=3,
        ).set_index("name")
        self.assertGreater(report.at["signal", "delta_race_log_loss_mean"], 0.1)
        self.assertAlmostEqual(0.0, report.at["noise", "delta_race_log_loss_mean"], places=8)


if __name__ == "__main__":
    unittest.main()