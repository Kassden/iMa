from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ima.inference import legacy_live_frame
from ima.operations import finalize_meeting


class OperationsInferenceTests(unittest.TestCase):
    def test_legacy_live_frame_marks_first_timer_unratable(self):
        source = pd.DataFrame({
            "race_id": ["r1", "r1"], "horse_no_x": [1, 2], "horse_no_y": [2, 2],
            "win_odds": [2.0, 4.0], "race_class": ["4", "4"],
            "prediction_ready": [True, False],
        })
        frame = legacy_live_frame(source)
        self.assertEqual([True, False], frame["ratable"].tolist())
        self.assertAlmostEqual(1.0, frame["market_probability"].sum())

    def test_finalize_meeting_versions_official_data_and_model(self):
        official = [{
            "race_no": 1,
            "runners": [{"place": 1, "horse_no": 2}, {"place": 2, "horse_no": 1}],
        }]
        report = {"recommended_champion": "logit", "candidates": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = finalize_meeting(
                "2026/07/25", "ST", root / "meetings", root / "runs.csv", root / "races.csv",
                root / "models", root / "registry",
                collector=lambda *args, **kwargs: official,
                trainer=lambda *args, **kwargs: report,
            )
            self.assertEqual(1, manifest["race_count"])
            self.assertTrue((root / "registry" / f"{manifest['meeting_version']}.json").exists())


if __name__ == "__main__":
    unittest.main()
