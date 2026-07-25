from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ima.inference import legacy_live_frame, predict_live_pools
from ima.operations import finalize_meeting
from ima.pools import OrderExponents


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

    def test_live_pool_predictions_use_market_combined_runner_strengths(self):
        class Model:
            def predict_proba(self, frame):
                return np.array([0.7, 0.2, 0.1])

        class Calibrator:
            def transform(self, probabilities, race_ids):
                return probabilities

        class Blend:
            def transform(self, fundamental, market, race_ids):
                return market

        frame = pd.DataFrame({
            "race_id": ["R1"] * 3,
            "horse_no": ["1", "2", "3"],
            "market_probability": [0.2, 0.6, 0.2],
            "ratable": [True, True, True],
        })
        artifact = {
            "model": Model(),
            "calibrator": Calibrator(),
            "blend": Blend(),
            "order_exponents": OrderExponents(),
        }
        pools = predict_live_pools(
            frame, artifact, pools=["WIN", "PLACE", "QIN", "QPL", "TRIO"]
        )
        race = pools["R1"]
        self.assertEqual({"WIN", "PLACE", "QIN", "QPL", "TRI"}, set(race))
        self.assertEqual(("2",), race["WIN"][0].runners)
        self.assertEqual(("1", "2"), race["QIN"][0].runners)
        self.assertAlmostEqual(sum(item.probability for item in race["WIN"]), 1.0)
        self.assertAlmostEqual(sum(item.probability for item in race["QPL"]), 3.0)

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
