import unittest

import numpy as np
import pandas as pd

from ima.auxiliary import train_auxiliary_bundle
from ima.data import RaceSplits
from ima.feature_sets import FeatureSchema


class AuxiliaryModelTests(unittest.TestCase):
    @staticmethod
    def frame(prefix: str, offset: float) -> pd.DataFrame:
        rows = []
        for race in range(4):
            for horse in range(3):
                result = horse + 1
                rows.append({
                    "race_id": f"{prefix}{race}",
                    "horse_no": horse + 1,
                    "signal": 1.0 - horse * 0.3 + offset,
                    "venue": "ST" if race % 2 else "HV",
                    "result": result,
                    "finish_seconds": 70.0 + horse + race * 0.1,
                    "win_odds": 2.0 + horse * 2.0,
                    "place_odds": 1.2 + horse * 0.5,
                })
        return pd.DataFrame(rows)

    def test_tunes_and_predicts_all_auxiliary_targets(self):
        schema = FeatureSchema("test-aux", ("signal",), ("venue",))
        splits = RaceSplits(
            self.frame("T", 0.0), self.frame("V", 0.05), self.frame("E", 0.1),
        )
        grid = (
            {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 2, "max_iter": 20},
            {"learning_rate": 0.10, "max_leaf_nodes": 15, "min_samples_leaf": 2, "max_iter": 20},
        )
        bundle = train_auxiliary_bundle(splits, schema, grid)
        self.assertEqual({"position", "finish_time", "win_odds", "place_odds"}, set(bundle.models))
        prediction = bundle.predict(splits.test)
        self.assertIn("predicted_position_rank", prediction)
        self.assertTrue(prediction["predicted_win_odds"].gt(1).all())
        self.assertEqual(2, len(bundle.models["position"].candidate_scores))
        self.assertIn("top_pick_win_rate", bundle.test_metrics["position"])

    def test_current_market_prices_are_not_prediction_inputs(self):
        schema = FeatureSchema("test-aux", ("signal",), ("venue",))
        splits = RaceSplits(
            self.frame("T", 0.0), self.frame("V", 0.05), self.frame("E", 0.1),
        )
        grid = ({
            "learning_rate": 0.05, "max_leaf_nodes": 7,
            "min_samples_leaf": 2, "max_iter": 15,
        },)
        bundle = train_auxiliary_bundle(splits, schema, grid)
        changed = splits.test.copy()
        changed["win_odds"] *= 100
        changed["place_odds"] *= 100
        first = bundle.predict(splits.test)
        second = bundle.predict(changed)
        np.testing.assert_allclose(first["predicted_position"], second["predicted_position"])
        np.testing.assert_allclose(first["predicted_finish_time"], second["predicted_finish_time"])


if __name__ == "__main__":
    unittest.main()
