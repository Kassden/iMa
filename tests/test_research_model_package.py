import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_model_package import (
    ResearchModelPackage,
    load_research_package,
    prediction_readback,
)
from ima.research_specs import PipelineRecipe


class DummyProbabilityModel:
    def predict_proba(self, frame):
        return np.full(len(frame), 0.5)


class ResearchModelPackageTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "race_id": ["R1", "R1", "R2", "R2"],
            "field_size": [2, 2, 2, 2],
        })

    def test_package_round_trip_and_prediction_readback(self):
        package = ResearchModelPackage(
            DummyProbabilityModel(),
            PipelineRecipe(),
            protocol_id="protocol-1",
            code_revision="abc123",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = package.save(Path(directory) / "pkg")
            loaded = load_research_package(root)
            direct = package.predict_proba(self.frame())
            round_trip = loaded.predict_proba(self.frame())
            readback = prediction_readback(root, self.frame())
        np.testing.assert_allclose(direct, round_trip)
        self.assertEqual(1.0, readback["probability_sum_min"])
        self.assertEqual(1.0, readback["probability_sum_max"])

    def test_incomplete_races_are_rejected(self):
        package = ResearchModelPackage(
            DummyProbabilityModel(),
            PipelineRecipe(),
            protocol_id="protocol-1",
            code_revision="abc123",
        )
        frame = self.frame()
        frame.loc[0, "field_size"] = 3
        with self.assertRaisesRegex(ValueError, "incomplete races"):
            package.predict_proba(frame)


if __name__ == "__main__":
    unittest.main()
