import tempfile
import unittest
from pathlib import Path
import json
import hashlib

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


class NestedDummyProbabilityModel(DummyProbabilityModel):
    def __init__(self, artifact):
        self.fit_report = {"learned_pools": [{"selection_mode": "nested_selection",
            "selection": {"selection_id": artifact["selection_id"],
                "selection_sha256": artifact["selection_sha256"],
                "selection_contract_sha256": artifact["protocol_sha256"],
                "selected_candidate_id": artifact["selected_candidate_id"],
                "selection_cutoff": artifact["selection_cutoff"], "artifact": artifact}}]}


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

    def test_nested_selection_identity_is_bound_to_package_manifest_and_replay(self):
        payload = {"protocol_sha256": "a"*64, "selected_candidate_id": "candidate-a",
                   "selection_cutoff": "2020-01-01", "selection_mode": "nested_selection",
                   "selected_graph": {"graph_id": "candidate-a"},
                   "selected_graph_sha256": hashlib.sha256(
                       json.dumps({"graph_id": "candidate-a"}, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        payload.update(selection_id="selection-"+digest[:24], selection_sha256=digest)
        package = ResearchModelPackage(NestedDummyProbabilityModel(payload), PipelineRecipe(),
            protocol_id="protocol-nested", code_revision="selection-replay")
        with tempfile.TemporaryDirectory() as directory:
            root = package.save(Path(directory) / "nested")
            loaded = load_research_package(root)
            np.testing.assert_allclose(package.predict_proba(self.frame()), loaded.predict_proba(self.frame()))
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(payload["selection_id"], manifest["selection_references"][0]["selection_id"])
            recipe_path = root / "recipe.json"
            saved_recipe = json.loads(recipe_path.read_text())
            self.assertEqual(payload["selection_id"], saved_recipe["selection_references"][0]["selection_id"])
            self.assertEqual(payload["selected_graph"], saved_recipe["selected_candidate_contracts"][0]["graph"])
            saved_recipe["selected_candidate_contracts"][0]["graph"]["graph_id"] = "forged"
            recipe_path.write_text(json.dumps(saved_recipe))
            with self.assertRaisesRegex(ValueError, "recipe/selected candidate contract"):
                load_research_package(root)
            recipe_path.write_text(json.dumps(package._recipe_package_payload(), indent=2, sort_keys=True))
            manifest["selection_references"][0]["selection_id"] = "selection-forged"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest does not match|selection references"):
                load_research_package(root)


if __name__ == "__main__":
    unittest.main()
