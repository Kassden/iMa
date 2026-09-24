import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.research_model_package import ResearchModelPackage, prediction_readback
from ima.research_specs import PipelineRecipe


class DummyProbabilityModel:
    def predict_proba(self, frame):
        return np.full(len(frame), 0.5)


class AgenticOptimizerE2ETests(unittest.TestCase):
    def test_terminal_agentic_dry_run_resume_and_package_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.optimize",
                    "run",
                    "--policy",
                    "agentic",
                    "--campaign",
                    str(campaign),
                    "--max-trials",
                    "6",
                    "--proposal-batch-size",
                    "3",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, first.returncode, first.stderr)
            second = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.optimize",
                    "run",
                    "--policy",
                    "agentic",
                    "--campaign",
                    str(campaign),
                    "--max-trials",
                    "6",
                    "--proposal-batch-size",
                    "3",
                    "--dry-run",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, second.returncode, second.stderr)
            payload = json.loads((campaign / "dry-run.json").read_text(encoding="utf-8"))
            self.assertEqual(3, len(payload["proposals"]))
            self.assertIn("recipe_hash", payload["proposals"][0])
            journal = campaign / "search" / "optuna-journal.log"
            self.assertTrue(journal.exists())

            package = ResearchModelPackage(
                DummyProbabilityModel(),
                PipelineRecipe(),
                protocol_id="fixture-protocol",
                code_revision="fixture-revision",
            )
            package_dir = package.save(root / "package")
            frame = pd.DataFrame({
                "race_id": ["R1", "R1", "R2", "R2"],
                "field_size": [2, 2, 2, 2],
            })
            readback = prediction_readback(package_dir, frame)
            self.assertEqual(1.0, readback["probability_sum_min"])
            self.assertEqual(1.0, readback["probability_sum_max"])


if __name__ == "__main__":
    unittest.main()
