import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.verify_official_snapshot import check_baseline_roster


class BaselineRosterTests(unittest.TestCase):
    def test_larger_snapshot_cannot_hide_baseline_runner_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "runners.parquet"
            manifest = root / "manifest.json"
            baseline = pd.DataFrame({"race_id": ["R1", "R1"],
                "horse_id": ["HK_2022_H033", "HK_2022_H034"], "source": ["official:hkjc-results"] * 2})
            baseline.to_parquet(path, index=False)
            data = {"source_policy": "HKJC-only; raw replay; no third-party values",
                    "files": {"runners.parquet": hashlib.sha256(path.read_bytes()).hexdigest()}}
            manifest.write_text(json.dumps(data))
            larger = pd.concat([baseline.iloc[:1], baseline.assign(race_id="R2")], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "attrition: 1"):
                check_baseline_roster(larger, path, manifest)
            complete = pd.concat([baseline, larger], ignore_index=True)
            self.assertEqual(check_baseline_roster(complete, path, manifest)["baseline_runner_keys_retained"], 2)
            data["files"]["runners.parquet"] = "invalid"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_baseline_roster(complete, path, manifest)


if __name__ == "__main__":
    unittest.main()
