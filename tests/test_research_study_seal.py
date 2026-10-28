from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ima.research_study_seal import create_study_seal, evaluate_study_seal


class ResearchStudySealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.candidate = self.root / "candidate.csv"
        self.baseline = self.root / "baseline.csv"
        rows = []
        for day in range(3):
            for horse, probability in (("1", .7), ("2", .3)):
                rows.append({"date": f"2099-01-0{day+1}", "race_id": f"r{day}",
                             "horse_no": horse, "model_probability": probability})
        pd.DataFrame(rows).to_csv(self.candidate, index=False)
        pd.DataFrame(rows).assign(model_probability=[.6, .4] * 3).to_csv(self.baseline, index=False)
        self.source = self.root / "history.json"
        self.source.write_text('{"reviewed": true}\n')
        self.seal = self.root / "study.seal.json"
        self.lock = self.root / "study.lock.json"
        self.result = self.root / "result.json"
        self.labels = self.root / "labels.csv"
        pd.DataFrame([{"date": f"2099-01-0{d}", "race_id": f"r{d-1}",
                       "horse_no": h, "target_win": int(h == "1")} 
                      for d in range(1, 4) for h in ("1", "2")]).to_csv(self.labels, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self):
        source = {"name": "manual-and-campaign-review", "path": str(self.source),
                  "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(), "status": "resolved"}
        return {"study_id": "study-1", "selection_cutoff": "2098-12-31",
                "objective": "win_race_log_loss",
                "candidate": {"prediction_csv": str(self.candidate),
                              "identity": {"candidate_id": "candidate", "revision": "rev-1"}},
                "baseline": {"prediction_csv": str(self.baseline),
                             "identity": {"candidate_id": "baseline", "revision": "rev-1"}},
                "split_id": "split-fixed", "recipe_id": "recipe-fixed", "dataset_id": "dataset-fixed",
                "code_id": "git:abc", "environment_id": "env:xyz",
                "exposure_inventory": {"scope": "all manually reviewed and campaign sources",
                                        "sources": [source], "known_exposures": []}}

    def create(self, manifest=None):
        with patch("ima.research_confirmation._utc_now",
                   return_value=datetime(2098, 12, 31, tzinfo=timezone.utc)):
            return create_study_seal(manifest or self.manifest(), self.lock, self.seal)

    def test_requires_named_hashed_exposure_inventory_and_readable_sources(self):
        manifest = self.manifest()
        del manifest["exposure_inventory"]
        with self.assertRaisesRegex(ValueError, "explicit non-empty exposure source inventory"):
            self.create(manifest)
        manifest = self.manifest()
        manifest["exposure_inventory"]["sources"][0]["path"] = str(self.root / "absent")
        with self.assertRaisesRegex(FileNotFoundError, "unreadable"):
            self.create(manifest)

    def test_unresolved_history_is_qualificationheld_and_never_clean(self):
        manifest = self.manifest()
        manifest["exposure_inventory"]["sources"][0]["status"] = "unresolved"
        self.create(manifest)
        payload = json.loads(self.seal.read_text())
        self.assertEqual("qualificationheld", payload["status"])
        self.assertEqual(["exposure_source_unresolved:manual-and-campaign-review"], payload["qualification_holds"])
        self.assertFalse(payload["exposure_inventory"]["history_completeness_claimed"])
        self.assertEqual("qualificationheld", evaluate_study_seal(self.seal, self.result)["status"])

    def test_exposed_race_date_is_quarantined(self):
        manifest = self.manifest()
        source = manifest["exposure_inventory"]["sources"][0]
        manifest["exposure_inventory"]["known_exposures"] = [{
            "date": "2099-01-02", "race_id": "r1", "source_name": source["name"],
            "source_sha256": source["sha256"],
        }]
        with self.assertRaisesRegex(ValueError, "overlaps known exposed"):
            self.create(manifest)
        self.assertFalse(self.lock.exists())

    def test_exposure_must_reference_inventoried_source_and_exact_hash(self):
        manifest = self.manifest()
        manifest["exposure_inventory"]["known_exposures"] = [{
            "date": "2099-01-02", "race_id": "unrelated", "source_name": "unlisted",
            "source_sha256": "0" * 64,
        }]
        with self.assertRaisesRegex(ValueError, "uninventoried source"):
            self.create(manifest)
        exposure = manifest["exposure_inventory"]["known_exposures"][0]
        exposure["source_name"] = manifest["exposure_inventory"]["sources"][0]["name"]
        with self.assertRaisesRegex(ValueError, "does not match inventory"):
            self.create(manifest)

    def test_seal_retry_is_immutable_and_binds_artifacts_and_ids(self):
        self.create()
        before = self.seal.read_bytes()
        self.create()
        self.assertEqual(before, self.seal.read_bytes())
        payload = json.loads(before)
        self.assertEqual("split-fixed", payload["split_id"])
        self.assertEqual("recipe-fixed", payload["recipe_id"])
        self.assertEqual("dataset-fixed", payload["dataset_id"])
        self.assertEqual("git:abc", payload["code_id"])
        self.assertEqual("env:xyz", payload["environment_id"])
        self.assertEqual("scheduled", evaluate_study_seal(self.seal, self.result)["status"])
        with self.assertRaisesRegex(ValueError, "Immutable artifact conflict"):
            create_study_seal(self.manifest() | {"code_id": "git:changed"}, self.lock, self.seal)

    def test_corrupted_seal_lock_or_prediction_artifact_is_rejected(self):
        self.create()
        self.candidate.write_text(self.candidate.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "candidate prediction artifact is missing or corrupted"):
            evaluate_study_seal(self.seal, self.result)

    def test_label_availability_waits_then_evaluates_at_fixed_horizon(self):
        self.create()
        with patch("ima.research_study_seal._utc_today", return_value=date(2099, 1, 4)), \
                patch("ima.research_confirmation._utc_today", return_value=date(2099, 1, 4)):
            self.assertEqual("pending_labels", evaluate_study_seal(self.seal, self.result)["status"])
            evaluated = evaluate_study_seal(self.seal, self.result, self.labels)
        self.assertEqual("evaluated", evaluated["status"])
        self.assertTrue(self.result.is_file())
        result = json.loads(self.result.read_text())
        self.assertEqual("confirmed_temporal_improvement", result["status"])
        self.assertEqual(hashlib.sha256(self.labels.read_bytes()).hexdigest(), result["labels_sha256"])
        self.assertEqual(3, result["comparison"]["race_count"])


if __name__ == "__main__":
    unittest.main()
