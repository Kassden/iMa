from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap

from ima import research_storage
from ima.research_confirmation import create_lock, evaluate_lock, _write_immutable
from ima.research_storage import StorageBlocked


ROOT = Path(__file__).resolve().parents[1]


def prediction_rows(count: int, candidate: bool, *, include_field_size: bool = False) -> pd.DataFrame:
    rows = []
    for day in range(count):
        for horse in range(3):
            winner = int(horse == day % 3)
            probability = [0.70, 0.20, 0.10][(horse - day % 3) % 3]
            if not candidate:
                probability = 0.40 if winner else 0.30
            rows.append({
                "date": f"2026-10-{day + 2:02d}", "race_id": f"race-{day}",
                "horse_no": str(horse + 1), "target_win": winner,
                "model_probability": probability,
            })
            if include_field_size:
                rows[-1]["field_size"] = 3
    return pd.DataFrame(rows)


class ResearchConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / "candidate.csv"
        self.baseline = self.root / "baseline.csv"
        prediction_rows(12, True).to_csv(self.candidate, index=False)
        prediction_rows(12, False).to_csv(self.baseline, index=False)
        self.lock = self.root / "comparison.lock.json"
        self.result = self.root / "result.json"

    def tearDown(self):
        self.temporary.cleanup()

    def create_lock(self, **overrides):
        arguments = {
            "candidate_path": self.candidate,
            "baseline_path": self.baseline,
            "lock_path": self.lock,
            "candidate_identity": {"candidate_id": "cand-a", "revision": "abc123"},
            "baseline_identity": {"candidate_id": "market", "revision": "market-v1"},
            "selection_cutoff": "2026-10-01",
            "bootstrap_resamples": 500,
            "random_seed": 77,
        }
        arguments.update(overrides)
        return create_lock(**arguments)

    def test_help_and_invalid_cli_inputs(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "scripts.research_confirmation", "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(0, help_result.returncode)
        self.assertIn("lock", help_result.stdout)
        self.assertIn("evaluate", help_result.stdout)
        lock_help = subprocess.run(
            [sys.executable, "-m", "scripts.research_confirmation", "lock", "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        evaluate_help = subprocess.run(
            [sys.executable, "-m", "scripts.research_confirmation", "evaluate", "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertIn("--prospective", lock_help.stdout)
        self.assertIn("--labels", evaluate_help.stdout)
        invalid = subprocess.run(
            [sys.executable, "-m", "scripts.research_confirmation", "lock",
             "--candidate", str(self.candidate), "--baseline", str(self.baseline),
             "--candidate-identity", "[]"], cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(2, invalid.returncode)

    def test_temporal_comparison_improvement_and_idempotent_replay(self):
        self.create_lock()
        evaluate_lock(self.lock, self.result)
        first_bytes = self.result.read_bytes()
        evaluate_lock(self.lock, self.result)
        self.assertEqual(first_bytes, self.result.read_bytes())
        result = json.loads(first_bytes)
        self.assertEqual("temporal_comparison_improvement", result["status"])
        self.assertLess(result["comparison"]["paired_race_bootstrap_interval"][1], 0)
        self.assertEqual("unsupported", result["place_metrics"]["status"])
        self.assertEqual("operator_declared", result["provenance"]["selection_cutoff_source"])
        self.assertEqual("retrospective_comparison_only", result["provenance"]["status"])
        self.assertFalse(result["provenance"]["feature_training_leakage_verified"])
        self.assertFalse(result["provenance"]["automatic_promotion_evidence"])
        self.assertIn("automatic promotion evidence", result["scope"])
        self.result.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Immutable artifact conflict"):
            evaluate_lock(self.lock, self.result)

    def test_storage_denial_precedes_lock_creation_and_preserves_sources(self):
        import os
        candidate_before=self.candidate.read_bytes()
        baseline_before=self.baseline.read_bytes()
        output=self.root/'missing'/'nested'/'lock.json'
        control=self.root.resolve()/'control'
        real_authority=research_storage.StorageAuthority
        real_authority(control,headroom_bytes=0,headroom_inodes=0)
        with patch.dict(os.environ,{'IMA_STORAGE_ROOT':str(control)}):
            with patch.object(research_storage,'StorageAuthority',
                    side_effect=lambda root:real_authority(root,headroom_bytes=10**15,headroom_inodes=0)):
                with self.assertRaises(StorageBlocked):
                    self.create_lock(lock_path=output)
        self.assertFalse(output.parent.exists())
        self.assertEqual(candidate_before,self.candidate.read_bytes())
        self.assertEqual(baseline_before,self.baseline.read_bytes())

        with patch.dict(os.environ,{'IMA_STORAGE_ROOT':str(control)}):
            with patch.object(research_storage,'StorageAuthority',
                    side_effect=lambda root:real_authority(root,headroom_bytes=0,headroom_inodes=0)):
                self.create_lock(lock_path=output)
                locked=output.read_bytes()
                self.create_lock(lock_path=output)
        self.assertEqual(locked,output.read_bytes())
        self.assertEqual(candidate_before,self.candidate.read_bytes())
        self.assertEqual(baseline_before,self.baseline.read_bytes())

    def test_late_directory_sync_failure_keeps_accepted_artifact_and_releases_lease(self):
        import os
        import stat

        output=self.root.resolve()/'late'/'artifact.json'
        control=self.root.resolve()/'control'
        real_authority=research_storage.StorageAuthority
        real_authority(control,headroom_bytes=0,headroom_inodes=0)
        payload={'artifact':'payload'}
        expected=b'{"artifact":"payload"}\n'
        real_fsync=os.fsync

        def fail_directory_sync(descriptor):
            info=os.fstat(descriptor)
            parent=output.parent.stat() if output.parent.exists() else None
            if (stat.S_ISDIR(info.st_mode) and parent is not None
                    and (info.st_dev,info.st_ino)==(parent.st_dev,parent.st_ino)):
                raise OSError('injected directory sync failure')
            return real_fsync(descriptor)

        with patch.dict(os.environ,{'IMA_STORAGE_ROOT':str(control)}):
            with patch.object(research_storage,'StorageAuthority',
                    side_effect=lambda root:real_authority(root,headroom_bytes=0,headroom_inodes=0)):
                with patch('ima.research_confirmation.os.fsync',side_effect=fail_directory_sync):
                    with self.assertRaisesRegex(OSError,'injected directory sync failure'):
                        _write_immutable(output,payload)
                self.assertTrue(output.is_file())
                self.assertEqual(expected,output.read_bytes())
                status=real_authority(control,headroom_bytes=0,headroom_inodes=0).status()
                self.assertEqual({},status['reservations'])
                retry_syncs=[]

                def record_retry_sync(descriptor):
                    self.assertTrue(stat.S_ISDIR(os.fstat(descriptor).st_mode))
                    retry_syncs.append(descriptor)
                    return real_fsync(descriptor)

                with patch('ima.research_confirmation.os.fsync',side_effect=record_retry_sync):
                    self.assertEqual(output,_write_immutable(output,payload))
                self.assertEqual(1,len(retry_syncs))
        self.assertEqual(expected,output.read_bytes())
        self.assertEqual([],list(output.parent.glob('.*.tmp')))

    def test_immutable_admission_estimate_covers_missing_directory_allocation(self):
        output=self.root.resolve()/'estimate'/'nested'/'artifact.json'
        with patch('ima.research_confirmation.admitted_output',
                   wraps=research_storage.admitted_output) as admitted:
            _write_immutable(output,{'artifact':'payload'})
        estimate=admitted.call_args.kwargs['estimated_bytes']
        # Two absent parent directories need conservative allocation on top
        # of the bytes actually staged and published.
        self.assertGreaterEqual(estimate,
            output.stat().st_size + 2 * 64 * 1024)

    def test_null_comparison_is_provisional(self):
        baseline = prediction_rows(12, True)
        baseline.to_csv(self.baseline, index=False)
        self.create_lock()
        evaluate_lock(self.lock, self.result)
        result = json.loads(self.result.read_text())
        self.assertEqual("provisional", result["status"])
        self.assertGreaterEqual(result["comparison"]["paired_race_bootstrap_interval"][1], 0)

    def test_changed_prediction_bytes_rejected(self):
        self.create_lock()
        changed = pd.read_csv(self.candidate)
        changed.loc[0, "model_probability"] = 0.69
        changed.to_csv(self.candidate, index=False)
        with self.assertRaisesRegex(ValueError, "hash differs from lock"):
            evaluate_lock(self.lock, self.result)

    def test_duplicate_and_partial_population_rejected_at_lock(self):
        candidate = prediction_rows(4, True)
        pd.concat([candidate, candidate.iloc[[0]]]).to_csv(self.candidate, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate runner keys"):
            self.create_lock()

        prediction_rows(4, True).to_csv(self.candidate, index=False)
        prediction_rows(3, False).to_csv(self.baseline, index=False)
        with self.assertRaisesRegex(ValueError, "populations differ"):
            self.create_lock()

        candidate = prediction_rows(4, True, include_field_size=True).iloc[1:].copy()
        baseline = prediction_rows(4, False, include_field_size=True).iloc[1:].copy()
        candidate.to_csv(self.candidate, index=False)
        baseline.to_csv(self.baseline, index=False)
        with self.assertRaisesRegex(ValueError, "Partial or inconsistent field_size"):
            self.create_lock()

    def test_minimum_two_races(self):
        prediction_rows(1, True).to_csv(self.candidate, index=False)
        prediction_rows(1, False).to_csv(self.baseline, index=False)
        with self.assertRaisesRegex(ValueError, "at least two races"):
            self.create_lock()

    def test_cutoff_overlap_rejected(self):
        candidate = prediction_rows(3, True)
        candidate.loc[0, "date"] = "2026-10-01"
        candidate.to_csv(self.candidate, index=False)
        with self.assertRaisesRegex(ValueError, "strictly after"):
            self.create_lock()

    def test_label_mismatch_and_partial_winner_race_rejected(self):
        self.create_lock()
        baseline = pd.read_csv(self.baseline)
        baseline.loc[0, "target_win"] = 0
        baseline.to_csv(self.baseline, index=False)
        with self.assertRaisesRegex(ValueError, "hash differs from lock"):
            evaluate_lock(self.lock, self.result)

        prediction_rows(12, True).to_csv(self.candidate, index=False)
        prediction_rows(12, False).to_csv(self.baseline, index=False)
        self.lock.unlink()
        baseline = pd.read_csv(self.baseline)
        baseline.loc[:2, "target_win"] = [0, 1, 0]
        baseline.to_csv(self.baseline, index=False)
        self.create_lock()
        with self.assertRaisesRegex(ValueError, "labels differ"):
            evaluate_lock(self.lock, self.result)

        prediction_rows(12, True).to_csv(self.candidate, index=False)
        prediction_rows(12, False).to_csv(self.baseline, index=False)
        self.lock.unlink()
        self.result.unlink(missing_ok=True)
        self.create_lock()
        candidate = pd.read_csv(self.candidate)
        candidate.loc[:2, "target_win"] = 0
        candidate.to_csv(self.candidate, index=False)
        self.lock.unlink()
        self.create_lock()
        with self.assertRaisesRegex(ValueError, "exactly one winner"):
            evaluate_lock(self.lock, self.result)

    def test_nonfinite_probability_and_immutable_conflict(self):
        candidate = prediction_rows(4, True)
        candidate.loc[0, "model_probability"] = float("inf")
        candidate.to_csv(self.candidate, index=False)
        prediction_rows(4, False).to_csv(self.baseline, index=False)
        self.create_lock()
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_lock(self.lock, self.result)

        prediction_rows(4, True).to_csv(self.candidate, index=False)
        prediction_rows(4, False).to_csv(self.baseline, index=False)
        self.lock.unlink()
        self.create_lock()
        original = self.lock.read_bytes()
        with self.assertRaisesRegex(ValueError, "Immutable artifact conflict"):
            self.create_lock(random_seed=78)
        self.assertEqual(original, self.lock.read_bytes())

    def test_rehashed_lock_with_invalid_inference_bounds_rejected(self):
        self.create_lock()
        payload = json.loads(self.lock.read_text())
        payload.pop("lock_id")
        payload["inference"]["confidence_level"] = 1.5
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False) + "\n"
        payload["lock_id"] = hashlib.sha256(canonical.encode()).hexdigest()
        self.lock.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaisesRegex(ValueError, "Invalid lock inference settings"):
            evaluate_lock(self.lock, self.result)

    def test_bootstrap_uses_fixed_batch_and_rejects_nonfinite_interval(self):
        self.create_lock()
        fake_interval = SimpleNamespace(confidence_interval=SimpleNamespace(low=float("nan"), high=0.0))
        with patch("ima.research_confirmation.bootstrap", return_value=fake_interval) as mocked:
            with self.assertRaisesRegex(ValueError, "endpoints must be finite"):
                evaluate_lock(self.lock, self.result)
        self.assertEqual(256, mocked.call_args.kwargs["batch"])

    def test_lock_change_during_evaluation_rejected(self):
        self.create_lock()

        def mutate_after_bootstrap(*args, **kwargs):
            result = scipy_bootstrap(*args, **kwargs)
            self.lock.write_bytes(self.lock.read_bytes() + b" ")
            return result

        with patch("ima.research_confirmation.bootstrap", side_effect=mutate_after_bootstrap):
            with self.assertRaisesRegex(ValueError, "lock changed during evaluation"):
                evaluate_lock(self.lock, self.result)
        self.assertFalse(self.result.exists())

    def _write_prospective_inputs(self):
        candidate = prediction_rows(12, True).drop(columns="target_win")
        baseline = prediction_rows(12, False).drop(columns="target_win")
        candidate.to_csv(self.candidate, index=False)
        baseline.to_csv(self.baseline, index=False)
        labels = prediction_rows(12, True)[["date", "race_id", "horse_no", "target_win"]]
        labels_path = self.root / "labels.csv"
        labels.to_csv(labels_path, index=False)
        return labels_path

    def _create_prospective_lock(self):
        return self.create_lock(
            selection_cutoff="2026-09-30", prospective=True,
        )

    def test_prospective_lock_waits_until_all_locked_race_dates_have_passed(self):
        labels_path = self._write_prospective_inputs()
        lock_time = datetime(2026, 10, 1, 23, 59, tzinfo=timezone.utc)
        with patch("ima.research_confirmation._utc_now", return_value=lock_time):
            self._create_prospective_lock()
        locked_candidate_bytes = self.candidate.read_bytes()
        locked_baseline_bytes = self.baseline.read_bytes()
        payload = json.loads(self.lock.read_text())
        self.assertEqual("2026-10-01T23:59:00.000000Z", payload["created_at_utc"])
        self.assertEqual("prospective", payload["mode"])
        original_lock_bytes = self.lock.read_bytes()
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 1, 23, 59, 30, tzinfo=timezone.utc)):
            self._create_prospective_lock()
        self.assertEqual(payload["created_at_utc"], json.loads(self.lock.read_text())["created_at_utc"])
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 5, 12, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "requires a separate"):
                evaluate_lock(self.lock, self.result)
            with self.assertRaisesRegex(ValueError, "has not passed in UTC"):
                evaluate_lock(self.lock, self.result, labels_path)
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 20, 12, tzinfo=timezone.utc)):
            evaluate_lock(self.lock, self.result, labels_path)
        result = json.loads(self.result.read_text())
        self.assertEqual("confirmed_temporal_improvement", result["status"])
        self.assertEqual("prospective_predictions_frozen_before_race_dates",
                         result["provenance"]["status"])
        self.assertEqual(payload["created_at_utc"], result["temporal_population"]["lock_created_at_utc"])
        self.assertFalse(result["provenance"]["feature_training_leakage_verified"])
        self.assertFalse(result["provenance"]["automatic_promotion_evidence"])
        self.assertEqual(hashlib.sha256(labels_path.read_bytes()).hexdigest(), result["labels_sha256"])
        self.assertEqual(locked_candidate_bytes, self.candidate.read_bytes())
        self.assertEqual(locked_baseline_bytes, self.baseline.read_bytes())
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 20, 12, tzinfo=timezone.utc)):
            self._create_prospective_lock()
        self.assertEqual(original_lock_bytes, self.lock.read_bytes())

    def test_prospective_lock_rejects_labels_and_nonfuture_dates(self):
        labels_path = self._write_prospective_inputs()
        candidate = pd.read_csv(self.candidate)
        candidate["target_win"] = 0
        candidate.to_csv(self.candidate, index=False)
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "must not contain target_win"):
                self._create_prospective_lock()

        candidate = prediction_rows(12, True).drop(columns="target_win")
        candidate.loc[candidate.index[0], "date"] = "2026-10-01"
        candidate.to_csv(self.candidate, index=False)
        prediction_rows(12, False).drop(columns="target_win").to_csv(self.baseline, index=False)
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "strictly after the current UTC date"):
                self._create_prospective_lock()

    def test_prospective_labels_must_match_locked_population(self):
        labels_path = self._write_prospective_inputs()
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 1, tzinfo=timezone.utc)):
            self._create_prospective_lock()
        labels = pd.read_csv(labels_path).iloc[1:]
        labels.to_csv(labels_path, index=False)
        with patch("ima.research_confirmation._utc_now", return_value=datetime(
                2026, 10, 20, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                evaluate_lock(self.lock, self.result, labels_path)


if __name__ == "__main__":
    unittest.main()
