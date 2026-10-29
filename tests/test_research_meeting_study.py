from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from ima import research_meeting_study as study
import numpy as np
import pandas as pd

from ima.research_confirmation import ExposureLedger
from ima.research_model_package import FeatureReplayContext, ResearchModelPackage
from ima.research_specs import PipelineRecipe


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def write_csv(path: Path, headers, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader(); writer.writerows(rows)
    return path


def timestamp(day: str, hour=11, minute=0) -> str:
    return f"{day}T{hour:02d}:{minute:02d}:00+00:00"


class PowerPilotTests(unittest.TestCase):
    def test_bootstrap_upper_sign_matches_direct_ratio_percentile(self):
        rng = np.random.default_rng(20261029)
        numerator_sums = rng.normal(size=(256, 40))
        # Include exact zeros and boundary cases around the linear 97.5th rank.
        numerator_sums[0] = -np.arange(1, 41, dtype=float)
        numerator_sums[1] = np.r_[np.full(39, -1.0), 0.0]
        numerator_sums[2] = np.r_[np.full(39, -1.0), 1.0]
        denominator_sums = rng.integers(1, 100, size=numerator_sums.shape)
        denominator_sums[0] = 1
        denominator_sums[1] = 1
        denominator_sums[2] = 1
        numerator_sums[3] = np.r_[np.full(39, -1.0), 1.0]
        denominator_sums[3] = np.r_[np.full(38, 1.0), 10_000.0, 100.0]
        direct = study._rowwise_linear_quantile(numerator_sums / denominator_sums, .975) < 0
        optimized, boundary = study._bootstrap_upper_sign_rank(numerator_sums, .975)
        optimized[boundary] = study._ratio_percentile_upper_below_zero(
            numerator_sums[boundary], denominator_sums[boundary], .975)
        np.testing.assert_array_equal(optimized, direct)

    def test_power_pilot_does_not_force_a_thirty_meeting_horizon(self):
        blocks = [{"meeting_id": f"d{i}", "races": 10, "difference_sum": -0.02}
                  for i in range(study.MIN_MEETINGS)]
        result = study._power_pilot(blocks, max_meetings=study.MIN_MEETINGS - 1)
        self.assertEqual(result["status"], "power_insufficient")
        self.assertIsNone(result["selected_meetings"])
        self.assertEqual(result["maximum_meetings_tested"], study.MIN_MEETINGS - 1)
        self.assertEqual(result["simulations"], [])


class ColumnProbabilityModel:
    def __init__(self, column):
        self.column = column

    def predict_proba(self, frame):
        return pd.to_numeric(frame[self.column]).to_numpy(float)


class MeetingStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_root = self.root / "trusted-packages"
        self.packages = []
        for package_id, column in (("challenger", "p_candidate"), ("reference", "p_reference")):
            package = ResearchModelPackage(ColumnProbabilityModel(column),
                                           PipelineRecipe(schema_version=3),
                                           protocol_id="protocol-shared", code_revision="fixture-r1",
                                           feature_context=FeatureReplayContext(training_cutoff="2026-10-01"))
            self.packages.append(package.save(self.source_root / package_id))
        self.meetings = []
        base = datetime(2027, 1, 1, tzinfo=timezone.utc)
        for i in range(30):
            day = (base.date()).fromordinal(base.date().toordinal() + i * 2).isoformat()
            mid, rid = f"m{i:02d}", f"r{i:02d}"
            self.meetings.append({"meeting_id": mid, "date": day,
                                  "races": [{"race_id": rid, "scheduled_post": f"{day}T20:00:00+08:00"}]})
        page_by_url = {}
        calendar_rows = []
        for meeting in self.meetings:
            year, month = meeting["date"].split("-")[:2]
            url = f"https://racing.hkjc.com/en-us/local/information/fixture?calmonth={month}&calyear={year}"
            if url not in page_by_url:
                raw_path = f"/owned-capture/raw/{hashlib.sha256(url.encode()).hexdigest()}.html.gz"
                body_hash = hashlib.sha256((url + " synthetic fixture body").encode()).hexdigest()
                fetched_at = "2026-10-28T02:00:00+00:00"
                page_by_url[url] = {
                    "url": url, "raw_path": raw_path, "body_sha256": body_hash,
                    "tables": [{"text": "synthetic official fixture month table"}],
                    "metadata": {"source_url": url, "requested_url": url,
                                 "official_source": "official:hkjc", "http_status": 200,
                                 "body_hash": body_hash, "raw_path": raw_path,
                                 "fetched_at": fetched_at},
                }
            page = page_by_url[url]
            calendar_rows.append({
                "meeting_date": meeting["date"], "venue": "HV", "session_icons": ["HV", "D", "turf"],
                "source_url": url, "source_body_sha256": page["body_sha256"],
                "source_raw_path": page["raw_path"], "observed_at": page["metadata"]["fetched_at"],
                "race_ids": None, "post_times": None,
            })
        self.calendar_source_readback = write_json(
            self.root / "calendar-source-readback.json",
            {"read_only": True, "driver_sha256": "a" * 64, "pages": list(page_by_url.values())})
        self.calendar = write_json(self.root / "calendar.json", {
            "schema": study.CALENDAR_SCHEMA,
            "source_readback_sha256": hashlib.sha256(self.calendar_source_readback.read_bytes()).hexdigest(),
            "timezone": "Asia/Hong_Kong",
            "meetings": calendar_rows,
        })
        self.package_identities = [study._verify_package(path, self.source_root)
                                   for path in self.packages]
        self.package_ids = [identity["manifest"]["package_id"]
                            for identity in self.package_identities]
        self.package_hashes = [study._package_identity_sha256(identity)
                               for identity in self.package_identities]
        self.dev = write_csv(self.root / "development.csv",
                             ["meeting_id", "date", "races", "difference_sum",
                              "candidate_package_sha256", "reference_package_sha256"],
                             [{"meeting_id": f"d{i}", "date": f"2025-01-{(i % 28)+1:02d}",
                               "races": 10, "difference_sum": -0.02,
                               "candidate_package_sha256": self.package_hashes[0],
                               "reference_package_sha256": self.package_hashes[1]} for i in range(30)])
        self.ledger = self.root / "exposures.sqlite"
        audit_source = write_json(self.root / "fixture-exposure-audit.json",
                                  {"scope": "complete synthetic fixture history; no live research assertion"})
        ExposureLedger(self.ledger).record(first_date="2025-01-01", last_date="2026-10-28",
                                          campaign="synthetic-fixture", purpose="development",
                                          evidence=audit_source)
        self.inventory = write_json(self.root / "exposure-inventory.json", {
            "schema": "ima.research_exposure_inventory.v1", "status": "complete",
            "selection_cutoff": "2026-10-28", "unresolved_sources": [],
            "ledger_path": str(self.ledger.resolve()),
            "ledger_sha256": hashlib.sha256(self.ledger.read_bytes()).hexdigest(),
            "sources": [{"scope": "complete synthetic fixture history",
                         "evidence": str(audit_source),
                         "evidence_sha256": hashlib.sha256(audit_source.read_bytes()).hexdigest()}]})
        self.manifest = {"study_id": "study-1", "selection_cutoff": "2026-10-28",
                         "calendar": str(self.calendar),
                         "calendar_source_readback": str(self.calendar_source_readback),
                         "trusted_package_root": str(self.source_root),
                         "candidate_package": str(self.packages[0]),
                         "reference_package": str(self.packages[1]),
                         "development_blocks": str(self.dev), "exposure_ledger": str(self.ledger),
                         "exposure_inventory": str(self.inventory),
                         "pilot_max_meetings": 30}
        # The test data are synthetic protocol fixtures; they are never asserted to be live racecards.
        self.lock = study.create_meeting_study(self.manifest, self.root / "lock.json")
        for i, meeting in enumerate(self.meetings):
            self._seal(i, 50)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_or_incomplete_exposure_inventory_holds_before_power(self):
        for mode in ("missing", "unresolved", "incomplete", "wrong-ledger", "empty-sources"):
            manifest = dict(self.manifest)
            inventory = json.loads(self.inventory.read_text())
            if mode == "missing":
                del manifest["exposure_inventory"]
            else:
                if mode == "unresolved": inventory["unresolved_sources"] = ["manual research not inventoried"]
                if mode == "incomplete": inventory["status"] = "incomplete"
                if mode == "wrong-ledger": inventory["ledger_sha256"] = "0" * 64
                if mode == "empty-sources": inventory["sources"] = []
                manifest["exposure_inventory"] = str(write_json(self.root / (mode + ".json"), inventory))
            with self.subTest(mode=mode), patch.object(study, "_power_pilot") as pilot:
                with self.assertRaises(ValueError):
                    study.create_meeting_study(manifest, self.root / (mode + "-lock.json"))
                pilot.assert_not_called()

    def test_changed_inventory_blocks_later_forecast_collection(self):
        self.inventory.write_text(self.inventory.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "exposure inventory"):
            study.collection_status(self.lock, self.root / "versions")

    def _inputs(self, index, *, leak=False, mismatch=False, two_races=False):
        meeting = self.meetings[index]; rid = meeting["races"][0]["race_id"]
        races = list(meeting["races"])
        if two_races:
            races.append({"race_id": rid + "-late", "scheduled_post": f"{meeting['date']}T21:00:00+08:00"})
        card = write_json(self.root / "official-cards" / f"{meeting['meeting_id']}.json", {
            "schema": study.SCHEDULE_SCHEMA, "source_url": "https://racing.example/official/card",
            "published_at": timestamp(meeting["date"], 9), "meeting_id": meeting["meeting_id"],
            "date": meeting["date"], "races": races,
        })
        runners = write_json(self.root / "cards" / f"{meeting['meeting_id']}.json", {
            "schema": "ima.actual_meeting_runners.v1", "meeting_id": meeting["meeting_id"],
            "date": meeting["date"],
            "captured_at": timestamp(meeting["date"], 10),
            "source_url": "https://racing.example/official/card",
            "official_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
            "runners": [{"race_id": race["race_id"], "horse_no": horse}
                        for race in races for horse in ("1", "2")],
        })
        feature_path = self.root / "features" / f"{meeting['meeting_id']}.parquet"
        rows = [(race["race_id"], horse, candidate, reference)
                for race in races for horse, candidate, reference in ((1, 0.6, 0.5), (2, 0.4, 0.5))]
        frame = pd.DataFrame({"date": [meeting["date"]] * len(rows),
                              "race_id": [row[0] for row in rows],
                              "horse_no": [row[1] for row in rows], "field_size": [2] * len(rows),
                              "p_candidate": [row[2] for row in rows],
                              "p_reference": [row[3] for row in rows]})
        if mismatch:
            frame.loc[0, "horse_no"] = 99
        if leak:
            frame["target_win"] = [1, 0]
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(feature_path, index=False)
        inputs = write_json(self.root / "inputs" / f"{meeting['meeting_id']}.json", {
            "schema": "ima.meeting_forecast_inputs.v1", "meeting_id": meeting["meeting_id"],
            "date": meeting["date"], "captured_at": timestamp(meeting["date"], 10),
            "feature_input_path": str(feature_path),
            "feature_input_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        })
        return card, runners, inputs

    def _seal(self, index, minute, *, leak=False, mismatch=False, two_races=False, root=None):
        card, runners, inputs = self._inputs(index, leak=leak, mismatch=mismatch, two_races=two_races)
        day = self.meetings[index]["date"]
        with patch.object(study, "_now", return_value=datetime.fromisoformat(timestamp(day, 11, minute))):
            return study.seal_meeting_version(self.lock, day, card, runners, inputs,
                                               root or self.root / "versions")

    def test_lock_binds_native_package_loader_receipts_calendar_and_power_contract(self):
        payload = json.loads(self.lock.read_text())
        self.assertEqual(payload["schema"], study.LOCK_SCHEMA)
        self.assertEqual(payload["candidate"]["loader_receipt"]["loader"],
                         "ima.research_model_package.load_research_package")
        self.assertEqual(payload["candidate"]["manifest"]["package_id"],
                         ResearchModelPackage(ColumnProbabilityModel("p_candidate"),
                                              PipelineRecipe(schema_version=3), "protocol-shared",
                                              "fixture-r1", feature_context=FeatureReplayContext(
                                                  training_cutoff="2026-10-01")).manifest().package_id)
        self.assertIn("model.joblib", payload["candidate"]["artifact_sha256"])
        self.assertEqual(payload["candidate"]["manifest_sha256"], payload["reference"]["manifest_sha256"])
        self.assertNotEqual(payload["candidate"]["artifact_sha256"], payload["reference"]["artifact_sha256"])
        self.assertEqual(payload["development"]["paired_package_ids"]["candidate"],
                         payload["development"]["paired_package_ids"]["reference"])
        self.assertNotEqual(payload["development"]["paired_package_sha256"]["candidate"],
                            payload["development"]["paired_package_sha256"]["reference"])
        self.assertIsNotNone(payload["candidate"]["artifact_sha256"]["feature-replay.json"])
        self.assertEqual(payload["candidate"]["feature_replay"]["training_cutoff"], "2026-10-01")
        self.assertEqual(len(payload["calendar"]["meeting_dates"]), 30)
        self.assertEqual(payload["calendar"]["source_readback_sha256"],
                         hashlib.sha256(self.calendar_source_readback.read_bytes()).hexdigest())
        self.assertEqual(payload["calendar"]["schema"], "ima.official_fixture_calendar_capture.v1")
        self.assertTrue(all(meeting["venue"] == "HV" and meeting["source_body_sha256"]
                            for meeting in payload["horizon"]["meetings"]))
        self.assertTrue(all("race_ids" not in meeting and "post_times" not in meeting
                            for meeting in payload["horizon"]["meetings"]))
        self.assertEqual(payload["power_pilot"]["selected_meetings"], 30)
        self.assertTrue(payload["power_pilot"]["assumptions"])
        self.assertEqual(payload["contract"]["resamples"], 20_000)
        self.assertEqual(payload["contract"]["csv_forecast_import"],
                         "held; no operator-supplied probability rows are accepted")
        self.assertFalse(payload["contract"]["optional_stopping"])
        self.assertEqual(payload["development"]["known_exposure_audit"]["overlapping_records"], [])

    def test_exposure_overlap_holds_lock(self):
        evidence = write_json(self.root / "exposure.json", {"evidence": "known selection"})
        ExposureLedger(self.ledger).record(first_date=self.meetings[0]["date"],
                                           last_date=self.meetings[0]["date"], campaign="prior",
                                           purpose="selection", evidence=evidence)
        with self.assertRaisesRegex(ValueError, "Known exposure"):
            study.create_meeting_study(self.manifest, self.root / "exposed-lock.json")

    def test_calendar_must_match_pinned_official_source_readback(self):
        readback = json.loads(self.calendar_source_readback.read_text())
        readback["pages"][0]["body_sha256"] = "0" * 64
        write_json(self.calendar_source_readback, readback)
        with self.assertRaisesRegex(ValueError, "source readback"):
            study.create_meeting_study(self.manifest, self.root / "calendar-source-mismatch-lock.json")

    def test_calendar_dates_must_match_zero_padded_official_source_month(self):
        calendar = json.loads(self.calendar.read_text())
        calendar["meetings"][0]["source_url"] = calendar["meetings"][0]["source_url"].replace(
            "calmonth=01", "calmonth=1")
        write_json(self.calendar, calendar)
        with self.assertRaisesRegex(ValueError, "zero-padded official month page"):
            study.create_meeting_study(self.manifest, self.root / "calendar-route-mismatch-lock.json")

    def test_development_blocks_must_match_locked_fitted_package_hashes(self):
        with self.dev.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["candidate_package_sha256"] = "0" * 64
        write_csv(self.dev, rows[0].keys(), rows)
        with self.assertRaisesRegex(ValueError, "locked candidate/reference package pair"):
            study.create_meeting_study(self.manifest, self.root / "development-pair-mismatch-lock.json")

    def test_candidate_and_reference_must_share_protocol_id(self):
        replacement = ResearchModelPackage(
            ColumnProbabilityModel("p_reference"), PipelineRecipe(schema_version=3),
            protocol_id="different-protocol", code_revision="fixture-r1",
            feature_context=FeatureReplayContext(training_cutoff="2026-10-01"),
        ).save(self.source_root / "other-protocol-reference")
        manifest = dict(self.manifest, reference_package=str(replacement))
        with self.assertRaisesRegex(ValueError, "share the frozen protocol_id"):
            study.create_meeting_study(manifest, self.root / "protocol-mismatch-lock.json")

    def test_insufficient_power_is_reported_without_creating_a_lock(self):
        with patch.object(study, "_power_pilot", return_value={"status": "power_insufficient",
                                                                  "selected_meetings": None,
                                                                  "simulations": []}):
            report_path = study.create_meeting_study(self.manifest, self.root / "underpowered-lock.json")
        report = json.loads(report_path.read_text())
        self.assertEqual(report["schema"], "ima.research_meeting_power_report.v1")
        self.assertFalse(report["lock_created"])
        self.assertFalse((self.root / "underpowered-lock.json").exists())

    def test_package_artifact_mutation_and_manifest_assertion_are_rejected(self):
        artifact = self.source_root / "challenger" / "model.joblib"
        artifact.write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "artifact hashes changed"):
            self._seal(0, 51)

    def test_untrusted_package_outside_declared_owned_boundary_is_rejected(self):
        self.manifest["trusted_package_root"] = str(self.root / "other-owned-directory")
        Path(self.manifest["trusted_package_root"]).mkdir()
        with self.assertRaisesRegex(ValueError, "inside trusted_package_root"):
            study.create_meeting_study(self.manifest, self.root / "outside-trust-lock.json")

    def test_group_or_other_writable_package_boundary_is_rejected(self):
        os.chmod(self.source_root, 0o777)
        with self.assertRaisesRegex(ValueError, "inside trusted_package_root"):
            study.create_meeting_study(self.manifest, self.root / "writable-trust-lock.json")

    def test_package_trained_after_selection_cutoff_is_rejected(self):
        future = ResearchModelPackage(
            ColumnProbabilityModel("p_candidate"), PipelineRecipe(schema_version=3),
            protocol_id="protocol-shared", code_revision="fixture-r1",
            feature_context=FeatureReplayContext(training_cutoff="2026-11-01"),
        ).save(self.source_root / "future-trained")
        manifest = dict(self.manifest, candidate_package=str(future))
        with self.assertRaisesRegex(ValueError, "training cutoff must not exceed"):
            study.create_meeting_study(manifest, self.root / "future-trained-lock.json")

    def test_sealed_input_and_package_bytes_are_rechecked(self):
        version_path = self.root / "versions" / self.meetings[0]["date"] / "versions" / "v0001.json"
        version = json.loads(version_path.read_text())
        Path(version["inputs"]["path"]).write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "input artifact changed"):
            study.collection_status(self.lock, self.root / "versions")

    def test_group_or_other_writable_feature_input_is_rejected(self):
        card, runners, inputs = self._inputs(0)
        receipt = json.loads(inputs.read_text())
        os.chmod(receipt["feature_input_path"], 0o666)
        day = self.meetings[0]["date"]
        with patch.object(study, "_now", return_value=datetime.fromisoformat(timestamp(day, 11, 51))):
            with self.assertRaisesRegex(ValueError, "non-group/other-writable"):
                study.seal_meeting_version(self.lock, day, card, runners, inputs,
                                           self.root / "unsafe-feature-version")

    def test_rehashed_forecast_version_cannot_be_reassigned_to_another_meeting(self):
        version_path = self.root / "versions" / self.meetings[0]["date"] / "versions" / "v0001.json"
        version = json.loads(version_path.read_text())
        version["meeting_date"] = self.meetings[1]["date"]
        # Rebuild the checksum to exercise the semantic identity check rather than hash validation.
        version.pop("content_sha256", None)
        study._add_content_hash(version)
        version_path.write_text(json.dumps(version, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaisesRegex(ValueError, "different meeting"):
            study.collection_status(self.lock, self.root / "versions")

    def test_runner_mismatch_and_label_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "population.*does not match"):
            self._seal(0, 51, mismatch=True)
        with self.assertRaisesRegex(ValueError, "label-free"):
            self._seal(0, 51, leak=True)

    def test_seal_replays_locked_packages_on_hashed_feature_frame(self):
        version_path = self._seal(0, 51)
        version = json.loads(version_path.read_text())
        feature_hash = version["inputs"]["feature_frame_sha256"]
        for side, expected in (("candidate", "challenger"), ("reference", "reference")):
            receipt = version["execution_receipts"][side]
            self.assertEqual(receipt["executor"],
                             "ima.research_model_package.load_research_package.predict_proba")
            self.assertEqual(receipt["feature_input_sha256"], feature_hash)
            self.assertEqual(receipt["package_id"], version[f"{side}_package"]["manifest"]["package_id"])
            self.assertEqual(len(version[f"{side}_forecast"]["rows"]), 2)
        self.assertAlmostEqual(float(version["candidate_forecast"]["rows"][0]["model_probability"]), 0.6)

    def test_cli_lock_seal_and_status_use_native_replay_flow(self):
        from scripts.research_confirmation import main
        manifest_path = write_json(self.root / "cli-manifest.json", self.manifest)
        lock_path = self.root / "cli-lock.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["meeting-lock", "--manifest", str(manifest_path),
                                   "--output", str(lock_path)]), 0)
        card, runners, inputs = self._inputs(0)
        day = self.meetings[0]["date"]
        output_root = self.root / "cli-versions"
        with patch.object(study, "_now", return_value=datetime.fromisoformat(timestamp(day, 11, 51))):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["meeting-seal", "--lock", str(lock_path),
                                       "--meeting-date", day, "--official-card", str(card),
                                       "--runners", str(runners), "--inputs", str(inputs),
                                       "--output-root", str(output_root)]), 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["meeting-status", "--lock", str(lock_path),
                                   "--versions-root", str(output_root)]), 0)
        self.assertTrue((output_root / day / "versions" / "v0001.json").is_file())

    def test_deadline_is_timezone_aware_and_last_valid_version_wins(self):
        first = self._seal(0, 51)
        second = self._seal(0, 54)
        self.assertNotEqual(first, second)
        exact_deadline = self._seal(2, 55)
        self.assertTrue(json.loads(exact_deadline.read_text())["valid_predeadline"])
        # Scheduled post 12:00 UTC => 11:55 UTC deadline. Equality is valid.
        late = self._seal(0, 56)
        self.assertEqual(json.loads(late.read_text())["valid_predeadline"], False)
        # setUp sealed v1; these add valid v2/v3, followed by invalid-late v4.
        versions = sorted((self.root / "versions" / self.meetings[0]["date"] / "versions").glob("v*.json"))
        self.assertEqual(len(versions), 4)
        self.assertFalse(json.loads(versions[-1].read_text())["valid_predeadline"])
        selected, _, _ = study._versions(json.loads(self.lock.read_text()), self.root / "versions")
        self.assertEqual(selected["r00"]["version"], 3)
        status = study.collection_status(self.lock, self.root / "versions")
        self.assertEqual(status["coverage"], 1.0)
        self.assertNotIn("delta_nats_per_race", status)
        self.assertNotIn("model_probability", json.dumps(status))
        self.assertEqual(study._instant("2027-01-01T11:55:00-05:00", "x").utcoffset(), timezone.utc.utcoffset(None))
        with self.assertRaisesRegex(ValueError, "timezone"):
            study._instant("2027-01-01T11:55:00", "x")

    def test_scheduled_post_requires_hong_kong_timezone(self):
        self.assertEqual(study._scheduled_post("2027-01-01T20:00:00+08:00"),
                         study._instant("2027-01-01T12:00:00+00:00", "x"))
        with self.assertRaisesRegex(ValueError, "Hong Kong timezone"):
            study._scheduled_post("2027-01-01T12:00:00+00:00")

    def test_late_only_forecast_is_an_operational_miss(self):
        card, runners, inputs = self._inputs(0)
        day = self.meetings[0]["date"]
        with patch.object(study, "_now", return_value=datetime.fromisoformat(timestamp(day, 11, 56))):
            study.seal_meeting_version(self.lock, day, card, runners, inputs, self.root / "late-versions")
        status = study.collection_status(self.lock, self.root / "late-versions")
        self.assertEqual(status["covered_races"], 0)
        self.assertFalse(status["operationally_qualified"])

    def test_seal_uses_each_official_race_post_deadline_independently(self):
        path = self._seal(0, 56, two_races=True, root=self.root / "per-race-versions")
        version = json.loads(path.read_text())
        self.assertEqual(version["eligible_race_ids"], ["r00-late"])
        status = study.collection_status(self.lock, self.root / "per-race-versions")
        self.assertEqual(status["scheduled_races"], 2)
        self.assertEqual(status["covered_races"], 1)
        self.assertEqual(status["coverage"], 0.5)

    def test_coverage_below_95_percent_is_operationally_unqualified(self):
        root = self.root / "low-coverage"
        self._seal(0, 56, root=root)
        self._seal(1, 56, root=root)
        for index in range(2, 30):
            self._seal(index, 50, root=root)
        status = study.collection_status(self.lock, root)
        self.assertEqual(status["coverage"], 28 / 30)
        self.assertFalse(status["operationally_qualified"])

    def test_early_evaluation_does_not_open_outcome_file(self):
        with patch.object(study, "_now", return_value=datetime(2026, 11, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "not complete"):
                study.evaluate_fixed_horizon(self.lock, self.root / "versions",
                                              self.root / "does-not-exist.json", self.root / "result.json")

    def test_fixed_horizon_block_weighted_evaluation_and_reason_ledger(self):
        races = []
        for i, meeting in enumerate(self.meetings):
            races.append({"race_id": meeting["races"][0]["race_id"], "status": "completed",
                          "winners": ["1"], "postdeadline_scratches": []})
        last_day = self.meetings[-1]["date"]
        outcomes = write_json(self.root / "outcomes.json", {"schema": "ima.meeting_outcomes.v1",
                                                               "published_at": timestamp(last_day, 16),
                                                               "races": races})
        output = self.root / "final.json"
        with patch.object(study, "_now", return_value=datetime.fromisoformat(timestamp(last_day, 16))):
            path = study.evaluate_fixed_horizon(self.lock, self.root / "versions", outcomes, output)
        result = json.loads(path.read_text())
        self.assertEqual(result["meeting_blocks"], 30)
        self.assertEqual(result["races"], 30)
        self.assertAlmostEqual(result["delta_nats_per_race"], -__import__("math").log(0.6) + __import__("math").log(0.5))
        self.assertEqual(result["exclusions"], [])
        self.assertEqual(result["bootstrap_resamples"], 20_000)

    def test_cancel_and_scratch_exclusions_require_reasons(self):
        outcomes = write_json(self.root / "bad-outcomes.json", {"schema": "ima.meeting_outcomes.v1",
            "published_at": "2027-04-01T00:00:00+00:00", "races": [
            {"race_id": m["races"][0]["race_id"], "status": "cancelled", "reason": ""}
            for m in self.meetings]})
        with patch.object(study, "_now", return_value=datetime(2027, 4, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "reason"):
                study.evaluate_fixed_horizon(self.lock, self.root / "versions", outcomes, self.root / "bad-result.json")
        bad_scratch = [{"race_id": m["races"][0]["race_id"], "status": "completed",
                        "winners": ["1"], "postdeadline_scratches": []} for m in self.meetings]
        bad_scratch[0]["postdeadline_scratches"] = [{"horse_no": "2", "reason": ""}]
        scratch_path = write_json(self.root / "bad-scratch.json", {"schema": "ima.meeting_outcomes.v1",
            "published_at": "2027-04-01T00:00:00+00:00", "races": bad_scratch})
        with patch.object(study, "_now", return_value=datetime(2027, 4, 1, tzinfo=timezone.utc)):
            with self.assertRaisesRegex(ValueError, "scratch exclusions require"):
                study.evaluate_fixed_horizon(self.lock, self.root / "versions", scratch_path, self.root / "bad-scratch-result.json")

    def test_total_difference_over_total_races_not_equal_meeting_average(self):
        import numpy as np
        value = study._weighted_ratio(np.asarray([0., -10.]), np.asarray([100., 1.]))
        self.assertAlmostEqual(float(value), -10 / 101)
        self.assertNotAlmostEqual(float(value), -5.)
        samples = np.random.default_rng(19).normal(size=(7, 20_000))
        np.testing.assert_allclose(study._rowwise_linear_quantile(samples, .975),
                                   np.quantile(samples, .975, axis=1), rtol=0, atol=1e-14)

    def test_immutable_version_conflict_and_cli_command_registration(self):
        path = self._seal(1, 52)
        payload = json.loads(path.read_text()); payload["version"] = 999
        with self.assertRaisesRegex(ValueError, "Immutable artifact conflict"):
            study._write_immutable(path, payload)
        from scripts.research_confirmation import build_parser
        parser = build_parser()
        for command in ("lock", "evaluate", "exposure", "seal", "evaluate-study",
                        "meeting-lock", "meeting-seal", "meeting-status", "meeting-evaluate"):
            self.assertIn(command, parser._subparsers._group_actions[0].choices)
        meeting_seal = parser._subparsers._group_actions[0].choices["meeting-seal"]
        option_strings = {option for action in meeting_seal._actions for option in action.option_strings}
        self.assertIn("--official-card", option_strings)
        self.assertIn("--meeting-date", option_strings)
        self.assertNotIn("--candidate-forecast", option_strings)
        self.assertNotIn("--reference-forecast", option_strings)


if __name__ == "__main__":
    unittest.main()
