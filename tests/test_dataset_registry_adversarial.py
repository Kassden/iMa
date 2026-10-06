"""Independent registry leakage and provenance tests; no source acquisition."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ima.dataset_registry import DatasetRegistry
from ima.dataset_specs import DatasetRequest


HORSES = ("HK_2020_A001", "HK_2020_A002", "HK_2020_A003")
OFFICIAL_URL = "https://racing.hkjc.com/en-us/local/information/localresults"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_rows():
    rows = []
    for number, day in enumerate(pd.date_range("2024-01-01", periods=8), start=1):
        for horse_no, horse in enumerate(HORSES, start=1):
            rows.append({
                "date": day.strftime("%Y-%m-%d"), "race_id": f"r{number}", "race_no": 1,
                "horse_id": horse, "horse_no": horse_no, "result": horse_no,
                "distance": 1200, "venue": "ST", "course": "TURF", "going": "GOOD",
                "race_class": "Class 4", "win_odds": 2.0 + horse_no,
                "finish_time": f"1:{horse_no - 1:02d}.00", "actual_weight": 120 + horse_no,
                "declared_weight": 1000 + horse_no, "draw": horse_no,
                "jockey_id": f"J{horse_no}", "trainer_id": f"T{horse_no}",
                "source": "official:hkjc-results", "source_url": OFFICIAL_URL,
                "source_body_hash": hashlib.sha256(f"synthetic-race-{number}".encode()).hexdigest(),
            })
    return pd.DataFrame(rows)


def event(identifier, family="trackwork", **changes):
    return {
        "event_id": identifier, "family": family, "horse_id": HORSES[0],
        "occurred_at": "2023-12-31T10:00:00+08:00",
        "published_at": "2023-12-31T12:00:00+08:00", "publication_verified": True,
        "source_url": "https://racing.hkjc.com/en-us/local/information/trackwork",
        "source_body_hash": hashlib.sha256(identifier.encode()).hexdigest(),
        "values": {"distance_metres": 1200, "workout_finish_seconds": 75},
    } | changes


class DatasetRegistryAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = DatasetRegistry(self.root / "registry")
        self.counter = 0

    def fixture(self, *, frame=None, events=None, target_only=None):
        self.counter += 1
        snapshot = self.root / f"snapshot-{self.counter}"
        snapshot.mkdir()
        frame = source_rows() if frame is None else frame.copy()
        events = [event("verified-workout")] if events is None else events
        frame.to_parquet(snapshot / "runners.parquet", index=False)
        (snapshot / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
        source_manifest = {
            "source_policy": "HKJC-only synthetic test evidence", "rows": len(frame),
            "races": int(frame.race_id.nunique()),
            "target_only_columns": target_only or ["result", "finish_time", "finish_seconds", "target_win", "target_probability"],
            "market_only_columns": ["win_odds", "market_raw", "market_probability"],
            "files": {name: digest(snapshot / name) for name in ("runners.parquet", "events.jsonl")},
        }
        (snapshot / "manifest.json").write_text(json.dumps(source_manifest))
        raw = snapshot / "raw-acquisition.json"
        raw.write_text(json.dumps({"manifest_id": f"raw-{self.counter}", "event_coverage": [],
                                   "fixture_only": True}))
        return snapshot, raw

    def request(self, identifier, raw, **changes):
        return DatasetRequest.model_validate({
            "request_id": identifier, "raw_corpus_manifest_id": digest(raw),
            "rationale": "Independent adversarial development fixture.", "evidence_watermark": "fixture-evidence-1",
            "protocol_id": "fixture-protocol-1", "evaluation_population_id": "fixture-development",
            "protocol": {"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                         "max_folds": 2, "whole_meeting_boundaries": True, "final_confirmation_races": 2},
        } | changes)

    def build(self, identifier, *, frame=None, events=None, **changes):
        snapshot, raw = self.fixture(frame=frame, events=events)
        request = self.request(identifier, raw, **changes)
        self.registry.submit(request)
        manifest = self.registry.build(identifier, source_snapshot=snapshot, raw_manifest=raw)
        return manifest, pd.read_parquet(self.registry.dataset_path(manifest["dataset_id"]) / "features.parquet")

    def test_real_build_is_hash_verified_idempotent_and_keeps_all_clean_development_keys(self):
        snapshot, raw = self.fixture()
        request = self.request("baseline", raw)
        self.registry.submit(request)
        first = self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        second = self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(first, second)
        self.assertEqual(self.registry.verify(first["dataset_id"]), first)
        self.assertEqual((first["rows"], first["races"]), (18, 6))
        frame = pd.read_parquet(first["features_path"])
        self.assertEqual(set(zip(frame.race_id, frame.horse_id)),
                         {(f"r{race}", horse) for race in range(1, 7) for horse in HORSES})
        self.assertEqual(first["validation"]["unaccounted_source_keys"], 0)
        self.assertFalse(first["validation"]["confirmation_labels_published"])

    def test_speed_predictors_have_source_specific_physical_units_not_batch_or_ratio_values(self):
        observations = [
            event("workout-ms", values={"distance": 1.2, "distance_unit": "km", "time_seconds": 75000,
                                        "time_unit": "ms", "time_basis": "individual"}),
            event("individual-trial", "barrier_trials", values={"distance_metres": 800, "trial_finish_seconds": 40}),
            event("batch-only", horse_id=HORSES[1], values={"distance_metres": 1200, "batch_winner_time_seconds": 60}),
            event("unknown-unit", horse_id=HORSES[2], values={"distance": 6, "distance_unit": "furlongs",
                                                            "workout_finish_seconds": 75}),
        ]
        manifest, features = self.build("physical-speeds", events=observations)
        # r1 outcomes become available at r2's midnight cutoff, so only r3 can use them.
        row = features.loc[(features.race_id == "r3") & (features.horse_id == HORSES[0])].iloc[0]
        self.assertAlmostEqual(row.past_race_speed_last_mps, 20.0)
        self.assertAlmostEqual(row.workout_speed_last_mps, 16.0)
        self.assertAlmostEqual(row.trial_speed_last_mps, 20.0)
        for family in ("past_race", "workout", "trial"):
            self.assertEqual(manifest["feature_catalog"][f"{family}_speed_last_mps"]["unit"], "m/s")
        for horse in HORSES[1:]:
            invalid = features.loc[(features.race_id == "r2") & (features.horse_id == horse)].iloc[0]
            self.assertTrue(pd.isna(invalid.workout_speed_last_mps))
            self.assertEqual(invalid.workout_speed_support, 0)

    def test_current_race_labels_and_market_prices_never_change_eligible_predictors(self):
        before, after = source_rows(), source_rows()
        current = after.race_id.eq("r4")
        after.loc[current, "result"] = [3, 2, 1]
        after.loc[current, "finish_time"] = ["1:12.00", "1:11.00", "1:10.00"]
        after.loc[current, "win_odds"] = [10, 11, 12]
        first, original = self.build("original-labels", frame=before)
        second, perturbed = self.build("perturbed-labels", frame=after)
        columns = sorted(set(first["eligible_numeric_predictors"]) | set(second["eligible_numeric_predictors"]))
        forbidden = {"result", "finish_seconds", "finish_time", "target_win", "target_probability",
                     "target_top3", "speed_ratio_raw", "market_raw", "market_probability", "win_odds"}
        self.assertFalse(forbidden & set(columns))
        left = original.loc[original.race_id.eq("r4")].sort_values("horse_id")[columns].reset_index(drop=True)
        right = perturbed.loc[perturbed.race_id.eq("r4")].sort_values("horse_id")[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)

    def test_delayed_publication_race_is_audited_out_of_development_and_history(self):
        before, after = source_rows(), source_rows()
        for frame in (before, after):
            frame["outcome_available_at"] = frame.date + "T18:00:00+08:00"
            frame.loc[frame.race_id == "r1", "outcome_available_at"] = "2024-01-05T12:00:00+08:00"
        after.loc[after.race_id == "r1", "result"] = [3, 2, 1]
        after.loc[after.race_id == "r1", "finish_time"] = ["1:12.00", "1:11.00", "1:10.00"]
        first, original = self.build("unpublished-prior-original", frame=before)
        second, perturbed = self.build("unpublished-prior-perturbed", frame=after)
        for manifest, features in ((first, original), (second, perturbed)):
            self.assertEqual(set(features.race_id), {"r2", "r3", "r4", "r5", "r6"})
            path = self.registry.dataset_path(manifest["dataset_id"])
            runners = pd.read_parquet(path / "runners.parquet")
            self.assertNotIn("r1", set(runners.race_id))
            exclusions = json.loads((path / "exclusions.json").read_text())
            rejected, = [row for row in exclusions if row["race_id"] == "r1"]
            self.assertEqual(rejected["rows"], 3)
            self.assertEqual(set(map(tuple, rejected["runner_keys"])), {("r1", horse) for horse in HORSES})
            self.assertIn("delayed_result_publication_requires_asof_history_rebuild", rejected["reasons"])
            self.assertEqual(manifest["validation"]["unaccounted_source_keys"], 0)
            self.assertEqual(manifest["validation"]["excluded_rows"], 9)
            self.assertTrue(any("delayed-label asof rebuild not yet supported" in note
                                for note in manifest["unsupported_requirements"]))
            self.assertTrue(features.loc[features.race_id == "r4", "prior_starts"].eq(2).all())
        columns = sorted(set(first["eligible_numeric_predictors"]) | set(second["eligible_numeric_predictors"]))
        left = original.loc[original.race_id.eq("r4")].sort_values("horse_id")[columns].reset_index(drop=True)
        right = perturbed.loc[perturbed.race_id.eq("r4")].sort_values("horse_id")[columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right,
                                      obj="Predictors rebuilt after whole-race delayed-publication exclusion")

    def test_corrupted_runner_quarantines_entire_race_with_accounted_exclusions(self):
        corrupt = source_rows()
        index = corrupt.index[(corrupt.race_id == "r2") & (corrupt.horse_no == 2)][0]
        corrupt.loc[index, "horse_id"] = HORSES[0]
        manifest, features = self.build("corrupt-identity", frame=corrupt)
        self.assertNotIn("r2", set(features.race_id))
        path = self.registry.dataset_path(manifest["dataset_id"])
        exclusions = json.loads((path / "exclusions.json").read_text())
        row, = [item for item in exclusions if item["race_id"] == "r2"]
        self.assertEqual(row["rows"], 3)
        self.assertIn("duplicate_or_missing_race_runner_identity", row["reasons"])
        self.assertEqual(manifest["validation"]["unaccounted_source_keys"], 0)
        self.assertEqual(set(features.groupby("race_id").size()), {3})

    def test_missing_runner_number_must_not_be_fabricated_into_a_verified_field(self):
        corrupt = source_rows()
        index = corrupt.index[(corrupt.race_id == "r2") & (corrupt.horse_no == 2)][0]
        corrupt.loc[index, "horse_no"] = np.nan
        manifest, features = self.build("missing-saddle", frame=corrupt)
        self.assertNotIn("r2", set(features.race_id), "Missing stable runner identity must exclude the entire race")
        self.assertEqual(manifest["validation"]["unaccounted_source_keys"], 0)

    def test_parent_confirmation_races_remain_quarantined_when_child_requests_zero(self):
        parent, _ = self.build("parent")
        path = self.registry.dataset_path(parent["dataset_id"])
        frozen = {tuple(key) for key in json.loads((path / "confirmation_keys.json").read_text())}
        self.assertEqual({race for race, _ in frozen}, {"r7", "r8"})
        child, features = self.build("child", parent_dataset_id=parent["dataset_id"], protocol={
            "min_train_races": 2, "calibration_races": 1, "score_races": 1, "max_folds": 2,
            "whole_meeting_boundaries": True, "final_confirmation_races": 0,
        })
        self.assertFalse(frozen & set(zip(features.race_id, features.horse_id)),
                         "A dataset extension cannot unlock the parent's confirmation labels")
        inherited = {tuple(key) for key in json.loads((self.registry.dataset_path(child["dataset_id"]) / "confirmation_keys.json").read_text())}
        self.assertTrue(frozen <= inherited)

    def test_artifact_extra_quarantine_is_inherited_but_does_not_expand_root_seed(self):
        self.build("operator-seed")
        seed_path = self.registry.root / "protected-confirmation-races.json"
        original_seed = seed_path.read_bytes()
        protocol = {"min_train_races": 2, "calibration_races": 1, "score_races": 1,
                    "max_folds": 2, "final_confirmation_races": 0,
                    "final_confirmation_race_ids": ["r6"]}
        extra, features = self.build("extra-local-quarantine", protocol=protocol)
        self.assertNotIn("r6", set(features.race_id))
        self.assertEqual(seed_path.read_bytes(), original_seed)
        child, inherited = self.build("inherit-local-extra", parent_dataset_id=extra["dataset_id"],
                                      protocol=protocol | {"final_confirmation_race_ids": []})
        self.assertNotIn("r6", set(inherited.race_id))
        _, independent = self.build("independent-seed-only", protocol=protocol | {"final_confirmation_race_ids": []})
        self.assertIn("r6", set(independent.race_id))
        self.assertEqual(seed_path.read_bytes(), original_seed)
        self.assertEqual(set(child["identity"]["protected_confirmation_races"]), {"r6", "r7", "r8"})

    def test_failed_oversized_quarantine_does_not_poison_existing_operator_seed(self):
        self.build("seed-before-malicious")
        seed_path = self.registry.root / "protected-confirmation-races.json"
        before = seed_path.read_bytes()
        snapshot, raw = self.fixture()
        request = self.request("oversized-quarantine", raw, protocol={
            "min_train_races": 2, "calibration_races": 1, "score_races": 1,
            "max_folds": 2, "final_confirmation_races": 7,
        })
        self.registry.submit(request)
        with self.assertRaises(ValueError):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")
        self.assertEqual(seed_path.read_bytes(), before)
        _, successor = self.build("after-malicious", protocol={
            "min_train_races": 2, "calibration_races": 1, "score_races": 1,
            "max_folds": 2, "final_confirmation_races": 0,
        })
        self.assertEqual(set(successor.race_id), {"r1", "r2", "r3", "r4", "r5", "r6"})

    def test_confirmation_identity_freezes_source_races_before_population_filtering(self):
        manifest, _ = self.build("filtered-operator-seed", race_population_spec={"end_date": "2024-01-06"})
        self.assertEqual(set(manifest["identity"]["protected_confirmation_races"]), {"r7", "r8"})
        expected_code = {"feature_definitions.py", "feature_expressions.py", "research_targets.py", "research_evaluation.py"}
        self.assertTrue(expected_code <= set(manifest["identity"]["code"]))
        path = self.registry.dataset_path(manifest["dataset_id"])
        self.assertEqual((path / "manifest.sha256").read_text().strip(), digest(path / "manifest.json"))
        confirmation = json.loads((path / "confirmation_keys.json").read_text())
        self.assertEqual({key[0] for key in confirmation}, {"r7", "r8"})

    def test_known_nonfinisher_is_zero_for_placing_but_excludes_whole_ranking_race(self):
        from ima.research_targets import apply_target_contract, target_contract

        frame = source_rows()
        frame["finishing_status"] = "FINISHED"
        index = frame.index[(frame.race_id == "r2") & (frame.horse_no == 3)][0]
        frame.loc[index, ["result", "finish_time"]] = [np.nan, None]
        frame.loc[index, "finishing_status"] = "DNF"
        manifest, features = self.build("known-nonfinisher", frame=frame, target_contracts=["top3", "position"])
        self.assertEqual(len(features.loc[features.race_id == "r2"]), 3)
        placed = apply_target_contract(features, target_contract("placing_top_k"))
        row = placed.loc[(placed.race_id == "r2") & (placed.horse_id == HORSES[2])].iloc[0]
        label = target_contract("placing_top_k").label_column
        self.assertEqual(row[label], 0)
        ranked = apply_target_contract(features, target_contract("ranking_strength"))
        self.assertNotIn("r2", set(ranked.race_id))
        eligibility = json.loads((self.registry.dataset_path(manifest["dataset_id"]) / "target-eligibility.json").read_text())
        self.assertEqual(eligibility["position"]["excluded_race_ids"], ["r2"])
        self.assertEqual(eligibility["top3"]["eligible_rows"], len(features))

    def test_unknown_nonfinisher_status_rejects_the_whole_race(self):
        frame = source_rows()
        frame["finishing_status"] = "FINISHED"
        index = frame.index[(frame.race_id == "r2") & (frame.horse_no == 3)][0]
        frame.loc[index, ["result", "finish_time"]] = [np.nan, None]
        frame.loc[index, "finishing_status"] = "UNKNOWN"
        manifest, features = self.build("unknown-nonfinisher", frame=frame, target_contracts=["top3", "position"])
        self.assertNotIn("r2", set(features.race_id))
        exclusions = json.loads((self.registry.dataset_path(manifest["dataset_id"]) / "exclusions.json").read_text())
        rejected, = [row for row in exclusions if row["race_id"] == "r2"]
        self.assertEqual(rejected["rows"], 3)
        self.assertIn("invalid_finish_order_or_nonfinisher", rejected["reasons"])

    def test_nonexistent_parent_fails_closed_instead_of_publishing_a_successor(self):
        snapshot, raw = self.fixture()
        request = self.request("orphan-child", raw, parent_dataset_id="dataset-" + "0" * 64)
        self.registry.submit(request)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")

    def test_child_cannot_bypass_parent_checksum_verification(self):
        parent, _ = self.build("parent-to-tamper")
        path = self.registry.dataset_path(parent["dataset_id"]) / "confirmation_keys.json"
        path.write_text("[]")
        snapshot, raw = self.fixture()
        request = self.request("child-after-tamper", raw, parent_dataset_id=parent["dataset_id"])
        self.registry.submit(request)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")

    def test_declared_target_only_predictor_cannot_enter_fundamental_catalog(self):
        snapshot, raw = self.fixture(target_only=["result", "finish_time", "finish_seconds", "draw"])
        request = self.request("declared-target-only", raw)
        self.registry.submit(request)
        manifest = self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        unsafe = {"draw", "poly_draw_sq", "poly_draw_distance"} & set(manifest["eligible_numeric_predictors"])
        self.assertFalse(unsafe,
                         "Source-contract target-only classification must exclude direct and derived predictors")

    def test_catalog_compound_units_match_materialized_formula_dimensions(self):
        manifest, _ = self.build("compound-units")
        expected = {"poly_distance_sq": "m^2", "poly_draw_distance": "m",
                    "poly_weight_change_distance": "lb*m", "prize": "HKD"}
        for name, unit in expected.items():
            with self.subTest(name=name):
                self.assertEqual(manifest["feature_catalog"][name]["unit"], unit)
                self.assertEqual(manifest["predictor_catalog"][name]["unit"], unit)

    def test_target_only_current_weight_and_distance_taint_all_actual_dependencies(self):
        dependencies = {
            "actual_weight": {"actual_weight", "carried_weight_rank", "carried_weight_change",
                              "carried_weight_change_rank", "poly_weight_change_distance"},
            "declared_weight": {"declared_weight", "body_weight_rank", "body_weight_change",
                                "body_weight_change_pct", "weight_change_per_day"},
            "distance": {"distance", "distance_change", "distance_band_starts", "distance_experience_rank",
                         "avg_same_distance_finish_time_183d", "poly_distance_sq", "poly_draw_distance",
                         "poly_weight_change_distance", "past_race_speed_same_distance_mean_mps",
                         "workout_speed_same_distance_mean_mps", "trial_speed_same_distance_mean_mps"},
        }
        for source, derived in dependencies.items():
            with self.subTest(source=source):
                snapshot, raw = self.fixture(target_only=["result", "finish_time", source])
                request = self.request(f"target-only-{source}", raw)
                self.registry.submit(request)
                manifest = self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
                leaked = derived & set(manifest["eligible_numeric_predictors"])
                self.assertFalse(leaked, f"Current {source} is target-only but dependent predictors remain eligible")

    def test_pre_normalized_events_cannot_bypass_strict_provenance_or_unknown_tier_checks(self):
        from ima.historical_events import normalize_events

        normalized = normalize_events([event("normalize-first")]).to_dict("records")[0]
        variants = ({"source_url": "https://untrusted.invalid/event"},
                    {"source_body_sha256": "not-a-body-hash"}, {"availability_tier": "unknown"})
        for number, changes in enumerate(variants):
            with self.subTest(changes=changes):
                _, features = self.build(f"normalized-untrusted-{number}", events=[normalized | changes])
                self.assertTrue(features.trackwork_usable_observations.eq(0).all())
                self.assertTrue(features.workout_speed_support.eq(0).all())
                self.assertTrue(features.workout_speed_last_mps.isna().all())

    def test_normalized_nullable_event_json_roundtrip_accepts_valid_provenance(self):
        from ima.historical_events import normalize_events

        normalized = normalize_events([event("valid-normalized-nullable")]).to_dict("records")[0]
        self.assertIsNone(normalized["valid_from"])
        self.assertIsNone(normalized["correction_of"])
        _, features = self.build("valid-normalized-roundtrip", events=[normalized])
        self.assertTrue(features.loc[features.horse_id.eq(HORSES[0]), "trackwork_usable_observations"].eq(1).all())
        self.assertTrue(features.loc[features.horse_id.ne(HORSES[0]), "trackwork_usable_observations"].eq(0).all())
        self.assertTrue(features.loc[features.horse_id.eq(HORSES[0]), "workout_speed_last_mps"].eq(16).all())
        self.assertTrue(features.loc[features.horse_id.ne(HORSES[0]), "workout_speed_last_mps"].isna().all())
        self.assertTrue(features.loc[features.horse_id.ne(HORSES[0]), "workout_speed_support"].eq(0).all())

    def test_pre_normalized_conflicting_observation_payloads_reject_registry_build(self):
        from ima.historical_events import normalize_events

        normalized = normalize_events([event("same-observation")]).to_dict("records")[0]
        changed = normalized | {"typed_values": {"distance_metres": 1200, "workout_finish_seconds": 30}}
        with self.assertRaisesRegex(ValueError, "Conflicting source observation"):
            normalize_events([normalized, changed])
        snapshot, raw = self.fixture(events=[normalized, changed])
        request = self.request("conflicting-normalized-observation", raw)
        self.registry.submit(request)
        with self.assertRaisesRegex(ValueError, "Conflicting source observation"):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")

    def test_mixed_date_only_and_timestamp_observations_keep_conservative_eligibility(self):
        manifest, features = self.build("date-only-json", events=[
            event("timestamp"),
            event("date-only", occurred_at=None, occurred_date="2024-01-02",
                  published_at="2024-01-01T12:00:00+08:00"),
        ])
        events = pd.read_parquet(self.registry.dataset_path(manifest["dataset_id"]) / "events.parquet")
        normalized = events.loc[events.event_id == "date-only"].iloc[0]
        self.assertEqual(pd.Timestamp(normalized.occurrence_eligible_at), pd.Timestamp("2024-01-03T00:00:00+08:00"))
        cutoff = features.loc[(features.race_id == "r3") & (features.horse_id == HORSES[0])].iloc[0]
        later = features.loc[(features.race_id == "r4") & (features.horse_id == HORSES[0])].iloc[0]
        self.assertEqual(cutoff.trackwork_usable_observations, 1)
        self.assertEqual(later.trackwork_usable_observations, 2)

    def test_strict_events_require_both_occurrence_and_availability_before_cutoff(self):
        observations = [
            event("unknown-publication", published_at=None, publication_verified=False),
            event("late-capture", published_at=None, publication_verified=False, first_seen_at="2024-01-10T12:00:00+08:00"),
            event("exact-cutoff", published_at="2024-01-02T00:00:00+08:00"),
            event("same-day-date-only", occurred_at=None, occurred_date="2024-01-02", published_at="2024-01-01T12:00:00+08:00"),
        ]
        manifest, features = self.build("strict-cutoffs", events=observations)
        early = features.loc[(features.race_id == "r2") & (features.horse_id == HORSES[0])].iloc[0]
        self.assertEqual(early.trackwork_usable_observations, 0)
        self.assertEqual(early.workout_speed_support, 0)
        self.assertTrue(pd.isna(early.trackwork_7d), "No complete coverage evidence is not a zero count")
        self.assertEqual(manifest["event_audit"]["tiers"]["unknown"], 1)
        late = features.loc[(features.race_id == "r4") & (features.horse_id == HORSES[0])].iloc[0]
        self.assertEqual(late.trackwork_usable_observations, 2)

    def test_unknown_runner_cutoff_cannot_publish_verified_predictors(self):
        frame = source_rows()
        frame["cutoff_at"] = frame.date + "T00:00:00+08:00"
        frame.loc[frame.race_id == "r2", "cutoff_at"] = None
        snapshot, raw = self.fixture(frame=frame)
        request = self.request("unknown-cutoff", raw)
        self.registry.submit(request)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")

    def test_source_checksum_tampering_rejects_build_without_verified_artifacts(self):
        snapshot, raw = self.fixture()
        request = self.request("tampered-source", raw)
        self.registry.submit(request)
        with (snapshot / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event("unmanifested-event")) + "\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.registry.build(request.request_id, source_snapshot=snapshot, raw_manifest=raw)
        self.assertEqual(self.registry.get(request.request_id)["status"], "rejected")
        self.assertEqual(list((self.registry.root / "datasets").iterdir()), [])

    def test_verified_payload_tampering_is_detected_on_replay(self):
        manifest, _ = self.build("tampered-verified")
        path = self.registry.dataset_path(manifest["dataset_id"]) / "confirmation_keys.json"
        path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.registry.verify(manifest["dataset_id"])


if __name__ == "__main__":
    unittest.main()
