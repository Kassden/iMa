import unittest

from ima.event_evidence_audit import audit_event_evidence


class EventEvidenceAuditTests(unittest.TestCase):
    def test_family_season_waterfall_is_exclusive_and_preserves_unknowns(self):
        base = {"family": "trackwork", "horse_id": "HK_2025_A123",
                "occurred_date": "2025-09-01", "source_observation_id": "row",
                "source_url": "https://racing.hkjc.com/sample", "source_body_sha256": "a" * 64,
                "first_seen_at": "2025-09-02T00:00:00Z"}
        rows = [base, dict(base, occurred_date="2025-08-31", horse_id="Unknown"),
                dict(base, occurred_date="invalid"), dict(base, source_body_sha256="bad"),
                dict(base, published_at="2025-09-01T00:00:00Z", publication_verified=True),
                dict(base, published_at="invalid", publication_verified=True),
                dict(base, first_seen_at=None)]
        result = audit_event_evidence(rows)
        slices = result["availability_waterfall"]["slices"]
        self.assertEqual(sum(row["observations"] for row in slices), len(rows))
        for row in slices:
            self.assertEqual(sum(row["exclusive_reasons"].values()), row["observations"])
        by_season = {row["season"]: row["exclusive_reasons"] for row in slices}
        self.assertEqual(by_season["2024-2025"], {"horse_identity_unresolved": 1})
        self.assertEqual(by_season["unknown"], {"occurrence_missing_or_invalid": 1})
        self.assertEqual(by_season["2025-2026"]["prospective_capture_only"], 2)
        self.assertEqual(result["availability"]["verified_publication_available"], 1)

    def test_normalized_null_instant_uses_occurrence_date(self):
        result = audit_event_evidence([{"occurred_at": None, "occurred_date": "2025-11-20"}])
        self.assertEqual(result["summary"]["occurrence_timestamp_present"], 1)
        self.assertEqual(result["summary"]["occurrence_timestamp_valid"], 1)

    def test_separates_missing_false_and_true_physical_evidence(self):
        rows = [
            {"family": "trackwork", "event_id": "missing"},
            {"family": "trackwork", "event_id": "defaulted", "physical_identity_verified": False},
            {"family": "trackwork", "event_id": "verified", "physical_identity_verified": True,
             "physical_identity_evidence": "official batch and timestamp match"},
        ]
        result = audit_event_evidence(rows)
        physical = result["physical_grouping"]
        self.assertEqual(physical["verified_flag_true"], 1)
        self.assertEqual(physical["explicit_false_unknown_not_rejected"], 1)
        self.assertEqual(physical["flag_absent_unknown"], 1)
        self.assertEqual(physical["separate_evidence_payload_present"], 1)
        self.assertEqual(result["groups_by_event_family"]["trackwork"]["observations"], 3)

    def test_publication_capture_and_occurrence_are_separate(self):
        result = audit_event_evidence([
            {"family": "veterinary", "published_at": None, "publication_verified": False,
             "first_seen_at": "2026-01-02T00:00:00Z", "occurred_at": "2026-01-01T00:00:00Z"},
            {"family": "veterinary", "published_at": "2026-01-01T00:00:00Z",
             "publication_verified": True, "first_seen_at": "2026-01-02T00:00:00Z"},
            {"family": "veterinary", "published_at": "2025-12-31T00:00:00Z",
             "availability_tier": "observed_prospective", "first_seen_at": "2026-01-02T00:00:00Z"},
        ])
        availability = result["availability"]
        self.assertEqual(availability["publication_timestamp_present"], 2)
        self.assertEqual(availability["verified_publication_available"], 1)
        self.assertEqual(availability["first_capture_timestamp_present"], 3)
        self.assertEqual(availability["occurrence_timestamp_present"], 1)

    def test_source_row_identity_and_field_denominators_are_reported(self):
        result = audit_event_evidence([
            {"family": "movements", "horse_id": "HK_2025_A123", "source_observation_id": "row-1",
             "source_url": "https://racing.hkjc.com/item/1", "source_body_sha256": "a" * 64,
             "physical_event_group_id": "group-1"},
            {"family": "incidents", "horse_id": "Unknown", "source_url": "https://racing.hkjc.com/item/2"},
        ])
        self.assertEqual(result["summary"]["horse_id_exact"], 1)
        self.assertEqual(result["summary"]["horse_id_unresolved_or_missing"], 1)
        self.assertEqual(result["source_linkage"]["rows_with_source_observation_id_and_url"], 1)
        self.assertEqual(result["field_availability"]["source_observation_id"],
                         {"present": 1, "denominator": 2})
        self.assertEqual(set(result["groups_by_event_family"]), {"incidents", "movements"})

    def test_truthy_strings_and_malformed_flags_do_not_become_evidence(self):
        result = audit_event_evidence([
            {"physical_identity_verified": "false", "publication_verified": "true",
             "published_at": "2026-01-01", "physical_identity_evidence": ""},
        ])
        self.assertEqual(result["physical_grouping"]["verified_flag_true"], 0)
        self.assertEqual(result["physical_grouping"]["malformed_flag_unknown"], 1)
        self.assertEqual(result["availability"]["publication_verification_true"], 0)
        self.assertEqual(result["availability"]["verified_publication_available"], 0)

    def test_effective_availability_is_not_counted_as_first_capture(self):
        result = audit_event_evidence([
            {"family": "trackwork", "available_at": "2026-01-02T00:00:00Z"},
            {"family": "trackwork", "first_fetched_at": "2026-01-03T00:00:00Z"},
        ])
        availability = result["availability"]
        self.assertEqual(availability["first_capture_timestamp_present"], 1)
        self.assertEqual(availability["first_capture_timestamp_valid"], 1)
        self.assertEqual(availability["effective_availability_timestamp_present"], 1)
        self.assertEqual(availability["effective_availability_timestamp_valid"], 1)

    def test_empty_payloads_and_malformed_timestamps_are_not_valid_evidence(self):
        result = audit_event_evidence([
            {"family": "incidents", "physical_identity_evidence": {},
             "physical_grouping_evidence": [], "published_at": "not-a-timestamp",
             "first_seen_at": "also-not-a-timestamp"},
        ])
        self.assertEqual(result["summary"]["physical_evidence_present"], 0)
        self.assertEqual(result["field_availability"]["physical_identity_evidence"]["present"], 0)
        self.assertEqual(result["availability"]["publication_timestamp_present"], 1)
        self.assertEqual(result["availability"]["publication_timestamp_valid"], 0)
        self.assertEqual(result["availability"]["first_capture_timestamp_present"], 1)
        self.assertEqual(result["availability"]["first_capture_timestamp_valid"], 0)

    def test_rejects_malformed_row_containers(self):
        with self.assertRaisesRegex(TypeError, "row 0"):
            audit_event_evidence(["not a mapping"])
        with self.assertRaisesRegex(TypeError, "iterable of row mappings"):
            audit_event_evidence(42)


if __name__ == "__main__":
    unittest.main()
