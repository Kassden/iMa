"""Evidence and temporal boundary tests for the public historical-event APIs."""
import unittest

import pandas as pd

from ima.historical_events import (
    CoverageRecord,
    build_event_features,
    event_views,
    instant,
    normalize_events,
    select_as_of,
)


HORSE = "HK_2022_H033"
OTHER_HORSE = "HK_2023_J044"
SOURCE = "https://racing.hkjc.com/en-us/local/information/trackworkresult"


def event(**changes):
    return {
        "horse_id": HORSE,
        "family": "trackwork",
        "event_date": "2026-09-01",
        "source_url": SOURCE,
        "source_body_hash": "a" * 64,
        "values": {"Type": "Gallop", "distance_metres": 800},
        "table_index": 1,
        **changes,
    }


def runners(*cutoffs, horse=HORSE):
    return pd.DataFrame({"horse_id": [horse] * len(cutoffs), "cutoff_at": list(cutoffs)},
                        index=[10 + i * 3 for i in range(len(cutoffs))])


def coverage(**changes):
    return CoverageRecord.model_validate({
        "coverage_id": "horse-workouts-complete",
        "family": "trackwork",
        "source_url": SOURCE,
        "horse_id": HORSE,
        "start_at": "2026-01-01T00:00:00+08:00",
        "end_at": "2026-09-10T00:00:00+08:00",
        "available_at": "2026-09-09T23:00:00+08:00",
        "status": "complete",
        "collector_watermark": "official-complete-interval-v1",
        **changes,
    })


class HistoricalEventTests(unittest.TestCase):
    def test_future_only_fast_path_matches_unrelated_past_event_control(self):
        frame = runners("2026-09-03", "2026-09-10")
        future = event(first_seen_at="2026-10-02")
        fast = build_event_features(frame, [future])
        unrelated = event(horse_id=OTHER_HORSE, published_at="2026-09-02",
                          publication_verified=True, source_body_hash="b" * 64)
        control = build_event_features(frame, [future, unrelated])
        pd.testing.assert_frame_equal(fast, control)
        self.assertTrue(fast.trackwork_available.eq(0).all())
        self.assertTrue(fast.trackwork_7d_observed_count.eq(0).all())
        self.assertTrue(fast.trackwork_7d.isna().all())
        self.assertTrue(fast.days_since_trackwork.isna().all())

    def test_empty_coverage_fast_path_still_rejects_unknown_cutoffs(self):
        with self.assertRaisesRegex(ValueError, "Unknown race cutoff"):
            build_event_features(runners(None), [])

    def test_occurrence_does_not_fabricate_publication_or_capture(self):
        normalized = normalize_events([event()])
        row = normalized.iloc[0]
        self.assertEqual(row.availability_tier, "unknown")
        self.assertIsNone(row.published_at)
        self.assertIsNone(row.first_seen_at)
        self.assertIsNone(row.available_at)
        features = build_event_features(runners("2026-09-10"), normalized)
        self.assertEqual(features.iloc[0].trackwork_available, 0)
        self.assertTrue(pd.isna(features.iloc[0].trackwork_7d))

    def test_capture_of_old_page_is_only_prospective(self):
        observed = event(first_seen_at="2026-10-02T10:00:00+08:00")
        normalized = normalize_events([observed])
        self.assertEqual(normalized.iloc[0].availability_tier, "observed_prospective")
        self.assertIsNone(normalized.iloc[0].published_at)
        frame = runners("2026-09-15", "2026-10-02T10:00:00+08:00",
                        "2026-10-02T10:00:01+08:00")
        result = build_event_features(frame, normalized)
        self.assertEqual(result.trackwork_available.tolist(), [0, 0, 1])
        self.assertEqual(result.trackwork_usable_observations.tolist(), [0, 0, 1])
        self.assertEqual(result.index.tolist(), frame.index.tolist())

    def test_unverified_publication_timestamp_is_not_historical_evidence(self):
        raw = event(published_at="2026-09-01T08:00:00+08:00",
                    fetched_at="2026-10-02T10:00:00+08:00")
        normalized = normalize_events([raw])
        self.assertEqual(normalized.iloc[0].availability_tier, "observed_prospective")
        result = build_event_features(runners("2026-09-15"), normalized)
        self.assertEqual(result.iloc[0].trackwork_available, 0)

    def test_verified_publication_can_precede_later_capture(self):
        raw = event(published_at="2026-09-02T08:00:00+08:00", publication_verified=True,
                    first_seen_at="2026-10-02T10:00:00+08:00")
        normalized = normalize_events([raw])
        self.assertEqual(normalized.iloc[0].availability_tier, "verified_point_in_time")
        result = build_event_features(runners("2026-09-03"), normalized)
        self.assertEqual(result.iloc[0].trackwork_available, 1)

    def test_both_occurrence_and_publication_must_precede_cutoff(self):
        raw = event(occurred_at="2026-09-02T09:00:00+08:00",
                    published_at="2026-09-02T08:00:00+08:00", publication_verified=True)
        frame = runners("2026-09-02T08:30:00+08:00", "2026-09-02T09:00:00+08:00",
                        "2026-09-02T09:00:01+08:00")
        self.assertEqual(build_event_features(frame, [raw]).trackwork_available.tolist(), [0, 0, 1])
        raw = event(occurred_at="2026-09-02T08:00:00+08:00",
                    published_at="2026-09-02T09:00:00+08:00", publication_verified=True)
        self.assertEqual(build_event_features(frame, [raw]).trackwork_available.tolist(), [0, 0, 1])

    def test_date_only_occurrence_has_conservative_next_day_eligibility(self):
        raw = event(published_at="2026-09-01T06:00:00+08:00", publication_verified=True)
        frame = runners("2026-09-01T23:59:59+08:00", "2026-09-02T00:00:00+08:00",
                        "2026-09-02T00:00:01+08:00")
        self.assertEqual(build_event_features(frame, [raw]).trackwork_available.tolist(), [0, 0, 1])

    def test_date_only_publication_and_naive_instants_use_hong_kong_time(self):
        self.assertEqual(instant("2026-09-02"), pd.Timestamp("2026-09-01T16:00:00Z"))
        raw = event(published_at="2026-09-02", publication_verified=True)
        normalized = normalize_events([raw])
        self.assertEqual(normalized.iloc[0].available_at, "2026-09-02T16:00:00+00:00")
        frame = runners("2026-09-03T00:00:00+08:00", "2026-09-03T00:00:01+08:00")
        self.assertEqual(build_event_features(frame, normalized).trackwork_available.tolist(), [0, 1])

    def test_name_only_and_explicitly_ambiguous_identities_never_join(self):
        for changes in ({"horse_id": None, "horse_name": "SAME NAME"},
                        {"horse_id": "H033"},
                        {"identity_status": "ambiguous_official_confirmation"}):
            with self.subTest(changes=changes):
                raw = event(first_seen_at="2026-09-02T08:00:00+08:00", **changes)
                normalized = normalize_events([raw])
                self.assertIsNone(normalized.iloc[0].horse_id)
                self.assertIn("unresolved_identity", normalized.iloc[0].exclusion_reasons)
                self.assertEqual(build_event_features(runners("2026-09-10"), normalized).iloc[0].trackwork_available, 0)

    def test_source_domain_or_hash_gaps_keep_evidence_but_exclude_features(self):
        for changes in ({"source_url": "https://racing.hkjc.com.example.org/workouts"},
                        {"source_url": "http://racing.hkjc.com/workouts"},
                        {"source_body_hash": "not-a-body-digest"}):
            with self.subTest(changes=changes):
                normalized = normalize_events([event(first_seen_at="2026-09-02", **changes)])
                self.assertEqual(len(normalized), 1)
                self.assertIn("unverified_official_source", normalized.iloc[0].exclusion_reasons)
                self.assertFalse(event_views(normalized))

    def test_duplicate_identical_source_observation_is_idempotent(self):
        raw = event(first_seen_at="2026-09-02T08:00:00+08:00")
        normalized = normalize_events([raw, dict(raw)])
        self.assertEqual(len(normalized), 1)
        result = build_event_features(runners("2026-09-10"), [raw, dict(raw)])
        self.assertEqual(result.iloc[0].trackwork_usable_observations, 1)

    def test_same_day_equal_text_different_rows_are_not_one_physical_event(self):
        raw = event(first_seen_at="2026-09-02T08:00:00+08:00")
        normalized = normalize_events([raw, raw | {"table_index": 2}])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized.physical_event_group_id.nunique(), 2)
        result = build_event_features(runners("2026-09-10"), normalized)
        self.assertEqual(result.iloc[0].trackwork_usable_observations, 2)

    def test_conflicting_values_for_one_source_observation_are_rejected(self):
        raw = event(event_id="event-one", source_observation_id="observation-one",
                    first_seen_at="2026-09-02T08:00:00+08:00")
        changed = raw | {"values": {"Type": "Gallop", "distance_metres": 1200}}
        with self.assertRaises(ValueError):
            normalize_events([raw, changed])

    def test_modern_revision_cannot_rewrite_a_past_cutoff(self):
        original = event(event_id="original", source_observation_id="original-observation",
                         first_seen_at="2026-09-02T08:00:00+08:00")
        correction = event(event_id="revision", source_observation_id="revision-observation",
                           source_body_hash="b" * 64, correction_of="original",
                           first_seen_at="2026-09-10T08:00:00+08:00",
                           values={"distance_metres": 1200})
        views = event_views([original, correction])[(HORSE, "trackwork")]
        before = select_as_of(views, instant("2026-09-10T08:00:00+08:00"))
        after = select_as_of(views, instant("2026-09-10T08:00:01+08:00"))
        self.assertEqual([r["event_id"] for r in before], ["original"])
        self.assertEqual([r["event_id"] for r in after], ["revision"])
        result = build_event_features(runners("2026-09-03", "2026-09-11"), [original, correction])
        self.assertEqual(result.trackwork_last_distance_metres.tolist(), [800, 1200])
        self.assertEqual(result.trackwork_usable_observations.tolist(), [1, 1])

    def test_revision_cycle_is_rejected_and_orphan_revision_is_unusable(self):
        first = event(event_id="first", source_observation_id="first-observation", correction_of="second")
        second = event(event_id="second", source_observation_id="second-observation", correction_of="first")
        with self.assertRaisesRegex(ValueError, "cycle"):
            normalize_events([first, second])
        orphan = event(correction_of="missing", first_seen_at="2026-09-02")
        self.assertFalse(event_views([orphan]))

    def test_unknown_coverage_is_missing_even_inside_observed_event_range(self):
        events = [event(event_date="2026-09-01", first_seen_at="2026-09-02"),
                  event(event_date="2026-09-20", first_seen_at="2026-09-21", table_index=2)]
        result = build_event_features(runners("2026-09-10"), events).iloc[0]
        self.assertEqual(result.trackwork_available, 1)
        self.assertTrue(pd.isna(result.trackwork_7d))
        self.assertEqual(result.trackwork_7d_coverage, 0)
        self.assertEqual(result.trackwork_7d_observed_count, 0)

    def test_verified_complete_horse_interval_can_produce_zero(self):
        result = build_event_features(runners("2026-09-10"), [], [coverage()]).iloc[0]
        self.assertEqual(result.trackwork_7d, 0)
        self.assertEqual(result.trackwork_7d_coverage, 1)
        other = build_event_features(runners("2026-09-10", horse=OTHER_HORSE), [], [coverage()]).iloc[0]
        self.assertTrue(pd.isna(other.trackwork_7d))

    def test_incomplete_stale_or_gapped_coverage_cannot_produce_zero(self):
        variants = [{"status": "unknown"}, {"status": "incomplete"}, {"missing_pages": ("page-2",)},
                    {"identity_gaps": 1}, {"publication_gaps": 1}, {"physical_identity_gaps": 1},
                    {"available_at": "2026-09-10"}, {"end_at": "2026-09-09"},
                    {"start_at": "2026-09-04"}]
        for changes in variants:
            with self.subTest(changes=changes):
                result = build_event_features(runners("2026-09-10"), [], [coverage(**changes)]).iloc[0]
                self.assertTrue(pd.isna(result.trackwork_7d))
                self.assertEqual(result.trackwork_7d_coverage, 0)

    def test_retrospective_policy_is_explicit_and_does_not_change_evidence(self):
        normalized = normalize_events([event()])
        frame = runners("2026-09-03T00:00:00+08:00", "2026-09-03T00:00:01+08:00")
        result = build_event_features(frame, normalized, policy="assumed_retrospective",
                                      retrospective_lags={"trackwork": 1})
        self.assertEqual(result.trackwork_available.tolist(), [0, 1])
        self.assertEqual(normalized.iloc[0].availability_tier, "unknown")
        self.assertIsNone(normalized.iloc[0].published_at)
        with self.assertRaisesRegex(ValueError, "Strict"):
            build_event_features(frame, normalized, retrospective_lags={"trackwork": 1})

    def test_future_clearance_cannot_close_a_known_condition(self):
        vet = event(family="veterinary", first_seen_at="2026-09-02T08:00:00+08:00")
        clear = event(family="veterinary_clearance", event_date="2026-09-08",
                      first_seen_at="2026-09-12T08:00:00+08:00")
        result = build_event_features(runners("2026-09-10", "2026-09-13"), [vet, clear])
        self.assertEqual(result.veterinary_condition_open.tolist(), [1, 0])

    def test_ongoing_stay_is_truncated_at_cutoff_without_completed_future_stay(self):
        move = event(family="movements", first_seen_at="2026-09-02T08:00:00+08:00",
                     values={"arrived_in_conghua": "2026-09-01", "returned_to_hk": "2026-09-20"})
        result = build_event_features(runners("2026-09-10"), [move]).iloc[0]
        self.assertEqual(result.movement_known_stay_days, 9)
        self.assertTrue(pd.isna(result.movement_completed_stay_days))
        self.assertTrue(pd.isna(result.movement_days_since_hk_return))

    def test_normalized_dataframe_roundtrip_keeps_observation_and_availability(self):
        raw = event(first_seen_at="2026-09-02T08:00:00+08:00")
        once = normalize_events([raw])
        twice = normalize_events(once)
        pd.testing.assert_frame_equal(once, twice)


if __name__ == "__main__":
    unittest.main()
