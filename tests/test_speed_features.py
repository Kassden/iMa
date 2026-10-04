"""Physical units and source-specific cutoff tests for speed feature APIs."""
import unittest

import numpy as np
import pandas as pd

from ima.speed_features import (
    build_speed_features,
    finish_seconds,
    normalize_speed_measurement,
    speed_feature_catalog,
)


HORSE = "HK_2022_H033"


def runner(date="2026-09-10", **changes):
    return {"horse_id": HORSE, "race_id": "race-current", "date": date,
            "distance": 1200, "surface": "TURF", "venue": "ST", "finish_time": "1.10.00",
            **changes}


def timed_event(family="barrier_trials", **changes):
    return {"horse_id": HORSE, "family": family, "event_date": "2026-09-01",
            "first_seen_at": "2026-09-02T08:00:00+08:00",
            "source_url": "https://racing.hkjc.com/en-us/local/information/trackworkresult",
            "source_body_hash": "a" * 64, "table_index": 1,
            "values": {"distance_metres": 1200, "trial_finish_seconds": 75,
                       "venue": "ST", "surface": "TURF"}, **changes}


class SpeedFeatureTests(unittest.TestCase):
    def test_official_finish_time_formats_are_seconds(self):
        for value in ("1.10.25", "1:10.25", "70.25", 70.25):
            with self.subTest(value=value):
                self.assertAlmostEqual(finish_seconds(value), 70.25)
        for value in (None, "", "---", "PU", "1.60.00", 0, -1, np.nan, np.inf):
            with self.subTest(value=value):
                self.assertTrue(pd.isna(finish_seconds(value)))

    def test_explicit_metres_and_individual_seconds_produce_physical_speed(self):
        for family, field in (("barrier_trials", "trial_finish_seconds"),
                              ("trackwork", "workout_finish_seconds"), ("past_races", "finish_seconds")):
            with self.subTest(family=family):
                self.assertEqual(normalize_speed_measurement({"distance_metres": 1200, field: 75}, family),
                                 (1200, 75, 16, None))

    def test_explicit_distance_and_time_units_are_converted(self):
        for value, unit in ((75, "s"), (75, "seconds"), (75000, "ms"), (1.25, "minutes")):
            with self.subTest(unit=unit):
                result = normalize_speed_measurement({"distance": 1.2, "distance_unit": "km",
                    "time": value, "time_unit": unit, "time_basis": "individual"}, "trackwork")
                self.assertEqual(result, (1200, 75, 16, None))

    def test_missing_units_effort_prose_and_invalid_values_do_not_create_speed(self):
        variants = [({"distance": 1200, "trial_finish_seconds": 75}, "distance_unit_unknown"),
                    ({"distance_metres": 1200, "time": 75, "time_unit": "s"}, "individual_time_missing"),
                    ({"distance_metres": 1200, "time": 75, "time_unit": "laps", "time_basis": "individual"}, "individual_time_missing"),
                    ({"Workouts": "1 Round - Fast", "Type": "Gallop"}, "distance_unit_unknown"),
                    ({"distance_metres": 1200, "trial_finish_seconds": 0}, "invalid_measurement"),
                    ({"distance_metres": np.inf, "trial_finish_seconds": 75}, "invalid_measurement"),
                    ({"distance_metres": 1200, "trial_finish_seconds": np.nan}, "invalid_measurement")]
        for values, reason in variants:
            with self.subTest(values=values):
                result = normalize_speed_measurement(values, "barrier_trials")
                self.assertEqual(result[3], reason)
                self.assertTrue(all(pd.isna(v) for v in result[:3]))

    def test_batch_or_winner_time_is_never_an_individual_trial_measurement(self):
        for values in ({"distance_metres": 1200, "batch_winner_time_seconds": 70},
                       {"distance_metres": 1200, "time": 70, "time_unit": "s", "time_basis": "batch_winner"}):
            with self.subTest(values=values):
                result = normalize_speed_measurement(values, "barrier_trials")
                self.assertEqual(result[3], "individual_time_missing")
                features = build_speed_features(pd.DataFrame([runner()]), [timed_event(values=values)]).iloc[0]
                self.assertTrue(pd.isna(features.trial_speed_last_mps))
                self.assertEqual(features.trial_speed_support, 0)

    def test_individual_trial_time_wins_over_different_batch_winner_time(self):
        raw = timed_event(values={"distance_metres": 1200, "trial_finish_seconds": 75,
                                  "batch_winner_time_seconds": 60})
        result = build_speed_features(pd.DataFrame([runner()]), [raw]).iloc[0]
        self.assertEqual(result.trial_speed_last_mps, 16)
        self.assertEqual(result.trial_speed_support, 1)

    def test_race_workout_and_trial_speeds_remain_distinct_families(self):
        frame = pd.DataFrame([runner("2026-09-01", race_id="race-prior", finish_time="1.00.00"), runner()])
        trial = timed_event()
        workout = timed_event("trackwork", values={"distance_metres": 800, "workout_finish_seconds": 50})
        result = build_speed_features(frame, [trial, workout]).iloc[1]
        self.assertEqual(result.past_race_speed_last_mps, 20)
        self.assertEqual(result.trial_speed_last_mps, 16)
        self.assertEqual(result.workout_speed_last_mps, 16)
        self.assertEqual(result.past_race_speed_support, 1)
        self.assertEqual(result.trial_speed_support, 1)
        self.assertEqual(result.workout_speed_support, 1)

    def test_current_and_future_race_outcomes_cannot_change_current_features(self):
        prior = runner("2026-09-01", race_id="race-prior", finish_time="1.00.00")
        current = runner()
        future = runner("2026-09-20", race_id="race-future", finish_time="0.01.00")
        original = build_speed_features(pd.DataFrame([prior, current])).iloc[1]
        changed = build_speed_features(pd.DataFrame([prior, current | {"finish_time": "0.01.00"}, future])).iloc[1]
        pd.testing.assert_series_equal(original, changed)
        first = build_speed_features(pd.DataFrame([current])).iloc[0]
        self.assertTrue(pd.isna(first.past_race_speed_last_mps))
        self.assertEqual(first.past_race_speed_support, 0)
        self.assertTrue(pd.isna(first.past_race_speed_std_5_mps))

    def test_date_only_past_race_is_unavailable_at_exact_next_day_boundary(self):
        frame = pd.DataFrame([runner("2026-09-01", race_id="race-prior", finish_time="1.00.00"),
                              runner("2026-09-02", cutoff_at="2026-09-02T00:00:00+08:00")])
        frame.loc[0, "cutoff_at"] = "2026-09-01T00:00:00+08:00"
        self.assertEqual(build_speed_features(frame).iloc[1].past_race_speed_support, 0)
        frame.loc[1, "cutoff_at"] = "2026-09-02T00:00:01+08:00"
        self.assertEqual(build_speed_features(frame).iloc[1].past_race_speed_last_mps, 20)

    def test_public_capture_controls_historical_event_speed_eligibility(self):
        frame = pd.DataFrame([runner(), runner("2026-10-03", race_id="race-later")])
        raw = timed_event(first_seen_at="2026-10-02T08:00:00+08:00")
        result = build_speed_features(frame, [raw])
        self.assertTrue(pd.isna(result.iloc[0].trial_speed_last_mps))
        self.assertEqual(result.iloc[1].trial_speed_last_mps, 16)
        unknown = raw | {"first_seen_at": None}
        result = build_speed_features(frame, [unknown])
        self.assertTrue(result.trial_speed_last_mps.isna().all())

    def test_duplicated_capture_does_not_inflate_speed_support(self):
        raw = timed_event()
        frame = pd.DataFrame([runner()], index=[71])
        single = build_speed_features(frame, [raw])
        repeated = build_speed_features(frame, [raw, dict(raw)])
        pd.testing.assert_frame_equal(single, repeated)
        self.assertEqual(repeated.index.tolist(), [71])
        self.assertEqual(repeated.iloc[0].trial_speed_support, 1)
        self.assertTrue(pd.isna(repeated.iloc[0].trial_speed_std_5_mps))

    def test_distinct_timed_observations_have_independent_support_and_dispersion(self):
        first = timed_event()
        second = timed_event(event_date="2026-09-03", first_seen_at="2026-09-04T08:00:00+08:00",
                             table_index=2, values={"distance_metres": 1200, "trial_finish_seconds": 60})
        result = build_speed_features(pd.DataFrame([runner()]), [first, second]).iloc[0]
        self.assertEqual(result.trial_speed_support, 2)
        self.assertEqual(result.trial_speed_last_mps, 20)
        self.assertEqual(result.trial_speed_mean_3_mps, 18)
        self.assertEqual(result.trial_speed_trend_mps, 4)
        self.assertAlmostEqual(result.trial_speed_std_5_mps, np.sqrt(8))

    def test_match_features_require_matching_distance_surface_and_venue(self):
        trial = timed_event()
        matching = build_speed_features(pd.DataFrame([runner()]), [trial]).iloc[0]
        self.assertEqual(matching.trial_speed_same_distance_mean_mps, 16)
        self.assertEqual(matching.trial_speed_same_surface_mean_mps, 16)
        self.assertEqual(matching.trial_speed_same_venue_mean_mps, 16)
        different = build_speed_features(pd.DataFrame([runner(distance=1400, surface="AWT", venue="HV")]), [trial]).iloc[0]
        self.assertTrue(pd.isna(different.trial_speed_same_distance_mean_mps))
        self.assertTrue(pd.isna(different.trial_speed_same_surface_mean_mps))
        self.assertTrue(pd.isna(different.trial_speed_same_venue_mean_mps))
        self.assertEqual(different.trial_speed_last_mps, 16)

    def test_catalog_names_are_predictors_with_source_specific_speed_units(self):
        catalog = speed_feature_catalog()
        features = build_speed_features(pd.DataFrame([runner()]))
        self.assertEqual(set(catalog), set(features))
        self.assertNotIn("finish_seconds", catalog)
        for name, metadata in catalog.items():
            self.assertEqual(metadata["cutoff_rule"], "occurrence_and_availability_strictly_before_cutoff")
            self.assertFalse(metadata["fit_required"])
            if name.endswith("mps"):
                self.assertEqual(metadata["unit"], "m/s")


if __name__ == "__main__":
    unittest.main()
