import unittest

import pandas as pd

from ima.data import FEATURES
from ima.feature_sets import (
    BASELINE_SCHEMA, BENTER_COVERAGE, FEATURE_FAMILIES, NOTEBOOK_FEATURE_FAMILIES,
    NOTEBOOK_RICH_SCHEMA, RICH_SCHEMA, validate_feature_contract,
)
from ima.rich_features import parse_lengths, prepare_rich_runner_dataset


class RichFeatureContractTests(unittest.TestCase):
    @staticmethod
    def source(second_winner: str = "A") -> pd.DataFrame:
        rows = []
        for race_id, date, results in (
            ("R1", "2020-01-01", {"A": 1, "B": 2}),
            ("R2", "2020-01-10", {second_winner: 1, "B" if second_winner == "A" else "A": 2}),
        ):
            for number, horse in enumerate(("A", "B"), start=1):
                result = results[horse]
                rows.append({
                    "date": date, "race_id": race_id, "race_no": int(race_id[-1]),
                    "horse_no": number, "horse_id": horse, "result": result,
                    "win_odds": 2.5 + number, "actual_weight": 120 + number,
                    "declared_weight": 1000 + 10 * number, "draw": number,
                    "finish_time": "1:20.00" if result == 1 else "1:20.50",
                    "lengths_behind": "---" if result == 1 else "1-1/2",
                    "running_position": f"{number + 2} {result}", "horse_rating": 60,
                    "distance": 1400, "race_class": "Class 3", "prize": 1000000,
                    "venue": "ST", "course": 'TURF - "A" Course', "going": "GOOD",
                    "jockey_id": "J1", "trainer_id": f"T{number}",
                    "source": "official:hkjc-results",
                })
        return pd.DataFrame(rows)

    def test_baseline_schema_remains_compatible(self):
        self.assertEqual(tuple(FEATURES), BASELINE_SCHEMA.features)

    def test_rich_contract_is_unique_and_family_complete(self):
        validate_feature_contract()
        assigned = {feature for values in FEATURE_FAMILIES.values() for feature in values}
        self.assertEqual(set(RICH_SCHEMA.features), assigned)
        self.assertGreater(len(RICH_SCHEMA.features), 60)
        self.assertGreater(len(NOTEBOOK_RICH_SCHEMA.features), len(RICH_SCHEMA.features))
        self.assertEqual(len(NOTEBOOK_RICH_SCHEMA.features), len(set(NOTEBOOK_RICH_SCHEMA.features)))
        notebook_assigned = [
            feature for values in NOTEBOOK_FEATURE_FAMILIES.values() for feature in values
        ]
        self.assertEqual(set(NOTEBOOK_RICH_SCHEMA.features), set(notebook_assigned))
        self.assertEqual(len(notebook_assigned), len(set(notebook_assigned)))

    def test_benter_coverage_discloses_unsupported_factors(self):
        coverage = {row["factor"]: row["status"] for row in BENTER_COVERAGE}
        self.assertEqual("partial", coverage["horse age"])
        self.assertEqual("unsupported", coverage["bad luck adjustment"])
        self.assertEqual("supported", coverage["lengths behind winner"])

    def test_lengths_parser_handles_hkjc_fractions(self):
        self.assertEqual(1.5, parse_lengths("1-1/2"))
        self.assertEqual(0.3, parse_lengths("NECK"))

    def test_current_race_outcome_cannot_change_current_features(self):
        first = prepare_rich_runner_dataset(self.source("A"))
        second = prepare_rich_runner_dataset(self.source("B"))
        columns = [
            "prior_starts", "prior_win_rate", "last_result", "jockey_starts",
            "jockey_win_rate", "draw_bias_starts", "distance_band_starts",
            "prior_second_count", "prior_third_count", "prior_second_rate",
            "prior_third_rate", "avg_result_6", "avg_result_183d",
        ]
        pd.testing.assert_frame_equal(
            first[first.race_id.eq("R2")][columns].reset_index(drop=True),
            second[second.race_id.eq("R2")][columns].reset_index(drop=True),
        )

    def test_entity_history_excludes_all_runners_in_current_race(self):
        frame = prepare_rich_runner_dataset(self.source())
        self.assertEqual([0.0, 0.0], frame[frame.race_id.eq("R1")]["jockey_starts"].tolist())
        self.assertEqual([2.0, 2.0], frame[frame.race_id.eq("R2")]["jockey_starts"].tolist())
        self.assertEqual([0.5, 0.5], frame[frame.race_id.eq("R2")]["jockey_win_rate"].tolist())

    def test_auxiliary_events_are_strictly_before_race_date(self):
        trackwork = pd.DataFrame({
            "horse_id": ["A", "A"], "event_date": ["2020-01-09", "2020-01-10"],
        })
        frame = prepare_rich_runner_dataset(self.source(), trackwork=trackwork)
        runner = frame[(frame.race_id.eq("R2")) & (frame.horse_id.eq("A"))].iloc[0]
        self.assertEqual(1.0, runner["trackwork_7d"])
        self.assertEqual(1.0, runner["days_since_trackwork"])

    def test_notebook_features_use_prior_results_and_named_polynomials(self):
        frame = prepare_rich_runner_dataset(self.source())
        runner = frame[(frame.race_id.eq("R2")) & (frame.horse_id.eq("A"))].iloc[0]
        self.assertEqual(0.0, runner["prior_second_count"])
        self.assertEqual(0.0, runner["prior_third_count"])
        self.assertEqual(1.0, runner["avg_result_6"])
        self.assertEqual(1.0, runner["avg_result_183d"])
        self.assertEqual(runner["horse_rating"] ** 2, runner["poly_rating_sq"])
        self.assertEqual(
            runner["horse_rating"] * runner["prior_win_rate"],
            runner["poly_rating_prior_win"],
        )

    def test_profile_values_are_asof_and_not_backfilled_from_future(self):
        profiles = pd.DataFrame({
            "horse_id": ["A", "A"],
            "snapshot_at": ["2019-12-01", "2020-01-10"],
            "season_stake": [1000, 999999], "total_stake": [5000, 999999],
            "start_of_season_rating": [55, 99], "no_of_start_past_10_meetings": [2, 9],
            "colour": ["Bay", "Future"], "import_type": ["PPG", "Future"],
            "sire": ["S1", "Future"], "dam_sire": ["D1", "Future"],
        })
        frame = prepare_rich_runner_dataset(self.source(), profiles=profiles)
        runner = frame[(frame.race_id.eq("R2")) & (frame.horse_id.eq("A"))].iloc[0]
        self.assertEqual(1000, runner["season_stakes"])
        self.assertEqual("PPG", runner["import_type"])
        self.assertEqual("Bay", runner["horse_colour"])

    def test_veterinary_and_movement_features_exclude_race_day_events(self):
        veterinary = pd.DataFrame({
            "horse_id": ["A", "A"],
            "event_date": ["2020-01-09", "2020-01-10"],
            "details": ["minor injury", "surgery"],
        })
        movements = pd.DataFrame({
            "horse_id": ["A"], "arrival_date": ["2020-01-08"],
            "from": ["Conghua"], "to": ["Hong Kong"],
        })
        frame = prepare_rich_runner_dataset(
            self.source(), veterinary=veterinary, movements=movements,
        )
        runner = frame[(frame.race_id.eq("R2")) & (frame.horse_id.eq("A"))].iloc[0]
        self.assertEqual(1.0, runner["veterinary_events_30d"])
        self.assertEqual(1.0, runner["injury_events_365d"])
        self.assertEqual(0.0, runner["surgery_events_365d"])
        self.assertEqual(2.0, runner["days_since_hk_arrival"])


if __name__ == "__main__":
    unittest.main()
