import unittest
import pandas as pd

from scripts.build_official_dataset import race_rows, merge_form, merge_country, resolve_workout_identities
from ima.rich_features import prepare_rich_runner_dataset


class OfficialDatasetTests(unittest.TestCase):
    def test_country_is_immutable_attributed_and_conflicts_remain_unknown(self):
        countries = {}
        source = {"source_url": "official-country", "body_hash": "a" * 64}
        merge_country(countries, "HK_2000_A018", "GB", source)
        self.assertEqual(countries["HK_2000_A018"]["country"], "GB")
        self.assertEqual(countries["HK_2000_A018"]["source_body_hash"], "a" * 64)
        merge_country(countries, "HK_2000_A018", "AUS", source)
        merge_country(countries, "HK_2000_A018", "GB", source)
        self.assertIsNone(countries["HK_2000_A018"]["country"])
    def test_workout_identity_requires_independent_compound_confirmation(self):
        values = {"Horse": "THUNDER KIT", "Type": "Trotting", "Racecourse_Track": "Sha Tin SmT",
                  "Workouts": "SmT 1 Round", "Gear": "H"}
        daily = {"family": "trackwork", "horse_id": None, "event_date": "2026-10-01", "values": values}
        confirmed = {"family": "trackwork", "horse_id": "HK_2025_L121", "horse_name": "THUNDER KIT",
                     "event_date": "2026-10-01", "values": values,
                     "identity_evidence": "requested_full_id_and_displayed_brand", "source_url": "official-horse-workout",
                     "source_body_hash": "a" * 64}
        self.assertEqual(resolve_workout_identities([daily, confirmed]), 1)
        self.assertEqual(daily["horse_id"], "HK_2025_L121")
        self.assertEqual(daily["identity_confirmation"]["source_body_hash"], "a" * 64)
        for alternative in (confirmed | {"event_date": "2026-10-02"},
                            confirmed | {"identity_evidence": "name_only"},
                            confirmed | {"values": values | {"Gear": "B"}}):
            candidate = daily | {"horse_id": None}
            self.assertEqual(resolve_workout_identities([candidate, alternative]), 0)
            self.assertIsNone(candidate["horse_id"])
        candidate = daily | {"horse_id": None}
        self.assertEqual(resolve_workout_identities([candidate, confirmed, confirmed | {"horse_id": "HK_2024_K999"}]), 0)
        self.assertEqual(candidate["identity_status"], "ambiguous_official_confirmation")
    def document(self):
        return {"status": "fetched_parsed", "source_url": "https://racing.hkjc.com/en-us/local/information/localresults",
            "body_hash": "a" * 64, "tables": [[["Pla.", "Horse No."] + ["x"] * 10,
                ["1", "1"] + ["x"] * 10]], "race": {"race_date": "2025-07-13", "venue": "ST",
                "race_no": 1, "distance": 1200, "course": "TURF", "going": "GOOD",
                "race_class": "Class 4", "prize": 1000, "runners": [{"horse_page_id": "HK_2022_H033",
                    "horse_no": 1, "horse_name": "HORSE", "place": 1, "win_odds": 3.0,
                    "finish_time": "1:10.00", "actual_weight": 120, "declared_weight": 1000,
                    "draw": 1, "lengths_behind": "---", "running_position": "1 1 1",
                    "jockey": "JOCKEY", "trainer": "TRAINER"}]}}

    def test_full_identity_and_raw_lineage_preserved(self):
        rows = race_rows(self.document())
        self.assertEqual(rows[0]["horse_id"], "HK_2022_H033")
        self.assertEqual(rows[0]["source_body_hash"], "a" * 64)
        self.assertIsNone(rows[0]["horse_rating"])

    def test_incomplete_field_is_quarantined(self):
        document = self.document()
        document["tables"][0].append(["DNF", "2"] + ["x"] * 10)
        with self.assertRaisesRegex(ValueError, "Incomplete runner"):
            race_rows(document)

    def test_brand_only_identity_is_not_guessed(self):
        document = self.document()
        document["race"]["runners"][0]["horse_page_id"] = "H033"
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            race_rows(document)

    def test_unparsed_race_cannot_enter_dataset(self):
        document = self.document()
        document["status"] = "fetched_unparsed"
        with self.assertRaises(ValueError):
            race_rows(document)

    def test_known_nonfinisher_stays_in_probability_field_without_fake_rank(self):
        document = self.document()
        runner = document["race"]["runners"][0].copy()
        runner.update(horse_no=2, horse_page_id="HK_2022_H034", place=None, finishing_status="PU", finish_time=None)
        document["race"]["runners"].append(runner)
        document["tables"][0].append(["PU", "2"] + ["x"] * 10)
        features = prepare_rich_runner_dataset(pd.DataFrame(race_rows(document)))
        self.assertEqual(len(features), 2)
        self.assertEqual(features["target_win"].tolist(), [1, 0])
        self.assertTrue(pd.isna(features.iloc[1]["result"]))
        self.assertEqual(features["field_size"].tolist(), [2.0, 2.0])
        self.assertAlmostEqual(features["market_probability"].sum(), 1.0)

    def test_rating_conflict_stays_quarantined(self):
        ratings = {}
        source = {"source_url": "official-url", "body_hash": "a" * 64}
        for rating in (50, 51, 50):
            merge_form(ratings, "HK_2022_H033", {"date": "01/10/2026", "rating": rating}, source)
        self.assertTrue(ratings[("HK_2022_H033", "2026-10-01")]["conflict"])


if __name__ == "__main__":
    unittest.main()
