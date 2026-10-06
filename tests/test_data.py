from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from ima.data import (
    build_canonical_runner_dataset, build_full_history_dataset, build_runner_dataset,
    chronological_race_split,
    validate_runner_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ARCHIVE = ROOT / "data/processed/historical/runners.csv.gz"
ARCHIVE_REASON = (
    "Integration test requires the gitignored canonical historical archive "
    "data/processed/historical/runners.csv.gz; synthetic CSV coverage runs in clean checkouts"
)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = build_runner_dataset(
            ROOT / "track/hkracing 2/runs.csv",
            ROOT / "track/hkracing 2/races.csv",
        )

    def test_dataset_invariants(self):
        validate_runner_dataset(self.frame)
        first_starts = self.frame.groupby("horse_id").head(1)
        self.assertTrue(first_starts["prior_starts"].eq(0).all())
        self.assertTrue(first_starts["last_result"].isna().all())

    def test_split_is_race_grouped_and_chronological(self):
        splits = chronological_race_split(self.frame)
        sets = [set(part["race_id"]) for part in (splits.train, splits.validation, splits.test)]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[1] & sets[2])
        self.assertLessEqual(splits.train["date"].max(), splits.validation["date"].min())
        self.assertLessEqual(splits.validation["date"].max(), splits.test["date"].min())

    @unittest.skipUnless(CANONICAL_ARCHIVE.is_file(), ARCHIVE_REASON)
    def test_canonical_archive_builds_point_in_time_features(self):
        frame = build_canonical_runner_dataset(
            ROOT / "data/processed/historical/runners.csv.gz"
        )
        validate_runner_dataset(frame)
        self.assertEqual(list(range(2005, 2026)), sorted(frame["date"].dt.year.unique()))
        self.assertTrue(frame.groupby("race_id")["market_probability"].sum().sub(1).abs().lt(1e-9).all())
        first_starts = frame.groupby("horse_id").head(1)
        self.assertTrue(first_starts["prior_starts"].eq(0).all())
        self.assertGreater(frame["prize"].notna().mean(), 0.80)
        self.assertTrue(set(frame["surface"].dropna().unique()).issubset({0.0, 1.0}))
        self.assertIn(1.0, set(frame["surface"].dropna().unique()))

    @unittest.skipUnless(CANONICAL_ARCHIVE.is_file(), ARCHIVE_REASON)
    def test_full_history_has_no_2005_overlap(self):
        frame = build_full_history_dataset(
            ROOT / "track/hkracing 2/runs.csv",
            ROOT / "track/hkracing 2/races.csv",
            ROOT / "data/processed/historical/runners.csv.gz",
        )
        validate_runner_dataset(frame)
        self.assertEqual("1997-06-02", frame["date"].min().date().isoformat())
        self.assertEqual("2025-12-27", frame["date"].max().date().isoformat())
        rows_2005 = frame[frame["date"].dt.year.eq(2005)]
        self.assertTrue(rows_2005["race_id"].str.startswith("canonical:").all())
        self.assertTrue(frame[frame["date"].dt.year.lt(2005)]["race_id"].str.startswith("legacy:").all())


class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.canonical_path = self.directory / "runners.csv"
        rows = []
        dates = ("2005-01-02", "2006-01-02", "2007-01-02")
        for index, date in enumerate(dates):
            for number, horse in enumerate(("A", "B"), start=1):
                rows.append({
                    "race_date": date, "venue": "ST", "race_no": 1,
                    "horse_id": horse, "horse_no": number,
                    "result": ((1, 2), (2, 1), (1, 2))[index][number - 1],
                    "win_odds": ((2, 4), (6, 2), (3, 6))[index][number - 1],
                    "finish_time": (("1.10.00", "1:11.50"), ("1:12.00", "1.11.00"),
                                    ("1.10.50", "1:13.00"))[index][number - 1],
                    "horse_age": 4 + index, "rating": 60,
                    "declared_weight": 1100, "actual_weight": 120,
                    "draw": number, "distance": 1200, "race_class": "Class 4",
                    "course": ("TURF", "AWT", "ALL WEATHER TRACK")[index],
                    "prize": 1000000, "going": "GOOD", "gear": "B",
                })
        self.canonical_source = pd.DataFrame(rows)
        self.canonical_source.iloc[::-1].to_csv(self.canonical_path, index=False)

    def test_canonical_csv_builds_only_previous_start_histories(self):
        frame = build_canonical_runner_dataset(self.canonical_path)
        validate_runner_dataset(frame)
        first = frame.groupby("horse_id").head(1)
        self.assertTrue(first.prior_starts.eq(0).all())
        self.assertTrue(first.last_result.isna().all())
        self.assertTrue(first.last_finish_time.isna().all())
        horse = frame[frame.horse_id.eq("A")].reset_index(drop=True)
        self.assertEqual(horse.prior_starts.tolist(), [0, 1, 2])
        self.assertEqual(horse.last_result.iloc[1:].tolist(), [1, 2])
        self.assertEqual(horse.last_win_odds.iloc[1:].tolist(), [2, 6])
        self.assertEqual(horse.last_finish_time.iloc[1:].tolist(), [70, 72])
        self.assertEqual(horse.prior_win_rate.iloc[1:].tolist(), [1, .5])
        self.assertEqual(horse.prior_avg_result.iloc[1:].tolist(), [1, 1.5])
        self.assertEqual(horse.prior_avg_odds.iloc[1:].tolist(), [2, 4])

    def test_canonical_csv_preserves_market_surface_and_time_semantics(self):
        frame = build_canonical_runner_dataset(self.canonical_path)
        validate_runner_dataset(frame)
        race = frame[frame.date.eq(pd.Timestamp("2005-01-02"))].sort_values("horse_no")
        self.assertEqual(race.finish_time.tolist(), [70, 71.5])
        self.assertAlmostEqual(race.iloc[0].market_probability, 2 / 3)
        self.assertAlmostEqual(race.iloc[1].market_probability, 1 / 3)
        self.assertTrue(frame.groupby("race_id").market_probability.sum().sub(1).abs().lt(1e-12).all())
        self.assertEqual(frame.groupby("date").surface.first().tolist(), [0, 1, 1])
        self.assertTrue(frame.field_size.eq(2).all())

    def test_future_outcome_changes_leave_earlier_canonical_histories_unchanged(self):
        original = build_canonical_runner_dataset(self.canonical_path)
        changed = self.canonical_source.copy()
        last_race = changed.race_date.eq("2007-01-02")
        changed.loc[last_race, "result"] = [2, 1]
        changed.loc[last_race, "win_odds"] = [20, 2]
        changed.loc[last_race, "finish_time"] = ["1:20.00", "1:15.00"]
        changed.to_csv(self.canonical_path, index=False)
        rebuilt = build_canonical_runner_dataset(self.canonical_path)
        history = ["prior_starts", "prior_win_rate", "prior_avg_result", "prior_avg_odds",
                   "last_result", "last_win_odds", "last_finish_time"]
        pd.testing.assert_frame_equal(original[history], rebuilt[history])
        earlier = original.date.lt(pd.Timestamp("2007-01-02"))
        pd.testing.assert_frame_equal(original.loc[earlier], rebuilt.loc[earlier])

    def test_full_history_csv_excludes_legacy_2005_overlap_and_namespaces_horses(self):
        runs_path = self.directory / "legacy_runs.csv"
        races_path = self.directory / "legacy_races.csv"
        runs, races = [], []
        for race_id, date in ((1, "2004-12-01"), (2, "2005-01-02")):
            races.append({"race_id": race_id, "date": date, "race_no": 1, "venue": "ST"})
            for number, horse in enumerate(("A", "B"), start=1):
                runs.append({"race_id": race_id, "horse_id": horse, "horse_no": number,
                             "horse_age": 3, "won": int(number == 1), "result": number,
                             "win_odds": 2 * number, "finish_time": 69 + number})
        pd.DataFrame(runs).to_csv(runs_path, index=False)
        pd.DataFrame(races).to_csv(races_path, index=False)
        frame = build_full_history_dataset(runs_path, races_path, self.canonical_path)
        validate_runner_dataset(frame)
        self.assertEqual(len(frame), 8)
        self.assertEqual(frame.race_id.nunique(), 4)
        self.assertEqual(frame.date.min(), pd.Timestamp("2004-12-01"))
        self.assertEqual(frame.date.max(), pd.Timestamp("2007-01-02"))
        older = frame[frame.date.dt.year.lt(2005)]
        newer = frame[frame.date.dt.year.ge(2005)]
        self.assertEqual(len(older), 2)
        self.assertTrue(older.race_id.str.startswith("legacy:").all())
        self.assertTrue(older.horse_id.str.startswith("legacy:").all())
        self.assertTrue(newer.race_id.str.startswith("canonical:").all())
        self.assertTrue(newer.horse_id.str.startswith("canonical:").all())
        self.assertEqual(len(newer[newer.date.dt.year.eq(2005)]), 2)
        self.assertNotIn("legacy:2", set(frame.race_id))
        first_canonical = newer.groupby("horse_id").head(1)
        self.assertTrue(first_canonical.prior_starts.eq(0).all())
        self.assertTrue(first_canonical.last_result.isna().all())


if __name__ == "__main__":
    unittest.main()
