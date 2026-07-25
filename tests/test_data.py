from pathlib import Path
import unittest

from ima.data import (
    build_canonical_runner_dataset, build_full_history_dataset, build_runner_dataset,
    chronological_race_split,
    validate_runner_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


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

    def test_canonical_archive_builds_point_in_time_features(self):
        frame = build_canonical_runner_dataset(
            ROOT / "data/processed/historical/runners.csv.gz"
        )
        validate_runner_dataset(frame)
        self.assertEqual(list(range(2005, 2026)), sorted(frame["date"].dt.year.unique()))
        self.assertTrue(frame.groupby("race_id")["market_probability"].sum().sub(1).abs().lt(1e-9).all())
        first_starts = frame.groupby("horse_id").head(1)
        self.assertTrue(first_starts["prior_starts"].eq(0).all())

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


if __name__ == "__main__":
    unittest.main()
