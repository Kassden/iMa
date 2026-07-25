import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ima.historical_sources import (
    CANONICAL_COLUMNS, extract_horse_identity, iter_mexwell_odds, normalize_mexwell_dividends,
    normalize_official_archive, reconcile_sources, validate_canonical,
)


class HistoricalSourceTests(unittest.TestCase):
    def row(self, source, odds):
        row = {column: None for column in CANONICAL_COLUMNS}
        row.update({
            "race_date": pd.Timestamp("2020-01-01"), "venue": "ST", "race_no": 1,
            "horse_no": 1, "result": 1, "win_odds": odds, "source": source,
        })
        return row

    def test_official_source_wins_conflicts(self):
        third = pd.DataFrame([self.row("kaggle:mexwell-hkjc", 5.0)])
        official = pd.DataFrame([self.row("official:hkjc-results", 4.8)])
        canonical, conflicts = reconcile_sources([third, official])
        self.assertEqual(4.8, canonical.iloc[0]["win_odds"])
        self.assertEqual(2, len(conflicts))
        self.assertEqual(1, validate_canonical(canonical)["races"])

    def test_horse_id_reconciles_sources_even_when_horse_number_is_missing(self):
        first = self.row("kaggle:mexwell-hkjc", 5.0)
        first.update({"race_date": pd.Timestamp("2009-01-01"), "horse_no": None, "horse_id": "H123"})
        second = self.row("third-party:swords-2008-2009", 4.8)
        second.update({"race_date": pd.Timestamp("2009-01-01"), "horse_no": 7, "horse_id": "H123"})
        canonical, conflicts = reconcile_sources([pd.DataFrame([first]), pd.DataFrame([second])])
        self.assertEqual(1, len(canonical))
        self.assertEqual(7, canonical.iloc[0]["horse_no"])
        self.assertEqual(2, len(conflicts))

    def test_mexwell_market_tables_are_exploded_and_labeled(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame([{
                "race_date": "2017-01-01", "race_location": "H", "race_country": "HK",
                "race_no": 1, "data": '{"win":{"1":"4.2"},"pla":{"1":"1.8"}}',
                "capture_time": "2017-01-01 04:00:00", "last_updated": "2017-01-01 04:00:01",
            }]).to_csv(root / "odds.csv", index=False)
            pd.DataFrame([{
                "race_date": "2017-01-01", "race_no": 1, "race_country": "HK",
                "dividends": (
                    '{"quinella":[{"combination":[1,2],"dividend":42.5},'
                    '{"combination":["-"],"dividend":10.0}]}'
                ),
                "last_updated": "2017-01-01 12:00:00",
            }]).to_csv(root / "dividends.csv", index=False)
            pd.DataFrame([{
                "race_date": "2017-01-01", "race_no": 1, "race_country": "HK",
                "race_location": "H",
            }]).to_csv(root / "races.csv", index=False)
            odds = pd.concat(iter_mexwell_odds(root / "odds.csv"), ignore_index=True)
            dividends = normalize_mexwell_dividends(root / "dividends.csv", root / "races.csv")
        self.assertEqual({"WIN", "PLACE"}, set(odds["pool"]))
        self.assertEqual("HV", odds.iloc[0]["venue"])
        self.assertEqual("QIN", dividends.iloc[0]["pool"])
        self.assertEqual({"1-2", "-"}, set(dividends["combination_key"]))

    def test_archive_horse_code_prefix_is_removed(self):
        identity = extract_horse_identity(pd.Series(["WAIT FOR ME(CE332)", "BATURO(H029)"]))
        self.assertEqual(["E332", "H029"], identity[1].tolist())
        self.assertEqual(["WAIT FOR ME", "BATURO"], identity[0].tolist())

    def test_official_normalization_keeps_race_metadata_and_runner_context(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "normalized/2020-01-01"
            destination.mkdir(parents=True)
            payload = [{
                "race_date": "2020-01-01", "venue": "ST", "race_no": 1,
                "race_class": "Class 3", "distance": 1400, "prize": 1860000,
                "going": "GOOD", "course": 'TURF - "A" Course',
                "runners": [{
                    "place": 1, "horse_no": 4, "horse_name": "SIGHT DREAMER",
                    "horse_code": "J542", "jockey": "A Atzeni", "trainer": "J Size",
                    "actual_weight": 134, "declared_weight": 1306, "draw": 8,
                    "lengths_behind": "---", "running_position": "5 4 1 1",
                    "finish_time": "1:21.99", "win_odds": 8.4,
                }],
            }]
            (destination / "ST.json").write_text(
                __import__("json").dumps(payload), encoding="utf-8"
            )
            frame = normalize_official_archive(root)
        self.assertEqual(1400, frame.iloc[0]["distance"])
        self.assertEqual("Class 3", frame.iloc[0]["race_class"])
        self.assertEqual("5 4 1 1", frame.iloc[0]["running_position"])


if __name__ == "__main__":
    unittest.main()