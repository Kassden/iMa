import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.backfill_horse_profiles import discover_horse_page_ids, fetch_one, write_tables


class HorseProfileBackfillTests(unittest.TestCase):
    def test_discovers_unique_profile_ids_from_compressed_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "2025-01-01/ST/R1.html.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(
                    '<a href="/en-us/local/information/horse?horseid=HK_2020_A001">A</a>'
                    '<a href="horse?horseid=HK_2021_B002&Option=1">B</a>'
                    '<a href="horse?horseid=HK_2020_A001">A again</a>'
                )
            horse_ids = discover_horse_page_ids(root)
        self.assertEqual(["HK_2020_A001", "HK_2021_B002"], horse_ids)

    def test_existing_raw_and_normalized_profile_is_resumable(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            raw = output / "raw/HK_2020_A001.html.gz"
            normalized = output / "normalized/HK_2020_A001.json"
            raw.parent.mkdir(parents=True)
            normalized.parent.mkdir(parents=True)
            raw.write_bytes(b"existing")
            normalized.write_text("{}", encoding="utf-8")
            horse_id, error = fetch_one("HK_2020_A001", output, retries=0)
        self.assertEqual("HK_2020_A001", horse_id)
        self.assertIsNone(error)

    def test_normalized_profiles_generate_profile_and_form_tables(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "profiles"
            processed = root / "processed"
            normalized = output / "normalized/HK_2020_A001.json"
            normalized.parent.mkdir(parents=True)
            normalized.write_text(json.dumps({
                "horse_page_id": "HK_2020_A001",
                "brand_code": "A001",
                "horse_name": "TEST HORSE",
                "country": "AUS",
                "age": 8,
                "colour": "Bay",
                "sex": "Gelding",
                "fetched_at": "2026-07-26T12:00:00+08:00",
                "source_urls": {"profile_and_form": "https://example.test/horse"},
                "form_records": [{
                    "date": "01/01/2023", "gear": "B/TT", "rating": 80,
                    "declared_weight": 1100, "actual_weight": 125,
                }],
            }), encoding="utf-8")
            report = write_tables(output, processed)
            profiles = pd.read_csv(processed / "horse-profiles.csv.gz")
            form = pd.read_csv(processed / "horse-form.csv.gz")
        self.assertEqual({"profiles": 1, "form_records": 1}, report)
        self.assertEqual(2026, profiles.iloc[0]["capture_year"])
        self.assertEqual("B/TT", form.iloc[0]["horse_gear"])


if __name__ == "__main__":
    unittest.main()
