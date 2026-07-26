from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scrapper.contracts import MODEL_FEATURE_COLUMNS
from scrapper.browser_odds import parse_rendered_odds
from scrapper.history import HistoricalFeatureStore
from scrapper.horse_pages import parse_profile_and_form, table_rows
from scrapper.pipeline import build_model_rows, normalize_snapshot, write_snapshot


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "race_snapshot.json"
RENDERED_FIXTURE = Path(__file__).parent / "fixtures" / "rendered_odds.html"


class ScraperTests(unittest.TestCase):
    def payload(self):
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_rendered_browser_odds_map_to_existing_contract(self):
        payload = parse_rendered_odds(
            RENDERED_FIXTURE.read_text(encoding="utf-8"),
            1,
            received_at="2026-07-25T00:00:00+00:00",
        )
        runners = normalize_snapshot(payload, 1)
        self.assertEqual(2, len(runners))
        self.assertEqual("HK_2023_J534", runners[0].horse_page_id)
        self.assertEqual(4.2, runners[0].win_odds)
        self.assertEqual(2.6, runners[1].place_odds)
        self.assertEqual(["QIN", "TRI", "WIN"], payload["provider"]["available_pool_ids"])
        self.assertEqual(228742.0, payload["provider"]["pool_turnover"]["QIN"])

    def test_odds_are_mapped_by_type_and_runner_number(self):
        runners = normalize_snapshot(self.payload(), 1, "2026-07-25T10:00:00+00:00")
        self.assertEqual(2, len(runners))
        self.assertEqual(9.7, runners[0].win_odds)
        self.assertEqual(3.7, runners[0].place_odds)
        self.assertEqual("2005-08-29T12:55:03+08:00", runners[0].odds_updated_at)

    def test_history_enrichment_matches_model_columns(self):
        history = HistoricalFeatureStore.from_csv(
            ROOT / "track/hkracing 2/runs.csv",
            ROOT / "track/hkracing 2/races.csv",
        )
        runners = normalize_snapshot(self.payload(), 1, "2026-07-25T10:00:00+00:00")
        rows = build_model_rows(runners, history)
        self.assertEqual(set(MODEL_FEATURE_COLUMNS), set(rows[0].values))
        self.assertEqual(1, rows[0].values["raced"])
        self.assertGreater(rows[0].values["exp"], 0)
        self.assertFalse(rows[1].prediction_ready)

    def test_missing_current_history_is_reported_not_zero_filled(self):
        history = HistoricalFeatureStore.from_csv(
            ROOT / "track/hkracing 2/runs.csv",
            ROOT / "track/hkracing 2/races.csv",
        )
        rows = build_model_rows(normalize_snapshot(self.payload(), 1), history)
        current = rows[1]
        self.assertIsNone(current.values["prev_resu"])
        self.assertIn("prev_resu", current.missing_features)
        self.assertFalse(current.prediction_ready)

    def test_snapshot_writes_raw_normalized_model_and_report_files(self):
        runners = normalize_snapshot(self.payload(), 1, "2026-07-25T10:00:00+00:00")
        rows = build_model_rows(runners)
        with tempfile.TemporaryDirectory() as directory:
            result = write_snapshot(self.payload(), runners, rows, Path(directory))
            self.assertTrue(result.raw_path.exists())
            self.assertTrue(result.runners_path.exists())
            self.assertTrue(result.model_path.exists())
            self.assertTrue(result.horse_details_path.exists())
            report = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(["2"], report["scratched_or_inactive"])
            self.assertEqual(0, report["prediction_ready_count"])

    def test_horse_page_form_records_supply_history_features(self):
        source = """
        <span class="title_text">TEST HORSE (J534)</span>
        <table>
          <tr><td>Country of Origin / Age</td><td>:</td><td>AUS / 5</td></tr>
          <tr><td>Colour / Sex</td><td>:</td><td>Bay / Gelding</td></tr>
          <tr><td>No. of 1-2-3-Starts*</td><td>:</td><td>3-1-3-19</td></tr>
          <tr><td>Current Rating</td><td>:</td><td>91</td></tr>
          <tr><td>310</td><td>06</td><td>01/01/26</td><td>ST / Turf / B+2</td>
          <td>1400</td><td>G</td><td>G3</td><td>4</td><td>91</td><td>Trainer</td>
          <td>Jockey</td><td>2-1/2</td><td>4.9</td><td>118</td><td>7 9 8 6</td>
          <td>1.21.01</td><td>1237</td><td>B/XB/TT</td></tr>
          <tr><td>263</td><td>01</td><td>14/12/25</td><td>ST / Turf / A</td>
          <td>1400</td><td>G</td><td>Class 2</td><td>2</td><td>88</td><td>Trainer</td>
          <td>Jockey</td><td>0</td><td>3.1</td><td>120</td><td>2 2 1</td>
          <td>1.20.00</td><td>1228</td><td>B</td></tr>
        </table>
        """
        page = parse_profile_and_form(source, "HK_2023_J534")
        features = page.model_features()
        self.assertEqual("AUS", features["horse_country"])
        self.assertEqual("Gelding", features["horse_type"])
        self.assertEqual(5, features["horse_age"])
        self.assertEqual(3.5, features["avg_2last"])
        self.assertEqual(81.01, features["fin_time"])
        self.assertEqual(1400.0, features["prev_dist"])
        self.assertAlmostEqual(17.282, features["last_speed"], places=3)

    def test_retired_horse_profile_and_four_digit_form_dates(self):
        source = """
        <span class="title_text">LUCKY RED (CJ059) (Retired)</span>
        <table>
          <tr><td>Country of Origin</td><td>:</td><td>AUS</td></tr>
          <tr><td>Colour / Sex</td><td>:</td><td>Bay / Brown / Gelding</td></tr>
          <tr><td>343</td><td>07</td><td>22/01/2014</td><td>HV / Turf / C+3</td>
          <td>1000</td><td>G</td><td>Class 5</td><td>3</td><td>24</td><td>Trainer</td>
          <td>Jockey</td><td>3</td><td>8.2</td><td>120</td><td>4 4 7</td>
          <td>0.57.50</td><td>1120</td><td>B/TT</td><td>Replay</td></tr>
        </table>
        """
        page = parse_profile_and_form(source, "HK_2007_J059")
        page.fetched_at = "2026-07-26T00:00:00+08:00"
        self.assertEqual("AUS", page.country)
        self.assertEqual("Bay / Brown", page.colour)
        self.assertEqual("Gelding", page.sex)
        self.assertEqual("B/TT", page.form_records[0].gear)
        self.assertEqual(7, page.model_features("2014-01-23")["prev_resu"])


if __name__ == "__main__":
    unittest.main()
