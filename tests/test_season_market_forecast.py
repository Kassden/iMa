"""Small forward-only fixtures; no model fitting or network calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ima.modeling import MarketBlend
from scripts.forecast_season_market import FAMILIES, build_forecast, forecast, verify_quote_sources


def fixtures():
    rows, quotes = [], {}
    for race in range(1, 6):
        for horse in range(1, 11):
            rows.append({"race_id": f"HKJC:2026-10-07:HV:R{race}", "date": pd.Timestamp("2026-10-07"),
                "venue": "HV", "race_no": race, "horse_no": horse, "horse_id": f"HK_2025_L{race * 10 + horse:03}",
                "horse_name": f"HORSE {race}-{horse}", "model_probability": horse / 55})
            quotes[("2026-10-07", "HV", race, "WIN", (str(horse),))] = {
                "odds_value_raw": str(horse + 2), "odds_value_numeric": horse + 2,
                "retrieved_at_utc": "2026-10-06T13:00:00+00:00", "source_url": "https://info.cld.hkjc.com/graphql/base/"}
    frames = {family: pd.DataFrame(rows) for family in FAMILIES}
    calibration = {family: {"blend": {"fundamental_weight": 0.3, "market_weight": 0.8}} for family in FAMILIES}
    scores = {family + "_market_blend_hindsight": {"order_exponents": {"second": 0.8, "third": 0.7}} for family in FAMILIES}
    units = {"WIN": {"conversion_status": "verified_primary_formula", "displayed_odds_to_D10_factor": 10, "displayed_odds_basis": "gross_return_multiple_per_HKD1"}}
    return frames, calibration, scores, quotes, units


class ForecastTests(unittest.TestCase):
    def test_200_rows_normalization_contract_ev_and_frozen_transform(self):
        frames, cal, scores, quotes, units = fixtures()
        with patch.object(MarketBlend, "fit", side_effect=AssertionError("No retraining")):
            runners, combinations = build_forecast(frames, cal, scores, quotes, units)
        self.assertEqual(len(runners), 200)
        self.assertEqual(len(combinations), 480)
        for _, race in runners.groupby(["family", "race_no"]):
            self.assertAlmostEqual(race.combinedp.sum(), 1)
            self.assertAlmostEqual(race.marketp.sum(), 1)
            self.assertAlmostEqual(race.pplace.sum(), 3)
            self.assertTrue(np.array_equal(race.combinedp, race.pwin))
            self.assertTrue(np.array_equal(race.combinedp, race.pcombined))
            expected = MarketBlend(0.3, 0.8).transform(race.fundp.to_numpy(), race.marketp.to_numpy(), race.race_id)
            np.testing.assert_allclose(race.combinedp, expected)
            np.testing.assert_allclose(race.quoted_ev_hkd, race.combinedp * race.quoted_gross_hkd_per_10 - 10)
        self.assertTrue(combinations.loc[combinations.pool.ne("WIN"), "quoted_ev_hkd"].isna().all())
        pd.testing.assert_frame_equal(runners, runners.sort_values(["family", "race_no", "rank"]).reset_index(drop=True))

    def test_rejects_missing_capped_unverified_or_extra_win_quotes(self):
        for mutation in ("missing", "cap", "unit", "extra", "numeric"):
            frames, cal, scores, quotes, units = fixtures()
            key = next(iter(quotes))
            if mutation == "missing":
                del quotes[key]
            elif mutation == "cap":
                quotes[key]["possible_display_ceiling"] = True
            elif mutation == "unit":
                units["WIN"]["conversion_status"] = "unverified"
            elif mutation == "extra":
                quotes[("2026-10-07", "HV", 1, "WIN", ("11",))] = quotes[key]
            else:
                quotes[key]["odds_value_numeric"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_forecast(frames, cal, scores, quotes, units)

    def test_rejects_duplicate_and_mismatched_family_population(self):
        frames, cal, scores, quotes, units = fixtures()
        frames["boosted"] = frames["boosted"].iloc[:-1]
        with self.assertRaisesRegex(ValueError, "populations differ"):
            build_forecast(frames, cal, scores, quotes, units)
        frames["boosted"] = pd.concat([frames["pool"], frames["pool"].iloc[:1]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_forecast(frames, cal, scores, quotes, units)

    def test_stale_query_is_rejected_before_forecasting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation").mkdir()
            (root / "fresh-predictions").mkdir()
            (root / "query-features.parquet").write_bytes(b"changed input")
            provenance = {"prediction_dir": str(root / "fresh-predictions"), "query_features_sha256": "stale"}
            (root / "evaluation/provenance.json").write_text(json.dumps(provenance))
            (root / "fresh-predictions/readback.json").write_text(json.dumps({"input_sha256": "stale"}))
            with self.assertRaisesRegex(ValueError, "stale"):
                forecast(root)

    def test_quote_raw_response_identity_and_value(self):
        quote = {"source_file": "raw.json", "source_event_index": 0, "meeting_date": "2026-10-07", "venue": "HV", "race_no": 1,
            "pool": "WIN", "official_odds_type": "WIN", "pool_id": "MTG_20261007_0001WIN1", "combination": "01",
            "odds_value_raw": "4.0", "odds_value_numeric": 4.0,
            "source_url": "https://info.cld.hkjc.com/graphql/base/", "retrieved_at_utc": "2026-10-06T13:00:00+00:00"}
        event = {"source_url": quote["source_url"], "http_status": 200,
            "request_body": {"variables": {"date": "2026-10-07", "venueCode": "HV", "raceNo": 1}},
            "response": {"data": {"raceMeetings": [{"pmPools": [{"id": quote["pool_id"], "oddsType": "WIN",
                "leg": {"races": [1]}, "oddsNodes": [{"combString": "01", "oddsValue": "4.0"}]}]}]}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw.json").write_text(json.dumps([event]))
            self.assertEqual(verify_quote_sources({"key": quote}, root), [root / "raw.json"])
            for field, value in (("meeting_date", "2026-10-04"), ("pool", "PLACE"), ("odds_value_raw", "8.0"),
                                 ("odds_value_numeric", 8.0), ("retrieved_at_utc", "2026-10-08T13:00:00+00:00")):
                bad = copy.deepcopy(quote)
                bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_quote_sources({"key": bad}, root)


if __name__ == "__main__":
    unittest.main()
