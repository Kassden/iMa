import unittest

from scripts.audit_official_coverage import audit, season


class CoverageTests(unittest.TestCase):
    def page(self, page, next_page, fetched_at="2026-10-02T01:00:00Z"):
        return {"source_url": f"https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202610011E.aspx?PageNum={page}",
                "family": "trackwork", "status": "fetched_parsed", "fetched_at": fetched_at,
                "next_page": next_page, "events": [{"horse_id": None, "event_date": "2026-10-01"}]}

    def test_partial_stream_not_complete(self):
        report = audit([self.page(1, 2), self.page(3, 0)])
        self.assertFalse(report["daily_streams"][0]["complete"])
        self.assertEqual(report["daily_streams"][0]["next_missing_or_invalid_page"], 2)

    def test_complete_chain_and_no_duplicate_captures(self):
        report = audit([self.page(1, 2), self.page(2, 0), self.page(1, 2)])
        self.assertTrue(report["daily_streams"][0]["complete"])
        self.assertEqual(report["daily_streams"][0]["rows"], 2)
        events = next(r for r in report["coverage"] if "parsed_events" in r)
        self.assertEqual(events["identity_unresolved_events"], 2)
        self.assertEqual(events["publication_unknown_events"], 2)
        self.assertIsNone(events["expected_count"])

    def test_null_venue_not_no_meeting(self):
        report = audit([{"source_url": "official-date-feed", "family": "race_census",
                         "status": "fetched_parsed", "date_candidates": [
                             {"date": "2026-10-01", "venues": [], "status": "venue_unresolved"}]}])
        self.assertEqual(report["date_candidates"][0]["status"], "venue_unresolved")

    def test_racing_season_boundary(self):
        self.assertEqual(season("2026-08-31"), "2025-2026")
        self.assertEqual(season("2026-09-01"), "2026-2027")
