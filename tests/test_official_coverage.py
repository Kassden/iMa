import unittest
from scrapy.http import HtmlResponse

from scripts.audit_official_coverage import audit, season, observed_race_numbers


class CoverageTests(unittest.TestCase):
    def test_meeting_links_reveal_missing_races_without_guessing_other_dates(self):
        url = "https://racing.hkjc.com/en-us/local/information/localresults?racedate=2026/09/23"
        body = b'''<a href="localresults?racedate=2026/09/23&amp;Racecourse=HV&amp;RaceNo=2">2</a>
          <a href="localresults?racedate=2026/09/23&amp;Racecourse=ST&amp;RaceNo=8">wrong venue</a>
          <a href="localresults?racedate=2026/09/24&amp;Racecourse=HV&amp;RaceNo=9">wrong day</a>
          <a href="https://example.com/localresults?racedate=2026/09/23&amp;Racecourse=HV&amp;RaceNo=7">offsite</a>'''
        race = {"race_date": "2026-09-23", "venue": "HV", "race_no": 1}
        numbers = observed_race_numbers(HtmlResponse(url, body=body, encoding="utf8"), race)
        self.assertEqual(numbers, [1, 2])
        document = {"source_url": url, "family": "results", "status": "fetched_parsed",
                    "race": race, "observed_race_numbers": numbers}
        report = audit([document])
        self.assertEqual(report["meeting_coverage"][0]["missing_observed_race_numbers"], [2])
        second = document | {"source_url": url + "&RaceNo=2", "race": race | {"race_no": 2}}
        self.assertEqual(audit([document, second])["meeting_coverage"][0]["missing_observed_race_numbers"], [])
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
