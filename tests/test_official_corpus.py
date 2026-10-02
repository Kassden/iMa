import unittest
import asyncio
import json
import tempfile
from scrapy.http import HtmlResponse, Request
from scrapper.official_corpus import CorpusSpider, canonical_url, parse_document, family, verify_race_context, parse_movement_text


class OfficialCorpusTests(unittest.TestCase):
    def test_movement_pdf_scope_is_exact(self):
        url = "https://racing.hkjc.com/general/-/media/Sites/JCRW/Page/content/conghua.pdf"
        self.assertIsNotNone(canonical_url(url))
        self.assertEqual(family(url), "movements")
        for unsafe in (url.replace("conghua", "other"), url + "?token=secret", url + "?sc_lang=zh-HK",
                       url + "?rev=not-a-source-revision", url.replace("racing.hkjc.com", "example.com")):
            self.assertIsNone(canonical_url(unsafe))

    def test_movement_stays_keep_context_and_standby_without_invented_id(self):
        text = """HORSES STABLED IN CONGHUA SINCE LAST START
SUNDAY 4 OCTOBER 2026 RACE MEETING
Race Horse Number Horse Name Arrived in Conghua Returned to HK
1 8 VIVA TASTE 06/08/2026 25/09/2026
13 GO GO GO 09/09/2026 17/09/2026
2 2 WOLF COMING 15/04/2026 11/06/2026
26/06/2026 01/10/2026
7 Standby 1 ISLAND HIGHFLYER 11/02/2026 01/10/2026
"""
        rows = parse_movement_text(text)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]["race_no"], 1)
        self.assertEqual(rows[3]["horse_name"], "WOLF COMING")
        self.assertEqual(rows[4]["standby_no"], 1)
        self.assertIsNone(rows[4]["horse_no"])
        self.assertTrue(all(r["horse_id"] is None and r["published_at"] is None for r in rows))
        self.assertTrue(all(r["venue"] == "unknown" for r in rows))
        other = parse_movement_text(text.replace("WOLF COMING", "ANOTHER HORSE"))
        self.assertNotEqual(rows[3]["values"], other[3]["values"])
        for bad in (text.replace("SUNDAY", "MONDAY"), text.replace("11/06/2026", "11/06/2027"),
                    text.replace("Race Horse Number", "Changed Columns"),
                    text + "BAD ROW 01/10/2026", text.replace("1 8 VIVA TASTE", "VIVA TASTE")):
            with self.assertRaises(ValueError):
                parse_movement_text(bad)

    def test_corrupt_movement_pdf_remains_explicitly_unparsed(self):
        from scrapy.http import Response
        url = "https://racing.hkjc.com/general/-/media/Sites/JCRW/Page/content/conghua.pdf"
        for body in (b"<html>No information</html>", b"%PDF-1.4\ncorrupt"):
            record = parse_document(Response(url, body=body))
            self.assertEqual(record["status"], "fetched_unparsed")
            self.assertIn("parse_error", record)
            self.assertNotIn("events", record)

    def test_current_movement_pdf_refreshes_without_html_navigation(self):
        from unittest.mock import patch
        from scrapy.http import Response
        url = "https://racing.hkjc.com/general/-/media/Sites/JCRW/Page/content/conghua.pdf"
        async def collect(spider):
            return [request async for request in spider.start()]
        with tempfile.TemporaryDirectory() as directory:
            spider = CorpusSpider(seeds=[url], output=directory)
            try:
                first = asyncio.run(collect(spider))[0]
                self.assertEqual(first.priority, 100000)
                with patch("scrapper.official_corpus.parse_document", return_value={
                        "family": "movements", "horse_ids": [], "status": "fetched_parsed", "events": []}):
                    self.assertEqual(list(spider.parse(Response(url, body=b"%PDF-1.4", request=first,
                        headers={"Content-Type": "application/pdf"}))), [])
                self.assertTrue(asyncio.run(collect(spider))[0].dont_filter)
            finally:
                spider.db.close()

    def test_legacy_eleven_columns_and_tnp_keep_complete_field(self):
        url = "https://racing.hkjc.com/en-us/local/information/localresults?racedate=2001/04/14&Racecourse=ST&RaceNo=1"
        body = '''<body><p>Race Meeting: 14/04/2001 Sha Tin</p><p>RACE 1 (512)</p>
          <table><tr><td>Class 4 - 1000M</td><td>Going :</td><td>GOOD</td></tr></table>
          <table><tr><td>Pla.</td><td>Horse No.</td><td>Horse</td><td>Jockey</td><td>Trainer</td>
          <td>Act. Wt.</td><td>Declar. Horse Wt.</td><td>Dr.</td><td>LBW</td><td>Finish Time</td><td>Win Odds</td></tr>
          <tr><td>1</td><td>1</td><td><a href="horse?horseid=HK_2000_A115">HORSE ONE</a> (CA115)</td>
          <td>J</td><td>T</td><td>120</td><td>1000</td><td>1</td><td>---</td><td>0:58.90</td><td>3</td></tr>
          <tr><td>TNP</td><td>2</td><td><a href="horse?horseid=HK_2000_A116">HORSE TWO</a> (CA116)</td>
          <td>J</td><td>T</td><td>120</td><td>---</td><td>2</td><td>---</td><td>---</td><td>8</td></tr>
          </table></body>'''
        row = parse_document(HtmlResponse(url, body=body.encode(), encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        self.assertEqual(len(row["race"]["runners"]), 2)
        self.assertEqual(row["race"]["runners"][0]["horse_page_id"], "HK_2000_A115")
        self.assertEqual(row["race"]["runners"][0]["running_position"], "")
        self.assertEqual(row["race"]["runners"][1]["finishing_status"], "TNP")
        self.assertIsNone(row["race"]["runners"][1]["declared_weight"])
        self.assertIsNone(row["race"]["runners"][1]["place"])
    def test_fresh_date_seed_can_overtake_persisted_general_frontier(self):
        url = "https://racing.hkjc.com/en-us/local/information/trackworkonedayresult?OneDay=1/10/2026"
        async def collect(spider):
            return [request async for request in spider.start()]
        with tempfile.TemporaryDirectory() as directory:
            spider = CorpusSpider(seeds=[url], output=directory)
            try:
                requests = asyncio.run(collect(spider))
                self.assertEqual(requests[0].priority, 100000)
                self.assertTrue(requests[0].dont_filter)
                spider.db.execute("INSERT INTO pages(url,status,metadata_json) VALUES (?,?,?)", (requests[0].url,"fetched_parsed","{}"))
                requests = asyncio.run(collect(spider))
                self.assertFalse(requests[0].dont_filter)
            finally:
                spider.db.close()

    def test_resume_repairs_unfetched_verified_json_frontier(self):
        url = "https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202610011E.aspx?PageNum=2"
        async def collect(spider):
            return [request async for request in spider.start()]
        with tempfile.TemporaryDirectory() as directory:
            spider = CorpusSpider(seeds=[], output=directory)
            try:
                spider.db.execute("INSERT INTO links VALUES (?,?,?)", (url.replace("=2", "=1"),url,"official_json_next"))
                requests = asyncio.run(collect(spider))
                self.assertEqual(len(requests), 1)
                self.assertTrue(requests[0].dont_filter)
                self.assertEqual(requests[0].priority, 200000)
                spider.db.execute("INSERT INTO pages(url,status,metadata_json) VALUES (?,?,?)", (url,"denied","{}"))
                self.assertEqual(asyncio.run(collect(spider)), [])
            finally:
                spider.db.close()
    def test_date_only_meeting_lead_overtakes_old_general_frontier(self):
        url = "https://racing.hkjc.com/en-us/local/information/localresults?racedate=2026/09/23"
        async def collect(spider):
            return [request async for request in spider.start()]
        with tempfile.TemporaryDirectory() as directory:
            spider = CorpusSpider(seeds=[url], output=directory)
            try:
                requests = asyncio.run(collect(spider))
                self.assertEqual(requests[0].priority, 75000)
                self.assertTrue(requests[0].dont_filter)
                spider.db.execute("INSERT INTO pages(url,status,metadata_json) VALUES (?,?,?)", (requests[0].url,"fetched_parsed","{}"))
                self.assertFalse(asyncio.run(collect(spider))[0].dont_filter)
            finally:
                spider.db.close()
    def test_race_identity_is_verified_from_displayed_header_not_filename(self):
        response = HtmlResponse("https://racing.hkjc.com/en-us/local/information/localresults",
            body=b'<body><div>Race Meeting: 13/07/2025 Sha Tin</div><td>RACE 10 (837)</td></body>', encoding="utf8")
        query = {"racedate": "2025/07/13", "racecourse": "ST", "raceno": "10"}
        self.assertEqual(verify_race_context(response, query), ("2025-07-13", "ST", 10))
        with self.assertRaisesRegex(ValueError, "race number"):
            verify_race_context(response, query | {"raceno": "1"})
        with self.assertRaisesRegex(ValueError, "date/venue"):
            verify_race_context(response, query | {"racecourse": "HV"})
        with self.assertRaisesRegex(ValueError, "date/venue"):
            verify_race_context(response, query | {"racedate": "2025/07/12"})
    def test_scope_and_canonicalization(self):
        self.assertIsNone(canonical_url("https://example.com/horse"))
        self.assertIsNone(canonical_url("https://racing.hkjc.com/en-us/login"))
        self.assertIsNone(canonical_url("https://user@racing.hkjc.com/en-us/local/information/horse"))
        self.assertIsNone(canonical_url("https://racing.hkjc.com:broken/en-us/local/information/horse"))
        self.assertEqual(canonical_url("horse?horseid=HK_2022_H033&b_cid=tracking#x"),
                         "https://racing.hkjc.com/en-us/local/information/horse?horseid=HK_2022_H033")

    def test_current_results_context_comes_from_displayed_header(self):
        url = "https://racing.hkjc.com/en-us/local/information/localresults"
        body = b'<body><p>Race Meeting: 01/10/2026 Sha Tin</p><p>RACE 10 (80)</p></body>'
        response = HtmlResponse(url, body=body, encoding="utf8")
        self.assertEqual(verify_race_context(response, {}), ("2026-10-01", "ST", 10))
        self.assertEqual(verify_race_context(response, {"raceno": "10"}), ("2026-10-01", "ST", 10))
        with self.assertRaisesRegex(ValueError, "date/venue"):
            verify_race_context(response, {"racecourse": "HV"})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            verify_race_context(HtmlResponse(url, body=body.replace(b'</body>',
                b'<p>Race Meeting: 30/09/2026 Happy Valley</p></body>'), encoding="utf8"), {})
        with self.assertRaisesRegex(ValueError, "race number"):
            verify_race_context(HtmlResponse(url, body=body.replace(b'</body>',
                b'<p>RACE 1 (71)</p></body>'), encoding="utf8"), {})
        with self.assertRaisesRegex(ValueError, "date/venue"):
            verify_race_context(HtmlResponse(url, body=b'<body>navigation only</body>', encoding="utf8"), {})

    def test_family_and_unparsed_are_not_empty(self):
        url = "https://racing.hkjc.com/en-us/local/information/ovehorse?horseid=HK_2022_H033"
        response = HtmlResponse(url,body=b"<html><table><tr><td>Header</td></tr></table></html>",encoding="utf8",request=Request(url))
        row = parse_document(response)
        self.assertEqual(family(url),"veterinary")
        self.assertEqual(row["status"],"fetched_unparsed")
        self.assertEqual(row["tables"],[[["Header"]]])

    def test_retired_route_recovers_official_identity(self):
        url = "https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2022_H033"
        body = b'<span class="title_text">EIGHTEEN PALMS (H033) (Retired)</span><select><option value="HK_2022_H033">horse</option></select>'
        row = parse_document(HtmlResponse(url,body=body,encoding="utf8",request=Request(url)))
        self.assertEqual(row["horse_ids"],["HK_2022_H033"])
        self.assertEqual(row["horse"]["horse_name"],"EIGHTEEN PALMS")
        self.assertEqual(row["status"],"fetched_parsed")

    def test_profile_metadata_retains_pedigree_and_quarantines_conflicting_labels(self):
        url = "https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2022_H033"
        body = b'''<span class="title_text">EIGHTEEN PALMS (H033) (Retired)</span>
          <table><tr><td>Sire</td><td>:</td><td>El Roca</td></tr>
          <tr><td>Dam</td><td>:</td><td>Buttermilk</td></tr>
          <tr><td>Last Rating</td><td>:</td><td>62</td></tr>
          <tr><td>Last Rating</td><td>:</td><td>63</td></tr></table>'''
        horse = parse_document(HtmlResponse(url, body=body, encoding="utf8"))["horse"]
        self.assertEqual(horse["profile_attributes"]["Sire"], "El Roca")
        self.assertEqual(horse["profile_attributes"]["Dam"], "Buttermilk")
        self.assertNotIn("Last Rating", horse["profile_attributes"])
        self.assertEqual(horse["profile_attribute_conflicts"]["Last Rating"], ["62", "63"])
        self.assertIn("snapshot-only", horse["profile_attributes_policy"])

    def test_legacy_cycle_alias_and_deregistered_title_keep_full_identity(self):
        url = "https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2006_H227"
        body = b'<span class="title_text">SUPER BABY (CH227) (Deregistered)</span>'
        row = parse_document(HtmlResponse(url, body=body, encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        self.assertEqual(row["horse"]["horse_name"], "SUPER BABY")
        self.assertEqual(row["horse"]["horse_page_id"], "HK_2006_H227")
        self.assertEqual(row["horse"]["brand_code"], "CH227")
        self.assertEqual(row["horse"]["registration_status_at_capture"], "inactive")
        bad = parse_document(HtmlResponse(url, body=body.replace(b"CH227", b"CH228"), encoding="utf8"))
        self.assertEqual(bad["status"], "fetched_unparsed")

    def test_workout_event_is_not_publication_proof(self):
        url = "https://racing.hkjc.com/en-us/local/information/trackworkresult?horseid=HK_2022_H033"
        body = b'<table><tr><th>Date</th><th>Type</th><th>Racecourse/Track</th><th>Workouts</th><th>Gear</th></tr><tr><td>01/10/2026</td><td>Gallop</td><td>Sha Tin</td><td>800m</td><td>B</td></tr></table>'
        row = parse_document(HtmlResponse(url,body=body,encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        self.assertEqual(row["events"][0]["event_date"], "2026-10-01")
        self.assertIsNone(row["events"][0]["published_at"])

    def test_wrong_trial_date_quarantines_events(self):
        url = "https://racing.hkjc.com/en-us/local/information/archive/btresult?Date=09/09/2025"
        body = b'<select><option selected>09/09/2026</option></select>'
        row = parse_document(HtmlResponse(url,body=body,encoding="utf8"))
        self.assertEqual(row["status"], "fetched_unparsed")
        self.assertNotIn("events", row)

    def test_trial_batch_context_is_not_confused_with_each_horses_time(self):
        url = "https://racing.hkjc.com/en-us/local/information/archive/btresult?Date=2025/09/09"
        body = b'''<body><select><option selected>09/09/2025</option></select>
          <table><tr><td>Batch 1 - SHA TIN ALL WEATHER TRACK - 1200m</td></tr>
          <tr><td>Going: WET SLOW</td><td>Time: 1.10.86</td></tr>
          <tr><td>Sectional Time: 24.5 23.2 23.1</td></tr></table>
          <table><tr><td>Horse</td><td>Jockey</td><td>Trainer</td><td>Draw</td><td>Time</td></tr>
          <tr><td><a href="horse?horseid=HK_2024_K362">HORSE</a></td><td>J</td><td>T</td><td>5</td><td>1.10.93</td></tr></table>
          <table><tr><td>Batch 2 - CONGHUA TURF TRACK - 1000m</td></tr></table>
          <table><tr><td>Horse</td><td>Jockey</td><td>Trainer</td><td>Draw</td><td>Time</td></tr>
          <tr><td><a href="horse?horseid=HK_2024_K363">OTHER HORSE</a></td><td>J</td><td>T</td><td>2</td><td>---</td></tr></table></body>'''
        row = parse_document(HtmlResponse(url, body=body, encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        first, second = row["events"]
        self.assertEqual(first["venue"], "ST")
        self.assertEqual(first["distance_metres"], 1200)
        self.assertEqual(first["going"], "WET SLOW")
        self.assertAlmostEqual(first["batch_winner_time_seconds"], 70.86)
        self.assertAlmostEqual(first["trial_finish_seconds"], 70.93)
        self.assertEqual(first["batch_sectional_seconds"], [24.5, 23.2, 23.1])
        self.assertEqual(second["trial_venue_name"], "CONGHUA")
        self.assertNotIn("going", second)
        self.assertIsNone(second["trial_finish_seconds"])
        self.assertIsNone(first["published_at"])

    def test_redirected_horse_identity_mismatch(self):
        url = "https://racing.hkjc.com/en-us/local/information/otherhorse?horseid=HK_2022_H033"
        body = b'<span class="title_text">SOME OTHER HORSE (J542)</span>'
        row = parse_document(HtmlResponse(url,body=body,encoding="utf8"))
        self.assertEqual(row["status"], "fetched_unparsed")

    def test_source_no_information_is_not_zero_events(self):
        url = "https://racing.hkjc.com/en-us/local/information/ovehorse?horseid=HK_2022_H033"
        row = parse_document(HtmlResponse(url,body=b'No information is found.',encoding="utf8"))
        self.assertEqual(row["status"], "source_reports_no_information")

    def test_public_daily_json_scope_is_narrow(self):
        self.assertIsNotNone(canonical_url("https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202610011E.aspx?PageNum=1"))
        self.assertIsNone(canonical_url("https://racing.hkjc.com/racing/information/json/PrivateAccount.aspx?PageNum=1"))
        self.assertIsNone(canonical_url("https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202610011E.aspx?PageNum=999999"))

    def test_daily_json_name_is_not_guessed_full_identity(self):
        url = "https://racing.hkjc.com/racing/information/json/TrackworkOneDayRecords/202610011E.aspx?PageNum=1"
        data = {"page": 1, "next": 2, "Records": [{"Horse": "HORSE NAME", "Type": "Trotting"}]}
        row = parse_document(HtmlResponse(url,body=json.dumps(data).encode(),encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        self.assertIsNone(row["events"][0]["horse_id"])
        self.assertEqual(row["identity_unresolved_count"], 1)
        data["next"] = 1
        bad = parse_document(HtmlResponse(url,body=json.dumps(data).encode(),encoding="utf8"))
        self.assertEqual(bad["status"], "fetched_unparsed")

    def test_date_feed_null_venues_remain_unresolved(self):
        url = "https://racing.hkjc.com/racing/information/json/DateList/LocalResults.aspx?lang=en-us"
        self.assertEqual(canonical_url(url), url)
        self.assertIsNone(canonical_url(url + "&token=secret"))
        data = {"MeetingDateList": [{"Key": "2026-10-01T00:00:00", "Value": None},
                                   {"Key": "2026-09-30T00:00:00", "Value": {"HV": "Happy Valley"}}]}
        row = parse_document(HtmlResponse(url, body=json.dumps(data).encode(), encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        self.assertEqual(row["date_candidates"][0]["status"], "venue_unresolved")
        self.assertEqual(row["date_candidates"][1]["venues"], ["HV"])
        data["MeetingDateList"][1]["Value"] = {"UNKNOWN": "Somewhere"}
        self.assertEqual(parse_document(HtmlResponse(url, body=json.dumps(data).encode(), encoding="utf8"))["status"], "fetched_unparsed")

    def test_sectional_components_and_full_identity(self):
        url = "https://racing.hkjc.com/en-us/local/information/archive/displaysectionaltime?racedate=13/07/2025&RaceNo=1"
        body = '''<body><div>Meeting Date:13/07/2025, Sha Tin</div><div>Race 1</div>
            <table><thead><tr><td>Finishing Order</td><td>Horse No.</td></tr></thead><tbody><tr>
            <td>1</td><td>4</td><td><a href="horse?horseid=HK_2023_J542">SIGHT DREAMER (J542)</a></td>
            <td><p><span>5</span><i>1</i></p><p>13.72</p></td>
            <td><p><span>4</span><i>3</i></p><p>21.71</p></td>
            <td><p><span>1</span><i>SH</i></p><p>23.45<span class="color_blue2"><span>11.73</span><span>11.72</span></span></p></td>
            <td><p><span>1</span><i>1</i></p><p>23.11</p></td><td></td><td></td><td>1:21.99</td>
            </tr></tbody></table></body>'''
        row = parse_document(HtmlResponse(url, body=body.encode(), encoding="utf8"))
        self.assertEqual(row["status"], "fetched_parsed")
        event = row["events"][0]
        self.assertEqual(event["horse_id"], "HK_2023_J542")
        self.assertEqual(event["venue"], "ST")
        self.assertEqual(event["values"]["splits"][0]["seconds"], 13.72)
        self.assertEqual(event["values"]["splits"][2]["subsplit_seconds"], [11.73, 11.72])
        for corrupted in (body.replace("HK_2023_J542", "J542"), body.replace("Race 1", "Race 2"),
                          body.replace("Meeting Date:13/07/2025", "Meeting Date:14/07/2025")):
            self.assertEqual(parse_document(HtmlResponse(url, body=corrupted.encode(), encoding="utf8"))["status"], "fetched_unparsed")


if __name__ == "__main__":
    unittest.main()
