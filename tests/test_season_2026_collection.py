"""Offline parser, coverage, acquisition-boundary and captured-source checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scrapy.http import HtmlResponse

from scripts.collect_season_2026 import (
    BASE, CARD_DATE, OUTPUT, POOLS, Fetcher, collect_race, coverage_manifest, fill_history, freeze_output,
    parse_card, parse_dividends, parse_fixture, parse_horse_history, parse_pdf_card_entries, parse_result, race_numbers, race_url,
)


def response(body, url=None):
    return HtmlResponse(url=url or race_url("results", "2026-10-04", "ST", 1),
                        body=body.encode(), encoding="utf-8")


def dividends(rows):
    return response("<body><table><tr><th colspan='3'>Dividend</th></tr>"
                    "<tr><th>Pool</th><th>Winning Combination</th><th>Dividend (HK$)</th></tr>"
                    + rows + "</table></body>")


def result_html():
    return """<body><p>Race Meeting: 04/10/2026 Sha Tin</p><p>RACE 1 (074)</p>
    <table><tr><td>Class 5 - 1650M</td><td>Going :</td><td>SEALED</td></tr>
    <tr><td>TEST HANDICAP</td><td>Course :</td><td>ALL WEATHER TRACK</td></tr>
    <tr><td>HK$875,000</td></tr></table>
    <table><tr><th>Pla.</th><th>Horse No.</th><th>Horse</th><th>Jockey</th><th>Trainer</th>
    <th>Act. Wt.</th><th>Declar. Horse Wt.</th><th>Dr.</th><th>LBW</th>
    <th>Running Position</th><th>Finish Time</th><th>Win Odds</th></tr>
    <tr><td>1 DH</td><td>1</td><td><a href='horse?horseid=HK_2023_J309'>DOUBLE BINGO</a> (J309)</td>
    <td>J</td><td>T</td><td>128</td><td>1084</td><td>3</td><td>---</td><td>1 1 1 1</td><td>1:39.83</td><td>11</td></tr>
    <tr><td>1 DH</td><td>2</td><td><a href='horse?horseid=HK_2023_J310'>OTHER</a> (J310)</td>
    <td>J</td><td>T</td><td>128</td><td>1084</td><td>4</td><td>---</td><td>2 2 2 2</td><td>1:39.83</td><td>5</td></tr>
    <tr><td>TNP</td><td>3</td><td><a href='horse?horseid=HK_2023_J311'>NONFINISHER</a> (J311)</td>
    <td>J</td><td>T</td><td>128</td><td>---</td><td>5</td><td>---</td><td></td><td>---</td><td>8</td></tr>
    </table></body>"""


class SeasonParserTests(unittest.TestCase):
    def test_archived_pdf_rows_use_exact_date_and_keep_blank_gear(self):
        text = "\n".join([
            "  3 6/9/26 G3 G3 Hcp ST A 1200 #115 (109) B A v 5 XB",
            "", "", "NAME H408", "COPARTNER PRANCE(AUS) 7 b g PPG",
            "  2 6/9/26 4 60-40 ST A 1000 120 (50) -7PNW 1 H/TT",
            "NAME L98", "RUBY THRIVE (AUS) 4 b g PPG",
            "  5 6/9/26 4 60-40 ST A 1200 128 (52) MFP 10",
            "", "", "", "", "NAME L359", "CAVA PIONEER (AUS) 4 b g PPG",
            "  5 6/9/25 4 60-40 ST A 1200 128 (51) MFP 10 TT",
            "NAME L359", "CAVA PIONEER (AUS) 3 b g PPG"])
        entries = parse_pdf_card_entries([text], "2026-09-06")
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["horse_rating"], 109)
        self.assertEqual(entries[0]["horse_country"], "AUS")
        self.assertEqual(entries[0]["horse_age"], 7)
        self.assertEqual(entries[0]["jockey_code_raw"], "B A v")
        self.assertEqual(entries[1]["horse_code"], "L098")
        self.assertEqual(entries[2]["gear"], "")
        self.assertEqual(entries[2]["gear_status"], "source_blank")

    def test_pdf_printed_race_and_horse_numbers(self):
        text = "B1\n3\n12.30 P.M.\n 3 6/9/26 G3 G3 Hcp ST A 1200 115 (109) B A v 5 XB\nNAME H408\nCOPARTNER PRANCE (AUS) 7 b g PPG\nH408\n 2 F\n"
        entry = parse_pdf_card_entries([text], "2026-09-06")[0]
        self.assertEqual(entry["pdf_race_no"], 3)
        self.assertEqual(entry["pdf_horse_no"], 2)
        unbranded = text.replace("H408\n 2 F", "25/26 : statistics\n24/25 : statistics\n 2 F")
        self.assertEqual(parse_pdf_card_entries([unbranded], "2026-09-06")[0]["pdf_horse_no"], 2)
    def test_rowspan_multiple_winners_and_exact_decimal(self):
        pools, rows = parse_dividends(dividends(
            "<tr><td rowspan='2'>WIN</td><td>1</td><td>118.50</td></tr>"
            "<tr><td>2</td><td>1,597.00</td></tr>"
            "<tr><td rowspan='3'>PLACE</td><td>1</td><td>28.00</td></tr>"
            "<tr><td>14</td><td>15.50</td></tr><tr><td>12</td><td>16.00</td></tr>"))
        self.assertEqual(set(pools), set(POOLS))
        self.assertEqual(len(pools["WIN"]["records"]), 2)
        self.assertEqual(pools["WIN"]["records"][1]["dividend_decimal"], "1597.00")
        self.assertEqual(pools["PLACE"]["records"][2]["combination"], [12])
        self.assertEqual(rows[-1]["expanded_cells"], ["PLACE", "12", "16.00"])
        self.assertEqual(rows[-1]["cells"], ["12", "16.00"])

    def test_ordered_and_unordered_keys(self):
        pools, _ = parse_dividends(dividends(
            "<tr><td>QUINELLA</td><td>14,1</td><td>205.50</td></tr>"
            "<tr><td>QUINELLA PLACE</td><td>14,1</td><td>75.00</td></tr>"
            "<tr><td>TRIO</td><td>14,1,12</td><td>182.00</td></tr>"
            "<tr><td>TIERCE</td><td>1,14,12</td><td>1,597.00</td></tr>"
            "<tr><td>FIRST 4</td><td>14,12,4,1</td><td>601.00</td></tr>"
            "<tr><td>QUARTET</td><td>1,14,12,4</td><td>15,907.00</td></tr>"))
        self.assertEqual(pools["QIN"]["records"][0]["combination_key"], "1,14")
        self.assertEqual(pools["FIRST4"]["records"][0]["combination_key"], "1,4,12,14")
        self.assertEqual(pools["QUARTET"]["records"][0]["combination_key"], "1,14,12,4")
        self.assertTrue(pools["TIERCE"]["ordered"])
        self.assertFalse(pools["TRIO"]["ordered"])

    def test_explicit_smaller_unit_is_reconstructable(self):
        pools, _ = parse_dividends(dividends(
            "<tr><td>QUARTET</td><td>12,1,6,10</td><td>349,150.00/$1.0</td></tr>"))
        record = pools["QUARTET"]["records"][0]
        self.assertEqual(record["status"], "payable")
        self.assertEqual(record["dividend_decimal"], "349150.00")
        self.assertEqual(record["unit_stake_decimal"], "1.0")
        self.assertEqual(record["dividend_per_hkd_10_decimal"], "3491500.0")
        self.assertEqual(record["unit_provenance"], "explicit_source_suffix")

    def test_refund_nonpayable_unavailable_are_not_numeric_zero(self):
        pools, _ = parse_dividends(dividends(
            "<tr><td>WIN</td><td>ALL</td><td>REFUND</td></tr>"
            "<tr><td>TRIO</td><td>1,2,3</td><td>NOT WON</td></tr>"
            "<tr><td>QUARTET</td><td>-</td><td>---</td></tr>"))
        self.assertEqual(pools["WIN"]["status"], "refund")
        self.assertEqual(pools["TRIO"]["status"], "nonpayable")
        self.assertEqual(pools["QUARTET"]["status"], "unavailable")
        self.assertEqual(pools["PLACE"]["reason"], "pool_not_published_in_dividend_table")
        self.assertTrue(all(r["dividend_hkd"] is None for p in pools.values() for r in p["records"]))

    def test_unknown_values_and_invalid_combinations_are_unparsed(self):
        for combination, amount in (("1,2", "10.00"), ("1", "12.3.4"), ("0", "10.00")):
            pools, _ = parse_dividends(dividends(
                f"<tr><td>WIN</td><td>{combination}</td><td>{amount}</td></tr>"))
            self.assertEqual(pools["WIN"]["status"], "unparsed")
            self.assertIsNone(pools["WIN"]["records"][0]["dividend_hkd"])

    def test_results_retain_deadheats_nonfinishers_and_raw_rows(self):
        record = parse_result(response(result_html()), "2026-10-04", "ST", 1)
        self.assertEqual(len(record["runners"]), 3)
        self.assertTrue(record["dead_heat"])
        self.assertEqual([r["place"] for r in record["runners"]], [1, 1, None])
        self.assertEqual(record["runners"][2]["finishing_status"], "TNP")
        self.assertEqual(record["runners"][0]["source_cells"][0], "1 DH")

    def test_wrong_requested_context_is_rejected_even_if_final_url_matches(self):
        for day, venue, number in (("2026-10-01", "ST", 1), ("2026-10-04", "HV", 1), ("2026-10-04", "ST", 2)):
            with self.assertRaises(ValueError):
                parse_result(response(result_html()), day, venue, number)

    def test_fixture_counts_and_cancellation(self):
        fixture = response("""<body>9/2026<table><tr><td class='calendar'>
          <p><span class='f_fs14'>6</span><img alt='ST'></p><p>1200(2) 60-40</p><p>1400(1) 80-60</p>
          </td><td class='calendar'><span class='f_fs14'>20</span><img alt='ST'></td></tr></table>
          The race meeting originally scheduled for Sunday, 20 September 2026 at Sha Tin Racecourse has been cancelled.</body>""")
        parsed = parse_fixture(fixture, 2026, 9)
        self.assertEqual(parsed["meetings"][0]["scheduled_race_count"], 3)
        self.assertEqual(parsed["meetings"][1]["status"], "cancelled")
        with self.assertRaises(ValueError):
            parse_fixture(fixture, 2026, 10)

    def test_navigation_ignores_other_pages_and_dates(self):
        page = response("""<a href='?RaceDate=2026/10/04&amp;Racecourse=ST&amp;RaceNo=2'>2</a>
            <a href='racecard?RaceDate=2026/10/04&amp;Racecourse=ST&amp;RaceNo=3'>3</a>
            <a href='?RaceDate=2026/10/01&amp;Racecourse=ST&amp;RaceNo=4'>4</a>""")
        self.assertEqual(race_numbers(page, "2026-10-04", "ST", "results", 1), [1, 2])
        with self.assertRaises(ValueError):
            race_numbers(response("<a href='?RaceDate=2026/10/04&amp;Racecourse=ST&amp;RaceNo=3'>3</a>"), "2026-10-04", "ST", "results", 1)

    def test_card_preserves_all_starter_columns_and_identity(self):
        body = """<body>Race 1 - TEST HANDICAP<br>Wednesday, October 07, 2026, Happy Valley, 18:35
        <p>Turf, "C+3" Course, 1800M, Good</p><p>Prize Money: $875,000, Rating: 40-0, Class 5</p>
        <table><tr><th>Horse No.</th><th>Last 6 Runs</th><th>Horse</th>
        <th>Brand No.</th><th>Draw</th><th>Jockey</th><th>Wt.</th><th>Gear</th></tr>
        <tr><td>1</td><td>1/2</td><td><a href='horse?horseid=HK_2023_J309'>DOUBLE BINGO</a></td>
        <td>J309</td><td>3</td><td>A Jockey</td><td>135</td><td>B/TT</td></tr></table></body>"""
        record = parse_card(response(body, race_url("racecard", CARD_DATE, "HV", 1)), CARD_DATE, "HV", 1)
        self.assertEqual(record["runners"][0]["horse_page_id"], "HK_2023_J309")
        self.assertEqual(record["runners"][0]["source_values"]["Gear"], "B/TT")
        self.assertEqual(record["distance"], 1800)
        self.assertEqual(record["course_configuration"], '"C+3" Course')
        self.assertEqual(record["surface"], "TURF")
        self.assertEqual(record["going"], "GOOD")
        self.assertEqual(record["going_raw"], "Good")
        self.assertEqual(record["prize"], 875000)
        self.assertEqual(record["prize_currency_symbol"], "$")
        self.assertIsNone(record["prize_currency"])
        with self.assertRaises(ValueError):
            parse_card(response(body), CARD_DATE, "HV", 2)
        annotated = parse_card(response(body.replace("A Jockey", "A Jockey (-3)")), CARD_DATE, "HV", 1)
        self.assertEqual(annotated["runners"][0]["jockey"], "A Jockey")
        self.assertEqual(annotated["runners"][0]["jockey_raw"], "A Jockey (-3)")
        self.assertEqual(annotated["runners"][0]["apprentice_allowance_lbs"], 3)
        self.assertEqual(annotated["runners"][0]["net_carried_weight_lbs"], 132)
        overweight = body.replace("<th>Gear</th>", "<th>Over Wt.</th><th>Gear</th>").replace("<td>B/TT</td>", "<td>+2</td><td>B/TT</td>").replace("A Jockey", "A Jockey (-3)")
        self.assertEqual(parse_card(response(overweight), CARD_DATE, "HV", 1)["runners"][0]["net_carried_weight_lbs"], 134)
        internal = parse_card(response(body.replace("A Jockey", "A (-3) Jockey")), CARD_DATE, "HV", 1)
        self.assertEqual(internal["runners"][0]["jockey"], "A (-3) Jockey")

    def test_horse_form_reconciles_withdrawals_and_rejects_future_data(self):
        headers = ["Race Index", "Pla.", "Date"] + ["Field"] * 15
        def row(index, place, day):
            cells = [index, place, day, 'HV / Turf / "B"', '1800', 'G', '4', '10', '42',
                     'Trainer', 'Jockey', '4-1/2', '25', '120', '1 4 2 1 8', '1.50.77', '1172', 'B/TT']
            return '<tr>' + ''.join('<td>' + c + '</td>' for c in cells) + '</tr>'
        page = response('<body><span class="title_text">TEST HORSE (H087)</span>'
            '<table><tr><td>No. of 1-2-3-Starts*</td><td>:</td><td>0-0-0-1</td></tr></table>'
            '<table><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr>'
            + row('031', '08', '16/09/26') + row('100', 'WX', '11/11/23')
            + row('090', '01', '07/10/26') + '</table></body>',
            BASE + '/en-us/local/information/horse?horseid=HK_2022_H087&Option=1')
        record = parse_horse_history(page, 'HK_2022_H087')
        self.assertEqual(record['form_record_count'], 2)
        self.assertEqual(record['captured_start_count'], 1)
        self.assertEqual(record['withdrawn_form_count'], 1)
        self.assertTrue(record['history_complete'])
        self.assertEqual(record['latest_form_date'], '2026-09-16')
        self.assertEqual(record['form_records'][1]['finishing_status'], 'WITHDRAWN')
        with self.assertRaises(ValueError):
            parse_horse_history(page, 'HK_2022_H088')


class AcquisitionTests(unittest.TestCase):
    def test_cache_replay_hash_integrity_and_request_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            fetcher = Fetcher(directory, max_requests=1, delay=0)
            reply = Mock(status_code=200, is_redirect=False, content=b"<body>Official</body>",
                         url=BASE + "/en-us/local/information/fixture", headers={})
            fetcher.session.get = Mock(return_value=reply)
            _, meta = fetcher.fetch(reply.url)
            fetcher.fetch(reply.url)
            self.assertEqual(fetcher.session.get.call_count, 1)
            replay = Fetcher(directory, offline=True)
            replay.session.get = Mock(side_effect=AssertionError("Offline network request"))
            self.assertEqual(replay.fetch(reply.url)[1]["sha256"], hashlib.sha256(reply.content).hexdigest())
            with self.assertRaisesRegex(ValueError, "budget"):
                fetcher.fetch(BASE + "/en-us/local/information/racecard")
            Path(directory, meta["raw_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                replay.fetch(reply.url)

    def test_official_host_boundary_and_redirects(self):
        with tempfile.TemporaryDirectory() as directory:
            fetcher = Fetcher(directory, delay=0)
            fetcher.session.get = Mock(return_value=Mock(status_code=302, is_redirect=True,
                url=BASE + "/fixture", headers={"Location": "https://example.com/redirect"}))
            with self.assertRaisesRegex(ValueError, "Outside official"):
                fetcher.fetch("https://example.com/race")
            self.assertEqual(fetcher.session.get.call_count, 0)
            with self.assertRaisesRegex(ValueError, "Outside official"):
                fetcher.fetch(BASE + "/fixture")
            self.assertEqual(fetcher.session.get.call_count, 1)

    def test_legacy_fallback_is_bounded(self):
        fetcher = Mock()
        fetcher.fetch.side_effect = [ValueError("access denied"), ValueError("legacy unavailable")]
        with self.assertRaises(ValueError):
            collect_race(fetcher, "results", "2026-10-04", "ST", 1)
        self.assertEqual(fetcher.fetch.call_count, 2)
        self.assertIn("LocalResults.aspx", fetcher.fetch.call_args.args[0])

    def test_coverage_never_marks_missing_races_or_pools_complete(self):
        meetings = [{"race_date": "2026-10-04", "venue": "ST", "status": "completed", "race_numbers": [1, 2]}]
        race = parse_result(response(result_html()), "2026-10-04", "ST", 1)
        manifest = coverage_manifest(meetings, [race], [], [], [{"date": "2026-10-04"}], [{}, {}], {})
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["missing_result_races"], [["2026-10-04", "ST", 2]])
        self.assertEqual(len(manifest["pool_gaps"]), 8)


class CapturedOfficialTests(unittest.TestCase):
    @unittest.skipUnless((OUTPUT / "races.json").exists(), "No captured sources yet")
    def test_every_published_record_reconstructs_from_hash_verified_source(self):
        fetcher = Fetcher(OUTPUT, offline=True)
        cases = [("races.json", parse_result), ("racecards.json", parse_card)]
        if (OUTPUT / "history-races.json").exists():
            cases.append(("history-races.json", parse_result))
        for name, parser in cases:
            records = json.loads((OUTPUT / name).read_text())
            for record in records:
                source = record["source"]
                raw = (OUTPUT / source["raw_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])
                page, _ = fetcher.fetch(source["source_url"])
                parsed = parser(page, record["race_date"], record["venue"], record["race_no"])
                self.assertEqual(record["runners"], parsed["runners"])
                if name in {"races.json", "history-races.json"}:
                    self.assertEqual(record["dividends"], parsed["dividends"])
                    self.assertEqual(record["dividend_rows"], parsed["dividend_rows"])

    @unittest.skipUnless((OUTPUT / "horse_histories.json").exists(), "No horse forms captured")
    def test_horse_form_replay_matches_saved_records_and_start_counts(self):
        fetcher = Fetcher(OUTPUT, offline=True)
        for record in json.loads((OUTPUT / "horse_histories.json").read_text()):
            page, _ = fetcher.fetch(record["source"]["source_url"])
            parsed = parse_horse_history(page, record["horse_page_id"])
            self.assertEqual(record["form_records"], parsed["form_records"])
            self.assertEqual(record["captured_start_count"], parsed["captured_start_count"])


class HistoryRepairTests(unittest.TestCase):
    def test_repair_retains_opponents_and_does_not_mutate_or_duplicate_baseline(self):
        import pandas as pd
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = output / "baseline.parquet"
            pd.DataFrame([{"date": "2026-01-01", "venue": "ST", "race_no": 1}]).to_parquet(baseline)
            original = baseline.read_bytes()
            first = parse_result(response(result_html()), "2026-10-04", "ST", 1)
            first.update(race_date="2026-01-01", source={"source_url": "official"})
            second = deepcopy(first)
            second.update(race_no=2, race_id="2026-01-01_ST_R2")
            fixtures = [{"meetings": [{"race_date": "2026-01-01", "venue": "ST", "status": "scheduled",
                                       "scheduled_race_count": 2}], "cancellations": []}]
            fixtures += [{"meetings": [], "cancellations": []} for _ in range(6)]
            with patch("scripts.collect_season_2026.Fetcher.fetch", return_value=(response("<body>fixture</body>"), {})), \
                    patch("scripts.collect_season_2026.parse_fixture", side_effect=fixtures), \
                    patch("scripts.collect_season_2026.collect_race", side_effect=[(first, [1, 2]), (second, [1, 2])]):
                result = fill_history(output, history_path=baseline)
            self.assertTrue(result["complete"])
            self.assertEqual(result["expected_races"], 2)
            saved = json.loads((output / "history-races.json").read_text())
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["race_no"], 2)
            self.assertEqual(len(saved[0]["runners"]), 3)
            self.assertEqual(baseline.read_bytes(), original)

    def test_deadline_checkpoint_is_explicitly_incomplete(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = output / "baseline.parquet"
            pd.DataFrame([{"date": "2026-01-01", "venue": "ST", "race_no": 1}]).to_parquet(baseline)
            result = fill_history(output, max_seconds=0, history_path=baseline)
            self.assertFalse(result["complete"])
            self.assertFalse(result["ready_for_freeze"])
            self.assertEqual(result["stopped_reason"], "wall_clock_deadline_reached")

    def test_freeze_checks_raw_integrity_and_refuses_incomplete_inputs(self):
        from scripts.collect_season_2026 import write_json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "raw").mkdir()
            raw = b"captured official source"
            (output / "raw/source.html").write_bytes(raw)
            source = {"raw_path": "raw/source.html", "sha256": hashlib.sha256(raw).hexdigest()}
            baseline = output / "baseline.parquet"
            baseline.write_bytes(b"original baseline")
            for filename in ("races.json", "racecards.json", "horse_histories.json", "history-races.json"):
                write_json(output / filename, [{"source": source}])
            write_json(output / "dividend_units.json", {"source": source})
            season = {"complete": True, "collected_result_races": 1, "collected_racecards": 1}
            horse = {"page_capture_complete": True, "captured_horses": 1}
            history = {"complete": True, "new_races": 1, "expected_races": 2,
                       "baseline": {"path": str(baseline), "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}}
            for filename, manifest in (("coverage_manifest.json", season),
                                       ("history_coverage_manifest.json", horse),
                                       ("history-race-coverage.json", history)):
                write_json(output / filename, manifest)
            freeze = freeze_output(output)
            self.assertTrue(freeze["original_baseline_verified_unchanged"])
            self.assertEqual(freeze["source_count"], 1)
            self.assertEqual(freeze["files"]["races.json"]["sha256"], hashlib.sha256((output / "races.json").read_bytes()).hexdigest())
            write_json(output / "calibration-runner-enrichment.json", [])
            write_json(output / "calibration-enrichment-coverage.json", {"complete": False, "acquisition_finished": False})
            with self.assertRaisesRegex(ValueError, "still running"):
                freeze_output(output)
            write_json(output / "calibration-enrichment-coverage.json", {"complete": False, "acquisition_finished": True, "whole_field_finite_rating_races": 0})
            self.assertFalse(freeze_output(output)["calibration_enrichment_complete"])
            (output / "raw/source.html").write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                freeze_output(output)
            (output / "raw/source.html").write_bytes(raw)
            write_json(output / "history-race-coverage.json", history | {"complete": False})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                freeze_output(output)


if __name__ == "__main__":
    unittest.main()
