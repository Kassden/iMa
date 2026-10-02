import signal
import json
import unittest
from unittest.mock import patch

from scripts.collect_official_corpus import run_segment
from scripts.discover_official_seeds import result_selector_seeds, trial_selector_seeds


class CollectorSupervisorTests(unittest.TestCase):
    def test_trial_selector_keeps_observed_archive_route_and_date_format(self):
        options = [{"value": "09/09/2025"}, {"value": "29/09/2026"}, {"value": "04/10/2026"}]
        observed = "https://racing.hkjc.com/en-us/local/information/archive/btresult?Date=2025/09/09"
        seeds = trial_selector_seeds(options, observed, "2026-10-02")
        self.assertEqual(len(seeds), 2)
        self.assertTrue(all("/archive/btresult" in s for s in seeds))
        self.assertTrue(seeds[0].endswith("Date=2025%2F09%2F09"))
        with self.assertRaisesRegex(ValueError, "displayed options"):
            trial_selector_seeds(options, observed.replace("2025/09/09", "2025/09/10"), "2026-10-02")
    def test_result_selector_never_guesses_venue_or_collects_future_dates(self):
        options = [{"value": json.dumps({"date": date, "venue": venue})} for date, venue in
                   [("01/10/2026", ""), ("23/09/2026", "HV"), ("04/10/2026", "")]]
        observed = "https://racing.hkjc.com/en-us/local/information/localresults?racedate=2026/10/01"
        seeds = result_selector_seeds(options, observed, "2026-10-02")
        self.assertEqual(len(seeds), 2)
        self.assertNotIn("Racecourse", seeds[0])
        self.assertIn("Racecourse=HV", seeds[1])
        with self.assertRaisesRegex(ValueError, "verified date-only"):
            result_selector_seeds(options, observed.replace("racing.hkjc.com", "example.com"), "2026-10-02")
        with self.assertRaisesRegex(ValueError, "does not match"):
            result_selector_seeds(options, observed.replace("2026/10/01", "2026/09/22"), "2026-10-02")
    @patch("scripts.collect_official_corpus.subprocess.Popen")
    def test_segment_exit_propagates(self, popen):
        popen.return_value.wait.return_value = 2
        self.assertEqual(run_segment(["collector", "--segment"]), 2)
        popen.assert_called_once_with(["collector", "--segment"], start_new_session=True)

    @patch("scripts.collect_official_corpus.subprocess.Popen")
    def test_interrupt_waits_for_crawler_to_save_queue(self, popen):
        child = popen.return_value
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with self.assertRaises(SystemExit) as stopped:
            run_segment(["collector"])
        self.assertEqual(stopped.exception.code, 2)
        child.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(child.wait.call_count, 2)
