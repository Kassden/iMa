from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scrapper.historical.bulk import BulkArchiveCollector


class BulkHistoryTests(unittest.TestCase):
    def test_pending_range_is_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = BulkArchiveCollector(Path(directory), workers=1, request_delay=0)
            pending = collector.pending(date(2025, 1, 1), date(2025, 1, 2))
            self.assertEqual(4, len(pending))
            collector.database.execute(
                "INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2025-01-01", "ST", "no_meeting", 0, 0, None, "now"),
            )
            collector.database.commit()
            self.assertEqual(3, len(collector.pending(date(2025, 1, 1), date(2025, 1, 2))))

    def test_targeted_meeting_scan_skips_completed_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = BulkArchiveCollector(Path(directory), workers=1, request_delay=0)
            collector.database.execute(
                "INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("2020-01-01", "ST", "complete", 10, 120, None, "now"),
            )
            collector.database.commit()
            with patch.object(collector, "_collect", return_value={
                "race_date": "2020-01-02", "venue": "HV", "status": "no_meeting",
            }) as collect:
                report = collector.run_targets(
                    [("2020-01-01", "ST"), ("2020-01-02", "HV")],
                    known_meetings=True,
                )
            collect.assert_called_once_with("2020-01-02", "HV")
            self.assertEqual(1, report["attempted"])
            self.assertEqual(1, report["archive_unavailable"])

    def test_transient_fetch_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = BulkArchiveCollector(
                Path(directory), workers=1, request_delay=0,
                request_retries=3, retry_backoff=0,
            )
            with patch("scrapper.historical.bulk.fetch_results") as fetch:
                fetch.side_effect = [TimeoutError("slow"), ("<html />", "https://example.test")]
                result = collector._fetch_with_retry("2020/01/01", "ST", 1)
            self.assertEqual(("<html />", "https://example.test"), result)
            self.assertEqual(2, fetch.call_count)


if __name__ == "__main__":
    unittest.main()
