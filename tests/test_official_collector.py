import signal
import unittest
from unittest.mock import patch

from scripts.collect_official_corpus import run_segment


class CollectorSupervisorTests(unittest.TestCase):
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
