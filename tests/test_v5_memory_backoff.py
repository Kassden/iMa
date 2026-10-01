import json
from pathlib import Path
import tempfile
import unittest

from scripts.v5_memory_backoff import apply_backoff, atomic_json


class MemoryBackoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.config, self.state, self.stop = root / "config", root / "state", root / "STOP"
        self.original = {"max_concurrent_trials": 12, "max_trials_per_decision": 260,
                         "model": "unchanged", "dataset_path": "unchanged"}
        atomic_json(self.config, self.original)
        self.unit = "ima-discovery-v5-supervisor.service"

    def event(self, time=1, unit=None, result="oom-kill"):
        return {"USER_UNIT": unit or self.unit, "UNIT_RESULT": result,
                "__REALTIME_TIMESTAMP": str(time), "__CURSOR": str(time)}

    def run_guard(self, events):
        return apply_backoff(self.config, self.state, self.stop, events, self.unit)

    def test_incremental_and_deduplicated(self):
        self.run_guard([self.event()])
        self.assertEqual(json.loads(self.config.read_text()),
                         {**self.original, "max_concurrent_trials": 10})
        self.assertFalse(self.run_guard([self.event()])["changed"])
        self.run_guard([self.event(2)])
        self.assertEqual(json.loads(self.config.read_text())["max_concurrent_trials"], 8)

    def test_non_oom_and_foreign_units_ignored(self):
        self.run_guard([self.event(unit="cortex-web.service"), self.event(result="exit-code")])
        self.assertEqual(json.loads(self.config.read_text()), self.original)

    def test_crash_transaction_replay(self):
        atomic_json(self.state, {"pending": True, "target_workers": 10,
                                "event_time": "1", "cursor": "1"})
        atomic_json(self.config, {**self.original, "max_concurrent_trials": 10})
        self.run_guard([self.event()])
        self.assertEqual(json.loads(self.config.read_text())["max_concurrent_trials"], 10)
        self.assertFalse(json.loads(self.state.read_text())["pending"])

    def test_floor_preserves_operator_stop(self):
        atomic_json(self.config, {**self.original, "max_concurrent_trials": 1})
        self.stop.write_text("operator pause")
        with self.assertRaises(RuntimeError):
            self.run_guard([self.event()])
        self.assertEqual(self.stop.read_text(), "operator pause")

    def test_floor_creates_stop_and_keeps_refusing(self):
        atomic_json(self.config, {**self.original, "max_concurrent_trials": 1})
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                self.run_guard([self.event()])
        self.assertTrue(self.stop.exists())

    def test_bad_config_fails_without_mutation(self):
        self.config.write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            self.run_guard([self.event()])
        self.assertFalse(self.state.exists())

    def test_out_of_order_events_choose_latest(self):
        self.run_guard([self.event(9), self.event(2)])
        self.assertEqual(json.loads(self.state.read_text())["event_time"], "9")
        self.assertFalse(self.run_guard([self.event(8)])["changed"])
