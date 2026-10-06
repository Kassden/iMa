import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from ima.research_resources import GIB, JobEstimator, JobMeasurement, JobWorkload
from ima.research_runtime_history import RuntimeHistory, read_own_unit_journal


UNIT = "ima-v6-openrouter-continuous-4864156.service"


def cgroup(peak=GIB, oom=0, kill=0, high=0, path="/own", boot="boot"):
    return {"supported": True, "path": path, "boot_id": boot,
            "memory.current": min(peak, GIB), "memory.peak": peak,
            "memory.max": 100_000_000_000,
            "memory.events": f"low 0\nhigh {high}\nmax 0\noom {oom}\noom_kill {kill}\noom_group_kill 0"}


def manager(message, cursor="cursor", **extra):
    return dict({"USER_UNIT": UNIT, "_PID": "484996", "_UID": "1001",
                 "_COMM": "systemd", "_EXE": "/usr/lib/systemd/systemd",
                 "MESSAGE": message, "__CURSOR": cursor,
                 "__REALTIME_TIMESTAMP": "1791141141000000"}, **extra)


class RuntimeHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "runtime.json"
        self.env = patch.dict(os.environ, {"IMA_RESEARCH_SYSTEMD_UNIT": UNIT, "INVOCATION_ID": "env-invocation"})
        self.env.start()
        self.addCleanup(self.env.stop)

    @patch("ima.research_runtime_history.subprocess.run")
    def test_first_startup_empty_history_no_journal_and_recovered_interruption(self, run):
        history = RuntimeHistory(self.path)
        self.assertEqual([], history.snapshot()["invocations"])
        snapshot = history.startup(invocation_id="first", cgroup={"supported": False},
                                   interrupted_attempts=["recovered-a"], now=1)
        self.assertEqual("first", snapshot["current_invocation_id"])
        self.assertEqual(0, snapshot["restart_count"])
        self.assertEqual(1, snapshot["interruption_count"])
        self.assertEqual([], snapshot["oom_facts"])
        self.assertIsNone(snapshot["invocations"][0]["previous_invocation_id"])
        self.assertIsNone(snapshot["invocations"][0]["cgroup"])
        self.assertEqual(1, history.startup(invocation_id="first", cgroup={"supported": False},
                                           interrupted_attempts=["recovered-a"], now=2)["interruption_count"])
        run.assert_not_called()

    @patch("ima.research_runtime_history.subprocess.run")
    @patch("ima.research_resources.observe_cgroup", return_value={"supported": False})
    def test_first_startup_defaults_without_previous_invocation(self, observe, run):
        history = RuntimeHistory(self.path)
        snapshot = history.startup()
        self.assertEqual("env-invocation", snapshot["current_invocation_id"])
        self.assertEqual(1, len(snapshot["invocations"]))
        self.assertEqual([], snapshot["interrupted_attempts"])
        self.assertEqual([], snapshot["oom_facts"])
        observe.assert_called_once_with()
        run.assert_not_called()

    def test_restart_preserves_peaks_counters_and_interrupts_once_without_oom_guess(self):
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup(peak=9 * GIB), now=1)
        history.observe(running_attempts=["a", "b"], cgroup=cgroup(peak=12 * GIB), now=2)
        restarted = RuntimeHistory(self.path)
        snapshot = restarted.startup(invocation_id="two", cgroup=cgroup(), interrupted_attempts=["a", "b"], now=3)
        self.assertEqual(12 * GIB, snapshot["cgroup_peak_bytes"])
        self.assertEqual(1, snapshot["restart_count"])
        self.assertEqual(2, snapshot["interruption_count"])
        self.assertEqual([], snapshot["oom_facts"])
        self.assertTrue(all(row["actual"] is None for row in snapshot["interrupted_attempts"]))
        snapshot = restarted.startup(invocation_id="two", cgroup=cgroup(), interrupted_attempts=["a", "b"], now=4)
        self.assertEqual(2, snapshot["interruption_count"])
        self.assertEqual(3, snapshot["invocations"][1]["started_at"])

    def test_actual_counter_deltas_and_resets_are_durable(self):
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup(oom=2, kill=1, high=4))
        history.observe(cgroup=cgroup(oom=3, kill=2, high=8))
        repeated = history.observe(cgroup=cgroup(oom=3, kill=2, high=8))
        self.assertEqual(4, len(repeated["oom_facts"]))
        restarted = RuntimeHistory(self.path)
        zero = restarted.startup(invocation_id="two", cgroup=cgroup())
        self.assertEqual(3, zero["cgroup_counter_totals"]["oom"])
        self.assertEqual(2, zero["cgroup_counter_totals"]["oom_kill"])
        self.assertEqual(8, zero["cgroup_counter_totals"]["high"])
        added = restarted.observe(cgroup=cgroup(oom=1, kill=1))
        self.assertEqual(3, added["cgroup_counter_totals"]["oom_kill"])
        self.assertEqual(6, len(added["oom_facts"]))
        self.assertEqual(6, len({row["fact_id"] for row in added["oom_facts"]}))
        self.assertTrue(all(row["victim_attempt_id"] is None for row in added["oom_facts"]))

    def test_surviving_cgroup_not_recounted_at_restart_but_new_boot_counted(self):
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup(oom=1, kill=1))
        restarted = RuntimeHistory(self.path)
        same = restarted.startup(invocation_id="two", cgroup=cgroup(oom=1, kill=1))
        self.assertEqual(1, same["cgroup_counter_totals"]["oom_kill"])
        new = restarted.observe(cgroup=cgroup(oom=1, kill=1, boot="new-boot"))
        self.assertEqual(2, new["cgroup_counter_totals"]["oom_kill"])

    def test_unavailable_cgroup_and_zero_counters_do_not_clear_incident(self):
        history = RuntimeHistory(self.path)
        history.startup(cgroup=cgroup(peak=20 * GIB, oom=1, kill=1))
        snapshot = history.observe(cgroup={"supported": False})
        self.assertEqual(20 * GIB, snapshot["cgroup_peak_bytes"])
        self.assertEqual(1, snapshot["cgroup_counter_totals"]["oom_kill"])
        self.assertEqual(1, snapshot["invocations"][0]["cgroup"]["counters"]["oom_kill"])

    def test_interrupted_private_bounds_feed_estimator_once_without_unit_peak(self):
        work = JobWorkload(stage="fit", family="ridge", rows=100, generated_features=4,
                           selected_features=4, implementation_revision="test", dependency_versions={"test": "1"})
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup())
        history.observe(running_attempts=["a"], attempt_measurements={"a": asdict(JobMeasurement(work.fingerprint(), 3 * GIB, 100))},
                        cgroup=cgroup(peak=90 * GIB))
        history.observe(attempt_measurements={"a": asdict(JobMeasurement(work.fingerprint(), GIB, 90))}, cgroup=cgroup())
        snapshot = RuntimeHistory(self.path).startup(invocation_id="two", cgroup=cgroup())
        actual = snapshot["interrupted_attempts"][0]["actual"]
        self.assertEqual(3 * GIB, actual["private_peak_bytes"])
        self.assertEqual(100, actual["wall_seconds"])
        self.assertTrue(actual["censored"])
        self.assertTrue(actual["failed"])
        estimator = JobEstimator(self.path.parent / "estimates.json")
        estimator.record_censored_history(work, snapshot)
        JobEstimator(estimator.path).record_censored_history(work, snapshot)
        self.assertEqual(1, len(estimator.samples))
        self.assertEqual(3 * GIB, estimator.estimate(work, margin=1).private_peak_bytes)

    def test_completed_attempt_is_not_interrupted_on_restart(self):
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup())
        history.observe(running_attempts=["a"], cgroup=cgroup())
        history.observe(running_attempts=[], cgroup=cgroup())
        snapshot = RuntimeHistory(self.path).startup(invocation_id="two", cgroup=cgroup())
        self.assertEqual([], snapshot["interrupted_attempts"])

    @patch("ima.research_runtime_history.os.geteuid", return_value=1001)
    def test_actual_user_manager_journal_shapes_idempotent_and_system_manager_supported(self, _):
        entries = [manager("The kernel OOM killer killed some processes in this unit.", "oom"),
                   manager("ima: Failed with result oom-kill", "failure", USER_UNIT=None, _SYSTEMD_USER_UNIT=UNIT),
                   manager("ima: Failed with result 'oom-kill'.", "system", USER_UNIT=None, UNIT=UNIT, _PID="1", _UID="0")]
        history = RuntimeHistory(self.path)
        first = history.startup(invocation_id="one", cgroup=cgroup(), journal_entries=entries)
        second = RuntimeHistory(self.path).startup(invocation_id="two", cgroup=cgroup(), journal_entries=entries)
        self.assertEqual(3, len(first["oom_facts"]))
        self.assertEqual(3, len(second["oom_facts"]))
        self.assertTrue(all(row["source"] == "own_unit_journal" for row in second["oom_facts"]))

    @patch("ima.research_runtime_history.os.geteuid", return_value=1001)
    def test_application_text_other_users_other_units_and_non_oom_results_ignored(self, _):
        oom = "The kernel OOM killer killed some processes in this unit."
        entries = [manager(oom, "application", _COMM="python", _EXE="/usr/bin/python"),
                   manager(oom, "other-user", _UID="1002"),
                   manager(oom, "other-unit", USER_UNIT="other.service"),
                   manager("still running; may be OOM", "speculation"),
                   manager("Failed with result 'exit-code'.", "exit"),
                   manager(oom, "spoof", _EXE="/usr/bin/python"),
                   manager([oom], "non-text"), None]
        snapshot = RuntimeHistory(self.path).startup(cgroup=cgroup(), journal_entries=entries)
        self.assertEqual([], snapshot["oom_facts"])

    @patch("ima.research_runtime_history.subprocess.run")
    def test_controller_methods_never_shell_for_journal(self, run):
        history = RuntimeHistory(self.path)
        history.startup(cgroup=cgroup())
        history.observe(cgroup=cgroup())
        history.snapshot()
        run.assert_not_called()

    def test_optional_unit_and_unsafe_names(self):
        with patch.dict(os.environ, {"IMA_RESEARCH_SYSTEMD_UNIT": ""}):
            snapshot = RuntimeHistory(self.path).startup(cgroup=cgroup(), journal_entries=[manager("Failed with result oom-kill")])
            self.assertEqual([], snapshot["oom_facts"])
        for value in ("other*", "--all.service", "../other.service", "own.service;id", "own service"):
            with self.subTest(value=value), patch.dict(os.environ, {"IMA_RESEARCH_SYSTEMD_UNIT": value}):
                with self.assertRaises(ValueError):
                    RuntimeHistory(self.path)

    def test_observe_before_startup_and_obsolete_invocation_fail(self):
        first = RuntimeHistory(self.path)
        with self.assertRaises(RuntimeError):
            first.observe(cgroup=cgroup())
        first.startup(invocation_id="one", cgroup=cgroup())
        RuntimeHistory(self.path).startup(invocation_id="two", cgroup=cgroup())
        with self.assertRaises(RuntimeError):
            first.observe(cgroup=cgroup())

    def test_failed_atomic_publication_preserves_previous_snapshot(self):
        history = RuntimeHistory(self.path)
        history.startup(invocation_id="one", cgroup=cgroup(), now=1)
        before = history.snapshot()
        with patch("ima.research_runtime_history.os.replace", side_effect=OSError("fixture publication failure")):
            with self.assertRaises(OSError):
                history.observe(cgroup=cgroup(peak=5 * GIB, oom=1, kill=1), now=2)
        self.assertEqual(before, history.snapshot())
        self.assertEqual([], list(self.path.parent.glob("runtime.json-*")))


class OwnJournalReaderTests(unittest.TestCase):
    @patch("ima.research_runtime_history.subprocess.run")
    def test_user_scope_exact_unit_no_other_user_query(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, "loaded\n"),
                           subprocess.CompletedProcess([], 0, json.dumps(manager("Failed with result oom-kill")))]
        self.assertEqual(1, len(read_own_unit_journal(UNIT)))
        args = run.call_args_list[1].args[0]
        self.assertIn("--user", args)
        self.assertIn("--user-unit=" + UNIT, args)
        self.assertNotIn("--system", args)

    @patch("ima.research_runtime_history.os.geteuid", return_value=1001)
    @patch("ima.research_runtime_history.subprocess.run")
    def test_system_scope_requires_own_user(self, run, _):
        run.side_effect = [subprocess.CompletedProcess([], 0, "not-found\n"),
                           subprocess.CompletedProcess([], 0, "1001\n"),
                           subprocess.CompletedProcess([], 0, "")]
        self.assertEqual([], read_own_unit_journal(UNIT))
        self.assertIn("--unit=" + UNIT, run.call_args_list[2].args[0])
        run.reset_mock()
        run.side_effect = [subprocess.CompletedProcess([], 0, "not-found\n"),
                           subprocess.CompletedProcess([], 0, "1002\n")]
        self.assertEqual([], read_own_unit_journal(UNIT))
        self.assertEqual(2, run.call_count)

    @patch("ima.research_runtime_history.subprocess.run")
    def test_unavailable_reader_is_optional_and_unsafe_unit_fails_before_shell(self, run):
        run.side_effect = FileNotFoundError()
        self.assertEqual([], read_own_unit_journal(UNIT))
        run.reset_mock()
        self.assertEqual([], read_own_unit_journal(None))
        with self.assertRaises(ValueError):
            read_own_unit_journal("*.service")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
