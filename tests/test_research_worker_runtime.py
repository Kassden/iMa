import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import Future, TimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ima.research_worker_runtime import (
    BoundedFitExecutor, _boot_identity, _signal_verified, _timed_call,
    cleanup_attempt_group, read_runtime, remaining_deadline,
)


def slow_child(path):
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
    Path(path).write_text(str(child.pid))
    time.sleep(60)


def quick(value):
    return value*2


def delayed_marker(path, delay):
    Path(path).write_text(str(os.getpid()))
    time.sleep(delay)
    return os.getpid()


def child_then_return(path, delay=0):
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
    Path(path).write_text(str(child.pid))
    time.sleep(delay)
    return os.getpid()


def child_then_crash(path):
    child_then_return(path, 2.3)
    os._exit(17)


def stubborn_detached_child(path):
    child = subprocess.Popen([sys.executable,"-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"],
        start_new_session=True)
    Path(path).write_text(str(child.pid))
    time.sleep(60)


def heartbeat_probe():
    from ima import research_worker_runtime as runtime
    sample, update = runtime.observe_process_tree, runtime._update_runtime
    measured = {'samples':[], 'publications':[]}

    def measure(name, function, *args, **kwargs):
        started = time.perf_counter()
        result = function(*args, **kwargs)
        measured[name].append(time.perf_counter()-started)
        return result

    with patch.object(runtime, 'observe_process_tree', side_effect=lambda: measure('samples', sample)), \
         patch.object(runtime, '_update_runtime', side_effect=lambda *a, **k: measure('publications', update, *a, **k)):
        time.sleep(3)
    return measured


def wait_for_file(path, timeout=10):
    until = time.monotonic()+timeout
    while not Path(path).is_file() and time.monotonic() < until:
        time.sleep(.02)
    if not Path(path).is_file():
        raise AssertionError(f"Worker did not publish {path}")


def is_running(pid):
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


class BoundedWorkers(unittest.TestCase):
    def test_timeout_isolated_replacement_and_child_cleanup(self):
        import psutil
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BoundedFitExecutor(2) as pool:
                slow = pool.submit(slow_child,str(root/"child"),runtime_path=root/"slow.json",timeout=3)
                good = pool.submit(quick,3,runtime_path=root/"good.json",timeout=15)
                self.assertEqual(good.result(20),6)
                with self.assertRaises(TimeoutError):
                    slow.result(20)
                cleanup_attempt_group(root/"slow.json")
                self.assertTrue((root/"child").exists(), "Timeout drill must actually start its child")
                pid = int((root/"child").read_text())
                self.assertTrue(not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE)
                self.assertEqual(pool.submit(quick,4,runtime_path=root/"replacement.json",timeout=15).result(20),8)

    def test_restart_does_not_reset_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"runtime.json"
            path.write_text(json.dumps({"deadline_at":"2001-01-01T00:00:00+00:00"}))
            self.assertEqual(remaining_deadline(path,1000),0)
            with BoundedFitExecutor(1) as pool:
                with self.assertRaises(TimeoutError):
                    pool.submit(quick,1,runtime_path=path,timeout=1000).result()

    def test_invalid_deadline_and_foreign_group(self):
        for timeout in (0, -1, float('inf'), float('nan'), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                remaining_deadline("missing",timeout)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"runtime.json"
            path.write_text(json.dumps({"pid":1,"process_group":2}))
            cleanup_attempt_group(path)

    def test_submit_durably_publishes_before_scheduling_and_preserves_restart_budget(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'runtime.json'
            raw = Future()

            def schedule(*args, **kwargs):
                state = read_runtime(path)
                self.assertEqual('queued', state['status'])
                self.assertTrue(state['task_nonce'])
                self.assertLessEqual(datetime.fromisoformat(state['started_at']).timestamp(), time.time())
                self.assertAlmostEqual(60, datetime.fromisoformat(state['deadline_at']).timestamp()
                                       - datetime.fromisoformat(state['started_at']).timestamp(), places=4)
                self.assertGreaterEqual(fsync.call_count, 2)
                return raw

            with patch('pebble.ProcessPool') as pool, patch('ima.research_worker_runtime.os.fsync', wraps=os.fsync) as fsync:
                pool.return_value.schedule.side_effect = schedule
                with BoundedFitExecutor(1) as executor:
                    future = executor.submit(quick, 1, runtime_path=path, timeout=60)
                    future.cancel()
            before = read_runtime(path)
            with patch('pebble.ProcessPool'):
                with BoundedFitExecutor(1) as executor:
                    future = executor.submit(quick, 1, runtime_path=path, timeout=1000)
                    after = read_runtime(path)
                    self.assertEqual(before['started_at'], after['started_at'])
                    self.assertEqual(before['deadline_at'], after['deadline_at'])
                    self.assertNotEqual(before['task_nonce'], after['task_nonce'])
                    future.cancel()

    def test_failed_atomic_publication_never_schedules_work(self):
        with tempfile.TemporaryDirectory() as root, patch('pebble.ProcessPool') as pool:
            path = Path(root)/'runtime.json'
            original = {'started_at':'2001-01-01T00:00:00+00:00', 'deadline_at':'2001-01-02T00:00:00+00:00'}
            path.write_text(json.dumps(original))
            with BoundedFitExecutor(1) as executor:
                with patch('ima.research_worker_runtime.os.replace', side_effect=OSError('publication fault')):
                    with self.assertRaisesRegex(OSError, 'publication fault'):
                        executor.submit(quick, 1, runtime_path=path, timeout=60)
            self.assertEqual(original, read_runtime(path))
            pool.return_value.schedule.assert_not_called()
            self.assertEqual([], list(Path(root).glob('runtime.json-*')))

    def test_queue_expiry_cancels_without_entering_function(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BoundedFitExecutor(1, max_tasks_per_child=3) as executor:
                first = executor.submit(delayed_marker, root/'first', 1,
                                        runtime_path=root/'first.json', timeout=15)
                wait_for_file(root/'first')
                began = time.monotonic()
                queued = executor.submit(delayed_marker, root/'should-not-run', 0,
                                         runtime_path=root/'queued.json', timeout=.2)
                with self.assertRaises(TimeoutError):
                    queued.result(2)
                self.assertLess(time.monotonic()-began, .9)
                self.assertFalse(first.done())
                first.result(10)
            self.assertFalse((root/'should-not-run').exists())

    def test_execution_rechecks_expired_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'runtime.json'
            path.write_text(json.dumps({'deadline_at':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),
                                        'task_nonce':'expired'}))
            function = Mock()
            with patch('ima.research_worker_runtime.os.setsid') as isolate:
                with self.assertRaisesRegex(TimeoutError, 'before execution'):
                    _timed_call(function, (), {}, path, 1000)
            function.assert_not_called()
            isolate.assert_not_called()

    def test_stale_birth_identity_never_receives_a_signal(self):
        identity = {'pid':999123,'create_time':100.,'uid':os.getuid(),
                    'process_group':999123,'session_id':999123}
        process = Mock(pid=identity['pid'])
        process.create_time.return_value = 200.
        process.uids.return_value = SimpleNamespace(real=os.getuid())
        with patch('psutil.Process', return_value=process), \
             patch('ima.research_worker_runtime.os.getpgid', return_value=999123), \
             patch('ima.research_worker_runtime.os.getsid', return_value=999123), \
             patch('ima.research_worker_runtime.os.pidfd_open', return_value=42, create=True), \
             patch('ima.research_worker_runtime.signal.pidfd_send_signal', create=True) as pinned_signal, \
             patch('ima.research_worker_runtime.os.close'):
            _signal_verified(identity, signal.SIGTERM)
        process.send_signal.assert_not_called()
        pinned_signal.assert_not_called()

    def test_boot_mismatch_and_legacy_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'runtime.json'
            for state in ({'pid':999123,'process_group':999123},
                          {'pid':999123,'process_group':999123,'boot_id':'other-boot'}):
                path.write_text(json.dumps(state))
                with patch('psutil.Process') as process, patch('ima.research_worker_runtime._signal_verified') as send:
                    cleanup_attempt_group(path)
                process.assert_not_called()
                send.assert_not_called()

    def test_old_task_nonce_cannot_signal_reused_worker_or_adopt_new_children(self):
        identity = {'pid':999123,'create_time':100.,'uid':os.getuid(),
                    'process_group':999123,'session_id':999123}
        with tempfile.TemporaryDirectory() as root:
            path, guard = Path(root)/'runtime.json', Path(root)/'assignment.json'
            guard.write_text(json.dumps({'task_nonce':'next-task','status':'running'}))
            for status in ('finished', 'failed', 'timed_out', 'running'):
                path.write_text(json.dumps({'pid':999123,'process_group':999123,'boot_id':_boot_identity(),
                                            'task_nonce':'old-task','process_identity':identity,'status':status,'children':[]}))
                with patch('ima.research_worker_runtime._task_guard_path', return_value=guard), \
                     patch('ima.research_worker_runtime._signal_verified') as send, \
                     patch('ima.research_worker_runtime._descendant_identities') as descendants:
                    cleanup_attempt_group(path)
                send.assert_not_called()
                descendants.assert_not_called()

    def test_orphan_cleanup_rejects_reused_child_pid_and_unrecorded_group_members(self):
        import psutil
        leader = {'pid':999120,'create_time':100.,'uid':os.getuid(),
                  'process_group':999120,'session_id':999120}
        child = dict(leader, pid=999121, create_time=101.)
        reused = dict(leader, pid=999122, create_time=102.)
        valid, replacement = Mock(pid=child['pid']), Mock(pid=reused['pid'])
        for process, created in ((valid,101.), (replacement,202.)):
            process.create_time.return_value = created
            process.uids.return_value = SimpleNamespace(real=os.getuid())
            process.children.return_value = []
            process.status.return_value = psutil.STATUS_ZOMBIE

        def process(pid):
            if pid == child['pid']:
                return valid
            if pid == reused['pid']:
                return replacement
            raise psutil.NoSuchProcess(pid)

        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'runtime.json'
            path.write_text(json.dumps({'pid':leader['pid'],'process_group':leader['pid'],
                                        'process_identity':leader,'boot_id':_boot_identity(),
                                        'task_nonce':'old-task','status':'failed','children':[child,reused]}))
            with patch('psutil.Process', side_effect=process), \
                 patch('psutil.process_iter') as enumerate_group, \
                 patch('ima.research_worker_runtime._task_guard_path', return_value=Path(root)/'missing-guard'), \
                 patch('ima.research_worker_runtime.os.getpgid', return_value=leader['pid']), \
                 patch('ima.research_worker_runtime.os.getsid', return_value=leader['pid']), \
                 patch('ima.research_worker_runtime.os.pidfd_open', side_effect=lambda pid: pid+10, create=True), \
                 patch('ima.research_worker_runtime.signal.pidfd_send_signal', create=True) as send, \
                 patch('ima.research_worker_runtime.os.close'):
                cleanup_attempt_group(path)
            self.assertTrue(send.called)
            self.assertEqual({child['pid']+10}, {call.args[0] for call in send.call_args_list})
            replacement.send_signal.assert_not_called()
            enumerate_group.assert_not_called()

    def test_root_reused_during_discovery_cannot_supply_unrelated_children(self):
        identity = {'pid':999123,'create_time':100.,'uid':os.getuid(),
                    'process_group':999123,'session_id':999123}
        with tempfile.TemporaryDirectory() as root:
            path, guard = Path(root)/'runtime.json', Path(root)/'assignment.json'
            guard.write_text(json.dumps({'task_nonce':'old-task','status':'idle'}))
            path.write_text(json.dumps({'pid':999123,'process_group':999123,'boot_id':_boot_identity(),
                                        'task_nonce':'old-task','process_identity':identity,'status':'finished','children':[]}))
            with patch('ima.research_worker_runtime._task_guard_path', return_value=guard), \
                 patch('ima.research_worker_runtime._verified_process', side_effect=[Mock(),None]), \
                 patch('ima.research_worker_runtime._isolated_group_identities', return_value=[dict(identity,pid=999124)]), \
                 patch('ima.research_worker_runtime._signal_verified') as send:
                cleanup_attempt_group(path)
            send.assert_not_called()

    def test_completed_cleanup_preserves_reused_root_and_next_task_children(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BoundedFitExecutor(1, max_tasks_per_child=3) as executor:
                first = executor.submit(child_then_return, root/'old-child',
                                        runtime_path=root/'old.json', timeout=15)
                first_pid = first.result(15)
                second = executor.submit(child_then_return, root/'new-child', 1,
                                         runtime_path=root/'new.json', timeout=15)
                try:
                    wait_for_file(root/'new-child')
                    cleanup_attempt_group(root/'old.json')
                    self.assertFalse(is_running(int((root/'old-child').read_text())))
                    self.assertTrue(is_running(int((root/'new-child').read_text())))
                    self.assertEqual(first_pid, second.result(15))
                finally:
                    second.result(15)
                    cleanup_attempt_group(root/'old.json')
                    cleanup_attempt_group(root/'new.json')

    def test_crashed_root_leaves_verified_children_cleanable(self):
        from pebble import ProcessExpired
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BoundedFitExecutor(1) as executor:
                future = executor.submit(child_then_crash, root/'child', runtime_path=root/'runtime.json', timeout=15)
                with self.assertRaises(ProcessExpired):
                    future.result(15)
                cleanup_attempt_group(root/'runtime.json')
                self.assertFalse(is_running(int((root/'child').read_text())))
                self.assertEqual(6, executor.submit(quick, 3, runtime_path=root/'replacement.json', timeout=15).result(15))

    def test_detached_stubborn_child_is_killed_with_bounded_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with BoundedFitExecutor(1) as executor:
                future = executor.submit(stubborn_detached_child, root/'child', runtime_path=root/'runtime.json', timeout=3)
                with self.assertRaises(TimeoutError):
                    future.result(15)
                began = time.monotonic()
                cleanup_attempt_group(root/'runtime.json')
                self.assertLess(time.monotonic()-began, 3)
                self.assertFalse(is_running(int((root/'child').read_text())))

    def test_economical_heartbeat_sampling_cadence(self):
        with tempfile.TemporaryDirectory() as root:
            with BoundedFitExecutor(1) as executor:
                measured = executor.submit(heartbeat_probe, runtime_path=Path(root)/'runtime.json', timeout=15).result(15)
            self.assertGreaterEqual(len(measured['samples']), 1)
            self.assertLessEqual(len(measured['samples']), 3)
            self.assertLessEqual(len(measured['publications']), 3)
            sampling = sum(measured['samples']) / len(measured['samples'])
            publication = sum(measured['publications']) / max(1,len(measured['publications']))
            print(f"heartbeat probe over 3s: {len(measured['samples'])} full samples, "
                  f"{len(measured['publications'])} publications; "
                  f"mean sampling={sampling*1000:.2f}ms, publication={publication*1000:.2f}ms", flush=True)
