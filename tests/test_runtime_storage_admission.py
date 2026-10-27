import errno
import fcntl
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from ima import research_storage
from ima.research_worker_runtime import _update_runtime, read_runtime


class RuntimeStorageAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.control = self.root / "control"
        self.control.mkdir()
        self.control.chmod(0o700)
        self.path = self.root / "nested" / "runtime.json"
        self.original_authority = research_storage.StorageAuthority

        def authority(root):
            return self.original_authority(root, headroom_bytes=0, headroom_inodes=0)

        self.authority_patch = patch.object(research_storage, "StorageAuthority", side_effect=authority)
        self.authority_patch.start()
        self.addCleanup(self.authority_patch.stop)
        self.env_patch = patch.dict(os.environ, {"IMA_STORAGE_ROOT": str(self.control)})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def reservations(self):
        return research_storage.StorageAuthority(self.control).status()["reservations"]

    @staticmethod
    def is_opened_path(stream, path):
        try:
            opened, current = os.fstat(stream.fileno()), Path(path).stat()
        except (OSError, ValueError):
            return False
        return (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)

    def test_admission_denial_precedes_parent_and_lock_creation(self):
        denial = research_storage.StorageBlocked("no capacity")
        attempted = []

        @contextmanager
        def reject(*args, **kwargs):
            attempted.append(kwargs)
            raise denial
            yield

        with patch.object(research_storage, "admitted_output", reject):
            with self.assertRaises(research_storage.StorageBlocked) as caught:
                _update_runtime(self.path, {"status": "new"})
        self.assertIs(caught.exception, denial)
        self.assertEqual(4, attempted[0]["inodes"])  # missing dir, lock, stage, and target
        self.assertGreaterEqual(attempted[0]["estimated_bytes"], 2 * 64 * 1024)
        self.assertFalse(self.path.parent.exists())
        self.assertFalse(Path(str(self.path) + ".lock").exists())

    def test_reads_merges_and_invokes_callable_under_admission(self):
        self.path.parent.mkdir()
        self.path.write_text('{"kept":1,"status":"old"}')
        seen = []

        def update(state):
            seen.append(dict(state))
            return {"status": "accepted", "call": len(seen)}

        result = _update_runtime(self.path, update)
        self.assertEqual({"kept": 1, "status": "old"}, seen[0])
        self.assertEqual({"kept": 1, "status": "accepted", "call": 1}, result)
        self.assertEqual(result, read_runtime(self.path))
        self.assertEqual({}, self.reservations())
        self.assertFalse(self.path.with_name("runtime.json.tmp").exists())

    def test_bounded_stream_extends_reservation_before_large_json_write(self):
        self.path.parent.mkdir()
        extended = []
        real_extend = self.original_authority.extend

        def record_extend(authority, token, **kwargs):
            extended.append(dict(kwargs))
            return real_extend(authority, token, **kwargs)

        with patch.object(self.original_authority, "extend", record_extend):
            result = _update_runtime(self.path, {"payload": "x" * (256 * 1024)})
        self.assertEqual(256 * 1024, len(result["payload"]))
        self.assertTrue(extended)
        self.assertGreater(extended[0]["extra_bytes"], 0)
        self.assertEqual({}, self.reservations())

    def test_partial_enospc_keeps_accepted_json_and_releases_lease(self):
        self.path.parent.mkdir()
        original = {"status": "accepted", "count": 4}
        self.path.write_text(json.dumps(original))
        real_dump = json.dump
        stage = self.path.with_name("runtime.json.tmp")

        def partial_dump(value, stream, **kwargs):
            if not self.is_opened_path(stream, stage):
                return real_dump(value, stream, **kwargs)
            stream.write('{"status":')
            raise OSError(errno.ENOSPC, "simulated full filesystem")

        with patch("ima.research_worker_runtime.json.dump", side_effect=partial_dump):
            with self.assertRaises(research_storage.StorageBlocked):
                _update_runtime(self.path, {"status": "rejected"})
        self.assertEqual(original, read_runtime(self.path))
        self.assertEqual({}, self.reservations())
        self.assertFalse(self.path.with_name("runtime.json.tmp").exists())
        self.assertTrue(callable(real_dump))

    def test_cleanup_failure_does_not_mask_primary_enospc(self):
        self.path.parent.mkdir()
        original = {"status": "accepted"}
        self.path.write_text(json.dumps(original))
        stage = self.path.with_name("runtime.json.tmp")
        real_unlink = Path.unlink
        real_dump = json.dump

        def fail_stage_cleanup(path, *args, **kwargs):
            if path == self.path.with_name("runtime.json.tmp"):
                raise OSError(errno.EACCES, "simulated cleanup failure")
            return real_unlink(path, *args, **kwargs)

        def partial_dump(value, stream, **kwargs):
            if not self.is_opened_path(stream, stage):
                return real_dump(value, stream, **kwargs)
            stream.write("{")
            raise OSError(errno.ENOSPC, "simulated primary full filesystem")

        with patch("ima.research_worker_runtime.json.dump", side_effect=partial_dump), \
             patch.object(Path, "unlink", fail_stage_cleanup):
            with self.assertRaises(research_storage.StorageBlocked) as caught:
                _update_runtime(self.path, {"status": "new"})
        self.assertEqual(errno.ENOSPC, caught.exception.__cause__.errno)
        self.assertTrue(any("cleanup also failed" in note for note in caught.exception.__notes__))
        self.assertEqual(original, read_runtime(self.path))
        self.assertTrue(self.path.with_name("runtime.json.tmp").exists())
        self.assertEqual({}, self.reservations())

    def test_failed_stage_cleanup_keeps_stable_lock_until_unlink(self):
        self.path.parent.mkdir()
        stage = self.path.with_name("runtime.json.tmp")
        real_dump, real_unlink = json.dump, Path.unlink
        checked = []

        def partial_dump(value, stream, **kwargs):
            if not self.is_opened_path(stream, stage):
                return real_dump(value, stream, **kwargs)
            stream.write("{")
            raise OSError(errno.ENOSPC, "simulated full filesystem")

        def locked_cleanup(path, *args, **kwargs):
            if path == stage:
                with open(str(self.path) + ".lock", "rb") as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                checked.append(True)
            return real_unlink(path, *args, **kwargs)

        with patch("ima.research_worker_runtime.json.dump", side_effect=partial_dump), \
             patch.object(Path, "unlink", locked_cleanup):
            with self.assertRaises(research_storage.StorageBlocked):
                _update_runtime(self.path, {"status": "new"})
        self.assertEqual([True], checked)
        self.assertFalse(stage.exists())

    def test_symlink_and_hardlink_runtime_state_are_refused(self):
        self.path.parent.mkdir()
        external = self.root / "external.json"
        external.write_text('{"status":"external"}')
        self.path.symlink_to(external)
        with self.assertRaises(OSError):
            _update_runtime(self.path, {"status": "changed"})
        self.assertEqual({"status": "external"}, read_runtime(external))
        self.path.unlink()
        os.link(external, self.path)
        with self.assertRaisesRegex(ValueError, "Unsafe runtime state"):
            _update_runtime(self.path, {"status": "changed"})
        self.assertEqual({"status": "external"}, read_runtime(external))

    def test_changed_lock_path_prevents_publication(self):
        self.path.parent.mkdir()
        original = {"status": "accepted"}
        self.path.write_text(json.dumps(original))

        def update(state):
            Path(str(self.path) + ".lock").unlink()
            Path(str(self.path) + ".lock").write_text("replacement lock")
            return {"status": "new"}

        with self.assertRaisesRegex(ValueError, "Runtime lock path changed"):
            _update_runtime(self.path, update)
        self.assertEqual(original, read_runtime(self.path))
        self.assertEqual({}, self.reservations())

    def test_replaced_stage_is_refused_and_cleanup_keeps_replacement(self):
        self.path.parent.mkdir()
        original = {"status": "accepted"}
        self.path.write_text(json.dumps(original))
        stage = self.path.with_name("runtime.json.tmp")
        real_dump = json.dump

        def replace_stage(value, stream, **kwargs):
            real_dump(value, stream, **kwargs)
            if self.is_opened_path(stream, stage):
                os.unlink(stage)
                stage.write_text("replacement")

        with patch("ima.research_worker_runtime.json.dump", side_effect=replace_stage):
            with self.assertRaisesRegex(ValueError, "staging path changed"):
                _update_runtime(self.path, {"status": "new"})
        self.assertEqual("replacement", stage.read_text())
        self.assertEqual(original, read_runtime(self.path))
        self.assertEqual({}, self.reservations())

    def test_disabled_mode_retains_legacy_update_and_temp_naming(self):
        self.env_patch.stop()
        self.addCleanup(self.env_patch.start)
        self.path.parent.mkdir()
        self.path.write_text('{"kept":true}')
        with patch.object(research_storage, "admitted_output", side_effect=AssertionError("unexpected admission")):
            self.assertEqual({"kept": True, "status": "legacy"},
                             _update_runtime(self.path, {"status": "legacy"}))
        self.assertEqual({"kept": True, "status": "legacy"}, read_runtime(self.path))
        self.assertEqual([], list(self.path.parent.glob("runtime.json-*.tmp")))


if __name__ == "__main__":
    unittest.main()
