import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock, Timeout
from scripts.refresh_official_snapshot import refresh, input_digest
from scrapper.official_corpus import PARSER_VERSION


class SnapshotRefreshTests(unittest.TestCase):
    def test_raw_input_changes_trigger_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            raw = root / "raw" / "one.html.gz"
            raw.write_bytes(b"first captured bytes")
            first = input_digest(root, root, [root])
            raw.write_bytes(b"corrected captured bytes")
            self.assertNotEqual(first, input_digest(root, root, [root]))

    @patch("scripts.refresh_official_snapshot.input_digest", return_value="input-a")
    @patch("scripts.refresh_official_snapshot.verify", return_value={"races": 2, "runners": 20})
    @patch("scripts.refresh_official_snapshot.build")
    def test_unchanged_inputs_skip_and_changed_inputs_reuse_verified_base(self, build, verify, digest):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = refresh(root, root, [root], root)
            self.assertEqual(first["action"], "built_and_verified")
            self.assertIsNone(build.call_args.args[0].base_snapshot)
            second = refresh(root, root, [root], root)
            self.assertEqual(second["action"], "unchanged_verified_inputs")
            self.assertEqual(build.call_count, 1)
            digest.return_value = "input-b"
            third = refresh(root, root, [root], root)
            self.assertEqual(build.call_args.args[0].base_snapshot, Path(first["snapshot"]))
            self.assertNotEqual(first["snapshot"], third["snapshot"])
            verify.side_effect = ValueError("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                refresh(root, root, [root], root)
            self.assertEqual(json.loads((root / "latest_verified.json").read_text())["snapshot"], third["snapshot"])
            verify.side_effect = None
            receipt = json.loads((root / "latest_verified.json").read_text())
            receipt["parser_version"] = "old-parser"
            (root / "latest_verified.json").write_text(json.dumps(receipt))
            digest.return_value = "input-c"
            refresh(root, root, [root], root)
            self.assertIsNone(build.call_args.args[0].base_snapshot)

    @patch("scripts.refresh_official_snapshot.input_digest", return_value="input-a")
    @patch("scripts.refresh_official_snapshot.verify", side_effect=ValueError("bad roster"))
    @patch("scripts.refresh_official_snapshot.build")
    def test_failed_readback_never_publishes_receipt(self, build, verify, digest):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "bad roster"):
                refresh(root, root, [root], root)
            self.assertFalse((root / "latest_verified.json").exists())

    def test_concurrent_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with FileLock(root / "refresh.lock"):
                with self.assertRaises(Timeout):
                    refresh(root, root, [root], root)


if __name__ == "__main__":
    unittest.main()
