"""Incremental snapshot tests for capture history and normalized race additions."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ima.research_objects import read_snapshot_table
from scripts import build_official_dataset
from scripts.refresh_official_snapshot import refresh, input_digest
from scrapper.official_corpus import PARSER_VERSION


URL_TEMPLATE = (
    "https://racing.hkjc.com/en-us/local/information/localresults"
    "?racedate={race_date}&Racecourse=ST&RaceNo=1"
)


def race_html(day, horse_a, horse_b):
    date = f"{day:02d}/07/2025"
    return (
        f"<body><p>Race Meeting: {date} Sha Tin</p><p>RACE 1 (512)</p>"
        "<table><tr><td>Class 4 - 1000M</td><td>Going :</td><td>GOOD</td></tr></table>"
        "<table><tr><td>Pla.</td><td>Horse No.</td><td>Horse</td><td>Jockey</td>"
        "<td>Trainer</td><td>Act. Wt.</td><td>Declar. Horse Wt.</td><td>Dr.</td>"
        "<td>LBW</td><td>Finish Time</td><td>Win Odds</td></tr>"
        f'<tr><td>1</td><td>1</td><td><a href="horse?horseid={horse_a}">HORSE ONE</a> '
        f"(CA{horse_a[-3:]})</td><td>J</td><td>T</td><td>120</td><td>1000</td>"
        "<td>1</td><td>---</td><td>0:58.90</td><td>3</td></tr>"
        f'<tr><td>2</td><td>2</td><td><a href="horse?horseid={horse_b}">HORSE TWO</a> '
        f"(CA{horse_b[-3:]})</td><td>J</td><td>T</td><td>120</td><td>1000</td>"
        "<td>2</td><td>1</td><td>0:59.10</td><td>8</td></tr></table></body>"
    ).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


class SnapshotCaptureProvenanceTests(unittest.TestCase):
    def test_changed_object_or_capacity_implementation_invalidates_refresh_fingerprint(self):
        original_read = Path.read_bytes
        for dependency in ('ima/research_objects.py', 'ima/research_object_lifecycle.py',
                           'ima/research_dataset_capacity.py', 'ima/snapshot_input_inventory.py'):
            before = input_digest(self.archive, self.profiles, [self.corpus])
            def changed_read(path):
                return b'changed consumer source' if path.as_posix() == dependency else original_read(path)
            with self.subTest(dependency=dependency), patch.object(Path, 'read_bytes', new=changed_read):
                self.assertNotEqual(before, input_digest(self.archive, self.profiles, [self.corpus]))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.archive = self.root / "archive"
        self.profiles = self.root / "profiles"
        self.corpus = self.root / "corpus"
        self.output = self.root / "snapshots"
        for path in (
            self.archive / "raw",
            self.archive / "normalized",
            self.profiles / "raw",
            self.profiles / "normalized",
            self.corpus / "raw",
            self.corpus / "documents",
        ):
            path.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def add_archive_seed(self, day, horse_a, horse_b):
        date = f"2025-07-{day:02d}"
        body = race_html(day, horse_a, horse_b)
        raw = self.archive / "raw" / date / "ST" / "R1.html.gz"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(gzip.compress(body, mtime=0))
        normalized = self.archive / "normalized" / date / "meeting.json"
        normalized.parent.mkdir(parents=True)
        normalized.write_text(json.dumps([{
            "race_date": date.replace("-", "/"),
            "race_no": 1,
            "source_url": URL_TEMPLATE.format(race_date=date.replace("-", "/")),
            "venue": "ST",
        }]))
        return body

    def add_corpus_capture(self, body, *, day, document_name=None):
        body_hash = sha256(body)
        source_url = URL_TEMPLATE.format(race_date=f"2025/07/{day:02d}")
        raw = self.corpus / "raw" / f"{body_hash}.html.gz"
        raw.write_bytes(gzip.compress(body, mtime=0))
        doc = {
            "body_hash": body_hash,
            "source_url": source_url,
            "fetched_at": "2026-10-28T10:00:00+00:00",
            "first_fetched_at": "2026-10-28T10:00:00+00:00",
            "collector_revision": "snapshot-capture-provenance-test",
        }
        name = document_name or f"capture-{body_hash}-{PARSER_VERSION}.json"
        (self.corpus / "documents" / name).write_text(
            json.dumps(doc, sort_keys=True)
        )
        return body_hash

    def refresh(self):
        return refresh(
            self.archive,
            self.profiles,
            [self.corpus],
            self.output,
            storage_format="objects-v2",
        )

    def assert_lineage_raws_are_exact(self, snapshot, expected_hashes):
        snapshot = Path(snapshot)
        lineage = json.loads((snapshot / "lineage.json").read_text())
        records = {item["body_hash"]: item for item in lineage}
        self.assertTrue(expected_hashes.issubset(records))
        for body_hash in expected_hashes:
            raw_path = Path(records[body_hash]["raw_path"])
            self.assertTrue(raw_path.is_file(), raw_path)
            self.assertEqual(sha256(gzip.decompress(raw_path.read_bytes())), body_hash)

    def assert_snapshot_input_inventory(self, snapshot):
        snapshot = Path(snapshot)
        manifest = json.loads((snapshot / "manifest.json").read_text())
        inventory_record = manifest["source_inputs"]["snapshot_input_inventory"]
        self.assertEqual(inventory_record["path"], "snapshot_input_inventory.json")
        inventory_path = snapshot / inventory_record["path"]
        inventory = json.loads(inventory_path.read_text())
        self.assertNotIn("inventory_sha256", inventory)
        self.assertEqual(sha256(inventory_path.read_bytes()), inventory_record["inventory_sha256"])
        self.assertEqual(manifest["files"][inventory_record["path"]], inventory_record["inventory_sha256"])
        self.assertIn("ima/snapshot_input_inventory.py", manifest["code_hashes"])
        self.assertEqual(manifest["code_hashes"]["ima/snapshot_input_inventory.py"],
                         sha256((snapshot / "code" / "snapshot_input_inventory.py").read_bytes()))
        return inventory

    def test_read_verified_hashes_the_same_compressed_capture_it_parses(self):
        raw = self.archive / "raw" / "2025-07-13" / "ST" / "R1.html.gz"
        raw.parent.mkdir(parents=True)
        parsed_body = race_html(13, "HK_2000_A115", "HK_2000_A116")
        replacement_body = race_html(13, "HK_2000_A117", "HK_2000_A118")
        captured_bytes = gzip.compress(parsed_body, mtime=0)
        replacement_bytes = gzip.compress(replacement_body, mtime=0)
        raw.write_bytes(captured_bytes)
        original_read_bytes = Path.read_bytes
        path_reads = []
        parsed_responses = []

        def read_then_replace(path):
            result = original_read_bytes(path)
            if path == raw:
                path_reads.append(path)
                # Simulate a writer replacing the capture after the builder's
                # one read. Parsing and both hashes must still describe the
                # exact same captured byte sequence.
                raw.write_bytes(replacement_bytes)
            return result

        def capture_response(response):
            parsed_responses.append(response)
            return {"status": "fetched_parsed"}

        with patch.object(Path, "read_bytes", new=read_then_replace), patch.object(
            build_official_dataset, "parse_document", side_effect=capture_response
        ):
            document = build_official_dataset.read_verified(
                raw, URL_TEMPLATE.format(race_date="2025/07/13")
            )

        self.assertEqual(path_reads, [raw])
        self.assertEqual(len(parsed_responses), 1)
        self.assertEqual(sha256(parsed_responses[0].body), sha256(parsed_body))
        self.assertEqual(document["body_hash"], sha256(parsed_body))
        self.assertEqual(document["capture_file_sha256"], sha256(captured_bytes))
        self.assertEqual(sha256(raw.read_bytes()), sha256(replacement_bytes))

    def test_changed_archive_capture_and_reversion_are_replayed_against_base(self):
        original = self.add_archive_seed(13, "HK_2000_A115", "HK_2000_A116")
        first = self.refresh()
        first_snapshot = Path(first['snapshot'])
        first_tree = tree_hash(first_snapshot)
        path = self.archive / 'raw' / '2025-07-13' / 'ST' / 'R1.html.gz'
        changed = original + b'\n<!-- corrected archive capture -->'
        path.write_bytes(gzip.compress(changed, mtime=0))
        second = self.refresh()
        second_snapshot = Path(second['snapshot'])
        second_tree = tree_hash(second_snapshot)
        self.assert_lineage_raws_are_exact(second_snapshot, {sha256(original), sha256(changed)})
        self.assertEqual(set(read_snapshot_table(second_snapshot, 'runners.parquet').source_body_hash),
                         {sha256(changed)})
        path.write_bytes(gzip.compress(original, mtime=0))
        third = self.refresh()
        self.assertEqual(set(read_snapshot_table(Path(third['snapshot']), 'runners.parquet').source_body_hash),
                         {sha256(original)})
        self.assertEqual(tree_hash(first_snapshot), first_tree)
        self.assertEqual(tree_hash(second_snapshot), second_tree)
        self.assert_lineage_raws_are_exact(Path(third['snapshot']), {sha256(original), sha256(changed)})
        inventory = self.assert_snapshot_input_inventory(third["snapshot"])
        self.assertEqual({row["sha256"] for row in inventory["files"]
                          if row["status"] == "content_referenced"},
                         {sha256(gzip.compress(original, mtime=0))})
        # A later unrelated addition must reuse only the currently materialized
        # capture, without parsing every old unchanged archive page again.
        self.add_archive_seed(20, 'HK_2000_A117', 'HK_2000_A118')
        original_reader = build_official_dataset.read_verified
        with patch.object(build_official_dataset, 'read_verified', wraps=original_reader) as replay:
            fourth = self.refresh()
        self.assertEqual(fourth['races'], 2)
        self.assertEqual([call.args[0].parent.parent.name for call in replay.call_args_list], ['2025-07-20'])

    def test_capture_only_revision_preserves_old_and_new_raw_provenance(self):
        original_body = self.add_archive_seed(13, "HK_2000_A115", "HK_2000_A116")
        first = self.refresh()
        first_snapshot = Path(first["snapshot"])
        first_tree = tree_hash(first_snapshot)
        original_hash = sha256(original_body)

        # The collector stores each immutable capture under its content hash.
        # The harmless comment changes raw identity without changing race rows.
        revised_body = original_body + b"\n<!-- later source capture -->\n"
        revised_hash = self.add_corpus_capture(revised_body, day=13)
        second = self.refresh()
        second_snapshot = Path(second["snapshot"])
        second_tree = tree_hash(second_snapshot)

        self.assertNotEqual(first["input_digest"], second["input_digest"])
        self.assertNotEqual(first["snapshot"], second["snapshot"])
        self.assertEqual(json.loads((self.output / "latest_verified.json").read_text())["snapshot"],
                         str(second_snapshot))
        self.assertEqual(tree_hash(first_snapshot), first_tree)
        self.assert_lineage_raws_are_exact(second_snapshot, {original_hash, revised_hash})
        inventory = self.assert_snapshot_input_inventory(second_snapshot)
        self.assertEqual({root["root_id"] for root in inventory["roots"]},
                         {"archive", "profiles", "corpus_000"})
        self.assertTrue(all(row["status"] == "content_referenced" and
                            row["deletion_eligible"] is False for row in inventory["files"]))

        before = read_snapshot_table(first_snapshot, "runners.parquet")
        after = read_snapshot_table(second_snapshot, "runners.parquet")
        pd.testing.assert_frame_equal(
            before.drop(columns=["source_body_hash"]).reset_index(drop=True),
            after.drop(columns=["source_body_hash"]).reset_index(drop=True),
        )
        self.assertEqual(set(after["source_body_hash"]), {revised_hash})

        # A -> B -> A is a real correction cycle. The first capture remains
        # independently addressable in the copied fixture; adding its later
        # recapture uses a new corpus document and never overwrites either
        # prior raw object. Reuse must compare the materialized rows too, or
        # the inherited A lineage entry could incorrectly leave B published.
        archive_raw = self.archive / "raw" / "2025-07-13" / "ST" / "R1.html.gz"
        self.assertEqual(sha256(archive_raw.read_bytes()), sha256(gzip.compress(original_body, mtime=0)))
        recaptured_a_hash = self.add_corpus_capture(
            original_body, day=13, document_name=f"zz-recapture-{original_hash}-{PARSER_VERSION}.json"
        )
        self.assertEqual(recaptured_a_hash, original_hash)
        third = self.refresh()
        third_snapshot = Path(third["snapshot"])
        third_readback = json.loads(Path(third["readback"]).read_text())

        self.assertNotEqual(second["input_digest"], third["input_digest"])
        self.assertNotEqual(second["snapshot"], third["snapshot"])
        self.assertEqual(third_readback["races"], 1)
        self.assertEqual(tree_hash(first_snapshot), first_tree)
        self.assertEqual(tree_hash(second_snapshot), second_tree)
        self.assertEqual(
            json.loads((self.output / "latest_verified.json").read_text())["snapshot"],
            str(third_snapshot),
        )
        self.assert_lineage_raws_are_exact(third_snapshot, {original_hash, revised_hash})
        reverted = read_snapshot_table(third_snapshot, "runners.parquet")
        pd.testing.assert_frame_equal(
            before.drop(columns=["source_body_hash"]).reset_index(drop=True),
            reverted.drop(columns=["source_body_hash"]).reset_index(drop=True),
        )
        self.assertEqual(set(reverted["source_body_hash"]), {original_hash})

    def test_new_normalized_corpus_race_is_added_to_verified_base(self):
        first_body = self.add_archive_seed(13, "HK_2000_A115", "HK_2000_A116")
        first = self.refresh()
        first_snapshot = Path(first["snapshot"])
        first_tree = tree_hash(first_snapshot)
        first_hash = sha256(first_body)

        # New incremental pages enter through corpus documents plus their
        # content-addressed raw capture, matching official_corpus output.
        new_body = race_html(20, "HK_2000_A117", "HK_2000_A118")
        new_hash = self.add_corpus_capture(new_body, day=20)
        second = self.refresh()
        second_snapshot = Path(second["snapshot"])
        readback = json.loads(Path(second["readback"]).read_text())

        self.assertNotEqual(first["input_digest"], second["input_digest"])
        self.assertEqual(second["races"], 2)
        self.assertEqual(readback["races"], 2)
        self.assertEqual(readback["runners"], 4)
        self.assertEqual(tree_hash(first_snapshot), first_tree)
        self.assertEqual(json.loads((self.output / "latest_verified.json").read_text())["snapshot"],
                         str(second_snapshot))
        self.assert_lineage_raws_are_exact(second_snapshot, {first_hash, new_hash})

    def test_invalid_new_archive_race_is_quarantined_but_capture_is_retained_and_retried(self):
        old_body = self.add_archive_seed(13, "HK_2000_A115", "HK_2000_A116")
        first = self.refresh()
        first_snapshot = Path(first["snapshot"])
        first_tree = tree_hash(first_snapshot)
        old_hash = sha256(old_body)

        # This follows the archive normalized/raw contract. The HTML is valid
        # and parseable but has no first-place runner, so race_rows must reject
        # it after the raw capture has entered lineage.
        invalid_body = race_html(20, "HK_2000_A117", "HK_2000_A118").replace(
            b"<td>1</td><td>1</td>", b"<td>2</td><td>1</td>", 1
        )
        invalid_hash = sha256(invalid_body)
        invalid_raw = self.archive / "raw" / "2025-07-20" / "ST" / "R1.html.gz"
        invalid_raw.parent.mkdir(parents=True)
        invalid_raw.write_bytes(gzip.compress(invalid_body, mtime=0))
        normalized = self.archive / "normalized" / "2025-07-20" / "meeting.json"
        normalized.parent.mkdir(parents=True)
        normalized.write_text(json.dumps([{
            "race_date": "2025/07/20", "race_no": 1,
            "source_url": URL_TEMPLATE.format(race_date="2025/07/20"), "venue": "ST",
        }]))

        second = self.refresh()
        second_snapshot = Path(second["snapshot"])
        second_tree = tree_hash(second_snapshot)
        readback = json.loads(Path(second["readback"]).read_text())
        exclusions = json.loads((second_snapshot / "exclusions.json").read_text())

        self.assertEqual(readback["races"], 1)
        self.assertEqual(readback["runners"], 2)
        self.assertTrue(any(item["path"] == str(invalid_raw) and "No winner" in item["reason"]
                            for item in exclusions))
        self.assertEqual(tree_hash(first_snapshot), first_tree)
        self.assert_lineage_raws_are_exact(second_snapshot, {old_hash, invalid_hash})
        invalid_lineage = next(
            item for item in json.loads((second_snapshot / "lineage.json").read_text())
            if item["body_hash"] == invalid_hash
        )
        self.assertEqual(sha256(Path(invalid_lineage["raw_path"]).read_bytes()),
                         sha256(invalid_raw.read_bytes()))

        # Change the input fingerprint while leaving the quarantined capture
        # untouched. Its race is absent from materialized rows, so the known
        # lineage tuple must not suppress a fresh parse attempt.
        changed_old_capture = self.add_corpus_capture(
            old_body + b"\n<!-- harmless recapture -->\n", day=13
        )
        original_read_verified = build_official_dataset.read_verified
        with patch.object(build_official_dataset, "read_verified", wraps=original_read_verified) as replay:
            third = self.refresh()
        self.assertNotEqual(second["input_digest"], third["input_digest"])
        self.assertEqual(tree_hash(second_snapshot), second_tree)
        self.assertTrue(any(str(call.args[0]) == str(invalid_raw) for call in replay.call_args_list))
        self.assert_lineage_raws_are_exact(third["snapshot"], {old_hash, invalid_hash, changed_old_capture})


if __name__ == "__main__":
    unittest.main()
