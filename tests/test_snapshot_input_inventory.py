import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from ima import research_storage
from ima import snapshot_input_inventory as inventory


def configure_admission(tmp_path, monkeypatch):
    control = tmp_path / "storage-control"
    control.mkdir()
    original = research_storage.StorageAuthority

    def test_authority(root, **kwargs):
        return original(root, headroom_bytes=0, headroom_inodes=0, **kwargs)

    monkeypatch.setattr(research_storage, "StorageAuthority", test_authority)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(control))
    return control


def setup_roots(tmp_path):
    roots = {}
    for name in ("archive", "profiles", "corpora"):
        root = tmp_path / "inputs" / name / "raw"
        root.mkdir(parents=True)
        roots[name] = root
    output_parent = tmp_path / "snapshot-parent"
    output_parent.mkdir()
    return roots, output_parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run_inventory(roots, output_parent, known=()):
    return inventory.inventory_snapshot_inputs(
        raw_roots=roots,
        output_parent=output_parent,
        verified_capture_file_sha256=set(known),
    )


def test_all_explicit_roots_hold_unresolved_exact_bytes_and_reuse_duplicates(
        tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    known_bytes = gzip.compress(b"lineage referenced raw", mtime=0)
    orphan_bytes = b"not a gzip stream; preserve these bytes exactly"
    (roots["archive"] / "known.html.gz").write_bytes(known_bytes)
    (roots["profiles"] / "nested").mkdir()
    (roots["profiles"] / "nested" / "orphan.html.gz").write_bytes(orphan_bytes)
    (roots["corpora"] / "duplicate.gz").write_bytes(orphan_bytes)
    (roots["archive"] / "ignored.txt").write_text("not a gzip raw input")

    first = run_inventory(roots, output_parent, {sha(known_bytes)})
    rows = {row["relative_path"]: row for row in first["files"]}
    assert rows["known.html.gz"]["status"] == "content_referenced"
    assert rows["known.html.gz"]["held_path"] is None
    assert rows["nested/orphan.html.gz"]["status"] == "unresolved_protected_raw"
    assert rows["duplicate.gz"]["status"] == "unresolved_protected_raw"
    for row in first["files"]:
        assert row["deletion_eligible"] is False
    unresolved = rows["nested/orphan.html.gz"]
    assert unresolved["training_eligible"] is False
    assert unresolved["reason"] == "descriptor_missing_or_not_resolved"
    assert unresolved["sha256"] == sha(orphan_bytes)
    assert Path(unresolved["held_path"]).read_bytes() == orphan_bytes
    assert not (output_parent / "unknown-raw" / f"{sha(known_bytes)}.gz").exists()
    assert len(list((output_parent / "unknown-raw").glob("*.gz"))) == 1

    second = run_inventory(roots, output_parent, {sha(known_bytes)})
    assert second == first
    assert len(list((output_parent / "unknown-raw").glob("*.gz"))) == 1
    encoded = inventory.inventory_manifest_bytes(first)
    assert encoded == inventory.inventory_manifest_bytes(second)
    assert first["inventory_sha256"] == sha(encoded)
    assert json.loads(encoded)["files"] == first["files"]


def test_referenced_only_inputs_do_not_create_unknown_raw_root(tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    raw = gzip.compress(b"known", mtime=0)
    (roots["archive"] / "known.gz").write_bytes(raw)

    result = run_inventory(roots, output_parent, {sha(raw)})

    assert result["files"][0]["status"] == "content_referenced"
    assert not (output_parent / "unknown-raw").exists()


def test_source_change_during_copy_fails_and_publishes_no_hold(tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    source = roots["archive"] / "racing.gz"
    original_bytes = b"x" * 32
    source.write_bytes(original_bytes)
    monkeypatch.setattr(inventory, "STREAM_CHUNK_BYTES", 4)
    real_reserved_output_file = inventory.reserved_output_file
    changed = False

    @contextmanager
    def mutate_source_after_first_write(path, mode="wb", **kwargs):
        nonlocal changed
        with real_reserved_output_file(path, mode, **kwargs) as stream:
            class Writer:
                def write(self, data):
                    nonlocal changed
                    result = stream.write(data)
                    if not changed and Path(path).suffix == ".tmp":
                        changed = True
                        source.write_bytes(original_bytes + b"changed")
                    return result

                def __getattr__(self, name):
                    return getattr(stream, name)

            yield Writer()

    monkeypatch.setattr(inventory, "reserved_output_file", mutate_source_after_first_write)
    with pytest.raises(ValueError, match="generation changed|bytes changed"):
        run_inventory(roots, output_parent)
    assert changed
    assert not list((output_parent / "unknown-raw").glob("*.gz"))
    assert not list((output_parent / "unknown-raw").glob("*.tmp"))


@pytest.mark.parametrize("kind", ["tampered", "symlink"])
def test_existing_destination_tamper_or_symlink_fails_closed(tmp_path, monkeypatch, kind):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    raw = b"raw bytes"
    source = roots["archive"] / "orphan.gz"
    source.write_bytes(raw)
    hold_root = output_parent / "unknown-raw"
    hold_root.mkdir(mode=0o700)
    destination = hold_root / f"{sha(raw)}.gz"
    if kind == "tampered":
        destination.write_bytes(b"bad bytes")
        destination.chmod(0o400)
    else:
        destination.symlink_to(source)

    with pytest.raises(ValueError, match="Noncanonical|Unsafe|collision|tampered|size mismatch"):
        run_inventory(roots, output_parent)
    assert source.read_bytes() == raw


def test_unresolved_copy_requires_existing_shared_admission_without_side_effects(
        tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    roots, output_parent = setup_roots(tmp_path)
    source = roots["archive"] / "orphan.gz"
    source.write_bytes(b"unresolved")

    with pytest.raises(RuntimeError, match="shared storage admission"):
        run_inventory(roots, output_parent)
    assert not (output_parent / "unknown-raw").exists()
    assert source.read_bytes() == b"unresolved"


def test_shared_quota_denial_prevents_payload_publication(tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    source = roots["archive"] / "orphan.gz"
    source.write_bytes(b"unresolved")
    hold_root = output_parent / "unknown-raw"
    hold_root.mkdir(mode=0o700)
    (hold_root / ".publication.lock").touch(mode=0o600)

    class DenyingAuthority:
        def __init__(self, root):
            self.root = Path(root)

        @contextmanager
        def admitted(self, **kwargs):
            raise research_storage.StorageBlocked("fixture quota denied")
            yield

    monkeypatch.setattr(research_storage, "StorageAuthority", DenyingAuthority)
    with pytest.raises(research_storage.StorageBlocked, match="quota denied"):
        run_inventory(roots, output_parent)
    assert source.read_bytes() == b"unresolved"
    assert not list(hold_root.glob("*.gz"))
    assert not list(hold_root.glob("*.tmp"))


@pytest.mark.parametrize("target", ["source_root", "source_child", "output_parent"])
def test_noncanonical_or_symlink_roots_fail_without_following_them(tmp_path, monkeypatch, target):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (external / "hidden.gz").write_bytes(b"do not walk")
    if target == "source_root":
        link = tmp_path / "input-link"
        link.symlink_to(roots["archive"], target_is_directory=True)
        roots["archive"] = link
    elif target == "source_child":
        (roots["archive"] / "outside").symlink_to(external, target_is_directory=True)
    else:
        link = tmp_path / "output-link"
        link.symlink_to(output_parent, target_is_directory=True)
        output_parent = link

    with pytest.raises(ValueError, match="Noncanonical|Symlink"):
        run_inventory(roots, output_parent)
    assert (external / "hidden.gz").read_bytes() == b"do not walk"


def test_output_parent_ownership_is_enforced(tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    with pytest.raises(ValueError, match="owned"):
        run_inventory(roots, output_parent)


def test_noncanonical_output_parent_is_rejected(tmp_path, monkeypatch):
    configure_admission(tmp_path, monkeypatch)
    roots, output_parent = setup_roots(tmp_path)
    with pytest.raises(ValueError, match="absolute|normalized|Noncanonical"):
        inventory.inventory_snapshot_inputs(
            raw_roots=roots,
            output_parent=output_parent / ".." / output_parent.name,
            verified_capture_file_sha256=set(),
        )


def test_absent_optional_raw_root_is_reported_without_creating_it(tmp_path):
    from ima.snapshot_input_inventory import inventory_snapshot_inputs
    parent = tmp_path.resolve()
    source = parent / "archive"
    source.mkdir(mode=0o700)
    result = inventory_snapshot_inputs(raw_roots={"archive": source / "raw"},
        output_parent=parent, verified_capture_file_sha256=set())
    assert result["files"] == []
    assert result["roots"] == [{"root_id": "archive", "path": str(source / "raw"),
                                "status": "absent_at_inventory"}]
    assert not (source / "raw").exists()
    assert not (parent / "unknown-raw").exists()
