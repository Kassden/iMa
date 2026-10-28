import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pandas as pd
import pytest

from ima.research_objects import ObjectStore, read_snapshot_table
from ima.research_object_lifecycle import LifecycleHold, ProtectedRoot, archive_object, lifecycle_lease, plan_retention, restore_object
from ima.research_storage import StorageAuthority

KINDS = ("mlflow", "hypothesis", "confirmation", "pending", "models", "raw", "unique_history")

def roots(base):
    result = []
    for kind in KINDS:
        path = base / "protected" / kind
        path.mkdir(parents=True, exist_ok=True)
        result.append(ProtectedRoot(kind, path))
    return result

def frame():
    return pd.DataFrame({"date": ["2024-01-01", "2025-02-02"], "race_id": ["r1", "r2"],
                         "horse_no": [1, 2], "x": [1.5, 2.5]})

def snapshot(base, store):
    path = base / "snapshot"
    path.mkdir()
    table = store.publish_table(frame())
    (path / "manifest.json").write_text(json.dumps({"storage_format": "ima.snapshot.objects.v2",
        "tables": {"runners.parquet": table}}))
    return path, table

def test_unknown_future_manifest_and_unreadable_reference_hold(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    path, _ = snapshot(tmp_path, store)
    mp = path / "manifest.json"
    data = json.loads(mp.read_text())
    data["storage_format"] = "ima.snapshot.objects.v99"
    mp.write_text(json.dumps(data))
    with pytest.raises(LifecycleHold, match="Unsupported snapshot format"):
        plan_retention(object_root=store.root, snapshot_roots=[path], protected_roots=roots(tmp_path))
    data["storage_format"] = "ima.snapshot.objects.v2"
    data["tables"]["runners.parquet"]["partitions"][0]["sha256"] = "f" * 64
    mp.write_text(json.dumps(data))
    with pytest.raises(LifecycleHold, match="unreadable"):
        plan_retention(object_root=store.root, snapshot_roots=[path], protected_roots=roots(tmp_path))

def test_registry_source_and_parent_chain_mark_objects(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    source, table = snapshot(tmp_path, store)
    registry = tmp_path / "registry"
    datasets = registry / "datasets"
    for name, parent in (("d1", None), ("d2", "d1")):
        item = datasets / name
        item.mkdir(parents=True)
        (item / "manifest.json").write_text(json.dumps({"dataset_id": name, "parent_dataset_id": parent,
                                                        "source_snapshot": str(source)}))
    report = plan_retention(object_root=store.root, dataset_registry_roots=[registry],
                            protected_roots=roots(tmp_path), now=time.time() + 30 * 86400)
    assert report["reference_count"] == len(table["partitions"])
    assert all(row["referenced"] and row["status"] == "held" for row in report["objects"])
    (datasets / "d2" / "manifest.json").write_text(json.dumps({"dataset_id": "d2",
        "parent_dataset_id": "absent", "source_snapshot": str(source)}))
    with pytest.raises(LifecycleHold, match="parent chain"):
        plan_retention(object_root=store.root, dataset_registry_roots=[registry], protected_roots=roots(tmp_path))

def test_missing_roots_incomplete_publication_and_grace_hold(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    with pytest.raises(LifecycleHold, match="Explicit MLflow"):
        plan_retention(object_root=store.root)
    store.put_table(frame())
    fresh = plan_retention(object_root=store.root, protected_roots=roots(tmp_path))
    assert all("initial_grace" in row["hold_reasons"] for row in fresh["objects"])
    report = plan_retention(object_root=store.root, protected_roots=roots(tmp_path),
                            now=time.time() + 30 * 86400)
    assert report["review_eligible"] == [] and report["phase_complete"] is False
    assert all(row["status"] == "held" for row in report["objects"])
    (store.root / ".parquet-partial").write_bytes(b"partial")
    with pytest.raises(LifecycleHold, match="Incomplete object publication"):
        plan_retention(object_root=store.root, protected_roots=roots(tmp_path))

def test_shared_lease_serializes_planner_and_crash_releases_kernel_lock(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
    def reader():
        with lifecycle_lease(store.root):
            entered.set()
            release.wait(3)
    def planner():
        with lifecycle_lease(store.root, exclusive=True):
            acquired.set()
    a = threading.Thread(target=reader); a.start()
    assert entered.wait(2)
    b = threading.Thread(target=planner); b.start()
    time.sleep(.1)
    assert not acquired.is_set()
    release.set(); a.join(2); b.join(2)
    assert acquired.is_set()
    code = ("import os; from ima.research_object_lifecycle import lifecycle_lease; "
            f"lease=lifecycle_lease({str(store.root)!r}); lease.__enter__(); os._exit(0)")
    subprocess.run([sys.executable, "-c", code], check=True)
    with lifecycle_lease(store.root, exclusive=True): pass

def test_actual_reader_and_object_publisher_wait_for_exclusive_marker(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")
    snap, _ = snapshot(tmp_path, store)
    original_read = ObjectStore.read_table
    entered_read = threading.Event()
    def gated_read(self, *args, **kwargs):
        entered_read.set()
        return original_read(self, *args, **kwargs)
    monkeypatch.setattr(ObjectStore, "read_table", gated_read)
    completed = threading.Event()
    def read():
        read_snapshot_table(snap, "runners.parquet")
        completed.set()
    with lifecycle_lease(store.root, exclusive=True):
        reader = threading.Thread(target=read); reader.start()
        time.sleep(.1)
        assert not entered_read.is_set()
        assert not completed.is_set()
    reader.join(3)
    assert entered_read.is_set() and completed.is_set()

    with lifecycle_lease(store.root, exclusive=True):
        publisher = threading.Thread(target=lambda: store.put_table(frame()))
        publisher.start()
        time.sleep(.1)
        assert publisher.is_alive()
    publisher.join(3)
    assert not publisher.is_alive()

def test_archive_failure_corruption_and_same_filesystem_zero_reclaim(tmp_path, monkeypatch):
    import ima.research_object_lifecycle as lifecycle
    store = ObjectStore(tmp_path / "objects")
    table = store.publish_table(frame())
    ref = table["partitions"][0]
    authority = StorageAuthority(tmp_path / "authority", headroom_bytes=0, headroom_inodes=0)
    destdir = tmp_path / "archive"; destdir.mkdir()
    dest = destdir / ref["sha256"]
    monkeypatch.setattr(lifecycle.shutil, "copyfileobj", lambda *_a, **_k: (_ for _ in ()).throw(OSError("copy failed")))
    with pytest.raises(OSError, match="copy failed"):
        archive_object(object_root=store.root, reference=ref, destination=dest, authority=authority)
    assert not dest.exists() and not list(destdir.glob(".*.tmp"))
    monkeypatch.undo()
    receipt = archive_object(object_root=store.root, reference=ref, destination=dest, authority=authority)
    assert receipt["same_filesystem"] and receipt["reclaimed_bytes"] == 0
    dest.chmod(0o600); dest.write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        restore_object(archive_path=dest, object_root=store.root, reference=ref, authority=authority)

def test_restore_checksum_and_actual_snapshot_consumer_parity(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    original = frame()
    snap, table = snapshot(tmp_path, store)
    ref = table["partitions"][0]
    authority = StorageAuthority(tmp_path / "authority", headroom_bytes=0, headroom_inodes=0)
    archive_dir = tmp_path / "archive"; archive_dir.mkdir()
    archived = archive_dir / ref["sha256"]
    receipt = archive_object(object_root=store.root, reference=ref, destination=archived, authority=authority)
    assert receipt["source_deleted"] is False and receipt["reclaimed_bytes"] == 0
    store.path(ref).unlink()  # isolated temporary fixture simulates an absent object
    restored = restore_object(archive_path=archived, object_root=store.root, reference=ref, authority=authority)
    assert restored["status"] == "restored_verified"
    pd.testing.assert_frame_equal(read_snapshot_table(snap, "runners.parquet"), original)

def test_symlinked_shard_holds_inventory_and_refuses_restore_before_writes(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    snap, table = snapshot(tmp_path, store)
    ref = table["partitions"][0]
    authority = StorageAuthority(tmp_path / "authority", headroom_bytes=0, headroom_inodes=0)
    archive = tmp_path / "archive"
    archive_object(object_root=store.root, reference=ref, destination=archive, authority=authority)
    shard = store.root / ref["sha256"][:2]
    outside = tmp_path / "outside-shard"
    shard.rename(outside)
    shard.symlink_to(outside, target_is_directory=True)
    with pytest.raises(LifecycleHold, match="Noncanonical object shard"):
        plan_retention(object_root=store.root, snapshot_roots=[snap], protected_roots=roots(tmp_path))
    with pytest.raises(LifecycleHold, match="Unknown object-store entry"):
        plan_retention(object_root=store.root, protected_roots=roots(tmp_path))
    (outside / ref["sha256"]).unlink()
    with pytest.raises(ValueError, match="canonical"):
        restore_object(archive_path=archive, object_root=store.root, reference=ref, authority=authority)
    assert list(outside.iterdir()) == []

def test_restore_collision_never_removes_preexisting_temporary(tmp_path, monkeypatch):
    import ima.research_object_lifecycle as lifecycle
    store = ObjectStore(tmp_path / "objects")
    ref = store.publish_table(frame())["partitions"][0]
    authority = StorageAuthority(tmp_path / "authority", headroom_bytes=0, headroom_inodes=0)
    archive = tmp_path / "archive"
    archive_object(object_root=store.root, reference=ref, destination=archive, authority=authority)
    store.path(ref).unlink()
    collision = store.root / ref["sha256"][:2] / f'.{ref["sha256"]}.0000000000000000.tmp'
    collision.write_bytes(b"belongs to another publisher")
    monkeypatch.setattr(lifecycle.os, "urandom", lambda size: bytes(size))
    with pytest.raises(FileExistsError):
        restore_object(archive_path=archive, object_root=store.root, reference=ref, authority=authority)
    assert collision.read_bytes() == b"belongs to another publisher"

def test_archive_collision_never_removes_preexisting_temporary(tmp_path, monkeypatch):
    import ima.research_object_lifecycle as lifecycle
    store = ObjectStore(tmp_path / "objects")
    ref = store.publish_table(frame())["partitions"][0]
    authority = StorageAuthority(tmp_path / "authority", headroom_bytes=0, headroom_inodes=0)
    archive = tmp_path / "archive"
    collision = tmp_path / ".archive.0000000000000000.tmp"
    collision.write_bytes(b"belongs to another publisher")
    monkeypatch.setattr(lifecycle.os, "urandom", lambda size: bytes(size))
    with pytest.raises(FileExistsError):
        archive_object(object_root=store.root, reference=ref, destination=archive, authority=authority)
    assert collision.read_bytes() == b"belongs to another publisher"
    assert not archive.exists()

def test_new_lifecycle_lock_is_admitted_before_creation(tmp_path, monkeypatch):
    import ima.research_object_lifecycle as lifecycle
    from ima.research_storage import StorageBlocked
    from contextlib import contextmanager
    store = ObjectStore(tmp_path / "objects")
    lock = tmp_path / ".ima-object-lifecycle.lock"
    @contextmanager
    def denied(*args, **kwargs):
        assert not lock.exists()
        raise StorageBlocked("denied lock admission")
        yield
    monkeypatch.setattr(lifecycle, "admitted_output", denied)
    with pytest.raises(StorageBlocked, match="denied lock admission"):
        with lifecycle_lease(store.root):
            pass
    assert not lock.exists()
