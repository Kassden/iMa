import json
from pathlib import Path

import pandas as pd
import pytest

from ima.research_objects import ObjectStore, read_snapshot_table, snapshot_storage_estimate
from ima.research_storage import StorageBlocked


def fixture():
    return pd.DataFrame({
        "date": ["2025-01-01", "2023-01-01", "2025-01-02"],
        "race_id": ["a", "b", "c"],
        "horse_no": [1, 2, 3],
        "x": [1.0, 2.0, 3.0],
    })


def test_partition_reuse_projection_and_original_order(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    frame = fixture()
    first = store.publish_table(frame)
    second = store.publish_table(frame.copy())
    assert first == second
    assert len(list(store.root.glob("[0-9a-f][0-9a-f]/*"))) == 2
    pd.testing.assert_frame_equal(store.read_table(first), frame)
    pd.testing.assert_frame_equal(
        store.read_table(first, columns=["x"], positions=[2, 0]),
        frame[["x"]].iloc[[2, 0]].reset_index(drop=True),
    )
    changed = frame.copy()
    changed.loc[0, "x"] = 4
    third = store.publish_table(changed)
    by_year = {ref["sha256"]: ref for ref in first["partitions"]}
    assert first["partitions"][0] in third["partitions"]
    assert first["partitions"][1] not in third["partitions"]
    assert set(by_year) == {ref["sha256"] for ref in first["partitions"]}
    pd.testing.assert_frame_equal(store.read_table(first), frame)


def test_corrupt_or_escaped_object_is_rejected(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    table = store.publish_table(fixture())
    path = store.path(table["partitions"][0])
    path.chmod(0o600)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.read_table(table)
    bad = dict(table["partitions"][0], sha256="../escape")
    with pytest.raises(ValueError):
        store.path(bad)


def test_snapshot_legacy_and_v2_readers_agree_on_values_schema_and_order(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    frame = fixture()
    frame.to_parquet(legacy / "runners.parquet", index=False, compression="zstd")
    (legacy / "manifest.json").write_text("{}")

    new = tmp_path / "new"
    new.mkdir()
    table = ObjectStore(tmp_path / "objects").publish_table(frame)
    (new / "manifest.json").write_text(json.dumps({
        "storage_format": "ima.snapshot.objects.v2",
        "tables": {"runners.parquet": table},
    }))

    old_read = read_snapshot_table(legacy, "runners.parquet")
    object_read = read_snapshot_table(new, "runners.parquet")
    pd.testing.assert_frame_equal(old_read, frame)
    pd.testing.assert_frame_equal(object_read, old_read)
    assert old_read.columns.tolist() == object_read.columns.tolist()


def test_existing_store_read_does_not_construct_storage_authority(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")
    frame = fixture()
    table = store.publish_table(frame)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(json.dumps({
        "storage_format": "ima.snapshot.objects.v2",
        "tables": {"runners.parquet": table},
    }))

    def denied(*args, **kwargs):
        raise StorageBlocked("simulated zero-space reservation denial")

    monkeypatch.setattr("ima.research_storage.StorageAuthority", denied)
    pd.testing.assert_frame_equal(read_snapshot_table(snapshot, "runners.parquet"), frame)
    # Constructor validation of an existing root is also read-only.
    ObjectStore(store.root)


def test_missing_root_is_denied_before_directory_creation(tmp_path, monkeypatch):
    root = tmp_path / "objects"

    def denied(*args, **kwargs):
        raise StorageBlocked("root capacity denied")

    monkeypatch.setattr("ima.research_objects.admitted_output", denied)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    with pytest.raises(StorageBlocked, match="root capacity denied"):
        ObjectStore(root)
    assert not root.exists()


def test_writer_reservation_denial_precedes_temporary_creation(tmp_path, monkeypatch):
    root = tmp_path / "objects"
    store = ObjectStore(root)

    def denied(*args, **kwargs):
        raise StorageBlocked("writer capacity denied")

    monkeypatch.setattr("ima.research_storage.admitted_output", denied)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    with pytest.raises(StorageBlocked, match="writer capacity denied"):
        store.put_table(fixture())
    assert list(root.iterdir()) == []


def test_shard_reservation_denial_precedes_shard_creation(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")

    def admitted(output, **kwargs):
        if kwargs.get("stage") == "snapshot_object_shard":
            raise StorageBlocked("shard capacity denied")
        from contextlib import nullcontext
        return nullcontext()

    monkeypatch.setattr("ima.research_objects.admitted_output", admitted)
    with pytest.raises(StorageBlocked, match="shard capacity denied"):
        store.put_table(fixture())
    assert not list(store.root.glob("[0-9a-f][0-9a-f]"))
    assert not list(store.root.glob(".parquet-*"))


def test_snapshot_estimator_delegates_to_committed_capacity_implementation(tmp_path, monkeypatch):
    expected = 12345
    monkeypatch.setattr(
        "ima.research_dataset_capacity.snapshot_storage_estimate",
        lambda snapshot: expected,
    )
    assert snapshot_storage_estimate(tmp_path / "snapshot") == expected


def test_native_partial_write_cleanup_failure_preserves_storage_blocked(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")
    blocked = StorageBlocked("native Parquet write blocked")

    def partial_then_blocked(self, stream, **kwargs):
        stream.write(b"partial parquet bytes")
        raise blocked

    original_unlink = Path.unlink

    def cleanup_failure(path, *args, **kwargs):
        if path.name.startswith(".parquet-"):
            raise OSError("simulated temporary cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", partial_then_blocked)
        patch.setattr(Path, "unlink", cleanup_failure)
        with pytest.raises(StorageBlocked) as caught:
            store.put_table(fixture())
        assert caught.value is blocked
        assert any('Object temporary retained' in note for note in getattr(blocked,'__notes__',[]))

    assert not list(store.root.glob("[0-9a-f][0-9a-f]/*"))
    for temporary in store.root.glob(".parquet-*"):
        original_unlink(temporary)


def test_publisher_failure_cannot_delete_an_object_already_reused(tmp_path,monkeypatch):
    import errno
    from ima import research_objects as objects, research_storage as storage
    authority=storage.StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda _:authority)
    store=ObjectStore(tmp_path/'objects')
    frame=fixture()
    sync=objects._fsync_directory
    references=[]
    entered=False

    def concurrent_reuse_then_full(path):
        nonlocal entered
        if Path(path).parent==store.root and not entered:
            entered=True
            references.append(store.put_table(frame))
            raise OSError(errno.ENOSPC,'first publisher lost capacity after concurrent acceptance')
        sync(path)

    monkeypatch.setattr(objects,'_fsync_directory',concurrent_reuse_then_full)
    with pytest.raises(StorageBlocked):
        store.put_table(frame)
    assert len(references)==1
    pd.testing.assert_frame_equal(pd.read_parquet(store.path(references[0])),frame)
    assert not list(store.root.glob('.parquet-*'))
    assert authority.status()['reservations']=={}


def test_content_collision_reuses_object_without_mutation(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    frame = fixture()
    first = store.put_table(frame)
    path = store.path(first)
    before_bytes = path.read_bytes()
    before_mode = path.stat().st_mode

    second = store.put_table(frame.copy())

    assert second == first
    assert path.read_bytes() == before_bytes
    assert path.stat().st_mode == before_mode
