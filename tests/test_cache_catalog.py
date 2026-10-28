import os
import stat
import hashlib
from types import SimpleNamespace

import pytest

from ima.research_cache_catalog import import_cache_inventory
from ima import research_cache_catalog as catalog
from ima.research_preparation import PreparationCache
from ima.research_storage import StorageAuthority, StorageBlocked
from tests.test_research_preparation import build, key


def _authority(path, *, limit=50 * 1024**3):
    return StorageAuthority(
        path,
        headroom_bytes=0,
        headroom_inodes=0,
        cache_limit_bytes=limit,
        cache_soft_bytes=limit,
        cache_target_bytes=limit,
        statvfs=lambda _: SimpleNamespace(
            f_frsize=1, f_blocks=10**12, f_bavail=10**12, f_favail=10**9
        ),
    )


def _cache(root, *, seed):
    return PreparationCache(root).prepare_fold(key(seed=seed), build)


def _payload_snapshot(path):
    result = {}
    for item in path.iterdir():
        info = item.lstat()
        contents = item.read_bytes() if stat.S_ISREG(info.st_mode) else None
        result[item.name] = (info.st_ino, info.st_size, info.st_mtime_ns,
                             contents, stat.S_ISLNK(info.st_mode))
    return result


def test_imports_all_explicit_roots_before_quota_blocks_growth(tmp_path):
    first = tmp_path / "cache-one"
    second = tmp_path / "cache-two"
    first.mkdir()
    second.mkdir()
    one = _cache(first, seed=1)
    two = _cache(second, seed=2)
    observed = sum(
        item.stat().st_blocks * 512 for artifact in (one, two)
        for item in (artifact.path, *artifact.path.iterdir())
    )
    authority = _authority(tmp_path / "authority", limit=max(1, observed - 1))

    report = import_cache_inventory(authority, [first, second])

    assert report["qualified_artifact_count"] == 2
    assert report["registered_artifact_count"] == 2
    assert report["observed_allocated_bytes"] == observed
    assert isinstance(report["max_mtime_ns"], int)
    assert isinstance(report["max_atime_ns"], int)
    assert report["quota_over_limit"]
    assert report["admission_ready"]
    assert authority.status()["cache_bytes"] >= observed
    with pytest.raises(StorageBlocked, match="cache"):
        authority.reserve(bytes_required=0, inodes_required=0, stage="test",
                          output=tmp_path, cache_bytes=1)


def test_import_is_idempotent_and_preserves_last_access_and_high_water_bytes(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    artifact = _cache(root, seed=3)
    authority = _authority(tmp_path / "authority")
    path = artifact.path
    authority.record_cache(root, artifact.artifact_id, 10**8, enforce_limit=False)
    with authority._locked() as state:
        state["caches"][str(path)]["last_access"] = 1234.5
        from ima.research_storage import _atomic_json
        _atomic_json(authority.root / "state.json", state)
    before_payload = _payload_snapshot(path)

    first = import_cache_inventory(authority, [root])
    second = import_cache_inventory(authority, [root])

    import json
    state = json.loads((authority.root / "state.json").read_text())
    imported = state["caches"][str(path)]
    assert imported["last_access"] == 1234.5
    assert imported["allocated_bytes"] == 10**8
    assert first["registered_artifact_count"] == second["registered_artifact_count"] == 1
    assert second["registered_inventory_bytes"] == 10**8
    assert _payload_snapshot(path) == before_payload


def test_live_pinned_artifact_is_reported_and_accounted_without_mutation(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    cache = PreparationCache(root)
    artifact = cache.prepare_fold(key(seed=4), build)
    authority = _authority(tmp_path / "authority")
    before = _payload_snapshot(artifact.path)

    with cache.pin(artifact):
        report = import_cache_inventory(authority, [root])

    assert report["pinned_artifacts"] == [str(artifact.path)]
    assert report["admission_ready"]
    assert authority.status()["cache_bytes"] > 0
    assert _payload_snapshot(artifact.path) == before


@pytest.mark.parametrize("entry_kind", ["model", "symlink"])
def test_unknown_or_linked_payload_is_held_not_registered(tmp_path, entry_kind):
    root = tmp_path / "cache"
    root.mkdir()
    artifact = _cache(root, seed=5)
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"protected bytes")
    if entry_kind == "model":
        (artifact.path / "model.bin").write_bytes(b"unknown model bytes")
    else:
        (artifact.path / "extra-link").symlink_to(outside)
    before = _payload_snapshot(artifact.path)
    outside_before = outside.read_bytes()
    authority = _authority(tmp_path / "authority")

    report = import_cache_inventory(authority, [root])

    assert not report["admission_ready"]
    assert report["qualified_artifact_count"] == 0
    assert report["held_allocated_bytes"] > 0
    assert report["held_accounting_complete"]
    assert authority.status()["cache_bytes"] == 0
    assert _payload_snapshot(artifact.path) == before
    assert outside.read_bytes() == outside_before


def test_unknown_non_hash_root_directory_is_held_and_blocks_registration(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    artifact = _cache(root, seed=51)
    unknown = root / "legacy-extra"
    unknown.mkdir()
    (unknown / "payload").write_bytes(b"unclassified")
    authority = _authority(tmp_path / "authority")
    before_state = (authority.root / "state.json").read_bytes() if (authority.root / "state.json").exists() else None

    report = import_cache_inventory(authority, [root])

    assert not report["admission_ready"]
    assert report["qualified_artifact_count"] == 1
    assert report["registered_artifact_count"] == 0
    assert report["qualified_artifacts"] == [str(artifact.path)]
    assert any(row["path"] == str(unknown) and row["reason"] == "unknown_root_entry"
               for row in report["held_artifacts"])
    after_state = (authority.root / "state.json").read_bytes() if (authority.root / "state.json").exists() else None
    assert after_state == before_state


def test_protected_native_artifact_is_qualified_but_never_registered(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    artifact = _cache(root, seed=52)
    authority = _authority(tmp_path / "authority")
    prior = _cache(root, seed=53)
    authority.record_cache(root, prior.artifact_id, 12345, enforce_limit=False)
    state_path = authority.root / "state.json"
    before_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()

    report = import_cache_inventory(authority, [root], protected_paths=[artifact.path])

    assert not report["admission_ready"]
    assert report["qualified_artifact_count"] == 2
    assert report["registered_artifact_count"] == 0
    assert {str(artifact.path), str(prior.path)} == set(report["qualified_artifacts"])
    assert any(row["path"] == str(artifact.path)
               and row["reason"] == "explicit_protected_native_artifact"
               for row in report["held_artifacts"])
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == before_hash


def test_incomplete_held_measurement_leaves_existing_authority_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    _cache(root, seed=54)
    (root / "unknown.tmp").write_bytes(b"unknown")
    authority = _authority(tmp_path / "authority")
    authority.record_cache(root, "f" * 64, 9876, enforce_limit=False)
    state_path = authority.root / "state.json"
    before = state_path.read_bytes()
    monkeypatch.setattr(catalog, "_measure_unknown", lambda *a, **k: {
        "allocated_bytes": 1, "inode_count": 1, "complete": False})

    report = import_cache_inventory(authority, [root])

    assert not report["admission_ready"]
    assert report["qualified_artifact_count"] == 1
    assert report["registered_artifact_count"] == 0
    assert not report["held_accounting_complete"]
    assert state_path.read_bytes() == before


def test_unknown_tree_measurement_stops_at_depth_and_foreign_device(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    cursor = root / "unknown"
    cursor.mkdir()
    for index in range(catalog._UNKNOWN_MAX_DEPTH + 4):
        cursor = cursor / f"d{index}"
        cursor.mkdir()

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        bounded = catalog._measure_unknown(
            root_fd, "unknown", root.stat().st_dev, os.getuid())
        foreign = catalog._measure_unknown(
            root_fd, "unknown", root.stat().st_dev + 1, os.getuid())
    finally:
        os.close(root_fd)

    assert not bounded["complete"]
    assert bounded["inode_count"] <= catalog._UNKNOWN_MAX_DEPTH + 1
    assert not foreign["complete"]
    assert foreign["inode_count"] == 1


def test_symlink_artifact_entry_is_held_without_following_target(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "payload").write_bytes(b"must not be traversed")
    identity = "a" * 64
    (root / identity).symlink_to(outside, target_is_directory=True)
    authority = _authority(tmp_path / "authority")

    report = import_cache_inventory(authority, [root])

    assert not report["admission_ready"]
    assert report["held_artifacts"][0]["reason"] == "linked_or_unopenable_artifact"
    assert report["held_allocated_bytes"] >= (root / identity).lstat().st_blocks * 512
    assert (outside / "payload").read_bytes() == b"must not be traversed"


def test_foreign_or_noncanonical_root_fails_before_authority_write(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    alias = tmp_path / "cache-alias"
    alias.symlink_to(root, target_is_directory=True)
    authority = _authority(tmp_path / "authority")

    with pytest.raises(ValueError, match="canonical"):
        import_cache_inventory(authority, [alias])
    assert not (authority.root / "state.json").exists()


def test_protected_paths_are_reported_and_block_without_authority_mutation(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    artifact = _cache(root, seed=6)
    protected = tmp_path / "campaign-output"
    protected.mkdir()
    authority = _authority(tmp_path / "authority")

    before_state = (authority.root / "state.json").read_bytes() if (authority.root / "state.json").exists() else None
    report = import_cache_inventory(authority, [root], protected_paths=[protected])

    assert report["protected_paths"] == [str(protected)]
    assert not report["admission_ready"]
    assert report["qualified_artifact_count"] == 1
    assert report["registered_artifact_count"] == 0
    after_state = (authority.root / "state.json").read_bytes() if (authority.root / "state.json").exists() else None
    assert after_state == before_state
    assert authority.status()["cache_bytes"] == 0
    assert artifact.path.is_dir()


@pytest.mark.parametrize('suffix',['.lock','.pin.lock'])
def test_nonempty_reserved_lock_name_is_held_and_never_hidden(tmp_path,suffix):
    root=tmp_path/'cache';root.mkdir()
    artifact=_cache(root,seed=9)
    unexpected=root/(artifact.artifact_id+suffix)
    unexpected.write_bytes(b'arbitrary bytes are not a native empty lock')
    authority=_authority(tmp_path/'authority')
    report=import_cache_inventory(authority,[root])
    assert report['admission_ready'] is False
    assert report['registered_artifact_count']==0
    assert any(row['path']==str(unexpected) and row['reason']=='unexpected_lock_payload'
               for row in report['held_artifacts'])
    assert unexpected.read_bytes()==b'arbitrary bytes are not a native empty lock'
    assert not (authority.root/'state.json').exists()
