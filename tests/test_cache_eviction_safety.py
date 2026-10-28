import os
from pathlib import Path

import pytest

from ima import research_preparation
from ima.research_preparation import PreparationCache
from ima.research_storage import StorageBlocked
from tests.test_research_preparation import build, key


def _snapshot_tree(path):
    result = {}
    for child in path.iterdir():
        info = child.lstat()
        result[child.name] = (
            info.st_mode,
            info.st_nlink,
            child.readlink() if child.is_symlink() else child.read_bytes(),
        )
    return result


def test_native_prepared_fold_eviction_retains_stable_lock_inodes(tmp_path):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    assert set(path.name for path in artifact.path.iterdir()) == {
        "manifest.json", "arrays.joblib", "fitted.joblib",
    }
    lock_paths = [
        cache.root / (artifact.artifact_id + ".pin.lock"),
        cache.root / (artifact.artifact_id + ".lock"),
    ]
    with cache.pin(artifact):
        assert cache.evict(artifact.artifact_id) is False
    inodes = [(path.stat().st_dev, path.stat().st_ino) for path in lock_paths]

    assert cache.evict(artifact.artifact_id)

    assert not artifact.path.exists()
    assert [(path.stat().st_dev, path.stat().st_ino) for path in lock_paths] == inodes


@pytest.mark.parametrize("entry_kind", ["model", "symlink", "hardlink"])
def test_unknown_or_linked_entry_holds_whole_artifact_unchanged(tmp_path, entry_kind):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    payload = artifact.path
    arrays = payload / "arrays.joblib"
    outside = tmp_path / "protected-model-payload.bin"

    if entry_kind == "model":
        extra = payload / "model.bin"
        extra.write_bytes(b"model bytes must be held")
    elif entry_kind == "symlink":
        outside.write_bytes(b"outside target must be held")
        extra = payload / "unrecognized-link"
        extra.symlink_to(outside)
    else:
        outside.write_bytes(b"outside hardlink payload must be held")
        extra = payload / "unrecognized-hardlink"
        os.link(outside, extra)

    before = _snapshot_tree(payload)
    outside_before = outside.read_bytes() if outside.exists() else None
    assert cache.evict(artifact.artifact_id) is False

    assert payload.is_dir()
    assert _snapshot_tree(payload) == before
    if outside_before is not None:
        assert outside.read_bytes() == outside_before


def test_active_pin_blocks_eviction_without_changing_artifact(tmp_path):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    before = _snapshot_tree(artifact.path)

    with cache.pin(artifact):
        assert cache.evict(artifact.artifact_id) is False
        assert artifact.path.is_dir()
        assert _snapshot_tree(artifact.path) == before


def test_capacity_denial_precedes_missing_eviction_lock_creation(tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    pin_lock = cache.root / (artifact.artifact_id + ".pin.lock")
    key_lock = cache.root / (artifact.artifact_id + ".lock")
    key_lock_inode = (key_lock.stat().st_dev, key_lock.stat().st_ino)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "storage-control"))

    def deny_capacity(*args, **kwargs):
        raise StorageBlocked("eviction lock inode admission denied")

    monkeypatch.setattr("ima.research_storage.StorageAuthority", deny_capacity)
    # Plain configured eviction holds missing metadata without trying to
    # recursively reserve under an authority lock. Explicit provisioning owns
    # the admitted creation path, outside the authority critical section.
    assert cache.evict(artifact.artifact_id) is False
    with pytest.raises(StorageBlocked, match="inode admission denied"):
        cache.prepare_eviction(artifact.artifact_id)

    assert not pin_lock.exists()
    assert (key_lock.stat().st_dev, key_lock.stat().st_ino) == key_lock_inode
    assert artifact.path.is_dir()


def test_directory_swap_after_initial_validation_never_deletes_outside_files(
        tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "arrays.joblib").write_bytes(b"protected array")
    (outside / "manifest.json").write_bytes(b"protected manifest")
    before = {path.name: path.read_bytes() for path in outside.iterdir()}
    displaced = tmp_path / "displaced-cache-artifact"
    original_open_locks = research_preparation._open_eviction_locks
    swapped = False

    def swap_after_validation(root, identity):
        nonlocal swapped
        descriptors = original_open_locks(root, identity)
        if descriptors is not None and not swapped:
            artifact.path.rename(displaced)
            artifact.path.symlink_to(outside, target_is_directory=True)
            swapped = True
        return descriptors

    monkeypatch.setattr(research_preparation, "_open_eviction_locks", swap_after_validation)
    assert cache.evict(artifact.artifact_id) is False
    assert swapped
    assert {path.name: path.read_bytes() for path in outside.iterdir()} == before


def test_directory_swap_after_second_qualification_never_traverses_symlink(
        tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "arrays.joblib").write_bytes(b"protected array")
    (outside / "manifest.json").write_bytes(b"protected manifest")
    before = {path.name: path.read_bytes() for path in outside.iterdir()}
    displaced = tmp_path / "displaced-cache-artifact"
    original_qualify = research_preparation._qualify_cache_directory
    qualification_count = 0

    def qualify_then_swap(root, root_fd, directory_fd, identity, root_identity,
                          directory_identity):
        nonlocal qualification_count
        entries = original_qualify(root, root_fd, directory_fd, identity,
                                   root_identity, directory_identity)
        qualification_count += 1
        if qualification_count == 2 and entries is not None:
            artifact.path.rename(displaced)
            artifact.path.symlink_to(outside, target_is_directory=True)
        return entries

    monkeypatch.setattr(research_preparation, "_qualify_cache_directory", qualify_then_swap)
    assert cache.evict(artifact.artifact_id) is False
    assert qualification_count == 2
    assert artifact.path.is_symlink()
    assert {path.name: path.read_bytes() for path in outside.iterdir()} == before
    assert displaced.is_dir()
    assert list(displaced.iterdir()) == []


@pytest.mark.parametrize("entry_kind", ["model", "symlink", "hardlink"])
def test_corrupt_rebuild_holds_unknown_or_linked_contents(tmp_path, entry_kind):
    cache = PreparationCache(tmp_path / "cache")
    artifact = cache.prepare_fold(key(), build)
    (artifact.path / "arrays.joblib").write_bytes(b"corrupt payload")
    outside = tmp_path / "protected-payload.bin"
    outside.write_bytes(b"outside payload must be held")

    if entry_kind == "model":
        (artifact.path / "model.bin").write_bytes(b"unrecognized model")
    elif entry_kind == "symlink":
        (artifact.path / "unrecognized-link").symlink_to(outside)
    else:
        os.link(outside, artifact.path / "unrecognized-hardlink")

    before = _snapshot_tree(artifact.path)
    outside_before = outside.read_bytes()
    with pytest.raises(ValueError, match="unknown or linked contents"):
        cache.prepare_fold(key(), build)

    assert artifact.path.is_dir()
    assert _snapshot_tree(artifact.path) == before
    assert outside.read_bytes() == outside_before
