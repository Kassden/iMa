import errno
import fcntl
import json
import os
import stat
from types import SimpleNamespace

import pytest

import ima.research_controller as controller
import ima.research_storage as storage


def _authority(tmp_path, monkeypatch, *, available=1024 * 1024):
    authority = storage.StorageAuthority(
        tmp_path / "control", headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(f_frsize=1, f_blocks=available,
            f_bavail=available, f_bfree=available, f_favail=available),
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


def test_lock_held_across_yield_after_reservation_release_and_inode_reused(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    controller._prepare_campaign_directory(campaign)
    lock_path = campaign / "controller.lock"

    with controller.campaign_lock(campaign):
        inode = lock_path.stat().st_ino
        assert authority.status()["reserved_bytes"] == 0
        assert authority.status()["reserved_inodes"] == 0
        assert storage._OUTPUT_BUDGET.get() is None
        contender = os.open(lock_path, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(contender)

    with controller.campaign_lock(campaign):
        assert lock_path.stat().st_ino == inode
        assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_denied_admission_creates_no_campaign_or_lock_and_preserves_existing_metadata(tmp_path, monkeypatch):
    available = {"value": 1024 * 1024}
    authority = storage.StorageAuthority(
        tmp_path / "control", headroom_bytes=0, headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(f_frsize=1, f_blocks=available["value"],
            f_bavail=available["value"], f_bfree=available["value"], f_favail=available["value"]),
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)

    missing = tmp_path / "absent" / "campaign"
    available["value"] = 0
    with pytest.raises(storage.StorageBlocked):
        controller._prepare_campaign_directory(missing)
    assert not missing.exists()
    assert not (missing / "controller.lock").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    lock = existing / "controller.lock"
    original = b'{"pid":123,"started_at":"prior"}\n'
    lock.write_bytes(original)
    available["value"] = 0
    with pytest.raises(storage.StorageBlocked):
        with controller.campaign_lock(existing):
            pytest.fail("denied lock admission must not yield")
    assert lock.read_bytes() == original
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_partial_native_enospc_is_classified_and_releases_lock(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    controller._prepare_campaign_directory(campaign)
    original_write = storage.ReservedFile.write

    def partial_then_full(stream, data):
        original_write(stream, data[:max(1, len(data) // 2)])
        raise OSError(errno.ENOSPC, "native lock metadata full")

    monkeypatch.setattr(storage.ReservedFile, "write", partial_then_full)
    with pytest.raises(storage.StorageBlocked) as caught:
        with controller.campaign_lock(campaign):
            pytest.fail("metadata failure must prevent yield")
    assert caught.value.diagnostic["stage"] == "controller_lock"
    assert caught.value.diagnostic["errno"] == errno.ENOSPC
    monkeypatch.setattr(storage.ReservedFile, "write", original_write)

    with controller.campaign_lock(campaign):
        fd = os.open(campaign / "controller.lock", os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


@pytest.mark.parametrize('error',[OSError(errno.ENOSPC,'fit failed'),BlockingIOError(errno.EAGAIN,'worker busy')])
def test_campaign_body_errors_are_not_relabelled_as_owner_failure(tmp_path,monkeypatch,error):
    authority=_authority(tmp_path,monkeypatch)
    campaign=tmp_path/'campaign';controller._prepare_campaign_directory(campaign)
    with pytest.raises(type(error)) as raised:
        with controller.campaign_lock(campaign):raise error
    assert raised.value is error
    assert authority.status()['reserved_bytes']==0
    with controller.campaign_lock(campaign):pass


@pytest.mark.parametrize("alias", ["symlink", "hardlink"])
def test_unsafe_lock_alias_is_refused_without_touching_target(tmp_path, monkeypatch, alias):
    _authority(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    protected = campaign / "protected"
    protected.write_bytes(b"preserve me")
    lock = campaign / "controller.lock"
    if alias == "symlink":
        lock.symlink_to(protected)
    else:
        os.link(protected, lock)

    with pytest.raises((OSError, ValueError)):
        with controller.campaign_lock(campaign):
            pytest.fail("unsafe lock alias must be refused")
    assert protected.read_bytes() == b"preserve me"
    if alias == "symlink":
        assert lock.is_symlink()
    else:
        assert stat.S_ISREG(lock.lstat().st_mode)


def test_status_probe_does_not_create_missing_lock(tmp_path):
    lock = tmp_path / "missing" / "controller.lock"
    assert controller._controller_lock_held(lock) is False
    assert not lock.parent.exists()
