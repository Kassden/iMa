import errno
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import ima.research_controller as controller
import ima.research_storage as storage


def _authority(tmp_path, monkeypatch, *, available=10 * 1024 * 1024):
    authority = storage.StorageAuthority(
        tmp_path / "control",
        headroom_bytes=0,
        headroom_inodes=0,
        statvfs=lambda _: SimpleNamespace(
            f_frsize=1, f_blocks=available, f_bavail=available,
            f_bfree=available, f_favail=1000,
        ),
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


def test_admission_denial_happens_before_parent_creation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch, available=32)
    destination = tmp_path / "not-created" / "records.jsonl"

    with pytest.raises(storage.StorageBlocked):
        controller._append_jsonl(destination, {"record": 1})

    assert not destination.parent.exists()
    assert authority.status()["reserved_bytes"] == 0


def test_enabled_append_roundtrips_unicode_and_releases_reservation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    destination = tmp_path / "campaign" / "records.jsonl"
    destination.parent.mkdir()
    destination.write_bytes(b'{"prior":true}\n')
    inode = destination.stat().st_ino

    controller._append_jsonl(destination, {"name": "香港賽馬", "emoji": "🐎"})

    records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert records == [{"prior": True}, {"emoji": "🐎", "name": "香港賽馬"}]
    assert destination.stat().st_ino == inode
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_partial_enospc_restores_original_prefix(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    destination = tmp_path / "records.jsonl"
    prefix = b'{"complete":"before"}\n'
    destination.write_bytes(prefix)
    original_write = storage.ReservedFile.write

    def partial_then_full(stream, data):
        original_write(stream, data[: max(1, len(data) // 2)])
        raise OSError(errno.ENOSPC, "native partial write")

    monkeypatch.setattr(storage.ReservedFile, "write", partial_then_full)

    with pytest.raises(storage.StorageBlocked) as raised:
        controller._append_jsonl(destination, {"complete": "after"})

    assert raised.value.diagnostic["stage"] == "controller_jsonl"
    assert raised.value.diagnostic["errno"] == errno.ENOSPC
    assert destination.read_bytes() == prefix
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_partial_enospc_preserves_record_appended_while_waiting_for_lock(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    destination = tmp_path / "records.jsonl"
    prefix = b'{"complete":"before"}\n'
    competitor = b'{"complete":"competitor"}\n'
    destination.write_bytes(prefix)
    original_flock = controller.fcntl.flock
    target_inode = destination.stat().st_ino
    injected = False

    def inject_before_lock(fd, operation):
        nonlocal injected
        descriptor = fd if isinstance(fd, int) else fd.fileno()
        info = os.fstat(descriptor)
        if not injected and info.st_ino == target_inode and operation == controller.fcntl.LOCK_EX:
            append_fd = os.open(destination, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(append_fd, competitor)
            finally:
                os.close(append_fd)
            injected = True
        return original_flock(fd, operation)

    def partial_then_full(stream, data):
        stream.stream.write(data[: max(1, len(data) // 2)])
        raise OSError(errno.ENOSPC, "native partial write")

    monkeypatch.setattr(controller.fcntl, "flock", inject_before_lock)
    monkeypatch.setattr(storage.ReservedFile, "write", partial_then_full)

    with pytest.raises(storage.StorageBlocked) as raised:
        controller._append_jsonl(destination, {"complete": "after"})

    assert injected
    assert raised.value.diagnostic["stage"] == "controller_jsonl"
    assert raised.value.diagnostic["errno"] == errno.ENOSPC
    assert destination.read_bytes() == prefix + competitor
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_hardlink_created_while_waiting_for_lock_is_rejected(tmp_path,monkeypatch):
    authority=_authority(tmp_path,monkeypatch)
    destination=tmp_path/'records.jsonl';prefix=b'{"accepted":true}\n'
    destination.write_bytes(prefix);alias=tmp_path/'alias.jsonl'
    original=controller.fcntl.flock;inode=destination.stat().st_ino
    def acquire(fd,operation):
        descriptor=fd if isinstance(fd,int) else fd.fileno()
        if operation==controller.fcntl.LOCK_EX and os.fstat(descriptor).st_ino==inode and not alias.exists():
            os.link(destination,alias)
        return original(fd,operation)
    monkeypatch.setattr(controller.fcntl,'flock',acquire)
    with pytest.raises(ValueError):controller._append_jsonl(destination,{'new':True})
    assert destination.read_bytes()==alias.read_bytes()==prefix
    assert authority.status()['reserved_bytes']==0


def test_missing_parent_directories_are_reserved_before_creation(tmp_path,monkeypatch):
    authority=_authority(tmp_path,monkeypatch)
    authority._statvfs=lambda _:SimpleNamespace(f_frsize=1,f_blocks=10*1024**2,
        f_bavail=10*1024**2,f_bfree=10*1024**2,f_favail=3)
    destination=tmp_path/'one'/'two'/'three'/'records.jsonl'
    with pytest.raises(storage.StorageBlocked):controller._append_jsonl(destination,{'new':True})
    assert not (tmp_path/'one').exists()
    assert authority.status()['reserved_inodes']==0


def test_rollback_error_keeps_primary_storage_classification(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    destination = tmp_path / "records.jsonl"
    prefix = b'{"complete":"before"}\n'
    destination.write_bytes(prefix)

    def partial_then_full(stream, data):
        stream.stream.write(data[: max(1, len(data) // 2)])
        raise OSError(errno.ENOSPC, "native partial write")

    monkeypatch.setattr(storage.ReservedFile, "write", partial_then_full)
    monkeypatch.setattr(controller.os, "ftruncate", lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "rollback failed")))

    with pytest.raises(storage.StorageBlocked) as raised:
        controller._append_jsonl(destination, {"complete": "after"})

    assert raised.value.diagnostic["stage"] == "controller_jsonl"
    assert raised.value.diagnostic["errno"] == errno.ENOSPC
    assert any("rollback failed" in note for note in getattr(raised.value.__cause__, "__notes__", []))
    assert authority.status()["reserved_bytes"] == 0


@pytest.mark.parametrize("target_kind", ["symlink", "hardlink"])
def test_unsafe_target_is_refused_without_changing_foreign_data(tmp_path, monkeypatch, target_kind):
    authority = _authority(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign.jsonl"
    original = b'{"protected":true}\n'
    foreign.write_bytes(original)
    destination = tmp_path / "target.jsonl"
    if target_kind == "symlink":
        destination.symlink_to(foreign)
    else:
        os.link(foreign, destination)

    with pytest.raises((ValueError, OSError)):
        controller._append_jsonl(destination, {"protected": False})

    assert foreign.read_bytes() == original
    assert authority.status()["reserved_bytes"] == 0
    assert authority.status()["reserved_inodes"] == 0


def test_storage_disabled_keeps_legacy_jsonl_serialization(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    destination = tmp_path / "legacy" / "records.jsonl"

    controller._append_jsonl(destination, {"value": float("nan")})

    assert math.isnan(json.loads(destination.read_text(encoding="utf-8"))["value"])
