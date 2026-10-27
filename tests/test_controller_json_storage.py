from types import SimpleNamespace
import errno
import json
import os
import stat
from pathlib import Path

import pytest

import ima.research_storage as storage
from ima.research_controller import _write_json_atomic


def authority_at(tmp_path,monkeypatch,available):
    authority=storage.StorageAuthority(tmp_path.resolve()/'authority',headroom_bytes=0,
        headroom_inodes=0,statvfs=lambda path:SimpleNamespace(f_frsize=1,
            f_blocks=available,f_bavail=available,f_bfree=available,f_favail=10000))
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    return authority


def test_denied_json_reservation_preserves_previous_status(tmp_path,monkeypatch):
    path=tmp_path/'status.json';path.write_text('{"status":"previous"}')
    authority=authority_at(tmp_path,monkeypatch,1024)
    with pytest.raises(storage.StorageBlocked):_write_json_atomic(path,{'status':'new'})
    assert json.loads(path.read_text())=={'status':'previous'}
    assert not path.with_suffix('.json.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_denied_json_reservation_does_not_create_parent(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch,1024)
    path=tmp_path/'not-created'/'status.json'
    with pytest.raises(storage.StorageBlocked):_write_json_atomic(path,{'status':'new'})
    assert not path.parent.exists()


def test_admitted_json_consumer_readback_and_release(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'statuses'/'status.json'
    _write_json_atomic(path,{'status':'training','description':'馬','completed':4})
    assert json.loads(path.read_text())=={'status':'training','description':'馬','completed':4}
    assert list(path.parent.iterdir())==[path]
    assert authority.status()['reserved_bytes']==0


def test_native_full_before_replace_preserves_previous_json(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'status.json';path.write_text('{"status":"previous"}')
    def full(self,data):raise OSError(errno.ENOSPC,'fixture full')
    monkeypatch.setattr(storage.ReservedFile,'write',full)
    with pytest.raises(storage.StorageBlocked) as error:_write_json_atomic(path,{'status':'new'})
    assert error.value.diagnostic['stage']=='controller_json'
    assert json.loads(path.read_text())=={'status':'previous'}
    assert not path.with_suffix('.json.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_admitted_json_refuses_staging_symlink_without_removing_it(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    protected=tmp_path/'protected';protected.write_text('preserve')
    path=tmp_path/'status.json';staging=path.with_suffix('.json.tmp')
    staging.symlink_to(protected)
    with pytest.raises(ValueError):_write_json_atomic(path,{'status':'new'})
    assert staging.is_symlink() and protected.read_text()=='preserve'
    assert authority.status()['reserved_bytes']==0


def test_invalid_nonfinite_json_rejected_before_reservation(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch,4*1024**2)
    with pytest.raises(ValueError):_write_json_atomic(tmp_path/'status.json',{'loss':float('nan')})
    assert not (tmp_path/'status.json').exists()


def test_ordinary_io_fault_is_not_misclassified_as_storage_pressure(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'status.json';path.write_text('{"status":"previous"}')
    def bad_io(self,data):raise OSError(errno.EIO,'fixture IO fault')
    monkeypatch.setattr(storage.ReservedFile,'write',bad_io)
    with pytest.raises(OSError) as error:_write_json_atomic(path,{'status':'new'})
    assert error.value.errno==errno.EIO
    assert json.loads(path.read_text())=={'status':'previous'}
    assert not path.with_suffix('.json.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_directory_sync_pressure_keeps_complete_published_json(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'status.json'
    original=os.fsync
    parent_identity=(tmp_path.stat().st_dev,tmp_path.stat().st_ino)
    def sync(fd):
        info=os.fstat(fd)
        if stat.S_ISDIR(info.st_mode) and (info.st_dev,info.st_ino)==parent_identity:
            raise OSError(errno.ENOSPC,'fixture directory sync full')
        return original(fd)
    monkeypatch.setattr(os,'fsync',sync)
    with pytest.raises(storage.StorageBlocked):_write_json_atomic(path,{'status':'new'})
    assert json.loads(path.read_text())=={'status':'new'}
    assert not path.with_suffix('.json.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_cleanup_failure_retains_primary_pressure_and_held_reservation(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'status.json';path.write_text('{"status":"previous"}')
    staging=path.with_suffix('.json.tmp')
    original_unlink=Path.unlink
    observed=[]
    def unlink(self,*args,**kwargs):
        if self==staging and self.exists():
            observed.append(authority.status()['reserved_bytes'])
            raise PermissionError(errno.EACCES,'fixture cleanup denied')
        return original_unlink(self,*args,**kwargs)
    def full(self,data):raise OSError(errno.ENOSPC,'primary full')
    monkeypatch.setattr(Path,'unlink',unlink)
    monkeypatch.setattr(storage.ReservedFile,'write',full)
    with pytest.raises(storage.StorageBlocked) as error:_write_json_atomic(path,{'status':'new'})
    assert error.value.diagnostic['errno']==errno.ENOSPC
    assert any('errno=13' in note for note in error.value.__cause__.__notes__)
    assert observed and all(value>0 for value in observed)
    assert json.loads(path.read_text())=={'status':'previous'}
    assert staging.exists()  # One bounded slot remains for the next retry.
    assert authority.status()['reserved_bytes']==0
