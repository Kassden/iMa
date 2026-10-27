"""The human-readable report follows the same admission/publication contract as JSON."""
import errno
import os
from types import SimpleNamespace

import pytest

import ima.research_controller as core
import ima.research_storage as storage
from tests.test_controller_json_storage import authority_at


def test_denied_report_does_not_create_parent(tmp_path,monkeypatch):
    authority_at(tmp_path,monkeypatch,1024)
    path=tmp_path/'new-campaign'/'summary.md'
    with pytest.raises(storage.StorageBlocked):core._write_text_atomic(path,'# Research\n')
    assert not path.parent.exists()


def test_report_readback_replaces_old_text_without_orphan(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'summary.md';path.write_text('old')
    text='# Research 馬\n\nCompleted: 3\n'
    core._write_text_atomic(path,text)
    assert path.read_text()==text and not path.with_suffix('.md.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_partial_report_failure_preserves_published_report(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/'summary.md';path.write_text('accepted report\n')
    original=storage.ReservedFile.write
    def partial(stream,data):
        original(stream,data[:3])
        raise OSError(errno.ENOSPC,'fixture native full')
    monkeypatch.setattr(storage.ReservedFile,'write',partial)
    with pytest.raises(storage.StorageBlocked) as error:core._write_text_atomic(path,'new report\n')
    assert error.value.diagnostic['stage']=='controller_text'
    assert path.read_text()=='accepted report\n'
    assert not path.with_suffix('.md.tmp').exists()
    assert authority.status()['reserved_bytes']==0


def test_deep_report_parent_inodes_are_admitted_before_creation(tmp_path,monkeypatch):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    authority._statvfs=lambda _:SimpleNamespace(f_frsize=1,f_blocks=4*1024**2,
        f_bavail=4*1024**2,f_bfree=4*1024**2,f_favail=4)
    path=tmp_path/'one'/'two'/'three'/'four'/'five'/'summary.md'
    with pytest.raises(storage.StorageBlocked):core._write_text_atomic(path,'report')
    assert not (tmp_path/'one').exists()
    assert authority.status()['reserved_inodes']==0


@pytest.mark.parametrize('format',['json','text'])
@pytest.mark.parametrize('replacement',['file','symlink'])
def test_changed_staging_path_is_neither_published_nor_cleaned(tmp_path,monkeypatch,format,replacement):
    authority=authority_at(tmp_path,monkeypatch,4*1024**2)
    path=tmp_path/('status.json' if format=='json' else 'summary.md')
    path.write_text('accepted before\n');temporary=path.with_suffix(path.suffix+'.tmp')
    other=tmp_path/'other';other.write_text('retained replacement\n')
    sync=os.fsync;changed=False
    def substitute(fd):
        nonlocal changed
        sync(fd)
        if not changed and temporary.exists() and os.path.samestat(os.fstat(fd),temporary.stat()):
            changed=True;temporary.unlink()
            if replacement=='file':temporary.write_text('retained replacement\n')
            else:temporary.symlink_to(other)
    monkeypatch.setattr(os,'fsync',substitute)
    with pytest.raises(ValueError,match='staging identity changed'):
        if format=='json':core._write_json_atomic(path,{'new':True})
        else:core._write_text_atomic(path,'new report\n')
    assert changed and path.read_text()=='accepted before\n'
    assert temporary.read_text()==other.read_text()=='retained replacement\n'
    assert temporary.is_symlink()==(replacement=='symlink')
    assert authority.status()['reserved_bytes']==0
