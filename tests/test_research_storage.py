import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ima.research_storage import StorageAuthority, StorageBlocked, payload_identity, reclaim_payload


def space(free=1000, inodes=100):
    return SimpleNamespace(f_bavail=free,f_bfree=free,f_frsize=1,f_favail=inodes)


def test_reservations_are_atomic_and_include_other_workers(tmp_path):
    a=StorageAuthority(tmp_path/'authority',headroom_bytes=100,headroom_inodes=2,statvfs=lambda _:space())
    one=a.reserve(bytes_required=600,inodes_required=5,stage='fit',output=tmp_path/'out')
    with pytest.raises(StorageBlocked): a.reserve(bytes_required=301,stage='acquisition',output=tmp_path/'other')
    with pytest.raises(StorageBlocked): a.extend(one,extra_bytes=301)
    a.extend(one,extra_bytes=100)
    assert a.status()['reserved_bytes']==700
    a.release(one)
    assert a.status()['reserved_bytes']==0


def test_live_reservations_never_expire_by_age(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0,statvfs=lambda _:space())
    token=a.reserve(bytes_required=10,stage='build',output=tmp_path/'out')
    state=json.loads((a.root/'state.json').read_text()); state['reservations'][token]['created_at']=0
    (a.root/'state.json').write_text(json.dumps(state))
    assert a.reconcile()['reserved_bytes']==10
    a.release(token)


def test_exception_releases_reservation_and_keeps_control_inode(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0)
    a.status(); inode=(a.root/'authority.lock').stat().st_ino
    with pytest.raises(OSError):
        with a.admitted(bytes_required=1,stage='fit',output=tmp_path/'out'):
            raise OSError(28,'ENOSPC')
    assert a.status()['reserved_bytes']==0
    assert (a.root/'authority.lock').stat().st_ino==inode


def test_inode_pressure_blocks_and_symlink_authority_rejected(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=10,statvfs=lambda _:space(inodes=10))
    with pytest.raises(StorageBlocked): a.reserve(bytes_required=1,stage='fit',output=tmp_path/'out')
    (tmp_path/'link').symlink_to(tmp_path/'a',target_is_directory=True)
    with pytest.raises(ValueError): StorageAuthority(tmp_path/'link')


@pytest.mark.parametrize('size',[-1,True,1.5])
def test_invalid_reservation_does_not_write(tmp_path,size):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0)
    with pytest.raises(ValueError): a.reserve(bytes_required=size,stage='fit',output=tmp_path/'out')
    assert not (a.root/'state.json').exists()


def test_reclaim_rejects_wrong_class_links_and_changed_content(tmp_path):
    root=tmp_path/'campaign'; cache=root/'fold-preparation-cache'; payload=cache/('a'*64)
    payload.mkdir(parents=True); (payload/'arrays.joblib').write_bytes(b'original')
    (cache/('a'*64+'.lock')).touch()
    record=payload_identity(payload,root)
    (payload/'arrays.joblib').write_bytes(b'changed!')
    with pytest.raises(ValueError,match='changed'):
        reclaim_payload(record,authority='approved exact payload',canary_receipt={'qualified':True,'package_id':'real'})
    assert payload.exists()
    other=root/'models'/('b'*64); other.mkdir(parents=True)
    with pytest.raises(ValueError): payload_identity(other,root)
    (payload/'link').symlink_to(payload/'arrays.joblib')
    with pytest.raises(ValueError): payload_identity(payload,root)


def _reserve_worker(root, output, ready, release):
    a=StorageAuthority(root,headroom_bytes=100,headroom_inodes=0,statvfs=lambda _:space())
    try:
        token=a.reserve(bytes_required=600,stage='fit',output=output)
        ready.put('reserved'); release.wait(10); a.release(token)
    except StorageBlocked: ready.put('blocked')


def test_processes_race_for_last_capacity(tmp_path):
    ctx=multiprocessing.get_context('fork'); q=ctx.Queue(); release=ctx.Event()
    args=(str(tmp_path/'a'),str(tmp_path/'output'),q,release)
    first=ctx.Process(target=_reserve_worker,args=args); first.start()
    assert q.get(timeout=10)=='reserved'
    second=ctx.Process(target=_reserve_worker,args=args); second.start()
    assert q.get(timeout=10)=='blocked'
    release.set(); first.join(10); second.join(10)
    assert first.exitcode==second.exitcode==0


def test_lexical_parent_escape_is_rejected(tmp_path):
    allowed=tmp_path/'allowed'; allowed.mkdir()
    outside=tmp_path/'outside/fold-preparation-cache'/('a'*64)
    outside.mkdir(parents=True); (outside/'arrays.joblib').write_bytes(b'protected')
    escaped=allowed/'..'/'outside'/'fold-preparation-cache'/('a'*64)
    with pytest.raises(ValueError): payload_identity(escaped,allowed)
    assert (outside/'arrays.joblib').read_bytes()==b'protected'


def test_pending_cache_growth_and_extensions_share_atomic_budget(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0,
                       cache_limit_bytes=50,statvfs=lambda _:space())
    token=a.reserve(bytes_required=40,cache_bytes=40,stage='fold',output=tmp_path/'one')
    with pytest.raises(StorageBlocked):
        a.reserve(bytes_required=40,cache_bytes=40,stage='fold',output=tmp_path/'two')
    with pytest.raises(StorageBlocked): a.extend(token,extra_bytes=20,extra_cache_bytes=20)
    a.release(token)
    token=a.reserve(bytes_required=40,cache_bytes=40,stage='fold',output=tmp_path/'two')
    a.release(token)


def test_cache_hysteresis_uses_soft_target_and_never_deletes_locks(tmp_path,monkeypatch):
    from ima.research_preparation import PreparationCache
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0,
        cache_limit_bytes=100,cache_soft_bytes=90,cache_target_bytes=80,
        statvfs=lambda _:space())
    root=tmp_path/'fold-preparation-cache'; root.mkdir()
    removed=[]
    def evict(cache,key):
        removed.append(key)
        return True
    monkeypatch.setattr(PreparationCache,'evict',evict)
    for i in range(3):
        key=f'{i:064x}'; (root/key).mkdir(); (root/(key+'.lock')).touch()
        a.record_cache(root,key,30)
    result=a.trim_caches(5)
    assert len(removed)==1 and result['remaining_bytes']==60
    assert len(list(root.glob('*.lock')))==3
    assert a.status()['cache_target_bytes']==80


def test_trim_reconciles_dead_reservations_before_admission(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0,
                       cache_limit_bytes=50,statvfs=lambda _:space())
    token=a.reserve(bytes_required=40,cache_bytes=40,stage='fold',output=tmp_path/'one')
    state=json.loads((a.root/'state.json').read_text())
    state['reservations'][token]['pid']=2147483647
    (a.root/'state.json').write_text(json.dumps(state))
    a.trim_caches(20)
    assert a.status()['reserved_bytes']==0


def test_existing_cache_over_limit_is_recorded_but_not_admitted(tmp_path):
    a=StorageAuthority(tmp_path/'a',headroom_bytes=0,headroom_inodes=0,
                       cache_limit_bytes=1,statvfs=lambda _:space())
    root=tmp_path/'fold-preparation-cache'; root.mkdir()
    (root/('a'*64)).mkdir()
    with pytest.raises(StorageBlocked,match='discovered cache'):
        a.record_cache(root,'a'*64,12288)
    assert a.status()['cache_bytes']==12288
