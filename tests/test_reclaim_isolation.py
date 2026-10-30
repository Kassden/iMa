import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import threading

import pytest

import ima.research_reclaim_isolation as isolation
from ima.research_storage import payload_identity


def test_bounded_protocol_roundtrip_and_rejects_large_frame():
    a,b=socket.socketpair()
    try:
        isolation._send(a,{'operation':'inspect','nonce':'abc'})
        assert isolation._receive(b)=={'operation':'inspect','nonce':'abc'}
        with pytest.raises(ValueError):isolation._send(a,{'huge':'x'*isolation.MAX_FRAME})
        a.sendall(struct.pack('!I',isolation.MAX_FRAME+1))
        with pytest.raises(ValueError):isolation._receive(b)
    finally:a.close();b.close()


def test_protocol_rejects_truncated_frame():
    a,b=socket.socketpair()
    a.sendall(struct.pack('!I',10)+b'{}');a.close()
    try:
        with pytest.raises(EOFError):isolation._receive(b)
    finally:b.close()


def credentials():
    return {'Uid':'1001 1001 1001 1001','Gid':'1001 1001 1001 1001','Groups':'',
            'NoNewPrivs':'1',**{k:'0000' for k in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')}}


@pytest.mark.parametrize('field,value',[('Uid','1001 1001 0 1001'),('Gid','1001 1001 0 1001'),
    ('Groups','1001'),('NoNewPrivs','0'),('CapBnd','01'),('CapEff','01')])
def test_credential_proof_rejects_saved_root_and_capabilities(monkeypatch,field,value):
    state=credentials();state[field]=value
    monkeypatch.setattr(isolation,'_status',lambda pid:state)
    with pytest.raises(PermissionError):isolation._unprivileged_status(123)


def test_credential_proof_accepts_permanent_drop(monkeypatch):
    monkeypatch.setattr(isolation,'_status',lambda pid:credentials())
    assert isolation._unprivileged_status(123)==credentials()


@pytest.fixture
def selected(tmp_path):
    root=tmp_path.resolve()/'campaigns';path=root/'one'/'fold-preparation-cache'/('a'*64)
    path.mkdir(parents=True)
    for name in ('arrays.joblib','fitted.joblib','manifest.json'):(path/name).write_bytes(b'fixture')
    lock=path.parent/(path.name+'.lock');lock.touch()
    return payload_identity(path,root),lock.stat().st_ino


def run_child(monkeypatch,record,reply,*,inspect_only=False,mutate=None):
    monkeypatch.setattr(isolation,'_trusted_root_peer',lambda channel:None)
    monkeypatch.setattr(isolation,'_unprivileged_status',lambda pid:credentials())
    parent,child=socket.socketpair();parent.settimeout(10);child.settimeout(10)
    result=[]
    thread=threading.Thread(target=lambda:result.append(isolation._child(child)))
    source=Path('ima/research_storage.py').read_text()
    thread.start()
    try:
        isolation._send(parent,{'operation':'begin','record':record,'storage_source':source,
            'authority':'fixture-only','canary_receipt':{'qualified':True,'package_id':'fixture'},
            'inspect_only':inspect_only})
        request=isolation._receive(parent)
        assert request['record']==record and request['operation']=='inspect'
        if mutate:mutate()
        isolation._send(parent,{'operation':'inspection','nonce':request['nonce'],'report':reply})
        terminal=isolation._receive(parent)
        thread.join(10);assert not thread.is_alive()
        return terminal
    finally:
        parent.close();child.close();thread.join(10)


def test_child_refuses_uncleared_inspection_without_deleting(monkeypatch,selected):
    record,_=selected
    result=run_child(monkeypatch,record,{'process_snapshot_complete':False,'error_count':1,'reference_count':0})
    assert result['ok'] is False
    assert payload_identity(record['path'],record['root'])==record


def test_child_reclaims_only_after_clear_and_retains_stable_lock(monkeypatch,selected):
    record,inode=selected
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0})
    assert result['ok'] is True
    path=Path(record['path'])
    assert not path.exists()
    assert path.with_name(path.name+'.lock').stat().st_ino==inode


def test_inspection_only_preserves_exact_payload(monkeypatch,selected):
    record,_=selected
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0},inspect_only=True)
    assert result['ok'] is True
    assert result['result']['mode']=='inspect_only'
    assert payload_identity(record['path'],record['root'])==record


def test_inspection_only_rejects_mutation_during_scan(monkeypatch,selected):
    record,_=selected
    path=Path(record['path'])/'arrays.joblib'
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0},
        inspect_only=True,mutate=lambda:path.write_bytes(b'changed'))
    assert result['ok'] is False
    assert Path(record['path']).exists()


def test_apply_rechecks_identity_after_privileged_inspection(monkeypatch,selected):
    record,_=selected
    path=Path(record['path'])/'arrays.joblib'
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0},
        mutate=lambda:path.write_bytes(b'changed during scan'))
    assert result['ok'] is False
    assert {item.name for item in Path(record['path']).iterdir()}=={'arrays.joblib','fitted.joblib','manifest.json'}


@pytest.mark.parametrize('companion',[False,True])
def test_apply_refuses_replaced_stable_key_or_pin_lock(monkeypatch,selected,companion):
    record,_=selected
    path=Path(record['path'])
    lock=path.with_name(path.name+('.pin.lock' if companion else '.lock'))
    if companion:lock.touch()
    def replace():
        lock.unlink();lock.write_bytes(b'replacement')
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0},mutate=replace)
    assert result['ok'] is False
    assert {item.name for item in path.iterdir()}=={'arrays.joblib','fitted.joblib','manifest.json'}


def test_partial_failure_remains_a_failure_with_exact_remaining_inventory(monkeypatch,selected):
    record,lock_inode=selected
    path=Path(record['path']);original=Path.unlink
    def fail_second(self,*args,**kwargs):
        if self==path/'fitted.joblib':raise PermissionError(errno.EACCES,'fixture second unlink denied')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',fail_second)
    result=run_child(monkeypatch,record,{'process_snapshot_complete':True,'error_count':0,'reference_count':0})
    assert result['ok'] is False and result['error_type']=='PermissionError'
    assert not (path/'arrays.joblib').exists()
    assert (path/'fitted.joblib').exists() and (path/'manifest.json').exists()
    assert path.with_name(path.name+'.lock').stat().st_ino==lock_inode
    remaining=isolation._remaining_payload(record)
    assert remaining['files']['arrays.joblib']=={'exists':False}
    assert remaining['files']['fitted.joblib']['ino']==next(f['ino'] for f in record['files'] if f['name']=='fitted.joblib')
