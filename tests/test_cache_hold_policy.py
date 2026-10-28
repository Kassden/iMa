import hashlib
import json

import pytest

from ima.research_cache_policy import cache_holds
from ima.research_preparation import PreparationCache
from ima.research_storage import StorageBlocked
from tests.test_research_preparation import build,key


def policy(monkeypatch,tmp_path,paths):
    source=tmp_path/'cache-holds.json'
    source.write_text(json.dumps({'schema_version':1,'protected_cache_paths':[str(p) for p in paths]}))
    source.chmod(0o400)
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY',str(source))
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY_SHA256',hashlib.sha256(source.read_bytes()).hexdigest())
    return source


def test_explicit_hold_blocks_eviction_and_rebuild_but_not_other_artifacts(tmp_path,monkeypatch):
    cache=PreparationCache(tmp_path/'fold-preparation-cache')
    held=cache.prepare_fold(key(seed=1),build)
    other=cache.prepare_fold(key(seed=2),build)
    policy(monkeypatch,tmp_path,[held.path])
    before={p.name:p.read_bytes() for p in held.path.iterdir()}
    assert cache.evict(held.artifact_id) is False
    assert cache.evict(other.artifact_id) is True
    assert {p.name:p.read_bytes() for p in held.path.iterdir()}==before
    (held.path/'arrays.joblib').write_bytes(b'corrupted but explicitly held')
    with pytest.raises(StorageBlocked,match='retention policy'):
        cache.prepare_fold(key(seed=1),build)
    assert (held.path/'arrays.joblib').read_bytes()==b'corrupted but explicitly held'


@pytest.mark.parametrize('fault',['changed','missing','symlink','hardlink','incomplete_env'])
def test_unverifiable_configured_policy_cannot_delete_cache(tmp_path,monkeypatch,fault):
    cache=PreparationCache(tmp_path/'fold-preparation-cache')
    artifact=cache.prepare_fold(key(),build)
    source=policy(monkeypatch,tmp_path,[artifact.path])
    if fault=='changed':
        source.chmod(0o600);source.write_text('{}')
    elif fault=='missing':source.unlink()
    elif fault=='symlink':
        moved=tmp_path/'moved';source.rename(moved);source.symlink_to(moved)
    elif fault=='hardlink':
        import os
        os.link(source,tmp_path/'extra-link')
    else:monkeypatch.delenv('IMA_CACHE_HOLD_POLICY_SHA256')
    with pytest.raises(StorageBlocked,match='cannot be verified'):cache.evict(artifact.artifact_id)
    assert artifact.path.is_dir()


@pytest.mark.parametrize('fault',['duplicate','outside'])
def test_duplicate_or_outside_cache_identities_fail_closed(tmp_path,monkeypatch,fault):
    root=tmp_path/'fold-preparation-cache';root.mkdir()
    path=root/('a'*64)
    paths=[path,path] if fault=='duplicate' else [tmp_path/'outside-model.joblib']
    policy(monkeypatch,tmp_path,paths)
    with pytest.raises(StorageBlocked):cache_holds()


def test_unconfigured_policy_retains_legacy_behavior(monkeypatch):
    monkeypatch.delenv('IMA_CACHE_HOLD_POLICY',raising=False)
    monkeypatch.delenv('IMA_CACHE_HOLD_POLICY_SHA256',raising=False)
    assert cache_holds()==frozenset()


@pytest.mark.parametrize('encoded',[
    '{"schema_version":true,"protected_cache_paths":[]}',
    '{"schema_version":1.0,"protected_cache_paths":[]}',
    '{"schema_version":1,"protected_cache_paths":[],"protected_cache_paths":[]}',
])
def test_ambiguous_policy_json_is_rejected_before_eviction(tmp_path,monkeypatch,encoded):
    cache=PreparationCache(tmp_path/'fold-preparation-cache')
    artifact=cache.prepare_fold(key(),build)
    source=tmp_path/'invalid-policy.json';source.write_text(encoded);source.chmod(0o400)
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY',str(source))
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY_SHA256',hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(StorageBlocked):cache.evict(artifact.artifact_id)
    assert artifact.path.is_dir()


def test_fifo_policy_is_rejected_without_waiting_for_a_writer(tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    source=tmp_path/'policy-fifo';os.mkfifo(source)
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY',str(source))
    monkeypatch.setenv('IMA_CACHE_HOLD_POLICY_SHA256','0'*64)
    code=('from ima.research_cache_policy import cache_holds\n'
          'from ima.research_storage import StorageBlocked\n'
          'try:cache_holds()\n'
          'except StorageBlocked:pass\n'
          'else:raise AssertionError("FIFO accepted")\n')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
