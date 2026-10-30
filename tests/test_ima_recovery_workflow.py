"""Deterministic terminal workflow and cooperating-worker safety, not live proof."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import io
import json
import fcntl
import os
from pathlib import Path
import time
import numpy as np
import pytest

from ima.research_preparation import PreparationCache, PreparationKey, PreparedFold
from ima.research_storage import ReservedFile, StorageAuthority, StorageBlocked
from scripts.research_storage import main


def test_operator_inspection_and_unapproved_plan_cannot_delete(tmp_path,capsys):
    root=tmp_path/'research'; root.mkdir()
    assert main(['inspect','--root',str(root)])==0
    inspection=json.loads(capsys.readouterr().out)
    assert inspection['reclaim_authorized'] is False
    cache=root/'campaign/fold-preparation-cache'; payload=cache/('a'*64)
    payload.mkdir(parents=True)
    (payload/'arrays.joblib').write_bytes(b'fixture')
    (cache/('a'*64+'.lock')).touch()
    manifest=tmp_path/'plan.json'
    assert main(['plan','--root',str(root),'--cache-root',str(cache),'--output',str(manifest)])==0
    plan=json.loads(manifest.read_text())
    assert plan['approved'] is False and len(plan['candidates'])==1
    assert payload.exists()


def test_admitted_identical_preparation_builds_once_and_loads_readonly(tmp_path,monkeypatch):
    control=tmp_path/'control'
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(control))
    key=PreparationKey(source_content_hash='source',training_row_keys=('a',),
        calibration_row_keys=('b',),score_row_keys=('c',),target_labels_hash='labels',
        availability_policy={'strict':True},feature_definitions=({'numeric':['x']},),
        selection_settings={},fitted_transform_settings={},fold_dates=('2020-01-01',),
        seed=42,implementation_revision='fixture',dependency_versions={'numpy':np.__version__})
    builds=[]
    def builder():
        builds.append(1); time.sleep(.05)
        return PreparedFold({name+'_x':np.array([[1.]]) for name in ('train','calibration','score')},
            ('x',),{'train':('a',),'calibration':('b',),'score':('c',)})
    def prepare(): return PreparationCache(tmp_path/'fold-preparation-cache').prepare_fold(key,builder)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs=list(pool.map(lambda _:prepare(),range(2)))
    assert len(builds)==1
    assert outputs[0].artifact_id==outputs[1].artifact_id
    assert not outputs[0].load_arrays()['train_x'].flags.writeable
    assert StorageAuthority(control).status()['reserved_bytes']==0


def test_growth_denied_before_any_underreserved_bytes_are_written():
    class Deny:
        def extend(self,*args,**kwargs): raise StorageBlocked('pressure')
    target=io.BytesIO()
    writer=ReservedFile(target,{'authority':Deny(),'token':'fixture','limit':0,'written':0,'cache':True})
    with pytest.raises(StorageBlocked): writer.write(b'cannot escape budget')
    assert target.getvalue()==b''


def test_publication_converts_growth_and_hands_pin_to_consumer(tmp_path,monkeypatch):
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'control'))
    key=PreparationKey(source_content_hash='source',training_row_keys=('a',),
        calibration_row_keys=('b',),score_row_keys=('c',),target_labels_hash='labels',
        availability_policy={'strict':True},feature_definitions=({'numeric':['x']},),
        selection_settings={},fitted_transform_settings={},fold_dates=('2020-01-01',),
        seed=42,implementation_revision='fixture',dependency_versions={'numpy':np.__version__})
    cache=PreparationCache(tmp_path/'fold-preparation-cache')
    commit=StorageAuthority.commit_cache
    observed=[]
    def interleaved(authority,token,root,identity,allocated):
        commit(authority,token,root,identity,allocated)
        status=authority.status()
        assert status['reservations'][token]['cache_growth_bytes']==0
        # The producer still owns its publication lock before the consumer pin.
        assert not PreparationCache(root).evict(identity)
        observed.append(identity)
    monkeypatch.setattr(StorageAuthority,'commit_cache',interleaved)
    def builder():
        return PreparedFold({name+'_x':np.array([[1.]]) for name in ('train','calibration','score')},
            ('x',),{'train':('a',),'calibration':('b',),'score':('c',)})
    with ExitStack() as pins:
        artifact=cache.prepare_fold(key,builder,pins=pins)
        assert observed==[artifact.artifact_id]
        assert not cache.evict(artifact.artifact_id)
        assert artifact.load_arrays()['train_x'][0,0]==1
    assert cache.evict(artifact.artifact_id)
    assert (cache.root/(artifact.artifact_id+'.lock')).exists()
    assert (cache.root/(artifact.artifact_id+'.pin.lock')).exists()


def test_companion_pin_covers_forced_nonatomic_downgrade_cold_and_warm(tmp_path,monkeypatch):
    key=PreparationKey(source_content_hash='source',training_row_keys=('a',),
        calibration_row_keys=('b',),score_row_keys=('c',),target_labels_hash='labels',
        availability_policy={'strict':True},feature_definitions=({'numeric':['x']},),
        selection_settings={},fitted_transform_settings={},fold_dates=('2020-01-01',),
        seed=42,implementation_revision='fixture',dependency_versions={'numpy':np.__version__})
    cache=PreparationCache(tmp_path/'fold-preparation-cache')
    lock=cache.root/(key.cache_id()+'.lock'); lock.touch()
    real=fcntl.flock; gaps=[]
    def flock(descriptor,operation):
        fd=descriptor if isinstance(descriptor,int) else descriptor.fileno()
        if operation==fcntl.LOCK_SH and os.fstat(fd).st_ino==lock.stat().st_ino:
            real(fd,fcntl.LOCK_UN)
            assert not cache.evict(key.cache_id())
            gaps.append(1)
        return real(descriptor,operation)
    monkeypatch.setattr(fcntl,'flock',flock)
    def builder():
        return PreparedFold({name+'_x':np.array([[1.]]) for name in ('train','calibration','score')},
            ('x',),{'train':('a',),'calibration':('b',),'score':('c',)})
    for expected in ('miss','hit'):
        with ExitStack() as pins:
            artifact=cache.prepare_fold(key,builder,pins=pins)
            assert artifact.cache_status==expected
            assert artifact.load_arrays()['train_x'][0,0]==1
    assert len(gaps)==2
    assert cache.evict(key.cache_id())
