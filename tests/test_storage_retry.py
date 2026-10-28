import errno
import json
import pickle
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from ima.research_storage import StorageAuthority, StorageBlocked, as_storage_blocked, storage_writer
from ima.research_store import ResearchLedger


def test_tagged_legacy_storage_failures_do_not_retire_preparation(tmp_path):
    from ima.research_expansion import _FoldDependencies
    state=tmp_path/'fold-dependency-errors.json'
    state.write_text(json.dumps({'schema_version':1,'errors':{'dependency':{
        'failures':3,'blocked':True,'diagnostic':{'failure_kind':'infrastructure_storage'}}}}))
    dependencies=_FoldDependencies(tmp_path)
    assert dependencies.errors['dependency']['scientific_failures']==0
    assert dependencies.errors['dependency']['blocked'] is False
    dependencies.failed('dependency','program',StorageBlocked('retry is still infrastructure'))
    assert dependencies.errors['dependency']['failures']==4
    assert dependencies.errors['dependency']['scientific_failures']==0
    assert dependencies.errors['dependency']['blocked'] is False


def test_typed_storage_error_survives_worker_transport():
    error=StorageBlocked('full',{'stage':'preparation','output':'/owned/output'})
    copied=pickle.loads(pickle.dumps(error))
    assert copied.diagnostic==error.diagnostic
    assert copied.diagnostic['retryable'] and str(copied)=='full'


def test_storage_normalization_requires_enabled_authority_and_actual_pressure(tmp_path,monkeypatch):
    full=sqlite3.OperationalError('database or disk is full')
    full.sqlite_errorcode=sqlite3.SQLITE_FULL
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    assert as_storage_blocked(full,stage='ledger',output=tmp_path) is None
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'control'))
    converted=as_storage_blocked(full,stage='ledger',output=tmp_path)
    assert isinstance(converted,StorageBlocked)
    assert converted.diagnostic['sqlite_errorcode']==sqlite3.SQLITE_FULL
    for error in (OSError(errno.EIO,'I/O error'),PermissionError('denied'),
                  sqlite3.OperationalError('no such table'),ValueError('bad dataset')):
        assert as_storage_blocked(error,stage='fixture',output=tmp_path) is None


@pytest.mark.parametrize('code',[errno.ENOSPC,errno.EDQUOT])
def test_guard_converts_os_storage_failure_and_releases_lease(tmp_path,monkeypatch,code):
    import ima.research_storage as storage
    authority=StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    @storage_writer(stage='fixture',output=lambda a:a['destination'],estimate=lambda a:1024)
    def operation(destination):raise OSError(code,'disk unavailable')
    with pytest.raises(StorageBlocked) as failure:operation(tmp_path/'artifact')
    assert failure.value.diagnostic['errno']==code
    assert authority.status()['reserved_bytes']==0


def test_deferred_trial_survives_restart_without_new_ask_or_terminal_outcome(tmp_path):
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    payload={'recipe':{'seed':42},'trial_number':7}
    row=ledger.reserve_attempt('same-scientific-identity',payload)
    ledger.mark_running(row.attempt_id)
    runtime={'status':'failed','task_nonce':'physical-one','deadline_at':'2026-01-01T00:00:00+00:00'}
    with patch('ima.research_store.time.time',return_value=100):
        receipt=ledger.defer_infrastructure(row.attempt_id,StorageBlocked('full').diagnostic,
                                            runtime_snapshot=runtime)
        assert receipt['retry_count']==1 and len(ledger.pending_attempts())==1
        assert not ledger.retry_ready(row.attempt_id)
    reopened=ResearchLedger(ledger.path)
    assert reopened.reserved_attempts()[0]['payload']==payload
    assert reopened.terminal_results()==reopened.pending_tells()==reopened.pending_outbox()==[]
    assert reopened.latest_infrastructure_runtime(row.attempt_id)==runtime
    with patch('ima.research_store.time.time',return_value=131):
        assert reopened.pending_attempts()[0]['attempt_id']==row.attempt_id
        assert reopened.retry_ready(row.attempt_id)
    assert reopened.reserve_attempt('same-scientific-identity',payload).attempt_id==row.attempt_id
    reopened.mark_running(row.attempt_id)
    assert reopened.infrastructure_waits()==[]
    reopened.complete_attempt(row.attempt_id,{'objective_value':2.1})
    assert len(reopened.terminal_results())==1
    assert reopened.infrastructure_history(row.attempt_id)[0]['runtime']==runtime
    with pytest.raises(ValueError,match='terminal'):
        reopened.defer_infrastructure(row.attempt_id,StorageBlocked('full').diagnostic)
    assert len(reopened.infrastructure_history(row.attempt_id))==1


def test_storage_prep_errors_do_not_trip_scientific_rejection_circuit(tmp_path):
    from ima.research_expansion import _FoldDependencies
    state=_FoldDependencies(tmp_path)
    for _ in range(5):state.failed('dependency','program',StorageBlocked('space'))
    assert not state.blocked('dependency')
    restarted=_FoldDependencies(tmp_path)
    assert not restarted.blocked('dependency')
    assert restarted.errors['dependency']['scientific_failures']==0
    for _ in range(3):restarted.failed('dependency','program',ValueError('invalid schema'))
    assert restarted.blocked('dependency')
    state.close();restarted.close()


def test_executor_propagates_storage_without_writing_a_scientific_failure(tmp_path,monkeypatch):
    import ima.research_executor as executor
    from ima.research_specs import PipelineRecipe
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    def blocked(request):raise StorageBlocked('no admission')
    monkeypatch.setattr(executor,'_load_recipe_features',blocked)
    recipe=PipelineRecipe.model_validate({'schema_version':3,'feature_schema':'baseline-v1',
        'model':{'kind':'benter_conditional_logit','parameters':{'max_iter':50}},'blend':{'kind':'none'}})
    request=executor.RecipeExecutionRequest('a','p',0,recipe,tmp_path/'data.csv',tmp_path/'trial')
    with pytest.raises(StorageBlocked):executor.execute_recipe(request)
    assert not (request.output_dir/'result.json').exists()


def test_executor_normalizes_raw_enospc_before_catch_all(tmp_path,monkeypatch):
    import ima.research_executor as executor
    from contextlib import contextmanager
    from ima.research_specs import PipelineRecipe
    import ima.research_storage as storage
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'control'))
    @contextmanager
    def admitted(*args,**kwargs):yield None
    monkeypatch.setattr(storage,'admitted_output',admitted)
    dataset=tmp_path/'dataset.csv';dataset.write_text('placeholder')
    def blocked(request):raise OSError(errno.ENOSPC,'raw disk full')
    monkeypatch.setattr(executor,'_load_recipe_features',blocked)
    recipe=PipelineRecipe.model_validate({'schema_version':3,'feature_schema':'baseline-v1',
        'model':{'kind':'benter_conditional_logit'},'blend':{'kind':'none'}})
    request=executor.RecipeExecutionRequest('a','p',0,recipe,dataset,tmp_path/'trial')
    with pytest.raises(StorageBlocked) as failure:executor.execute_recipe(request)
    assert failure.value.diagnostic['errno']==errno.ENOSPC
    assert not (request.output_dir/'result.json').exists()


def test_runtime_reset_preserves_archived_episode_and_rejects_replacement(tmp_path):
    from ima.research_expansion import _defer_storage_fit,_reset_storage_retry_runtime
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    attempt=ledger.reserve_attempt('one',{'trial_number':0}).attempt_id
    ledger.mark_running(attempt)
    path=tmp_path/'trials'/attempt/'runtime.json';path.parent.mkdir(parents=True)
    lock=path.with_suffix('.lock');lock.touch()
    before={'status':'failed','task_nonce':'old','deadline_at':'2000-01-01T00:00:00+00:00'}
    path.write_text(json.dumps(before))
    _defer_storage_fit(ledger,tmp_path,attempt,StorageBlocked('full'))
    path.write_text(json.dumps(before|{'task_nonce':'different-physical-owner'}))
    with pytest.raises(ValueError,match='changed'):_reset_storage_retry_runtime(ledger,tmp_path,attempt)
    path.write_text(json.dumps(before))
    _reset_storage_retry_runtime(ledger,tmp_path,attempt)
    assert not path.exists() and lock.exists()
    assert ledger.latest_infrastructure_runtime(attempt)==before
    assert len(ledger.infrastructure_history(attempt))==1


def test_no_runtime_reset_for_a_possibly_active_fit(tmp_path):
    from ima.research_expansion import _defer_storage_fit
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    attempt=ledger.reserve_attempt('active',{}).attempt_id;ledger.mark_running(attempt)
    path=tmp_path/'trials'/attempt/'runtime.json';path.parent.mkdir(parents=True)
    path.write_text('{"status":"running","task_nonce":"active"}')
    with pytest.raises(RuntimeError,match='active'):
        _defer_storage_fit(ledger,tmp_path,attempt,StorageBlocked('space'))
    assert ledger.infrastructure_waits()==[] and path.exists()


@pytest.mark.parametrize('stage',['dataset','program','fold'])
def test_artifact_retry_survives_restart_without_scientific_allocation(tmp_path,stage):
    from ima.research_expansion import _defer_storage_operation,_reset_operation_runtime
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    operation=stage+':immutable-input'
    path=tmp_path/stage/'runtime.json';path.parent.mkdir()
    runtime={'status':'failed','task_nonce':'first-physical-episode',
             'deadline_at':'2000-01-01T00:00:00+00:00'}
    path.write_text(json.dumps(runtime))
    with patch('ima.research_store.time.time',return_value=100):
        _defer_storage_operation(ledger,operation,path,StorageBlocked('disk full'))
        assert not ledger.retry_ready(operation)
    reopened=ResearchLedger(ledger.path)
    assert reopened.reserved_attempts()==reopened.terminal_results()==[]
    assert reopened.pending_tells()==reopened.pending_outbox()==[]
    assert reopened.infrastructure_waits()==[]
    assert reopened.operation_waits()[0]['operation_id']==operation
    with patch('ima.research_store.time.time',return_value=131):
        assert reopened.retry_ready(operation)
    path.write_text(json.dumps(runtime|{'task_nonce':'replacement-owner'}))
    with pytest.raises(ValueError,match='changed'):
        _reset_operation_runtime(reopened,operation,path)
    path.write_text(json.dumps(runtime))
    _reset_operation_runtime(reopened,operation,path)
    assert not path.exists()
    assert reopened.latest_infrastructure_runtime(operation)==runtime
    reopened.complete_operation(operation)
    assert reopened.operation_waits()==[]
    assert reopened.infrastructure_history(operation)[0]['runtime']==runtime


@pytest.mark.parametrize('operation',['dataset:','program:','fold:','dataset:  ','trial:x',None])
def test_nontrial_retry_requires_an_identified_supported_operation(tmp_path,operation):
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    with pytest.raises(ValueError,match='Namespaced'):
        ledger.defer_operation(operation,StorageBlocked('full').diagnostic)
    assert ledger.operation_waits()==[]
