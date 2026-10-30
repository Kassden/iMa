"""Real controller/control/outbox plumbing with fixture planner, fit and delivery."""
from dataclasses import replace
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ima.research_storage as storage
from ima import research_controller as core
from ima.research_control import ControlIntentStore,ControlPolicy
from ima.research_executor import RecipeExecutionResult
from ima.research_expansion import run_expansion_campaign,_refresh_control_queue,_planner_spend,_TrackingDiagnostics
from ima.research_telemetry import _outbox,_read_outbox


@pytest.fixture
def campaign_context(tmp_path,monkeypatch):
    from tests.test_research_expansion_resources import ExpansionResourceIntegrationTests
    case=ExpansionResourceIntegrationTests('test_final_completion_stop_waits_for_tracking_upload')
    case.setUp()
    a=storage.StorageAuthority(tmp_path/'authority',headroom_bytes=256*1024**2,headroom_inodes=4,
        statvfs=lambda path:SimpleNamespace(f_bavail=2*1024**3,f_frsize=1,f_favail=10000))
    policy=ControlPolicy(max_envelope_bytes=1024**2,page_size=4096,max_db_pages=1024,
        max_journal_bytes=80*1024**2,metadata_bytes=65536,allowance_bytes=96*1024**2)
    store=ControlIntentStore.bootstrap(a,policy)
    directory=case.config.campaign_dir.resolve();directory.mkdir(parents=True,exist_ok=True)
    config=replace(case.config,campaign_dir=directory,
        dataset_path=case.config.dataset_path.resolve(),
        protocol_path=case.config.protocol_path.resolve(),
        mlflow_tracking_uri='http://unused',timeout_minutes=1)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(a.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:a)
    try:yield case,store,directory,config
    finally:case.doCleanups()


def fixture_delivery(directory,config,role,number):
    receipt={'trace_id':'fixture-delivery','role':role,'number':number}
    conn=_outbox(directory)
    try:
        conn.execute('UPDATE traces SET delivered=1,receipt=? WHERE role=? AND number=?',
                     (json.dumps(receipt),role,number))
        conn.commit()
    finally:conn.close()
    return receipt


def test_controller_reserves_before_planner_and_clears_completed_intents(campaign_context):
    from tests.test_research_expansion import proposal
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    calls=[]
    def planned(evidence,config):
        pending=[json.loads(raw) for raw in ControlIntentStore(store.authority).pending()]
        reserved=[row for row in pending if row['role']=='decision']
        assert len(reserved)==1 and reserved[0]['state']=='reserved'
        source=directory/'evidence'/(evidence['decision_id']+'.json')
        identity=reserved[0]['evidence_identity']
        assert identity['evidence_id']==evidence['evidence_id']
        assert identity['evidence_source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        calls.append(evidence['decision_id'])
        item=proposal().model_copy(update={'evidence_ids':(evidence['evidence_id'],)})
        return {'decision':{'decision_id':evidence['decision_id'],'evidence_id':evidence['evidence_id'],
                'trial_budget':1,'programs':[item.model_dump(mode='json')]},'usage':{'cost_status':'fixture'}}
    def worker(request,*args):
        return RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,
            request.recipe.recipe_hash(),'win_probability','completed',
            'development_fundamental_race_log_loss',2.2,{},{},{},0.)
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision',side_effect=planned),patch('ima.research_expansion._worker',side_effect=worker),patch('ima.research_v5.upload_result',return_value={'run_id':'fixture-run'}),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    status=core._read_json(directory/'status.json')
    if result['mode']!='complete':
        from ima.research_hypotheses import HypothesisMemory
        diagnostics={key:status.get(key) for key in (
            'mode','counts','ledger','tracking_errors','last_planner_error','blocked_storage',
            'preparation_dependencies','preparing_programs','ready_programs','pending_trace_enqueues',
            'trace_control_state_unknown','pending_trace_control_reservations')}
        diagnostics['hypotheses']=HypothesisMemory(directory/'hypotheses.sqlite').retrieve(limit=20)
        pytest.fail(json.dumps(diagnostics,indent=2))
    assert len(calls)==1
    assert result['pending_trace_enqueues']==0
    assert result['pending_trace_control_reservations']==0
    assert ControlIntentStore(store.authority).pending()==()
    with _read_outbox(directory) as conn:
        assert conn.execute("SELECT COUNT(*) FROM traces WHERE role='decision'").fetchone()[0]==1
    case.tracing_startup.assert_called_once_with(config)


def test_expired_deadline_preserves_completed_trial_pending_upload(campaign_context):
    from tests.test_research_expansion import proposal
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    finished_fit=threading.Event()
    clock=SimpleNamespace(time=time.time,sleep=time.sleep,
        monotonic=lambda:time.monotonic()+(120 if finished_fit.is_set() else 0))
    def planned(evidence,config):
        item=proposal().model_copy(update={'evidence_ids':(evidence['evidence_id'],)})
        return {'decision':{'decision_id':evidence['decision_id'],'evidence_id':evidence['evidence_id'],
            'trial_budget':1,'programs':[item.model_dump(mode='json')]},'usage':{'cost_status':'fixture'}}
    def worker(request,*args):
        result=RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,
            request.recipe.recipe_hash(),'win_probability','completed',
            'development_fundamental_race_log_loss',2.2,{},{},{},0.)
        finished_fit.set()
        return result
    with patch('ima.research_expansion.time',clock),patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision',side_effect=planned),patch('ima.research_expansion._worker',side_effect=worker),patch('ima.research_v5.upload_result') as upload,patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    assert finished_fit.is_set()
    upload.assert_not_called()
    assert result['mode']=='blocked_tracking' and result['pending_tracking']==1
    from ima.research_store import ResearchLedger
    ledger=ResearchLedger(directory/'ledger.sqlite')
    assert len(ledger.pending_outbox())==1
    terminal=ledger.terminal_results()
    assert len(terminal)==1 and terminal[0]['status']=='completed'
    assert terminal[0]['uploaded_at'] is None
    assert terminal[0]['attempt_id']==ledger.pending_outbox()[0]['attempt_id']


def test_controller_startup_replays_stored_sequence_before_new_snapshot(campaign_context):
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    original={'evidence_id':'frozen-source','created_at':'2026-10-20T00:00:00Z','counts':{'completed':42}}
    with store.claim(directory,config.mlflow_tracking_uri):
        store.stage('execution',17,original,signature='source-signature')
    core._write_json_atomic(directory/'STOP',{})
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision') as planner,patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    planner.assert_not_called()
    assert result['mode']=='stopped'
    assert ControlIntentStore(store.authority).pending()==()
    with _read_outbox(directory) as conn:
        rows=conn.execute("SELECT number,payload FROM traces WHERE role='execution' ORDER BY number").fetchall()
    assert rows[0]['number']==17 and json.loads(rows[0]['payload'])==original
    assert len({row['number'] for row in rows})==len(rows)
    assert all(row['number']>=17 for row in rows)


def test_unresolved_reservation_never_repeats_planner_or_reports_known_zero_total(campaign_context):
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    with store.claim(directory,config.mlflow_tracking_uri):
        original=store.reserve_decision(7,{'decision_id':'D000007','evidence_id':'unknown-response'})
    core._write_json_atomic(directory/'STOP',{})
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision') as planner,patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    planner.assert_not_called()
    assert result['mode']=='blocked_tracking'
    assert result['pending_trace_control_reservations']==1
    assert result['planner_spend_unknown'] is True and result['planner_spend_usd'] is None
    assert ControlIntentStore(store.authority).pending()==(original,)


def test_unknown_control_read_keeps_planning_frozen_even_with_empty_normal_outbox(campaign_context):
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    config=replace(config,timeout_minutes=.05)
    original_pending=ControlIntentStore.pending
    observations=[]
    def pressure(self):
        observations.append(1)
        if len(observations)>=3:raise storage.StorageBlocked('control read fixture pressure')
        return original_pending(self)
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision') as planner,patch.object(ControlIntentStore,'pending',pressure),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    planner.assert_not_called()
    assert result['mode']=='blocked_tracking'
    assert result['trace_control_state_unknown'] is True
    assert result['pending_trace_control_reservations'] is None
    assert result['planner_spend_unknown'] is True and result['planner_spend_usd'] is None


def test_refresh_retains_last_known_control_state_and_marks_unknown():
    previous=[{'state':'reserved','sequence':8}]
    store=SimpleNamespace(pending=lambda:(_ for _ in ()).throw(storage.StorageBlocked('fixture')))
    pending,unknown,error=_refresh_control_queue(store,{},previous)
    assert pending is previous and unknown is True and error=='fixture'


def test_reserved_spend_does_not_replace_a_reported_receipt_with_unknown(tmp_path):
    responses=tmp_path/'planner-responses';responses.mkdir()
    (responses/'D000001.json').write_text(json.dumps({'response':{'usage':{'total_cost_usd':.07,'cost_status':'reported'}}}))
    known=_planner_spend(tmp_path,reserved_decision_ids=['D000001'])
    assert known['known_spend_usd']==.07 and known['total_cost_usd']==.07
    unknown=_planner_spend(tmp_path,reserved_decision_ids=['D000001','D000002'])
    assert unknown['known_spend_usd']==.07 and unknown['total_cost_usd'] is None
    assert unknown['unresolved_decision_ids']==['D000002']


def test_tracking_diagnostics_bound_both_count_and_individual_size():
    errors=_TrackingDiagnostics(str(number) for number in range(150))
    assert len(errors)==100 and errors[0]=='50'
    errors.extend(['x'*100000,'last'])
    assert len(errors)==100 and errors[-1]=='last'
    assert len(errors[-2])<=errors.max_entry_chars
    assert errors[-2].endswith('[truncated]')
    assert json.loads(json.dumps(errors))[-1]=='last'


def test_model_upload_storage_pressure_retries_past_normal_failure_limit(campaign_context):
    from tests.test_research_expansion import proposal
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    config=replace(config,timeout_minutes=1)
    calls=[]
    def planned(evidence,config):
        item=proposal().model_copy(update={'evidence_ids':(evidence['evidence_id'],)})
        return {'decision':{'decision_id':evidence['decision_id'],'evidence_id':evidence['evidence_id'],
            'trial_budget':1,'programs':[item.model_dump(mode='json')]},'usage':{'cost_status':'fixture'}}
    def worker(request,*args):
        return RecipeExecutionResult(1,request.attempt_id,request.proposal_id,request.trial_number,
            request.recipe.recipe_hash(),'win_probability','completed',
            'development_fundamental_race_log_loss',2.2,{},{},{},0.)
    def upload(*args):
        calls.append(1)
        if len(calls)<=4:raise storage.StorageBlocked('fixture upload capacity')
        return {'run_id':'recovered-upload'}
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision',side_effect=planned),patch('ima.research_expansion._worker',side_effect=worker),patch('ima.research_v5.upload_result',side_effect=upload),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    assert len(calls)==5 and result['mode']=='complete'
    assert result['pending_tracking']==0
    assert core._read_json(directory/'status.json')['upload_storage_blocked']=={}


def test_final_nonstorage_enqueue_failure_cannot_report_clean_stop(campaign_context):
    from tests.test_research_expansion_resources import executors
    from ima.research_telemetry import _enqueue_trace
    case,store,directory,config=campaign_context
    core._write_json_atomic(directory/'STOP',{})
    def enqueue(campaign,payload,config,*,role,number):
        if 'requested_stop_mode' in payload:
            raise RuntimeError('fixture final outbox unavailable')
        return _enqueue_trace(campaign,payload,config,role=role,number=number)
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_telemetry._enqueue_trace',side_effect=enqueue),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    assert result['mode']=='blocked_tracking'
    pending=[json.loads(raw) for raw in ControlIntentStore(store.authority).pending()]
    assert any(row['state']=='complete' and row['role']=='execution' for row in pending)
    assert result['pending_trace_enqueues']==1


def test_paid_response_survives_both_normal_receipt_and_outbox_pressure(campaign_context):
    from tests.test_research_expansion import proposal
    from tests.test_research_expansion_resources import executors
    from ima.research_telemetry import _enqueue_trace
    case,store,directory,config=campaign_context
    config=replace(config,timeout_minutes=.03)
    write=core._write_json_atomic
    def planned(evidence,config):
        write(directory/'STOP',{})
        item=proposal().model_copy(update={'evidence_ids':(evidence['evidence_id'],)})
        return {'decision':{'decision_id':evidence['decision_id'],'evidence_id':evidence['evidence_id'],
            'trial_budget':1,'programs':[item.model_dump(mode='json')]},'usage':{'total_cost_usd':.07,'cost_status':'reported'}}
    def receipt(path,value):
        if path.parent in {directory/'decisions',directory/'planner-responses'}:
            raise storage.StorageBlocked('fixture decision receipt full')
        return write(path,value)
    def enqueue(campaign,payload,config,*,role,number):
        if role=='decision':raise storage.StorageBlocked('fixture decision outbox full')
        return _enqueue_trace(campaign,payload,config,role=role,number=number)
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_expansion.plan_decision',side_effect=planned) as planner,patch.object(core,'_write_json_atomic',side_effect=receipt),patch('ima.research_telemetry._enqueue_trace',side_effect=enqueue),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    planner.assert_called_once()
    assert result['mode']=='blocked_tracking'
    pending=[json.loads(raw) for raw in ControlIntentStore(store.authority).pending()]
    decision=next(row for row in pending if row['role']=='decision')
    assert decision['state']=='complete'
    assert decision['payload']['planner_response']['usage']['total_cost_usd']==.07
    assert decision['payload']['planner_usage']['total_cost_usd']==.07


def test_frozen_usage_is_not_lost_without_normal_receipts(tmp_path):
    payload={'decision_id':'D000001','planner_usage':{'total_cost_usd':.07,'cost_status':'reported'},
        'planner_response':{'usage':{'total_cost_usd':.07,'cost_status':'reported'}}}
    result=_planner_spend(tmp_path,reserved_decision_ids=['D000001'],fallback_decisions=[payload])
    assert result['known_spend_usd']==.07 and result['total_cost_usd']==.07


def test_final_enqueue_and_control_read_failure_keep_returned_cost_unknown(campaign_context):
    from tests.test_research_expansion_resources import executors
    from ima.research_telemetry import _enqueue_trace
    case,store,directory,config=campaign_context
    core._write_json_atomic(directory/'STOP',{})
    pending=ControlIntentStore.pending
    failed=[]
    def read(self):
        if failed:raise storage.StorageBlocked('fixture final control unreadable')
        return pending(self)
    def enqueue(campaign,payload,config,*,role,number):
        if 'requested_stop_mode' in payload:
            failed.append(True)
            raise RuntimeError('fixture final outbox unavailable')
        return _enqueue_trace(campaign,payload,config,role=role,number=number)
    with patch('ima.research_expansion.ProcessPoolExecutor',side_effect=executors),patch('ima.research_telemetry._enqueue_trace',side_effect=enqueue),patch.object(ControlIntentStore,'pending',read),patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        result=run_expansion_campaign(config)
    assert result['mode']=='blocked_tracking' and result['trace_control_state_unknown'] is True
    assert result['planner_spend_unknown'] is True and result['planner_spend_usd'] is None
    assert core._read_json(directory/'status.json')['planner_spend']['total_cost_usd'] is None


@pytest.mark.parametrize('crash_window',['before_apply','after_apply'])
def test_saved_paid_response_recovers_reserved_intent_without_provider_replay(campaign_context,crash_window):
    from ima import research_expansion as expansion
    from tests.test_research_expansion import proposal
    from tests.test_research_expansion_resources import executors
    case,store,directory,config=campaign_context
    config=replace(config,max_pending_programs=1,max_new_programs_per_decision=1)
    class SimulatedCrash(BaseException):pass
    def planned(evidence,config):
        item=proposal().model_copy(update={'evidence_ids':(evidence['evidence_id'],)})
        return {'decision':{'decision_id':evidence['decision_id'],'evidence_id':evidence['evidence_id'],
            'trial_budget':1,'programs':[item.model_dump(mode='json')]},
            'usage':{'total_cost_usd':.07,'cost_status':'reported'}}
    apply=expansion.apply_decision
    publish=expansion._publish_campaign_trace
    def interrupted_apply(*args,**kwargs):
        if crash_window=='before_apply':raise SimulatedCrash()
        return apply(*args,**kwargs)
    def interrupted_publish(*args,**kwargs):
        if crash_window=='after_apply' and kwargs.get('role')=='decision':raise SimulatedCrash()
        return publish(*args,**kwargs)
    with patch.object(expansion,'ProcessPoolExecutor',side_effect=executors), \
         patch.object(expansion,'plan_decision',side_effect=planned) as planner, \
         patch.object(expansion,'apply_decision',side_effect=interrupted_apply), \
         patch.object(expansion,'_publish_campaign_trace',side_effect=interrupted_publish), \
         patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        with pytest.raises(SimulatedCrash):run_expansion_campaign(config)
    planner.assert_called_once()
    assert (directory/'planner-responses'/'D000001.json').is_file()
    pending=[json.loads(raw) for raw in ControlIntentStore(store.authority).pending()]
    assert any(row['role']=='decision' and row['state']=='reserved' for row in pending)
    core._write_json_atomic(directory/'STOP',{})
    with patch.object(expansion,'ProcessPoolExecutor',side_effect=executors), \
         patch.object(expansion,'plan_decision',side_effect=AssertionError('paid call repeated')) as planner, \
         patch('ima.research_telemetry._deliver_trace',side_effect=fixture_delivery):
        outcome=run_expansion_campaign(config)
    planner.assert_not_called()
    assert outcome['mode']=='stopped'
    assert ControlIntentStore(store.authority).pending()==()
    with _read_outbox(directory) as conn:
        rows=conn.execute("SELECT payload FROM traces WHERE role='decision'").fetchall()
    assert len(rows)==1
    recovered=json.loads(rows[0]['payload'])
    assert recovered['planner_status']=='accepted',recovered.get('error')
    assert recovered['saved_response_recovered'] is True and recovered['additional_planner_calls']==0
    assert recovered['planner_usage']['total_cost_usd']==.07
    with expansion.DecisionStore(directory/'decisions.sqlite')._read() as conn:
        assert conn.execute('SELECT COUNT(*) FROM decisions').fetchone()[0]==1
