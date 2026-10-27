import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ima.research_storage as storage
from ima.research_control import ControlIntentStore
from ima.research_telemetry import log_snapshot,replay_control_trace,_read_outbox
from tests.test_research_control import authority,policy


@pytest.fixture
def context(tmp_path,monkeypatch):
    # Delivery reserves its artifact envelope and an independent native SQLite
    # lease. Keep successful handoff fixtures above both peaks; denial paths
    # below inject capacity failures explicitly.
    a=authority(tmp_path/'authority', free=128 * 1024**2)
    store=ControlIntentStore.bootstrap(a,policy())
    campaign=tmp_path/'campaign';campaign.mkdir()
    config=SimpleNamespace(mlflow_tracking_uri='http://tracking.invalid',research_policy='expansion_v6')
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(a.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:a)
    return store,campaign,config


def rows(campaign):
    with _read_outbox(campaign) as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM traces ORDER BY role,number')]


@pytest.mark.parametrize('role',['execution','dataset','betting'])
def test_complete_operational_handoff_commits_once_before_delivery(context,role):
    store,campaign,config=context
    payload={'evidence_id':'e1','source_inputs':{'features':['speed','rating']}}
    def delivered(*args):
        assert store.pending()==()
        assert json.loads(rows(campaign)[0]['payload'])==payload
        return {'trace_id':'delivery-fixture'}
    with store.claim(campaign,config.mlflow_tracking_uri):
        with patch('ima.research_telemetry._deliver_trace',side_effect=delivered) as emit:
            result=log_snapshot(campaign,payload,config,role=role,number=1,control_store=store,
                                execution_signature='source-signature')
    assert result['trace_id']=='delivery-fixture'
    emit.assert_called_once()
    assert len(rows(campaign))==1


def test_decision_reservation_and_native_cost_are_reused_without_another_span(context):
    from tests.test_research_telemetry import SnapshotCostTests
    store,campaign,config=context
    case=SnapshotCostTests('test_paid_decision_native_cost_is_written_once_and_link_reused')
    case.setUp()
    identity={'decision_id':'decision-1','evidence_id':'evidence-1'}
    try:
        with store.claim(campaign,config.mlflow_tracking_uri):
            store.reserve_decision(1,identity)
            first=log_snapshot(campaign,case.payload(),config,role='decision',number=1,
                               control_store=store,decision_evidence_identity=identity)
            second=log_snapshot(campaign,case.payload(),config,role='decision',number=1,
                                control_store=store,decision_evidence_identity=identity)
            assert first==second and store.pending()==()
        assert case.mlflow.start_span.call_count==1
        cost_calls=[call for call in case.span.set_attribute.call_args_list if call.args[0]=='mlflow.llm.cost']
        assert len(cost_calls)==1
        assert json.loads(rows(campaign)[0]['payload'])['planner_usage']==case.payload()['planner_usage']
    finally:case.doCleanups()


def test_denied_enqueue_reopens_and_replays_frozen_evidence_not_changed_source(context):
    store,campaign,config=context
    evidence={'decision_id':'D000001','evidence_id':'input-1','terminal_watermark':42,'hypothesis':'original'}
    directory=campaign/'evidence';directory.mkdir()
    source=directory/'D000001.json';source.write_text(json.dumps(evidence))
    identity={key:evidence[key] for key in ('decision_id','evidence_id')}
    payload={'decision_id':'D000001','evidence_id':'output-2','planner_usage':{'total_cost_usd':.01,'input_tokens':12}}
    with store.claim(campaign,config.mlflow_tracking_uri):
        store.reserve_decision(1,identity)
        with patch('ima.research_telemetry._enqueue_trace',side_effect=storage.StorageBlocked('capacity')):
            with pytest.raises(storage.StorageBlocked) as failure:
                log_snapshot(campaign,payload,config,role='decision',number=1,
                             control_store=store,decision_evidence_identity=identity)
        assert failure.value.diagnostic['trace_publication_phase']=='outbox_enqueue'
        frozen,=store.pending()
    assert not (campaign/'trace-outbox.sqlite').exists()
    source.write_text(json.dumps(evidence|{'hypothesis':'changed','terminal_watermark':999}))
    reopened=ControlIntentStore(store.authority)
    with reopened.claim(campaign,config.mlflow_tracking_uri):
        assert reopened.pending()==(frozen,)
        with patch('ima.research_telemetry._trace_payload',side_effect=AssertionError('re-read normalized input')):
            with patch('ima.research_telemetry._deliver_trace',return_value={'trace_id':'fixture'}):
                replay_control_trace(reopened,frozen,config)
        assert reopened.pending()==()
    recorded=json.loads(rows(campaign)[0]['payload'])
    assert recorded['planner_evidence']==evidence
    assert recorded['input_terminal_watermark']==42
    assert recorded['planner_usage']==payload['planner_usage']


def test_outbox_commit_before_ack_reopens_without_duplicate_row(context):
    store,campaign,config=context
    payload={'evidence_id':'e1','complete_input':[1,2,3]}
    with store.claim(campaign,config.mlflow_tracking_uri):
        with patch.object(store,'acknowledge',side_effect=storage.StorageBlocked('ack capacity')):
            with patch('ima.research_telemetry._deliver_trace') as emit:
                with pytest.raises(storage.StorageBlocked) as failure:
                    log_snapshot(campaign,payload,config,role='execution',number=5,control_store=store)
                emit.assert_not_called()
        assert failure.value.diagnostic['trace_publication_phase']=='control_acknowledge'
        frozen,=store.pending()
    assert len(rows(campaign))==1
    reopened=ControlIntentStore(store.authority)
    with reopened.claim(campaign,config.mlflow_tracking_uri):
        with patch('ima.research_telemetry._deliver_trace',return_value={'trace_id':'fixture'}):
            replay_control_trace(reopened,frozen,config)
        assert reopened.pending()==()
    assert len(rows(campaign))==1 and json.loads(rows(campaign)[0]['payload'])==payload


def test_replay_rejects_reserved_changed_and_wrong_destination_envelopes(context):
    store,campaign,config=context
    with store.claim(campaign,config.mlflow_tracking_uri):
        reserved=store.reserve_decision(1,{'evidence_id':'e1'})
        complete=store.stage('execution',1,{'full':True})
        bad=json.loads(complete);bad['payload']={'full':False}
        with patch('ima.research_telemetry._enqueue_trace') as enqueue:
            for raw in (reserved,json.dumps(bad).encode(),complete.decode()):
                with pytest.raises(ValueError):replay_control_trace(store,raw,config)
            wrong=SimpleNamespace(mlflow_tracking_uri='http://wrong.invalid')
            with pytest.raises(ValueError):replay_control_trace(store,complete,wrong)
            enqueue.assert_not_called()
        assert len(store.pending())==2


def test_delivery_only_pressure_has_no_control_intent_to_restage(context):
    store,campaign,config=context
    identity={'decision_id':'D1','evidence_id':'e1'}
    payload=identity|{'planner_usage':{'total_cost_usd':.03}}
    with store.claim(campaign,config.mlflow_tracking_uri):
        store.reserve_decision(1,identity)
        with patch('ima.research_telemetry._deliver_trace',side_effect=storage.StorageBlocked('delivery capacity')):
            with pytest.raises(storage.StorageBlocked) as failure:
                log_snapshot(campaign,payload,config,role='decision',number=1,control_store=store,
                             decision_evidence_identity=identity)
        assert failure.value.diagnostic['trace_publication_phase']=='outbox_delivery'
        assert store.pending()==()
        with patch('ima.research_telemetry._deliver_trace',return_value={'trace_id':'existing'}):
            assert log_snapshot(campaign,payload,config,role='decision',number=1,
                                control_store=store)['trace_id']=='existing'
    assert len(rows(campaign))==1


def test_new_decision_without_pre_dispatch_reservation_cannot_enqueue(context):
    store,campaign,config=context
    with store.claim(campaign,config.mlflow_tracking_uri):
        with patch('ima.research_telemetry._enqueue_trace') as enqueue:
            with pytest.raises(ValueError,match='Reserve decision'):
                log_snapshot(campaign,{'evidence_id':'e1'},config,role='decision',number=1,
                             control_store=store,decision_evidence_identity={'evidence_id':'e1'})
            enqueue.assert_not_called()
        assert store.pending()==()


def test_wrong_claimed_campaign_is_rejected_before_creating_an_intent(context,tmp_path):
    store,campaign,config=context
    other=tmp_path/'other';other.mkdir()
    with store.claim(campaign,config.mlflow_tracking_uri):
        with pytest.raises(ValueError,match='claimed control'):
            log_snapshot(other,{'full':True},config,role='execution',number=1,control_store=store)
        assert store.pending()==()
    assert not (other/'trace-outbox.sqlite').exists()
