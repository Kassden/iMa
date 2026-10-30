import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ima.research_storage as storage
from ima.mlflow_tracking import _write_json_atomic
from ima.research_telemetry import _enqueue_trace, drain_trace_outbox, log_snapshot, next_trace_number
from tests.test_storage_exports import authority_at


def enabled_authority(tmp_path,monkeypatch,available=100*1024**2):
    authority=authority_at(tmp_path,available)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(authority.root))
    monkeypatch.setattr(storage,'StorageAuthority',lambda root:authority)
    return authority


def test_tracking_json_export_roundtrip_and_admission_denial_preserve_previous(tmp_path,monkeypatch):
    authority=enabled_authority(tmp_path,monkeypatch)
    path=tmp_path/'tracking'/'link.json'
    _write_json_atomic(path,{'tokens':120,'cost_usd':.02})
    original=path.read_bytes()
    assert json.loads(original)['tokens']==120
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('blocked_storage')):
        with pytest.raises(storage.StorageBlocked):
            _write_json_atomic(path,{'tokens':240})
    assert path.read_bytes()==original
    assert list(path.parent.iterdir())==[path]


def test_outbox_estimate_includes_full_planner_evidence_before_database_creation(tmp_path,monkeypatch):
    enabled_authority(tmp_path,monkeypatch,available=32*1024**2)
    campaign=tmp_path/'campaign'
    config=SimpleNamespace(mlflow_tracking_uri='http://tracking.invalid')
    evidence={'decision_id':'d1','evidence_id':'e1','research':'x'*(5*1024**2)}
    with patch('ima.research_telemetry._emit_trace') as emit:
        with pytest.raises(storage.StorageBlocked):
            log_snapshot(campaign,{'decision_id':'d1'},config,role='decision',number=1,
                         planner_evidence=evidence)
        emit.assert_not_called()
    assert not campaign.exists()


def test_storage_blocked_outbox_replay_keeps_pending_row_and_does_not_emit(tmp_path,monkeypatch):
    authority=enabled_authority(tmp_path,monkeypatch)
    campaign=tmp_path/'campaign'
    config=SimpleNamespace(mlflow_tracking_uri='http://tracking.invalid')
    _enqueue_trace(campaign,{'evidence_id':'e1'},config,role='execution',number=1)
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('blocked_storage')):
        with patch('ima.research_telemetry._emit_trace') as emit:
            with pytest.raises(storage.StorageBlocked):
                drain_trace_outbox(campaign,config)
            emit.assert_not_called()
    with sqlite3.connect(campaign/'trace-outbox.sqlite') as conn:
        assert conn.execute('SELECT delivered,attempts FROM traces').fetchone()==(0,0)
    with patch('ima.research_telemetry._outbox',side_effect=AssertionError('write connection')):
        assert next_trace_number(campaign,'execution')==2
    assert authority.status()['reserved_bytes']==0


def test_sequence_read_reserves_before_sqlite_can_create_wal_sidecars(tmp_path,monkeypatch):
    authority=enabled_authority(tmp_path,monkeypatch)
    campaign=tmp_path/'campaign'
    config=SimpleNamespace(mlflow_tracking_uri='http://tracking.invalid')
    _enqueue_trace(campaign,{'evidence_id':'e1'},config,role='execution',number=1)
    before={path.name:path.stat().st_size for path in campaign.iterdir()}
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('blocked_storage')):
        with patch('ima.research_telemetry.sqlite3.connect') as connect:
            with pytest.raises(storage.StorageBlocked):next_trace_number(campaign,'execution')
            connect.assert_not_called()
    assert before=={path.name:path.stat().st_size for path in campaign.iterdir()}
    with patch.object(authority,'reserve',wraps=authority.reserve) as reserve:
        assert next_trace_number(campaign,'execution')==2
        assert reserve.call_args.kwargs['stage']=='tracking_outbox_read'


def test_tracking_delivery_with_admission_preserves_existing_cost_and_replay_contract(tmp_path,monkeypatch):
    from tests.test_research_telemetry import SnapshotCostTests
    enabled_authority(tmp_path,monkeypatch)
    case=SnapshotCostTests('test_paid_decision_native_cost_is_written_once_and_link_reused')
    case.setUp()
    case.campaign=case.campaign.resolve()
    try:
        case.test_paid_decision_native_cost_is_written_once_and_link_reused()
    finally:
        case.doCleanups()


def test_coverage_report_admission_preserves_previous_report(tmp_path,monkeypatch):
    from scripts.audit_official_coverage import publish_report
    authority=enabled_authority(tmp_path,monkeypatch)
    output=tmp_path/'coverage'/'report.json'
    publish_report(output,{'coverage':[{'family':'trackwork','complete':False}]})
    original=output.read_bytes()
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('blocked_storage')):
        with pytest.raises(storage.StorageBlocked):
            publish_report(output,{'coverage':[]})
    assert output.read_bytes()==original
    assert list(output.parent.iterdir())==[output]


def test_delivery_estimate_supports_outboxes_before_trace_json_migration(tmp_path,monkeypatch):
    from ima.research_telemetry import _delivery_bytes
    enabled_authority(tmp_path,monkeypatch)
    campaign=tmp_path/'legacy';campaign.mkdir()
    with sqlite3.connect(campaign/'trace-outbox.sqlite') as conn:
        conn.execute('CREATE TABLE traces(role TEXT,number INTEGER,payload TEXT)')
        conn.execute('INSERT INTO traces VALUES(?,?,?)',('execution',1,'{"old":true}'))
    assert _delivery_bytes({'campaign':campaign,'role':'execution','number':1})>16*1024**2
    with sqlite3.connect(campaign/'trace-outbox.sqlite') as conn:
        assert 'trace_json' not in {row[1] for row in conn.execute('PRAGMA table_info(traces)')}


def test_controller_pending_count_is_unknown_not_zero_when_read_is_blocked(tmp_path,monkeypatch):
    from ima.research_expansion import _pending_trace_count
    authority=enabled_authority(tmp_path,monkeypatch)
    campaign=tmp_path/'campaign'
    config=SimpleNamespace(mlflow_tracking_uri='http://tracking.invalid')
    _enqueue_trace(campaign,{'evidence_id':'e1'},config,role='execution',number=1)
    assert _pending_trace_count(campaign)==1
    with patch.object(authority,'reserve',side_effect=storage.StorageBlocked('blocked_storage')):
        assert _pending_trace_count(campaign) is None
    assert _pending_trace_count(campaign)==1


def test_summary_exposes_control_pending_and_unknown_delivery_counts():
    from ima.research_summary import summary,summary_markdown
    report=summary({'trace_role':'execution','status':'blocked_storage','trace_storage_blocked':True,
                    'pending_trace_delivery':None,'pending_trace_enqueues':1,
                    'pending_trace_control_reservations':1})
    assert report['execution']['pending_trace_delivery'] is None
    assert report['execution']['pending_trace_enqueues']==1
    assert report['execution']['pending_trace_control_reservations']==1
    assert report['health']=='degraded'
    assert 'Unresolved decision reservation' in summary_markdown(report)
