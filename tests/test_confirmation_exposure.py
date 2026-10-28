import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch
from ima.research_confirmation import ExposureLedger
from ima import research_storage
from ima.research_storage import StorageBlocked
from scripts.research_confirmation import main


def test_known_exposure_is_durable_idempotent_and_cross_target(tmp_path):
    evidence=tmp_path/'research.json'; evidence.write_text('{"target":"placing_top_k"}')
    ledger=ExposureLedger(tmp_path/'exposure.sqlite')
    kwargs=dict(first_date='2026-09-01',last_date='2026-10-19',campaign='manual',
                purpose='label_inspection',evidence=evidence)
    assert ledger.record(**kwargs)==ledger.record(**kwargs)
    assert len(ExposureLedger(ledger.path).overlapping(['2026-10-07']))==1
    assert not ledger.overlapping(['2026-10-20'])
    with pytest.raises(ValueError): ledger.record(**(kwargs|{'last_date':'2026-08-01'}))


def test_exposure_cli_receipt_reads_back(tmp_path,capsys):
    evidence=tmp_path/'evidence.json'; evidence.write_text('{}')
    path=tmp_path/'exposure.sqlite'
    assert main(['exposure','--ledger',str(path),'--first-date','2026-10-07',
        '--last-date','2026-10-07','--campaign','manual','--purpose','analysis',
        '--evidence',str(evidence)])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['receipt_id'] and receipt['ledger']==str(path)
    assert len(ExposureLedger(path).overlapping(['2026-10-07']))==1


def test_missing_overlap_read_does_not_create_ledger_or_parent(tmp_path):
    path=tmp_path/'not-created'/'exposure.sqlite'
    assert ExposureLedger(path).overlapping(['2026-10-07'])==[]
    assert not path.parent.exists()


def test_existing_overlap_read_is_readonly_and_does_not_create_directories(tmp_path):
    evidence=tmp_path/'evidence.json'; evidence.write_text('{}')
    path=tmp_path/'existing'/'exposure.sqlite'
    ledger=ExposureLedger(path)
    ledger.record(first_date='2026-10-01',last_date='2026-10-10',campaign='manual',
                  purpose='analysis',evidence=evidence)
    original=path.read_bytes()
    statements=[]
    real_connect=sqlite3.connect

    def connect_readonly(database,*args,**kwargs):
        assert kwargs.get('uri') is True
        assert database.endswith('?mode=ro')
        connection=real_connect(database,*args,**kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    with patch('ima.research_confirmation.sqlite3.connect',side_effect=connect_readonly):
        with patch.object(Path,'mkdir',side_effect=AssertionError('read attempted mkdir')):
            assert len(ledger.overlapping(['2026-10-07']))==1
    assert statements and all(statement.lstrip().upper().startswith('SELECT')
                              for statement in statements)
    assert path.read_bytes()==original
    assert not Path(str(path)+'-journal').exists()


def test_exposure_storage_denial_precedes_creation_and_retry_is_idempotent(
        tmp_path,monkeypatch):
    evidence=tmp_path/'evidence.json'
    evidence.write_text('{"unchanged":true}')
    original_evidence=evidence.read_bytes()
    ledger_path=tmp_path/'nested'/'exposure.sqlite'
    control=tmp_path/'control'
    real_authority=research_storage.StorageAuthority
    real_authority(control,headroom_bytes=0,headroom_inodes=0)
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(control))
    monkeypatch.setattr(research_storage,'StorageAuthority',
        lambda root:real_authority(root,headroom_bytes=10**15,headroom_inodes=0))
    ledger=ExposureLedger(ledger_path)
    kwargs=dict(first_date='2026-09-01',last_date='2026-10-19',campaign='manual',
                purpose='label_inspection',evidence=evidence)
    with pytest.raises(StorageBlocked):
        ledger.record(**kwargs)
    assert not ledger_path.parent.exists()
    assert evidence.read_bytes()==original_evidence

    monkeypatch.setattr(research_storage,'StorageAuthority',
        lambda root:real_authority(root,headroom_bytes=0,headroom_inodes=0))
    receipt=ledger.record(**kwargs)
    assert ledger.record(**kwargs)==receipt
    assert len(ledger.overlapping(['2026-10-07']))==1
    assert evidence.read_bytes()==original_evidence
