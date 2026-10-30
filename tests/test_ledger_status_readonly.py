"""The status consumer must remain read-only even when normal admission is unavailable."""
import hashlib
import sqlite3

from ima.research_controller import campaign_status
import ima.research_storage as storage


def test_status_reads_legacy_core_schema_without_authority_or_migration(tmp_path,monkeypatch):
    campaign=tmp_path/'campaign';campaign.mkdir();database=campaign/'ledger.sqlite'
    with sqlite3.connect(database) as conn:
        conn.execute('CREATE TABLE attempts (status TEXT,uploaded_at TEXT,told_at TEXT)')
        conn.execute("INSERT INTO attempts VALUES ('completed',NULL,NULL)")
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    control=tmp_path/'control-not-created'
    monkeypatch.setenv('IMA_STORAGE_ROOT',str(control))
    def denied(*args,**kwargs):raise AssertionError('A status read must not initialize storage authority')
    monkeypatch.setattr(storage,'StorageAuthority',denied)
    status=campaign_status(campaign)
    assert status['ledger']=={'completed':1,'pending_tracking':1,'pending_tells':1}
    assert status['controller_online'] is False and status['controller_lock'] is None
    assert not control.exists() and not (campaign/'controller.lock').exists()
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[('attempts',)]
