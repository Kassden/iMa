import sqlite3

import pytest

from ima import research_storage
from ima.research_sqlite import admitted_sqlite
from ima.research_store import ResearchLedger
from ima.research_storage import StorageAuthority, StorageBlocked


def test_deferred_logical_allocation_and_history_survive_reopening(tmp_path,monkeypatch):
    available=[256*1024**2]
    _bounded_authority(monkeypatch,tmp_path/'control',available)
    database=tmp_path/'ledger.sqlite';ledger=ResearchLedger(database)
    original={'trial_number':4,'recipe':{'seed':42}}
    attempt=ledger.reserve_attempt('same-logical-trial',original)
    ledger.mark_running(attempt.attempt_id)
    runtime={'status':'failed','task_nonce':'retained-episode'}
    receipt=ledger.defer_infrastructure(attempt.attempt_id,StorageBlocked('capacity').diagnostic,
        now=100,delay_seconds=30,runtime_snapshot=runtime)
    reopened=ResearchLedger(database)
    assert receipt['retry_count']==1 and not reopened.retry_ready(attempt.attempt_id,now=129)
    assert reopened.retry_ready(attempt.attempt_id,now=131)
    assert reopened.pending_attempts()[0]['payload']==original
    assert reopened.terminal_results()==[]
    assert reopened.latest_infrastructure_runtime(attempt.attempt_id)==runtime
    reopened.mark_running(attempt.attempt_id)
    assert reopened.infrastructure_waits()==[]
    assert len(reopened.infrastructure_history(attempt.attempt_id))==1


def test_nontrial_retry_is_durable_without_model_allocations(tmp_path,monkeypatch):
    available=[256*1024**2]
    _bounded_authority(monkeypatch,tmp_path/'control',available)
    database=tmp_path/'ledger.sqlite';ledger=ResearchLedger(database)
    ledger.defer_operation('fold:shared-matrix',StorageBlocked('capacity').diagnostic,
        now=100,delay_seconds=30,runtime_snapshot={'phase':'preparation'})
    reopened=ResearchLedger(database)
    assert reopened.reserved_attempts()==[] and reopened.terminal_results()==[]
    assert len(reopened.operation_waits())==1
    reopened.complete_operation('fold:shared-matrix')
    assert reopened.operation_waits()==[]


def test_mark_running_cannot_clear_a_live_retry_deadline(tmp_path,monkeypatch):
    available=[256*1024**2]
    _bounded_authority(monkeypatch,tmp_path/'control',available)
    ledger=ResearchLedger(tmp_path/'ledger.sqlite')
    attempt=ledger.reserve_attempt('waiting',{'seed':42})
    ledger.mark_running(attempt.attempt_id,now=90)
    ledger.defer_infrastructure(attempt.attempt_id,StorageBlocked('capacity').diagnostic,
        now=100,delay_seconds=30)
    with pytest.raises(StorageBlocked) as error:ledger.mark_running(attempt.attempt_id,now=129)
    assert error.value.diagnostic['retry_at_epoch']==130
    assert ledger.pending_attempts()[0]['status']=='reserved'
    assert len(ledger.infrastructure_waits())==1 and ledger.terminal_results()==[]
    assert ledger.mark_running(attempt.attempt_id,now=131).status=='running'
    assert ledger.infrastructure_waits()==[]


def test_enabled_wal_readback_refuses_before_opening_sqlite(tmp_path,monkeypatch):
    import hashlib
    database=tmp_path/'ledger.sqlite'
    writer=sqlite3.connect(database)
    try:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE attempts (status TEXT,uploaded_at TEXT,told_at TEXT)')
        writer.execute("INSERT INTO attempts VALUES ('completed',NULL,NULL)");writer.commit()
        files=[database, database.with_name(database.name+'-wal'),database.with_name(database.name+'-shm')]
        before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        monkeypatch.setenv('IMA_STORAGE_ROOT',str(tmp_path/'unused-control'))
        def must_not_open(*args,**kwargs):raise AssertionError('WAL must be refused before SQLite opens')
        monkeypatch.setattr(sqlite3,'connect',must_not_open)
        with pytest.raises(RuntimeError,match='WAL ledger readback'):
            ResearchLedger(database,read_only=True)
        assert {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}==before
        assert not (tmp_path/'unused-control').exists()
    finally:writer.close()


def _bounded_authority(monkeypatch, root, available):
    real_statvfs = research_storage.os.statvfs

    def statvfs(path):
        current = real_statvfs(path)
        return type("Stat", (), {
            "f_bavail": available[0] // current.f_frsize,
            "f_favail": current.f_favail,
            "f_frsize": current.f_frsize,
            "f_blocks": current.f_blocks,
        })()

    class TestAuthority(StorageAuthority):
        def __init__(self, path):
            super().__init__(path, headroom_bytes=0, headroom_inodes=0, statvfs=statvfs)

    monkeypatch.setattr(research_storage, "StorageAuthority", TestAuthority)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(root))


def test_ledger_init_admission_denial_precedes_database_parent_creation(tmp_path, monkeypatch):
    available = [0]
    _bounded_authority(monkeypatch, tmp_path / "control", available)
    database = tmp_path / "not-created" / "nested" / "ledger.sqlite"

    with pytest.raises(StorageBlocked):
        ResearchLedger(database)

    assert not database.parent.exists()


def test_large_reserve_and_completion_payloads_round_trip_with_admission(tmp_path, monkeypatch):
    available = [2 * 1024**3]
    _bounded_authority(monkeypatch, tmp_path / "control", available)
    ledger = ResearchLedger(tmp_path / "ledger.sqlite")
    payload = {"serialized": "p" * (3 * 1024**2)}
    result = {"serialized_result": "r" * (4 * 1024**2)}
    signature = "s" * (256 * 1024)

    attempt = ledger.reserve_attempt(signature, payload)
    completed = ledger.complete_attempt(attempt.attempt_id, result)

    assert completed.status == "completed"
    assert completed.signature == signature
    assert ledger.reserved_attempts()[0]["payload"] == payload
    assert ledger.terminal_results()[0]["result"] == result
    assert StorageAuthority(tmp_path / "control", headroom_bytes=0,
                            headroom_inodes=0).status()["reservations"] == {}


def test_native_admission_denial_preserves_existing_attempt_status(tmp_path, monkeypatch):
    available = [256 * 1024**2]
    _bounded_authority(monkeypatch, tmp_path / "control", available)
    ledger = ResearchLedger(tmp_path / "ledger.sqlite")
    attempt = ledger.reserve_attempt("kept", {"x": 1})
    ledger.mark_running(attempt.attempt_id)
    available[0] = 0

    with pytest.raises(StorageBlocked):
        ledger.complete_attempt(attempt.attempt_id, {"result": "must not be stored"})

    assert ledger.reserved_attempts()[0]["status"] == "running"


def test_read_methods_remain_available_under_admission_pressure_without_sidecars(tmp_path, monkeypatch):
    available = [256 * 1024**2]
    _bounded_authority(monkeypatch, tmp_path / "control", available)
    ledger = ResearchLedger(tmp_path / "ledger.sqlite")
    attempt = ledger.reserve_attempt("read", {"x": 1})
    before = {path.name for path in tmp_path.iterdir()}
    available[0] = 0

    assert ledger.snapshot()["reserved"] == 1
    assert ledger.reserved_attempts()[0]["attempt_id"] == attempt.attempt_id
    assert ledger.terminal_results() == []
    assert ledger.pending_outbox() == []
    assert ledger.pending_tells() == []
    after = {path.name for path in tmp_path.iterdir()}
    assert after == before
    assert StorageAuthority(tmp_path / "control", headroom_bytes=0,
                            headroom_inodes=0).status()["reservations"] == {}


def test_read_only_ledger_reads_core_schema_without_migrating_or_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    database = tmp_path / "core.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("""CREATE TABLE attempts (
            attempt_id TEXT PRIMARY KEY, signature TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL, payload_json TEXT NOT NULL, result_json TEXT,
            remote_id TEXT, uploaded_at TEXT, told_at TEXT, updated_at TEXT NOT NULL
        )""")
        conn.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("attempt-core", "core", "reserved", '{"n":1}', None, None, None, None, "now"))

    ledger = ResearchLedger(database, read_only=True)

    assert ledger.reserved_attempts() == [{
        "attempt_id": "attempt-core", "status": "reserved", "payload": {"n": 1},
    }]
    assert ledger.snapshot()["reserved"] == 1
    with pytest.raises(PermissionError, match="read-only"):
        ledger.reserve_attempt("new", {})
    with sqlite3.connect(database) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"attempts"}


def test_read_only_ledger_rejects_schema_that_requires_migration(tmp_path):
    database = tmp_path / "incomplete.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE attempts (attempt_id TEXT, signature TEXT, status TEXT, payload_json TEXT)")

    with pytest.raises(RuntimeError, match="cannot read snapshot"):
        ResearchLedger(database, read_only=True)

    with sqlite3.connect(database) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "told_at" not in columns
    assert tables == {"attempts"}


def test_read_only_snapshot_accepts_legacy_core_columns_without_retry_tables(tmp_path):
    database = tmp_path / "legacy-core.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE attempts (status TEXT, uploaded_at TEXT, told_at TEXT)")
        conn.execute("INSERT INTO attempts VALUES ('reserved', NULL, NULL)")

    ledger = ResearchLedger(database, read_only=True)

    assert ledger.snapshot() == {"reserved": 1, "pending_tracking": 0, "pending_tells": 0}
    with sqlite3.connect(database) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"attempts"}


@pytest.mark.parametrize("enabled", [False, True])
def test_admitted_helper_row_factory_and_read_only_table_info(enabled, tmp_path, monkeypatch):
    if enabled:
        available = [256 * 1024**2]
        _bounded_authority(monkeypatch, tmp_path / "control", available)
    else:
        monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    database = tmp_path / "helper.sqlite"
    with admitted_sqlite(database, estimated_growth_bytes=8192, row_factory=sqlite3.Row) as conn:
        conn.execute("CREATE TABLE sample (value TEXT)")
        assert conn.execute("PRAGMA table_info(sample)").fetchone()["name"] == "value"
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("PRAGMA journal_mode=WAL")
