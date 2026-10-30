import hashlib
import errno
import os
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from ima import research_storage
from ima.research_hypotheses import HypothesisMemory
from ima.research_sqlite import admitted_sqlite
from ima.research_storage import StorageBlocked, StorageAuthority


def enable_storage(monkeypatch, *, available_bytes=None):
    original = os.statvfs

    class TestAuthority(StorageAuthority):
        def __init__(self, root, **kwargs):
            kwargs.setdefault("headroom_bytes", 0)
            kwargs.setdefault("headroom_inodes", 0)
            if available_bytes is not None:
                def bounded_statvfs(path):
                    value = original(path)
                    return SimpleNamespace(f_bavail=available_bytes // value.f_frsize,
                                           f_favail=value.f_favail, f_frsize=value.f_frsize,
                                           f_blocks=value.f_blocks)
                kwargs["statvfs"] = bounded_statvfs
            super().__init__(root, **kwargs)

    monkeypatch.setattr(research_storage, "StorageAuthority", TestAuthority)


def test_denial_precedes_parent_creation(tmp_path, monkeypatch):
    control = tmp_path / "control"
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(control))
    enable_storage(monkeypatch, available_bytes=0)
    output = tmp_path / "not-created" / "hypotheses.sqlite"
    with pytest.raises(StorageBlocked):
        with admitted_sqlite(output, estimated_growth_bytes=1024):
            pass
    assert not output.parent.exists()


def test_hypothesis_records_are_durable_idempotent_and_conflicts_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    memory = HypothesisMemory(tmp_path / "nested" / "hypotheses.sqlite")
    memory.record("event-1", "hyp-1", "screen", {"score": 0.7})
    memory.record("event-1", "hyp-1", "screen", {"score": 0.7})
    rows = memory.retrieve("score", 10)
    assert len(rows) == 1
    assert rows[0]["hypothesis_id"] == "hyp-1"
    assert rows[0]["payload"] == {"score": 0.7}
    with pytest.raises(ValueError, match="Conflicting"):
        memory.record("event-1", "hyp-1", "screen", {"score": 0.8})


def test_hypothesis_memory_growth_envelope_handles_many_splits_and_long_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "many-records.sqlite"
    memory = HypothesisMemory(db)
    ids = []
    for index in range(600):
        event_id = f"event-{index:04d}-" + ("e" * 240)
        hypothesis_id = f"hypothesis-{index:04d}-" + ("h" * 180)
        ids.append(event_id)
        memory.record(event_id, hypothesis_id, "small-event", {
            "marker": "split-check", "sequence": index, "long_event_id": event_id,
        })
    rows = memory.retrieve("split-check", 100)
    assert len(rows) == 100
    assert rows[0]["payload"]["sequence"] == 599
    assert rows[0]["payload"]["long_event_id"] == ids[-1]
    assert rows[-1]["payload"]["sequence"] == 500
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 600


def test_sqlite_full_rolls_back_and_is_classified(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "events.sqlite"
    with admitted_sqlite(db, estimated_growth_bytes=64 * 1024) as conn:
        conn.execute("CREATE TABLE events (value TEXT)")
        conn.execute("INSERT INTO events VALUES (?)", ("preserved",))
    with pytest.raises(StorageBlocked) as error:
        with admitted_sqlite(db, estimated_growth_bytes=0) as conn:
            conn.execute("INSERT INTO events VALUES (?)", ("x" * 100_000,))
    assert error.value.diagnostic["failure_kind"] == "infrastructure_storage"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("preserved",)]


def test_integrity_errors_remain_ordinary_with_admission_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "integrity.sqlite"
    with admitted_sqlite(db, estimated_growth_bytes=16 * 1024) as conn:
        conn.execute("CREATE TABLE events (value TEXT UNIQUE)")
        conn.execute("INSERT INTO events VALUES ('same')")
    with pytest.raises(sqlite3.IntegrityError):
        with admitted_sqlite(db, estimated_growth_bytes=4096) as conn:
            conn.execute("INSERT INTO events VALUES ('same')")


def test_primary_sqlite_full_survives_rollback_and_close_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "cleanup-fault.sqlite"
    with admitted_sqlite(db, estimated_growth_bytes=64 * 1024) as conn:
        conn.execute("CREATE TABLE events (value TEXT)")
        conn.execute("INSERT INTO events VALUES (?)", ("preserved",))

    real_connect = sqlite3.connect

    class CleanupFaultConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def rollback(self):
            raise OSError(errno.ENOSPC, "injected rollback cleanup failure")

        @property
        def in_transaction(self):
            return True

        def close(self):
            self.connection.close()
            raise OSError(errno.ENOSPC, "injected close cleanup failure")

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        if kwargs.get("uri"):
            return connection
        return CleanupFaultConnection(connection)

    monkeypatch.setattr("ima.research_sqlite.sqlite3.connect", connect)
    with pytest.raises(StorageBlocked) as error:
        with admitted_sqlite(db, estimated_growth_bytes=0) as conn:
            conn.execute("INSERT INTO events VALUES (?)", ("x" * 100_000,))
    primary = error.value.__cause__
    assert isinstance(primary, sqlite3.Error)
    assert primary.sqlite_errorcode & 255 == sqlite3.SQLITE_FULL
    assert any("rollback cleanup also failed" in note for note in primary.__notes__)
    assert any("close cleanup also failed" in note for note in primary.__notes__)
    monkeypatch.setattr("ima.research_sqlite.sqlite3.connect", real_connect)
    with real_connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("preserved",)]


def test_concurrent_writers_cannot_overbook_last_shared_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch, available_bytes=1_500_000)
    entered = threading.Event()
    release = threading.Event()
    first_error = []

    def first_writer():
        try:
            with admitted_sqlite(tmp_path / "first.sqlite", estimated_growth_bytes=900_000):
                entered.set()
                release.wait(5)
        except BaseException as exc:  # surfaced in the main test thread
            first_error.append(exc)

    worker = threading.Thread(target=first_writer)
    worker.start()
    assert entered.wait(5)
    try:
        with pytest.raises(StorageBlocked):
            with admitted_sqlite(tmp_path / "second.sqlite", estimated_growth_bytes=900_000):
                pass
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert not first_error


def test_wal_refusal_does_not_modify_main_or_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "wal.sqlite"
    source = sqlite3.connect(db)
    assert source.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    source.execute("PRAGMA wal_autocheckpoint=0")
    source.execute("CREATE TABLE events (value TEXT)")
    source.execute("INSERT INTO events VALUES ('kept in wal')")
    source.commit()
    wal = tmp_path / "wal.sqlite-wal"
    assert wal.exists()
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (db, wal)}
    try:
        with pytest.raises(RuntimeError, match="WAL mode is unsupported"):
            with admitted_sqlite(db, estimated_growth_bytes=4096):
                pass
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (db, wal)}
        assert after == before
    finally:
        source.close()


def test_hot_rollback_journal_is_preserved_without_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "hot.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE events (value TEXT)")
    journal = tmp_path / "hot.sqlite-journal"
    journal.write_bytes(b"unrecognized journal header")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    journal_before = journal.read_bytes()
    with pytest.raises(RuntimeError, match="preserving database"):
        with admitted_sqlite(db, estimated_growth_bytes=4096):
            pass
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert journal.read_bytes() == journal_before


def test_transaction_surface_blocks_escape_and_page_cap_changes(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    with admitted_sqlite(tmp_path / "restricted.sqlite", estimated_growth_bytes=16 * 1024) as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("PRAGMA max_page_count=1000000")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("ATTACH DATABASE ':memory:' AS escaped")
        with pytest.raises(AttributeError):
            _ = conn.execute("SELECT 1").connection


@pytest.mark.parametrize("mode", ["DELETE", "TRUNCATE", "PERSIST"])
def test_supported_rollback_journal_modes(tmp_path, monkeypatch, mode):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    db = tmp_path / f"{mode.lower()}.sqlite"
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"PRAGMA journal_mode={mode}").fetchone()[0].upper() == mode
        conn.execute("CREATE TABLE events (value TEXT)")
    with admitted_sqlite(db, estimated_growth_bytes=8192) as conn:
        conn.execute("INSERT INTO events VALUES (?)", (mode,))
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT value FROM events").fetchone() == (mode,)


def test_rollback_and_close_release_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path / "control"))
    enable_storage(monkeypatch)
    db = tmp_path / "rollback.sqlite"
    with pytest.raises(ValueError, match="abort"):
        with admitted_sqlite(db, estimated_growth_bytes=32 * 1024) as conn:
            conn.execute("CREATE TABLE transient (value TEXT)")
            raise ValueError("abort")
    assert not (tmp_path / "rollback.sqlite-journal").exists()
    authority = StorageAuthority(tmp_path / "control", headroom_bytes=0, headroom_inodes=0)
    assert authority.status()["reservations"] == {}
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='transient'").fetchone() is None


def test_storage_disabled_preserves_hypothesis_memory_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    memory = HypothesisMemory(tmp_path / "created" / "events.sqlite")
    memory.record("e", "h", "kind", {"nested": [1, 2]})
    assert memory.retrieve("nested")[0]["payload"] == {"nested": [1, 2]}
    assert not (tmp_path / "created" / "events.sqlite.admission-lock").exists()


def test_storage_disabled_preserves_legacy_symlink_parent(tmp_path,monkeypatch):
    monkeypatch.delenv('IMA_STORAGE_ROOT',raising=False)
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    memory=HypothesisMemory(alias/'events.sqlite')
    memory.record('legacy-event','legacy-hypothesis','kind',{'source':'alias'})
    assert memory.retrieve('alias')[0]['payload']=={'source':'alias'}
    assert (real/'events.sqlite').is_file()
    assert not (real/'events.sqlite.admission-lock').exists()
