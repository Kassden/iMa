import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from ima.feature_definitions import FeatureDefinition
from ima.feature_exploration import ExplorationQueue, exploration_lock
from ima import research_sqlite, research_storage


def _definition():
    return FeatureDefinition(name="bounded_ratio", unit="1", input_refs=("a", "b"),
        expression_ast={"op": "divide", "args": [{"op": "ref", "ref": "a"},
                                                   {"op": "ref", "ref": "b"}]})


@pytest.fixture
def enabled_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path/"control"))
    real_authority = research_storage.StorageAuthority

    class SmallHeadroomAuthority(real_authority):
        def __init__(self, root, **kwargs):
            kwargs.setdefault("headroom_bytes", 0)
            kwargs.setdefault("headroom_inodes", 0)
            super().__init__(root, **kwargs)

    monkeypatch.setattr(research_storage, "StorageAuthority", SmallHeadroomAuthority)
    return tmp_path


def _sidecars(database):
    return {path.name: (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in database.parent.glob(database.name+"*")}


def test_capacity_denial_precedes_queue_parent_and_database_creation(tmp_path, monkeypatch):
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path/"control"))

    class DeniedAuthority:
        def __init__(self, root):
            pass

        @contextmanager
        def admitted(self, **kwargs):
            raise research_storage.StorageBlocked("capacity denied")
            yield

    monkeypatch.setattr(research_storage, "StorageAuthority", DeniedAuthority)
    database = tmp_path/"not-created"/"nested"/"queue.sqlite"
    with pytest.raises(research_storage.StorageBlocked):
        ExplorationQueue(database)
    assert not database.parent.exists()
    assert not database.exists()


def test_dynamic_utf8_envelope_large_payload_reopen_and_read_without_admission(
        enabled_storage, monkeypatch):
    database = enabled_storage/"queue.sqlite"
    queue = ExplorationQueue(database)
    identifier = queue.enqueue([_definition()])
    payload = "雪"*500_000
    queue.update(identifier, "screened", {"payload": payload})
    reopened = ExplorationQueue(database)
    before_read = _sidecars(database)

    def denied_read_admission(*args, **kwargs):
        raise AssertionError("read path attempted storage admission")

    monkeypatch.setattr(research_sqlite, "admitted_sqlite", denied_read_admission)
    assert reopened.rows()[0]["result"]["payload"] == payload
    assert _sidecars(database) == before_read
    assert not Path(str(database)+"-wal").exists()
    assert not Path(str(database)+"-shm").exists()


def test_native_full_rolls_back_queue_update(enabled_storage, monkeypatch):
    database = enabled_storage/"queue.sqlite"
    queue = ExplorationQueue(database)
    identifier = queue.enqueue([_definition()])
    real_admitted = research_sqlite.admitted_sqlite

    def no_growth(path, *, estimated_growth_bytes, stage, row_factory=None):
        return real_admitted(path, estimated_growth_bytes=0, stage=stage,
                             row_factory=row_factory)

    monkeypatch.setattr(research_sqlite, "admitted_sqlite", no_growth)
    with pytest.raises(research_storage.StorageBlocked) as failure:
        queue.update(identifier, "screened", {"payload": "x"*500_000})
    assert failure.value.diagnostic['sqlite_errorcode'] & 255 == sqlite3.SQLITE_FULL
    monkeypatch.setattr(research_sqlite, "admitted_sqlite", real_admitted)
    row = ExplorationQueue(database).rows()[0]
    assert row["status"] == "queued"
    assert row["result"] == {}


def test_admitted_queue_preserves_existing_wal_without_migration(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    database = tmp_path/"queue.sqlite"
    queue = ExplorationQueue(database)
    queue.enqueue([_definition()])
    before = _sidecars(database)
    with database.open("rb") as stream:
        stream.seek(18)
        assert stream.read(2) == b"\x02\x02"

    monkeypatch.setenv("IMA_STORAGE_ROOT", str(tmp_path/"control"))
    with pytest.raises(RuntimeError, match="WAL"):
        ExplorationQueue(database)
    assert _sidecars(database) == before


def test_disabled_mode_keeps_legacy_wal_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    database = tmp_path/"legacy"/"queue.sqlite"
    queue = ExplorationQueue(database)
    identifier = queue.enqueue([_definition()])
    assert queue.rows()[0]["id"] == identifier
    with database.open("rb") as stream:
        stream.seek(18)
        assert stream.read(2) == b"\x02\x02"


def test_live_queue_reader_honors_sqlite_exclusive_writer_lock(enabled_storage, monkeypatch):
    from ima import feature_exploration
    database = enabled_storage/'queue.sqlite'
    queue = ExplorationQueue(database)
    queue.enqueue([_definition()])
    real_connect = sqlite3.connect
    writer = real_connect(database)
    writer.execute('BEGIN EXCLUSIVE')

    def quick_read(*args, **kwargs):
        kwargs['timeout'] = .01
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(feature_exploration.sqlite3, 'connect', quick_read)
    try:
        with pytest.raises(sqlite3.OperationalError) as failure:
            queue.rows()
        assert failure.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
    finally:
        writer.rollback()
        writer.close()


def test_loop_lock_admission_ends_before_body_and_contention_is_reported(
        enabled_storage, monkeypatch):
    real_admitted_output = research_storage.admitted_output
    reservations = []

    @contextmanager
    def tracked(*args, **kwargs):
        with real_admitted_output(*args, **kwargs):
            reservations.append(True)
            try:
                yield
            finally:
                reservations.pop()

    monkeypatch.setattr(research_storage, "admitted_output", tracked)
    root = enabled_storage/"campaign"
    with exploration_lock(root):
        assert not reservations
        with pytest.raises(RuntimeError, match="Another feature exploration"):
            with exploration_lock(root):
                pass
    assert not reservations
