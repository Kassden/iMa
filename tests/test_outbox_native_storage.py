import sqlite3
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ima.research_storage as storage
import ima.research_sqlite as research_sqlite
from ima.research_telemetry import _enqueue_trace, _outbox, _read_outbox, _deliver_trace
from ima.research_sqlite import native_sqlite_connection


def authority_at(tmp_path,available):
    return storage.StorageAuthority(tmp_path/'control',headroom_bytes=0,headroom_inodes=0,
        statvfs=lambda _:SimpleNamespace(f_frsize=1,f_blocks=available,f_bavail=available,
            f_bfree=available,f_favail=100000))


def enabled(tmp_path, monkeypatch, available=64 * 1024**2):
    authority = authority_at(tmp_path, available)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(storage, "StorageAuthority", lambda root: authority)
    return authority


def test_admission_denial_precedes_campaign_mkdir_and_native_connect(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch, available=10_000)
    campaign = tmp_path / "absent" / "campaign"
    with patch.object(authority, "reserve", side_effect=storage.StorageBlocked("denied")):
        with patch("ima.research_sqlite.sqlite3.connect") as connect:
            with pytest.raises(storage.StorageBlocked):
                _outbox(campaign)
            connect.assert_not_called()
    assert not campaign.exists()


def test_connection_holds_exact_lease_until_confirmed_close_and_preserves_budget(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    before = storage._OUTPUT_BUDGET.get()
    expected_growth = 256 * 1024
    conn = _outbox(campaign, estimated_growth_bytes=expected_growth)
    assert storage._OUTPUT_BUDGET.get() is before
    with (campaign / "trace-outbox.sqlite").open("rb") as stream:
        stream.seek(18)
        assert stream.read(2) == b"\x01\x01"
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("PRAGMA max_page_count=999999")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("ATTACH DATABASE ':memory:' AS escaped")
    held = authority.status()["reserved_bytes"]
    rounded_growth = research_sqlite._rounded_growth_bytes(expected_growth)
    assert held == (research_sqlite._journal_peak_bytes(
        (campaign / "trace-outbox.sqlite").stat().st_size + rounded_growth) +
        rounded_growth + research_sqlite._METADATA_BYTES)
    with pytest.raises(RuntimeError, match="bounded execute"):
        conn.cursor()
    with pytest.raises(RuntimeError, match="executescript"):
        conn.executescript("CREATE TABLE bypass(value)")
    conn.close()
    assert authority.status()["reserved_bytes"] == 0


def test_later_transaction_extends_journal_peak_after_concurrent_database_growth(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    first = _outbox(campaign, estimated_growth_bytes=128 * 1024)
    first_lease = authority.status()["reserved_bytes"]
    second = _outbox(campaign, estimated_growth_bytes=2 * 1024**2)
    second.execute("BEGIN IMMEDIATE")
    second.execute("INSERT INTO traces(role,number,payload,tracking_uri) VALUES(?,?,?,?)",
                   ("execution", 1, "x" * (1024**2), "http://tracking.invalid"))
    second.commit()
    second.close()
    assert (campaign / "trace-outbox.sqlite").stat().st_size > 8192

    with patch.object(authority, "extend", wraps=authority.extend) as extend:
        first.execute("UPDATE traces SET attempts=attempts+1 WHERE role='execution' AND number=1")
        extend.assert_called_once()
    refreshed_lease = authority.status()["reserved_bytes"]
    assert refreshed_lease > first_lease
    assert first.in_transaction
    first.commit()
    first.close()
    assert authority.status()["reserved_bytes"] == 0


def test_delivery_identity_commit_releases_sql_transaction_before_network_and_marks_later(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    config = SimpleNamespace(mlflow_tracking_uri="http://tracking.invalid")
    _enqueue_trace(campaign, {"evidence_id": "e1"}, config, role="execution", number=1)
    connections = []
    original_outbox = _outbox

    def capture_outbox(*args, **kwargs):
        conn = original_outbox(*args, **kwargs)
        connections.append(conn)
        return conn

    class Mlflow:
        def set_tracking_uri(self, uri):
            pass

        def flush_trace_async_logging(self):
            assert connections[-1].in_transaction is False

    def emit(campaign, payload, config, *, role, number, persist):
        receipt = {"trace_id": "trace-1", "experiment_id": "exp-1"}
        body = {"trace_id": "trace-1", "spans": []}
        persist(receipt, body)
        assert connections[-1].in_transaction is False
        # A second writer can commit while the delivery connection and its
        # capacity lease remain alive, proving no database write lock spans IO.
        held_first = authority.status()["reserved_bytes"]
        second = original_outbox(campaign, estimated_growth_bytes=64 * 1024)
        second.execute("BEGIN IMMEDIATE")
        second.execute("UPDATE traces SET last_error='parallel' WHERE role=? AND number=?",
                       (role, number))
        second.commit()
        held_with_both = authority.status()["reserved_bytes"]
        second.close()
        assert held_with_both > held_first
        assert authority.status()["reserved_bytes"] == held_first
        return receipt, body

    with patch("ima.research_telemetry._outbox", side_effect=capture_outbox), \
            patch("ima.research_telemetry._emit_trace", side_effect=emit), \
            patch("ima.mlflow_tracking._mlflow", return_value=Mlflow()), \
            patch("ima.research_telemetry._verify_or_restore_trace"):
        receipt = _deliver_trace(campaign, config, "execution", 1)
    assert receipt["trace_id"] == "trace-1"
    assert authority.status()["reserved_bytes"] == 0
    with _read_outbox(campaign) as conn:
        row = conn.execute("SELECT delivered,last_error,receipt FROM traces").fetchone()
    assert row["delivered"] == 1
    assert row["last_error"] is None
    assert 'trace-1' in row["receipt"]


def test_native_full_keeps_prior_rows_and_outer_guard_classifies_it(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch)
    campaign = tmp_path / "campaign"
    config = SimpleNamespace(mlflow_tracking_uri="http://tracking.invalid")
    _enqueue_trace(campaign, {"id": "kept"}, config, role="execution", number=1)
    original = native_sqlite_connection

    def tiny_cap(path, *, estimated_growth_bytes, stage, row_factory=None):
        return original(path, estimated_growth_bytes=0, stage=stage, row_factory=row_factory)

    monkeypatch.setattr("ima.research_sqlite.native_sqlite_connection", tiny_cap)
    with pytest.raises(storage.StorageBlocked) as error:
        _enqueue_trace(campaign, {"body": "x" * (2 * 1024**2)}, config,
                       role="execution", number=2)
    assert error.value.diagnostic['sqlite_errorcode']==sqlite3.SQLITE_FULL
    with _read_outbox(campaign) as conn:
        rows = conn.execute("SELECT role,number FROM traces").fetchall()
    assert [tuple(row) for row in rows] == [("execution", 1)]
    assert authority.status()["reserved_bytes"] == 0


def test_disabled_mode_keeps_legacy_wal_and_enabled_refusal_preserves_triad(tmp_path, monkeypatch):
    campaign = tmp_path / "legacy"
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    conn = _outbox(campaign)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    conn.close()
    db = campaign / "trace-outbox.sqlite"
    before = {p.name: p.read_bytes() for p in campaign.iterdir() if p.is_file()}

    authority = enabled(tmp_path / "enabled", monkeypatch)
    # Use an authority rooted on the same filesystem as the campaign.
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    with pytest.raises(RuntimeError, match="WAL mode"):
        _outbox(campaign)
    after = {p.name: p.read_bytes() for p in campaign.iterdir() if p.is_file()}
    assert after == before
    assert authority.status()["reserved_bytes"] == 0


def test_close_release_failure_keeps_reservation_for_retry(tmp_path, monkeypatch):
    authority = enabled(tmp_path, monkeypatch)
    conn = _outbox(tmp_path / "campaign")
    release = authority.release
    calls = 0

    def fail_once(token):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fixture release failure")
        release(token)

    monkeypatch.setattr(authority, "release", fail_once)
    with pytest.raises(OSError, match="release failure"):
        conn.close()
    assert authority.status()["reserved_bytes"] > 0
    conn.close()
    assert authority.status()["reserved_bytes"] == 0
