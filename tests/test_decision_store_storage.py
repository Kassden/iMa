import json
import sqlite3
from types import SimpleNamespace

import pytest

import ima.research_storage as research_storage
import ima.research_sqlite as research_sqlite
from ima.research_expansion import DecisionStore, PlannerDecision
from ima.research_storage import StorageBlocked, StorageAuthority


def _decision(identifier="decision-1", *, evidence="evidence-1"):
    return PlannerDecision(
        decision_id=identifier,
        evidence_id=evidence,
        trial_budget=0,
        review_reason="Storage admission fixture.",
    )


def _authority(tmp_path, monkeypatch, *, available=64 * 1024 * 1024):
    capacity = {"bytes": available}

    def statvfs(_):
        return SimpleNamespace(f_frsize=1, f_blocks=capacity["bytes"],
                               f_bavail=capacity["bytes"], f_favail=100_000)

    authority = StorageAuthority(
        tmp_path / "control", headroom_bytes=0, headroom_inodes=0,
        statvfs=statvfs,
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(research_storage, "StorageAuthority", lambda root: authority)
    return authority, capacity


def _snapshot(path):
    return {
        item.name: (item.read_bytes() if item.is_file() else None)
        for item in path.parent.iterdir()
        if item.name in {path.name, path.name + "-wal", path.name + "-shm"}
    }


def test_denied_constructor_admission_precedes_parent_creation(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch, available=0)
    path = tmp_path / "absent" / "campaign" / "decisions.sqlite"

    with pytest.raises(StorageBlocked):
        DecisionStore(path)

    assert not path.parent.exists()


def test_admitted_receipts_completion_reopen_and_public_readback(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    store = DecisionStore(tmp_path / "campaign" / "decisions.sqlite")
    decision = _decision()
    assert store.record(decision) is True
    assert store.receipt(decision.decision_id, "program:probe", {
        "program_id": "probe", "allocated": 3,
    }) == {"program_id": "probe", "allocated": 3}
    assert store.receipt(decision.decision_id, "dataset:sample", {
        "request": {"request_id": "dataset-1"},
    }) == {"request": {"request_id": "dataset-1"}}
    assert store.receipt(decision.decision_id, "betting:sample", {
        "action_id": "action-1", "request": {"request_id": "paper-1"},
    }) == {"action_id": "action-1", "request": {"request_id": "paper-1"}}
    store.complete(decision.decision_id)

    reopened = DecisionStore(store.path)
    assert reopened.pending() == []
    assert reopened.action(decision.decision_id, "program:probe") == {
        "program_id": "probe", "allocated": 3,
    }
    assert reopened.authorized_budgets() == {"probe": 3}
    assert reopened.authorized_program_ids() == {"probe"}
    assert reopened.dataset_actions() == [{"request_id": "dataset-1"}]
    assert reopened.betting_actions() == [{
        "action_id": "action-1", "request": {"request_id": "paper-1"},
        "decision_id": decision.decision_id, "completion": None,
    }]
    assert reopened.betting_actions(pending_only=True) == reopened.betting_actions()


def test_conflicting_replay_and_unknown_receipt_leave_stored_application_unchanged(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    store = DecisionStore(tmp_path / "decisions.sqlite")
    original = _decision()
    store.record(original)
    store.receipt(original.decision_id, "program:probe", {"program_id": "probe"})
    before = store.pending()[0].model_dump(mode="json")

    with pytest.raises(ValueError, match="Conflicting decision replay"):
        store.record(_decision(evidence="changed-evidence"))
    with pytest.raises(ValueError, match="unknown decision"):
        store.receipt("missing", "program:other", {"program_id": "other"})

    assert store.pending()[0].model_dump(mode="json") == before
    assert store.action(original.decision_id, "program:probe") == {"program_id": "probe"}
    assert store.action("missing", "program:other") is None


def test_rejected_write_preserves_prior_application(tmp_path, monkeypatch):
    _, capacity = _authority(tmp_path, monkeypatch)
    store = DecisionStore(tmp_path / "decisions.sqlite")
    prior = _decision()
    store.record(prior)
    before = store.pending()[0].model_dump(mode="json")
    capacity["bytes"] = 0

    with pytest.raises(StorageBlocked):
        store.record(_decision("decision-2"))

    assert store.pending()[0].model_dump(mode="json") == before
    assert store.action(prior.decision_id, "unused") is None


def test_reads_under_storage_pressure_create_no_files(tmp_path, monkeypatch):
    authority, capacity = _authority(tmp_path, monkeypatch)
    store = DecisionStore(tmp_path / "campaign" / "decisions.sqlite")
    item = _decision()
    store.record(item)
    store.receipt(item.decision_id, "program:probe", {"program_id": "probe", "allocated": 1})
    store.complete(item.decision_id)
    before = set(store.path.parent.iterdir())
    capacity["bytes"] = 0

    assert store.pending() == []
    assert store.action(item.decision_id, "program:probe")["program_id"] == "probe"
    assert store.authorized_budgets() == {"probe": 1}
    assert store.dataset_actions() == []
    assert store.betting_actions() == []
    assert set(store.path.parent.iterdir()) == before
    assert authority.status()["reserved_bytes"] == 0


def test_enabled_wal_refusal_leaves_database_and_sidecars_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "wal.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE fixture (value TEXT)")
    _authority(tmp_path, monkeypatch)
    before = _snapshot(path)

    with pytest.raises(RuntimeError, match="WAL mode is unsupported"):
        DecisionStore(path)
    # Exercise the read-side refusal independently of constructor dispatch.
    reader = object.__new__(DecisionStore)
    reader.path = path
    with pytest.raises(RuntimeError, match="WAL mode is unsupported"):
        reader.pending()

    assert _snapshot(path) == before


def test_disabled_mode_retains_wal_compatibility(tmp_path, monkeypatch):
    monkeypatch.delenv("IMA_STORAGE_ROOT", raising=False)
    store = DecisionStore(tmp_path / "legacy" / "decisions.sqlite")
    assert store.record(_decision()) is True
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert len(DecisionStore(store.path).pending()) == 1


def test_growth_estimate_scales_with_large_utf8_payload(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    observed = []
    admitted = research_sqlite.admitted_sqlite

    def capture(path, *, estimated_growth_bytes, **kwargs):
        observed.append(estimated_growth_bytes)
        return admitted(path, estimated_growth_bytes=estimated_growth_bytes, **kwargs)

    monkeypatch.setattr(research_sqlite, "admitted_sqlite", capture)
    store = DecisionStore(tmp_path / "decisions.sqlite")
    long_evidence = "证据" * 40_000
    record = _decision(evidence=long_evidence)
    encoded = json.dumps(record.model_dump(mode="json"), sort_keys=True)
    assert store.record(record) is True
    assert observed[-1] >= 1024 * 1024 + 3 * len(encoded.encode("utf-8"))
