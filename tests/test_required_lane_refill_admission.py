"""Required-lane admission uses executable ancestry, before durable actions."""
import gc
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ima import research_expansion as expansion
from ima.research_controller import _portfolio_identity
from ima.research_search import ProgramSearchController
from ima.research_scheduler import ResourceAdmission
from tests import test_research_expansion as fixtures
from tests import test_research_expansion_resources as resource_fixtures

decision, proposal = fixtures.decision, fixtures.proposal


@pytest.fixture
def admission(tmp_path):
    store = expansion.DecisionStore(tmp_path / "decisions.sqlite")
    search = ProgramSearchController(tmp_path)
    config = SimpleNamespace(campaign_dir=tmp_path, proposal_batch_size=260,
                             max_new_programs_per_decision=8, max_pending_programs=64)
    evidence = {"evidence_id": "evidence-1", "completed_trial_index": [],
                "planning_reason": "required_lane_refill",
                "next_required_lane": expansion._lane(13),
                "remaining_program_capacity": {}}
    return store, search, set(), config, evidence


def apply(admission, item):
    store, search, retired, config, evidence = admission
    return expansion.apply_decision(item, store, search, retired, config, evidence, None)


def mixed(identifier="named-benter"):
    recipe = fixtures.PortfolioCompatibilityTests().graph_recipe(experimental=True)
    return proposal(identifier).model_copy(update={"recipe": recipe})


def snapshot(root):
    # Finish unrelated SQLite connection finalizers before comparing durable bytes.
    gc.collect()
    result = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.name.endswith(("-wal", "-shm")):
            continue
        if p.suffix == ".sqlite":
            # Read-only receipt lookup can create WAL bookkeeping, but must not change rows.
            with closing(sqlite3.connect(f"file:{p}?mode=ro", uri=True)) as conn:
                result[str(p.relative_to(root))] = tuple(conn.iterdump())
        else:
            result[str(p.relative_to(root))] = p.read_bytes()
    return result


def test_thirteen_reservations_with_experimental_backlog_require_benter(admission, monkeypatch):
    store, search, retired, config, evidence = admission
    ledger = expansion.ResearchLedger(config.campaign_dir / "ledger.sqlite")
    for index in range(13):
        row = ledger.reserve_attempt(str(index), {"program_id": "previous", "trial_number": index})
        ledger.complete_attempt(row.attempt_id, {"status": "completed"})
    reservations = ledger.reserved_attempts()
    assert len(reservations) == 13 and ledger.pending_attempts() == []
    pid = search.register(mixed("backlog"))
    capacity = {pid: 1}
    evidence["remaining_program_capacity"] = capacity
    required = expansion._lane(len(reservations))
    assert required == "benter"
    assert expansion._needs_lane_refill(capacity, search.programs, len(reservations))
    assert expansion.planning_reason(
        now=10, last_plan=1, terminal_since_plan=0, terminal_threshold=32,
        remaining_trials=1, low_watermark=8, active_jobs=0,
        available_program_slots=63, checkpoint_seconds=600,
        lane_refill_required=True) == "required_lane_refill"
    # The fit filter has no candidate even with an idle resource pool.
    ready = {pid}
    allowed = [p for p in capacity if capacity[p] > 0 and p in ready
               and _portfolio_identity(search.programs[p].recipe, "expansion_v6")["lane"] == required]
    assert allowed == []
    assert ResourceAdmission(2, 26, 93).peek_feasible({}) is None
    before = snapshot(config.campaign_dir)
    preflight = Mock(side_effect=AssertionError("reached later preflight"))
    monkeypatch.setattr(expansion, "_preflight_betting_requests", preflight)
    with pytest.raises(ValueError, match="zero eligible benter.*experimental"):
        apply(admission, decision(programs=(mixed("another-named-benter"),)))
    preflight.assert_not_called()
    assert snapshot(config.campaign_dir) == before
    assert set(search.programs) == {pid}
    assert retired == set()
    assert store.pending() == []


def test_new_classical_benter_is_durably_accepted(admission):
    accepted = apply(admission, decision())
    store, search, _, _, _ = admission
    assert len(accepted) == 1
    assert _portfolio_identity(search.programs[accepted[0]].recipe, "expansion_v6")["lane"] == "benter"
    assert store.authorized_budgets()[accepted[0]] == 1
    assert store.pending() == []
    assert apply(admission, decision()) == accepted  # durable replay


def test_extension_restores_exhausted_classical_capacity(admission):
    store, search, _, _, evidence = admission
    pid = search.register(proposal("existing", trials=2))
    evidence["remaining_program_capacity"][pid] = 0
    item = decision(programs=(), extensions={pid: 1})
    assert apply(admission, item) == []
    assert search.programs[pid].max_trials == 3
    assert store.authorized_budgets()[pid] == 3
    assert apply(admission, item) == []
    assert search.programs[pid].max_trials == 3


def test_extension_of_execution_exhausted_program_cannot_cover_lane(admission):
    _, search, _, config, evidence = admission
    pid = search.register(proposal("no-unique-recipes", trials=2))
    # The real controller omits execution-exhausted IDs from its capacity view;
    # budget-exhausted IDs remain present with zero, as tested above.
    assert pid not in evidence["remaining_program_capacity"]
    before = snapshot(config.campaign_dir)
    with pytest.raises(ValueError, match="required_lane_refill: zero eligible benter"):
        apply(admission, decision(programs=(), extensions={pid: 1}))
    assert snapshot(config.campaign_dir) == before


@pytest.mark.parametrize("retired_now", [False, True])
def test_retired_classical_capacity_does_not_cover_refill(admission, retired_now):
    _, search, retired, config, evidence = admission
    pid = search.register(proposal("retiring", trials=2))
    evidence["remaining_program_capacity"][pid] = 2
    if retired_now:
        retired.add(pid)
    item = decision(programs=(mixed(),), retire_program_ids=() if retired_now else (pid,))
    before = snapshot(config.campaign_dir)
    with pytest.raises(ValueError, match="zero eligible benter"):
        apply(admission, item)
    assert snapshot(config.campaign_dir) == before


def test_existing_required_capacity_can_cover_experimental_addition(admission):
    _, search, _, _, evidence = admission
    pid = search.register(proposal("existing", trials=2))
    evidence["remaining_program_capacity"][pid] = 1
    assert len(apply(admission, decision(programs=(mixed(),)))) == 1


@pytest.mark.parametrize("retired_extension", [False, True])
def test_extension_cannot_cover_wrong_lane_or_revive_retired_program(admission, retired_extension):
    _, search, retired, config, evidence = admission
    candidate = proposal("retired") if retired_extension else mixed("experimental")
    pid = search.register(candidate)
    evidence["remaining_program_capacity"][pid] = 0
    if retired_extension:
        retired.add(pid)
    before = snapshot(config.campaign_dir)
    with pytest.raises(ValueError, match="zero eligible benter"):
        apply(admission, decision(programs=(), extensions={pid: 1}))
    assert snapshot(config.campaign_dir) == before


@pytest.mark.parametrize("unallocated", [0, 3])
def test_explicit_no_allocation_hold_preserved(admission, unallocated):
    item = decision(trial_budget=unallocated, programs=(),
                    unallocated_trials=unallocated,
                    unallocated_reason="Hold for evidence" if unallocated else None,
                    review_reason="Hold for evidence")
    assert apply(admission, item) == []
    assert admission[0].authorized_budgets() == {}


def test_paper_only_hold_reaches_existing_paper_preflight(admission, monkeypatch):
    # Keep paper's own ledger/source validation intact; lane admission must pass it through.
    item = decision(trial_budget=0, programs=(), betting_requests=({
        "request_id": "paper", "evidence_id": "evidence-1", "attempt_ids": ["completed"],
    },))
    preflight = Mock(side_effect=ValueError("paper preflight reached"))
    monkeypatch.setattr(expansion, "_preflight_betting_requests", preflight)
    with pytest.raises(ValueError, match="paper preflight reached"):
        apply(admission, item)
    preflight.assert_called_once()


def test_other_planning_reasons_and_eighty_twenty_order_unchanged(admission):
    admission[4]["planning_reason"] = "startup_review"
    assert len(apply(admission, decision(programs=(mixed(),)))) == 1
    lanes = [expansion._lane(i) for i in range(15)]
    assert lanes == ["benter", "benter", "experimental", "benter", "benter"] * 3


def test_experimental_required_lane_uses_same_ancestry_contract(admission):
    admission[4]["next_required_lane"] = expansion._lane(12)
    with pytest.raises(ValueError, match="zero eligible experimental"):
        apply(admission, decision())
    assert len(apply(admission, decision(programs=(mixed(),)))) == 1


@pytest.mark.parametrize("retired_duplicate", [False, True])
def test_canonical_duplicate_does_not_restore_exhausted_or_retired_capacity(admission, retired_duplicate):
    _, search, retired, config, evidence = admission
    original = proposal("original", trials=1)
    pid = search.register(original)
    trial = search.studies[pid].ask()
    search.studies[pid].tell(trial, 2.0)
    evidence["remaining_program_capacity"][pid] = 0
    if retired_duplicate:
        retired.add(pid)
    renamed = original.model_copy(update={"proposal_id": "renamed", "max_trials": 4})
    assert search.register(renamed) == pid
    assert search.programs[pid].max_trials == 1
    before = snapshot(config.campaign_dir)
    with pytest.raises(ValueError, match="required_lane_refill:.*canonical duplicates"):
        apply(admission, decision(trial_budget=4, programs=(renamed,)))
    assert snapshot(config.campaign_dir) == before
    assert search.programs[pid].max_trials == 1


def test_lane_failure_streak_is_consecutive_and_reaches_existing_feedback(tmp_path):
    root = tmp_path / "decisions"
    root.mkdir()
    for number in range(1, 4):
        expansion.core._write_json_atomic(root / f"D{number:06d}.json", {
            "decision_id": f"D{number:06d}", "planner_status": "failed",
            "error": "required_lane_refill: zero eligible benter capacity; supply a valid extension",
        })
        assert expansion._required_lane_failure_streak(tmp_path) == number
    feedback = expansion._recent_planner_failures(tmp_path)
    assert len(feedback) == 3
    assert "supply a valid extension" in feedback[0]["error"]
    expansion.core._write_json_atomic(root / "D000004.json", {"planner_status": "accepted"})
    assert expansion._required_lane_failure_streak(tmp_path) == 0
    expansion.core._write_json_atomic(root / "D000004.json", {
        "planner_status": "failed", "error": "unrelated failure",
    })
    assert expansion._required_lane_failure_streak(tmp_path) == 0


def test_three_paid_lane_rejections_prevent_fourth_planner_call(monkeypatch):
    from dataclasses import replace
    fixture = resource_fixtures.ExpansionResourceIntegrationTests()
    fixture.setUp()
    try:
        root = fixture.config.campaign_dir
        for number in range(1, 4):
            expansion.core._write_json_atomic(root / "decisions" / f"D{number:06d}.json", {
                "planner_status": "failed", "error": "required_lane_refill: zero eligible benter capacity",
            })
        planner = Mock(side_effect=AssertionError("fourth paid decision must not start"))
        monkeypatch.setattr(expansion, "plan_decision", planner)
        monkeypatch.setattr(expansion, "ProcessPoolExecutor", resource_fixtures.executors)
        monkeypatch.setattr(expansion, "log_snapshot", Mock())
        monkeypatch.setattr(expansion, "_control_planner_spend", lambda *a, **k: {
            "spend_unknown": False, "known_spend_usd": .03, "total_cost_usd": .03,
            "unresolved_decision_ids": [], "not_dispatched_decision_ids": [],
        })
        monkeypatch.setattr(expansion, "_unsent_failure_streak", lambda _: 0)
        config = replace(fixture.config, planner_mode="openrouter", model="test/model",
                         max_total_cost_usd=1., max_trials=None, timeout_minutes=.03)
        result = expansion.run_expansion_campaign(config)
        planner.assert_not_called()
        assert result["mode"] == "timeout"
        status = expansion.core._read_json(root / "status.json")
        assert status["status"] == "timeout"
        assert "three rejected decisions" in status["last_planner_error"]
        assert not status["planner_inflight"]
    finally:
        fixture.doCleanups()
