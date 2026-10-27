"""Execution recovery cannot turn half an Optuna ask into a model recipe."""
from types import SimpleNamespace

from ima.research_expansion import _recover_orphan_asks
from ima.research_search import ProgramSearchController
from tests.test_research_journal_storage import _authority, _proposal


def test_execution_bridge_retires_incomplete_ask_before_any_allocation(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    controller = ProgramSearchController(tmp_path / 'campaign')
    program_id = controller.register(_proposal())
    trial = controller.studies[program_id].ask()
    trial.suggest_float('C', .1, 1.)

    def forbidden(*args, **kwargs):
        raise AssertionError('An incomplete ask reached execution allocation')

    ledger = SimpleNamespace(reserved_attempts=lambda: [], reserve_attempt=forbidden)
    assert _recover_orphan_asks(controller, ledger, {program_id: {}},
                                {program_id}, 'revision', 'environment') == 0
    assert controller.studies[program_id].trials[0].state.name == 'PRUNED'
    assert controller.remaining_capacity() == _proposal().max_trials
