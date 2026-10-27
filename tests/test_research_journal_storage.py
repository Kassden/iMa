from __future__ import annotations

import errno
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

optuna = pytest.importorskip("optuna")

from ima import research_storage
from ima.research_journal import (
    AdmittedJournalFileBackend, admitted_journal_backend,
)
from ima.research_search import ProgramSearchController, RecipeSearchController
from ima.research_specs import PipelineRecipe, ResearchProposal


def _authority(tmp_path: Path, monkeypatch, **kwargs):
    policy = {"headroom_bytes": 0, "headroom_inodes": 0}
    policy.update(kwargs)
    authority = research_storage.StorageAuthority(
        tmp_path / "control", **policy
    )
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(authority.root))
    monkeypatch.setattr(research_storage, "StorageAuthority", lambda _root: authority)
    return authority


def _proposal() -> ResearchProposal:
    return ResearchProposal(
        proposal_id="journal-storage",
        hypothesis="Exercise native journal admission.",
        changed_axes=("hyperparameters",),
        recipe=PipelineRecipe(),
        search_space={"C": {"kind": "float", "low": 0.1, "high": 1.0}},
        expected_observation="A persisted trial.",
        falsification_rule="The journal does not replay.",
        max_trials=2,
    )


def test_native_create_ask_tell_and_reopen_preserves_trial_data(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    program = ProgramSearchController(root)
    program_id = program.register(_proposal())
    suggestion = program.ask(1)[0]
    program.tell(program_id, suggestion.trial_number, 1.234)
    expected_params = program.studies[program_id].trials[suggestion.trial_number].params
    recipe_search = RecipeSearchController(root / "recipe")
    recipe_suggestion = recipe_search.ask(1)[0]
    recipe_search.tell(recipe_suggestion.trial_number, 2.345)

    reopened = ProgramSearchController(root)
    restored = reopened.studies[program_id].trials[suggestion.trial_number]
    assert restored.value == 1.234
    assert restored.params == expected_params
    recipe_reopened = RecipeSearchController(root / "recipe")
    restored_recipe = recipe_reopened.study.trials[recipe_suggestion.trial_number]
    assert restored_recipe.value == 2.345
    assert restored_recipe.user_attrs["recipe_hash"] == recipe_suggestion.recipe.recipe_hash()


def test_denied_admission_precedes_search_directory_and_journal_creation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch, headroom_bytes=10**30)
    campaign = tmp_path / "not-created"
    with pytest.raises(research_storage.StorageBlocked):
        RecipeSearchController(campaign)
    assert not campaign.exists()
    assert authority.status()["reservations"] == {}


def test_partial_native_append_rolls_back_and_releases_allocation(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    backend = admitted_journal_backend(tmp_path / "campaign" / "journal.log")
    path = Path(backend._file_path)
    before = path.read_bytes()

    @contextmanager
    def partial_writer(target, mode):
        fd = os.open(target, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)

        class Partial:
            def fileno(self):
                return fd

            def write(self, payload):
                os.write(fd, payload[: max(1, len(payload) // 2)])
                raise OSError(errno.ENOSPC, "injected full filesystem")

            def flush(self):
                pass

        try:
            yield Partial()
        finally:
            os.close(fd)

    monkeypatch.setattr(research_storage, "reserved_output_file", partial_writer)
    with pytest.raises(research_storage.StorageBlocked) as raised:
        backend.append_logs([{"op_code": 777, "worker_id": "test"}])
    assert raised.value.diagnostic['errno'] == errno.ENOSPC
    assert raised.value.diagnostic['stage'] == 'optuna_journal_append'
    assert path.read_bytes() == before
    assert list(backend.read_logs(0)) == []
    assert authority.status()["reservations"] == {}


def test_zero_byte_capacity_denies_journal_metadata_before_creation(tmp_path,monkeypatch):
    authority=_authority(tmp_path,monkeypatch)
    state=os.statvfs(authority.root)
    authority._statvfs=lambda _:SimpleNamespace(f_blocks=state.f_blocks,
        f_frsize=state.f_frsize,f_bavail=0,f_favail=state.f_favail)
    destination=tmp_path/'not-created'/'journal.log'
    with pytest.raises(research_storage.StorageBlocked):admitted_journal_backend(destination)
    assert not destination.parent.exists()


def test_optional_fallback_directory_also_requires_admission(tmp_path,monkeypatch):
    authority=_authority(tmp_path,monkeypatch,headroom_bytes=10**30)
    import ima.research_search as search
    monkeypatch.setattr(search,'optuna',None)
    with pytest.raises(research_storage.StorageBlocked):search.RecipeSearchController(tmp_path/'campaign')
    assert not (tmp_path/'campaign').exists()


def test_journal_data_symlink_and_hardlink_aliases_are_refused(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    directory = tmp_path / "campaign"
    directory.mkdir()
    target = directory / "target.log"
    target.write_bytes(b"")
    linked = directory / "linked.log"
    linked.symlink_to(target.name)
    with pytest.raises(ValueError, match="canonical|single-link"):
        admitted_journal_backend(linked)
    hardlinked = directory / "hardlinked.log"
    os.link(target, hardlinked)
    with pytest.raises(ValueError, match="single-link"):
        admitted_journal_backend(target)


def test_adapter_bytes_match_installed_optuna_serialization(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    backend = admitted_journal_backend(tmp_path / "journal.log")
    row = {"op_code": 41, "worker_id": "w", "payload": [1, "é"]}
    expected = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
    before = Path(backend._file_path).read_bytes()
    backend.append_logs([row])
    assert Path(backend._file_path).read_bytes() == before + expected
    assert list(backend.read_logs(0)) == [row]


def test_concurrent_native_appends_remain_complete_and_replayable(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    path = tmp_path / "journal.log"
    backends = [admitted_journal_backend(path) for _ in range(4)]

    def append(index):
        backends[index % len(backends)].append_logs([{"op_code": index, "worker_id": str(index)}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(40)))
    rows = list(backends[0].read_logs(0))
    assert len(rows) == 40
    assert {row["op_code"] for row in rows} == set(range(40))
    assert all(type(backend) is AdmittedJournalFileBackend for backend in backends)


def test_program_index_and_budget_metadata_use_controller_writers(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    controller = ProgramSearchController(tmp_path / "campaign")
    program_id = controller.register(_proposal())
    controller.set_budget(program_id, 1)
    assert json.loads(controller.index_path.read_text().splitlines()[0])["program_id"] == program_id
    assert json.loads(controller.budget_path.read_text())[program_id] == 1


def test_program_ask_holds_complete_journal_admission_before_first_append(tmp_path, monkeypatch):
    authority = _authority(tmp_path, monkeypatch)
    controller = ProgramSearchController(tmp_path / "campaign")
    controller.register(_proposal())
    backend = controller.storage._backend
    original = backend.append_logs
    reservations_seen_at_append = []

    def observe_append(logs):
        reservations_seen_at_append.append([
            row["stage"] for row in authority.status()["reservations"].values()
        ])
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", observe_append)
    assert len(controller.ask(1)) == 1
    assert reservations_seen_at_append
    assert all("optuna_program_ask" in stages for stages in reservations_seen_at_append)
    assert all("optuna_journal_append" not in stages for stages in reservations_seen_at_append)


def test_reopen_prunes_incomplete_ask_and_keeps_persisted_recipe_out_of_next_ask(
    tmp_path, monkeypatch
):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    program_id = controller.register(_proposal())
    backend = controller.storage._backend
    original = backend.append_logs
    injected = False

    def fail_recipe_hash(logs):
        nonlocal injected
        if not injected and any(
            row.get("op_code") == 8
            and "recipe_hash" in row.get("user_attr", {})
            for row in logs
        ):
            injected = True
            raise research_storage.StorageBlocked("injected ask metadata pressure")
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", fail_recipe_hash)
    with pytest.raises(research_storage.StorageBlocked, match="injected ask metadata"):
        controller.ask(1)
    partial = controller.studies[program_id].trials[0]
    assert partial.state.name == "PRUNED"
    assert "recipe" in partial.user_attrs
    assert "recipe_hash" not in partial.user_attrs
    assert partial.user_attrs["recovered_incomplete"] is True
    assert partial.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"
    partial_hash = PipelineRecipe.model_validate(partial.user_attrs["recipe"]).recipe_hash()

    reopened = ProgramSearchController(root)
    recovered = reopened.studies[program_id].trials[0]
    assert recovered.state.name == "PRUNED"
    assert recovered.user_attrs["recovered_incomplete"] is True
    assert recovered.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"
    assert reopened.remaining_capacity() == _proposal().max_trials
    next_suggestions = reopened.ask(1)
    assert len(next_suggestions) == 1
    assert next_suggestions[0].recipe.recipe_hash() == partial_hash
    assert reopened.studies[program_id].trials[0].state.name == "PRUNED"


def test_reopen_releases_budget_for_ask_failing_before_recipe_is_materialized(
    tmp_path, monkeypatch
):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    program_id = controller.register(_proposal())
    backend = controller.storage._backend
    original = backend.append_logs

    def fail_recipe(logs):
        if any(
            row.get("op_code") == 8 and "recipe" in row.get("user_attr", {})
            for row in logs
        ):
            raise research_storage.StorageBlocked("injected first recipe append pressure")
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", fail_recipe)
    with pytest.raises(research_storage.StorageBlocked, match="first recipe append"):
        controller.ask(1)
    partial = controller.studies[program_id].trials[0]
    assert partial.state.name == "PRUNED"
    assert partial.user_attrs["recovered_incomplete"] is True
    assert partial.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"

    reopened = ProgramSearchController(root)
    recovered = reopened.studies[program_id].trials[0]
    assert recovered.state.name == "PRUNED"
    assert recovered.user_attrs["recovered_incomplete"] is True
    assert recovered.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"
    assert reopened.has_capacity()
    assert reopened.remaining_capacity() == _proposal().max_trials


def test_recovery_marker_storage_failure_remains_typed_and_retryable(
    tmp_path, monkeypatch
):
    authority = _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    program_id = controller.register(_proposal())
    backend = controller.storage._backend
    original_append = backend.append_logs

    def fail_recipe(logs):
        if any(
            row.get("op_code") == 8 and "recipe" in row.get("user_attr", {})
            for row in logs
        ):
            raise research_storage.StorageBlocked("injected ask metadata pressure")
        if any(
            row.get("op_code") == 8
            and "recovered_incomplete" in row.get("user_attr", {})
            for row in logs
        ):
            raise research_storage.StorageBlocked("injected immediate cleanup pressure")
        return original_append(logs)

    monkeypatch.setattr(backend, "append_logs", fail_recipe)
    with pytest.raises(research_storage.StorageBlocked):
        controller.ask(1)
    monkeypatch.setattr(backend, "append_logs", original_append)

    original_reserve = authority.reserve

    def fail_recovery_reservation(**kwargs):
        if kwargs.get("stage") == "optuna_journal_append":
            raise research_storage.StorageBlocked("injected recovery marker pressure")
        return original_reserve(**kwargs)

    monkeypatch.setattr(authority, "reserve", fail_recovery_reservation)
    with pytest.raises(research_storage.StorageBlocked) as raised:
        ProgramSearchController(root)
    assert raised.value.diagnostic["retryable"] is True
    assert raised.value.diagnostic["failure_kind"] == "infrastructure_storage"
    assert "recovered_incomplete" not in controller.studies[program_id].trials[0].user_attrs

    monkeypatch.setattr(authority, "reserve", original_reserve)
    reopened = ProgramSearchController(root)
    recovered = reopened.studies[program_id].trials[0]
    assert recovered.state.name == "PRUNED"
    assert recovered.user_attrs["recovered_incomplete"] is True
    assert recovered.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"


def test_second_ask_reconciles_same_controller_after_partial_storage_failure(
    tmp_path, monkeypatch
):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    proposal = _proposal().model_copy(update={"max_trials": 1})
    program_id = controller.register(proposal)
    backend = controller.storage._backend
    original = backend.append_logs

    def fail_ask_and_cleanup(logs):
        if any(
            row.get("op_code") == 8 and "recipe" in row.get("user_attr", {})
            for row in logs
        ):
            raise research_storage.StorageBlocked("injected ask pressure")
        if any(
            row.get("op_code") == 8
            and "recovered_incomplete" in row.get("user_attr", {})
            for row in logs
        ):
            raise research_storage.StorageBlocked("injected cleanup pressure")
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", fail_ask_and_cleanup)
    with pytest.raises(research_storage.StorageBlocked, match="injected ask pressure") as raised:
        controller.ask(1)
    assert any("Immediate incomplete-ask recovery also failed" in note
               for note in getattr(raised.value, "__notes__", []))

    monkeypatch.setattr(backend, "append_logs", original)
    retry = controller.ask(1)
    assert len(retry) == 1
    orphan = controller.studies[program_id].trials[0]
    assert orphan.state.name == "PRUNED"
    assert orphan.user_attrs["recovered_incomplete"] is True
    assert orphan.user_attrs["reason"] == "infrastructure_storage/incomplete_ask"
    assert retry[0].trial_number == 1


def test_pruned_completed_recipe_remains_in_duplicate_seen_set(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    proposal = ResearchProposal.model_validate(
        _proposal().model_dump(mode="json")
        | {"search_space": {}, "fixed_parameters": True, "max_trials": 1}
    )
    controller = ProgramSearchController(tmp_path / "campaign")
    program_id = controller.register(proposal)
    first = controller.ask(1)[0]
    controller.studies[program_id].tell(first.trial_number, state=optuna.trial.TrialState.PRUNED)

    assert controller.ask(1) == []
    duplicates = controller.studies[program_id].get_trials(deepcopy=True)[1:]
    assert duplicates
    assert all(trial.state.name == "PRUNED" for trial in duplicates)
    assert all(trial.user_attrs["recipe_hash"] == first.recipe.recipe_hash() for trial in duplicates)


def test_count_batch_failure_keeps_earlier_complete_ask_for_outer_recovery(
    tmp_path, monkeypatch
):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    proposal = _proposal().model_copy(update={"max_trials": 3})
    program_id = controller.register(proposal)
    backend = controller.storage._backend
    original = backend.append_logs
    recipe_hash_writes = 0

    def fail_second_recipe_hash(logs):
        nonlocal recipe_hash_writes
        if any(
            row.get("op_code") == 8
            and "recipe_hash" in row.get("user_attr", {})
            for row in logs
        ):
            recipe_hash_writes += 1
            if recipe_hash_writes == 2:
                raise research_storage.StorageBlocked("injected later ask append pressure")
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", fail_second_recipe_hash)
    with pytest.raises(research_storage.StorageBlocked, match="later ask append"):
        controller.ask(2)

    reopened = ProgramSearchController(root)
    trials = reopened.studies[program_id].get_trials(deepcopy=True)
    assert trials[0].state.name == "RUNNING"
    assert {"recipe", "recipe_hash", "program_id"} <= set(trials[0].user_attrs)
    assert trials[1].state.name == "PRUNED"
    assert trials[1].user_attrs["recovered_incomplete"] is True
    assert reopened.remaining_capacity() == 2


def test_reopen_waits_for_active_program_ask_envelope(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    controller = ProgramSearchController(root)
    program_id = controller.register(_proposal())
    backend = controller.storage._backend
    original = backend.append_logs
    append_paused = threading.Event()
    release_append = threading.Event()
    opened = threading.Event()
    results = []

    def pause_after_trial_create(logs):
        if any(row.get("op_code") == 4 for row in logs):
            append_paused.set()
            if not release_append.wait(timeout=3):
                raise TimeoutError("Timed out waiting to resume the worker ask")
        return original(logs)

    monkeypatch.setattr(backend, "append_logs", pause_after_trial_create)

    def run_worker_ask():
        results.extend(controller.ask(1))

    reopened = []

    def reopen_during_ask():
        reopened.append(ProgramSearchController(root))
        opened.set()

    ask_worker = threading.Thread(target=run_worker_ask)
    reopen_worker = threading.Thread(target=reopen_during_ask)
    ask_worker.start()
    assert append_paused.wait(timeout=2)
    reopen_worker.start()
    try:
        assert not opened.wait(timeout=0.1)
    finally:
        release_append.set()
    ask_worker.join(timeout=3)
    reopen_worker.join(timeout=3)
    assert not ask_worker.is_alive()
    assert not reopen_worker.is_alive()
    assert opened.is_set()
    assert len(results) == 1
    assert reopened[0].studies[program_id].trials[0].state.name == "RUNNING"


def test_replay_never_observes_lines_from_a_rolled_back_append(tmp_path, monkeypatch):
    _authority(tmp_path, monkeypatch)
    backend = admitted_journal_backend(tmp_path / 'campaign' / 'journal.log')
    backend.append_logs([{'accepted': 1}])
    writer_started, finish_writer, reader_done = (threading.Event() for _ in range(3))
    reader_at_lock = threading.Event()
    observed, errors = [], []
    real_open = research_storage.reserved_output_file

    @contextmanager
    def interrupted(target, mode):
        with real_open(target, mode) as stream:
            class Partial:
                def fileno(self):
                    return stream.fileno()

                def write(self, data):
                    stream.write(data[:data.index(b'\n') + 1])
                    stream.flush()
                    writer_started.set()
                    assert finish_writer.wait(5)
                    raise OSError(errno.ENOSPC, 'append interrupted after a complete line')

                def flush(self):
                    stream.flush()
            yield Partial()

    monkeypatch.setattr(research_storage, 'reserved_output_file', interrupted)
    real_flock = __import__('fcntl').flock

    def observed_flock(fd, operation):
        if operation == __import__('fcntl').LOCK_SH:
            reader_at_lock.set()
        return real_flock(fd, operation)

    monkeypatch.setattr('ima.research_journal.fcntl.flock', observed_flock)

    def write():
        try:
            backend.append_logs([{'uncommitted': 2}, {'uncommitted': 3}])
        except research_storage.StorageBlocked as exc:
            errors.append(exc)

    def read():
        observed.extend(backend.read_logs(0))
        reader_done.set()

    writer = threading.Thread(target=write)
    reader = threading.Thread(target=read)
    writer.start()
    assert writer_started.wait(5)
    reader.start()
    try:
        assert reader_at_lock.wait(5), 'Reader never reached the native shared lock'
        assert not reader_done.is_set()
    finally:
        finish_writer.set()
        writer.join(5)
        reader.join(5)
    assert not writer.is_alive() and not reader.is_alive()
    assert len(errors) == 1
    assert observed == [{'accepted': 1}]
