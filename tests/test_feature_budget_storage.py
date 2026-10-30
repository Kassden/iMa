import contextlib
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess

import pytest


HELPER = Path(__file__).resolve().parents[1] / "deploy/systemd/ima-feature-exploration-budget.py"
SPEC = importlib.util.spec_from_file_location("feature_budget_storage_test_target", HELPER)
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)
REAL_AUTHORITY = BUDGET.research_storage.StorageAuthority


@pytest.mark.parametrize('native_exit,condition_exit', [(0, 0), (2, 2), (1, 255), (127, 255)])
def test_service_condition_preserves_storage_skip_and_fails_unexpected_errors(tmp_path, native_exit, condition_exit):
    unit = HELPER.with_name('ima-feature-exploration.service').read_text()
    condition = next(line.split('=', 1)[1] for line in unit.splitlines()
                     if line.startswith('ExecCondition='))
    assert 'ExecStartPre=' not in unit
    fake = tmp_path / 'fake-python'
    fake.write_text('#!/bin/sh\nexit "$TEST_NATIVE_EXIT"\n')
    fake.chmod(0o700)
    result = subprocess.run(shlex.split(condition), env={
        **os.environ, 'IMA_FEATURE_PYTHON': str(fake), 'IMA_FEATURE_RELEASE': str(tmp_path),
        'TEST_NATIVE_EXIT': str(native_exit)}, capture_output=True, text=True)
    assert result.returncode == condition_exit


def _fixture(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    control = home / "storage-control"
    control.mkdir(mode=0o700)
    control.chmod(0o700)
    monkeypatch.setattr(BUDGET, "OWN_HOME", home)
    monkeypatch.setattr(BUDGET, "ROOT", home / "research-v2")
    monkeypatch.setattr(BUDGET, "STATE", home / "research-v2" / "feature-exploration-budget.json")
    monkeypatch.setattr(BUDGET, "DROPIN", home / ".config/systemd/user" /
                        (BUDGET.MODEL_UNIT + ".d/95-feature-exploration-budget.conf"))
    monkeypatch.setattr(BUDGET, "STORAGE_ROOT", control)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(control))
    monkeypatch.setattr(BUDGET, "require_owner",
                        lambda: BUDGET.require_owner_identity(1001, Path("/home/imaopt")))

    def local_authority(path, *, headroom_bytes=0, headroom_inodes=0):
        return REAL_AUTHORITY(path, headroom_bytes=headroom_bytes,
                              headroom_inodes=headroom_inodes)

    monkeypatch.setattr(BUDGET.research_storage, "StorageAuthority", local_authority)
    unit = {"ActiveState": "failed", "MainPID": "0", "ControlGroup": "",
            "MemoryMax": str(BUDGET.TOTAL_BYTES), "CPUQuotaPerSecUSec": "26s",
            "FreezerState": "running"}
    manager_calls = []
    systemctl_calls = []

    def manager(*args):
        manager_calls.append(args)
        return unit[args[args.index("-p") + 1]]

    def run(command, **kwargs):
        systemctl_calls.append((tuple(command), kwargs))
        if "set-property" in command:
            for item in command:
                if item.startswith("MemoryMax="):
                    unit["MemoryMax"] = item.split("=", 1)[1]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(BUDGET, "manager", manager)
    monkeypatch.setattr(BUDGET.subprocess, "run", run)
    return home, control, unit, manager_calls, systemctl_calls


def _main(budget, action="reserve"):
    return budget.main(action)


def test_reserve_keeps_exact_partition_and_inactive_main_pid_zero_has_no_cgroup(tmp_path, monkeypatch):
    _, _, unit, manager_calls, calls = _fixture(tmp_path, monkeypatch)

    assert _main(BUDGET) == 0
    assert BUDGET.MODEL_BYTES + BUDGET.FEATURE_BYTES == 100_000_000_000
    assert int(unit["MemoryMax"]) == BUDGET.MODEL_BYTES
    assert BUDGET.DROPIN.read_text() == BUDGET.BODY
    receipt = json.loads(BUDGET.STATE.read_text())
    assert receipt["fits_cancelled"] == 0
    assert receipt["feature_bytes"] + receipt["model_bytes"] == 100_000_000_000
    assert all(call[0][4] == BUDGET.MODEL_UNIT for call in calls if "set-property" in call[0])
    assert not any(call[0] == "freeze" for call in manager_calls)
    assert not BUDGET.MODEL_CAMPAIGN.exists()


def test_capacity_denial_precedes_metadata_directories_and_set_property(tmp_path, monkeypatch):
    home, control, _, _, calls = _fixture(tmp_path, monkeypatch)
    def denied_authority(path):
        return REAL_AUTHORITY(path, headroom_bytes=10**30, headroom_inodes=0)

    monkeypatch.setattr(BUDGET.research_storage, "StorageAuthority", denied_authority)

    assert _main(BUDGET) == 2
    assert not BUDGET.ROOT.exists()
    assert not BUDGET.DROPIN.parent.exists()
    assert not any("set-property" in command for command, _ in calls)
    assert (control / "state.json").exists() is False


def test_atomic_metadata_replace_failure_preserves_old_state_and_cleans_owned_temp(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    BUDGET.ROOT.mkdir()
    BUDGET.STATE.write_text('{"status":"old"}')
    old_replace = BUDGET.os.replace

    @contextlib.contextmanager
    def admitted(*args, **kwargs):
        with BUDGET.research_storage.admitted_output(
                BUDGET.OWN_HOME, estimated_bytes=1_048_700, stage="test", inodes=2):
            yield

    monkeypatch.setattr(BUDGET, "_admit_metadata", admitted)
    with BUDGET._admit_metadata((b'{"status":"new"}',), []):
        def fail_target_replace(source, target):
            if Path(target) == BUDGET.STATE:
                raise OSError("replace failed")
            return old_replace(source, target)

        monkeypatch.setattr(BUDGET.os, "replace", fail_target_replace)
        with pytest.raises(OSError, match="replace failed"):
            BUDGET._atomic_owned_write(BUDGET.STATE, b'{"status":"new"}')
    monkeypatch.setattr(BUDGET.os, "replace", old_replace)
    assert BUDGET.STATE.read_text() == '{"status":"old"}'
    assert list(BUDGET.ROOT.glob(".*.tmp")) == []


def test_replaced_budget_staging_is_neither_published_nor_removed(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    BUDGET.ROOT.mkdir()
    BUDGET.STATE.write_text('accepted old state')
    real_write = BUDGET.research_storage.ReservedFile.write

    def replaced(stream, data):
        result = real_write(stream, data)
        for candidate in BUDGET.ROOT.glob('.feature-exploration-budget.json.*.tmp'):
            if os.path.samestat(os.fstat(stream.fileno()), candidate.stat()):
                candidate.unlink()
                candidate.write_bytes(b'replacement is held')
                break
        return result

    with BUDGET._admit_metadata((b'new state',), []):
        monkeypatch.setattr(BUDGET.research_storage.ReservedFile, 'write', replaced)
        with pytest.raises(RuntimeError, match='staging identity changed'):
            BUDGET._atomic_owned_write(BUDGET.STATE, b'new state')
    assert BUDGET.STATE.read_text() == 'accepted old state'
    held = list(BUDGET.ROOT.glob('.feature-exploration-budget.json.*.tmp'))
    assert len(held) == 1 and held[0].read_bytes() == b'replacement is held'


def test_inactive_main_start_race_is_detected_before_creating_metadata(tmp_path, monkeypatch):
    home, _, unit, _, calls = _fixture(tmp_path, monkeypatch)
    real_admission = BUDGET._admit_metadata

    @contextlib.contextmanager
    def raced_admission(*args, **kwargs):
        with real_admission(*args, **kwargs):
            unit["ActiveState"] = "active"
            unit["MainPID"] = "4321"
            unit["ControlGroup"] = "/user.slice/user-1001.slice/ima.service"
            yield

    monkeypatch.setattr(BUDGET, "_admit_metadata", raced_admission)
    with pytest.raises(RuntimeError, match="Main became active"):
        _main(BUDGET)
    assert not BUDGET.ROOT.exists()
    assert not BUDGET.DROPIN.parent.exists()
    assert not any("set-property" in command for command, _ in calls)
    assert home.is_dir()


def test_thaw_failure_does_not_replace_primary_error(monkeypatch):
    states = iter(["running", "frozen"])
    calls = []

    def manager(*args):
        if "-p" in args and args[args.index("-p") + 1] == "ActiveState":
            return "active"
        if "-p" in args and args[args.index("-p") + 1] == "FreezerState":
            return next(states)
        raise AssertionError(args)

    def run(*args, **kwargs):
        calls.append(args)
        if args[0] == "thaw":
            raise OSError("thaw failed")

    monkeypatch.setattr(BUDGET, "manager", manager)
    monkeypatch.setattr(BUDGET, "_run_systemctl", run)
    primary = RuntimeError("primary operation failed")
    with pytest.raises(RuntimeError, match="primary operation failed") as raised:
        with BUDGET.frozen_model():
            raise primary
    assert raised.value is primary
    assert any("model thaw also failed" in note for note in getattr(primary, "__notes__", []))
    assert len(calls) == 2


def test_systemctl_mutations_are_limited_to_fixed_model_unit():
    with pytest.raises(RuntimeError, match="fixed feature-budget scope"):
        BUDGET._run_systemctl("restart", BUDGET.MODEL_UNIT)
    with pytest.raises(RuntimeError, match="fixed feature-budget scope"):
        BUDGET._run_systemctl("set-property", "--runtime", "other.service",
                              f"MemoryMax={BUDGET.MODEL_BYTES}", "CPUQuota=2400%")
    with pytest.raises(RuntimeError, match="fixed model unit"):
        BUDGET.manager("show", "other.service", "-p", "MainPID", "--value")


def test_status_is_readonly_and_does_not_require_storage_environment(tmp_path, monkeypatch, capsys):
    _fixture(tmp_path, monkeypatch)
    BUDGET.STATE.parent.mkdir(parents=True)
    BUDGET.STATE.write_text('{"status":"reserved"}')
    monkeypatch.delenv("IMA_STORAGE_ROOT")

    assert _main(BUDGET, "status") == 0
    assert capsys.readouterr().out.strip() == '{"status":"reserved"}'


def test_mutation_requires_exact_canonical_storage_root_before_any_metadata(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("IMA_STORAGE_ROOT", str(BUDGET.STORAGE_ROOT) + "/alias")

    with pytest.raises(RuntimeError, match="canonical IMA_STORAGE_ROOT"):
        _main(BUDGET)
    assert not BUDGET.ROOT.exists()
    assert not BUDGET.DROPIN.parent.exists()


def test_owner_identity_helper_keeps_production_uid_and_home_guard():
    BUDGET.require_owner_identity(1001, Path("/home/imaopt"))
    with pytest.raises(RuntimeError, match="own imaopt user"):
        BUDGET.require_owner_identity(501, Path("/home/imaopt"))
