"""Launch-level checks for the shared optimizer/acquisition storage authority."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy/systemd/ima-research-expansion-supervisor"
CANONICAL_STORAGE_ROOT = "/home/imaopt/research-v2/storage-control"


def _fake_runtime(tmp_path: Path) -> tuple[Path, Path]:
    worker = tmp_path / "fake-worker"
    worker.write_text(
        "#!/bin/sh\n"
        "printf '%s' \"$IMA_STORAGE_ROOT\" > \"$IMA_STORAGE_CAPTURE\"\n"
    )
    worker.chmod(0o700)
    interpreter = tmp_path / "fake-python"
    interpreter.write_text(
        "#!/bin/sh\n"
        "if [ \"${1:-}\" = \"-\" ]; then\n"
        "  exec \"$IMA_FAKE_WORKER\"\n"
        "fi\n"
        "exit 97\n"
    )
    interpreter.chmod(0o700)
    return interpreter, worker


def _supervisor_env(tmp_path: Path, interpreter: Path) -> dict[str, str]:
    release = tmp_path / "release"
    release.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "IMA_V6_RELEASE": str(release),
            "IMA_V6_REVISION": "a" * 40,
            "IMA_V6_PYTHON": str(interpreter),
            "IMA_V6_DEPENDENCIES": str(tmp_path / "dependencies"),
            "IMA_V6_CAMPAIGN": str(tmp_path / "campaign"),
            "IMA_V6_CONFIG": str(tmp_path / "config.json"),
            "IMA_V6_RESEARCH_POLICY": "expansion_v6",
            "IMA_CACHE_HOLD_POLICY": "/reviewed-fixture-policy.json",
            "IMA_CACHE_HOLD_POLICY_SHA256": "b" * 64,
        }
    )
    return env


def test_supervisor_rejects_missing_storage_root_before_interpreter_or_campaign(tmp_path):
    interpreter, worker = _fake_runtime(tmp_path)
    env = _supervisor_env(tmp_path, interpreter)
    env.pop("IMA_STORAGE_ROOT", None)
    env["IMA_FAKE_WORKER"] = str(worker)
    env["IMA_STORAGE_CAPTURE"] = str(tmp_path / "worker-env")

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check"], env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "IMA_STORAGE_ROOT" in result.stderr
    assert not (tmp_path / "worker-env").exists()
    assert not (tmp_path / "campaign").exists()


def test_supervisor_passes_canonical_storage_root_to_child_process(tmp_path):
    interpreter, worker = _fake_runtime(tmp_path)
    env = _supervisor_env(tmp_path, interpreter)
    capture = tmp_path / "worker-env"
    env.update(
        {
            "IMA_STORAGE_ROOT": CANONICAL_STORAGE_ROOT,
            "IMA_FAKE_WORKER": str(worker),
            "IMA_STORAGE_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check"], env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text() == CANONICAL_STORAGE_ROOT


def test_research_expansion_rejects_noncanonical_storage_root_before_python(tmp_path):
    interpreter, worker = _fake_runtime(tmp_path)
    env = _supervisor_env(tmp_path, interpreter)
    env.update(
        {
            "IMA_STORAGE_ROOT": str(tmp_path / "other-root"),
            "IMA_FAKE_WORKER": str(worker),
            "IMA_STORAGE_CAPTURE": str(tmp_path / "worker-env"),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER), "--check"], env=env, text=True, capture_output=True
    )

    assert result.returncode == 78
    assert "canonical own-user research-v2 storage-control root" in result.stderr
    assert not (tmp_path / "worker-env").exists()


def test_acquisition_services_set_the_same_scoped_storage_root():
    for service_name in (
        "ima-hkjc-acquisition.service",
        "ima-hkjc-snapshot.service",
    ):
        service = (ROOT / "deploy/systemd" / service_name).read_text()
        assert (
            f"Environment=IMA_STORAGE_ROOT={CANONICAL_STORAGE_ROOT}" in service
        ), service_name


def test_supervisor_rejects_unpinned_holds_before_interpreter_or_campaign(tmp_path):
    interpreter,worker=_fake_runtime(tmp_path)
    env=_supervisor_env(tmp_path,interpreter)
    env.update(IMA_STORAGE_ROOT=CANONICAL_STORAGE_ROOT,IMA_FAKE_WORKER=str(worker),
               IMA_STORAGE_CAPTURE=str(tmp_path/'worker-env'))
    for missing in ('IMA_CACHE_HOLD_POLICY','IMA_CACHE_HOLD_POLICY_SHA256'):
        selected=dict(env);selected.pop(missing)
        result=subprocess.run(['bash',str(LAUNCHER),'--check'],env=selected,text=True,capture_output=True)
        assert result.returncode!=0 and missing in result.stderr
        assert not (tmp_path/'worker-env').exists() and not (tmp_path/'campaign').exists()
