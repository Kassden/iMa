import os
from types import SimpleNamespace
from pathlib import Path

import pytest

import ima.research_reclaim_inspection as inspection


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection, "_UID", os.getuid())
    root = tmp_path / "campaigns"
    path = root / "campaign-one" / "fold-preparation-cache" / ("a" * 64)
    path.mkdir(parents=True)
    files = []
    for name in sorted(inspection._FILES):
        target = path / name
        target.write_bytes(b"fixture")
        st = target.lstat()
        files.append({"name": name, "dev": st.st_dev, "ino": st.st_ino,
                      "uid": st.st_uid, "size": st.st_size,
                      "mtime_ns": st.st_mtime_ns,
                      "allocated_bytes": st.st_blocks * 512,
                      "sha256": "unused-by-inspector"})
    st = path.lstat()
    record = {"root": str(root), "path": str(path), "dev": st.st_dev,
              "ino": st.st_ino, "uid": st.st_uid, "files": files,
              "allocated_bytes": st.st_blocks * 512}
    return root, path, record


def _proc_stat(pid, start="1", state="S"):
    # Linux /proc/PID/stat field 22 is starttime (tail index 19).
    fields = [state] + ["0"] * 18 + [start]
    return f"{pid} (fixture process) " + " ".join(fields)


def _proc_fixture(tmp_path, record, *, pid="42", maps=None, fd_names=(), monkeypatch=None):
    proc = tmp_path / "proc"
    base = proc / pid
    (base / "fd").mkdir(parents=True)
    (base / "stat").write_text(_proc_stat(pid))
    (base / "maps").write_text(maps or "")
    target = record["path"]
    for name in fd_names:
        referent = Path(target) if name == "__directory__" else Path(target) / name
        os.symlink(str(referent), base / "fd" / str(name))
    return proc


def _map_line(item, alias="/tmp/renamed-by-bind-mount"):
    dev = os.major(item["dev"]), os.minor(item["dev"])
    return f"7f000000-7f001000 r--p 00000000 {dev[0]:02x}:{dev[1]:02x} {item['ino']} {alias}"


def test_scanner_matches_alias_mapping_by_device_and_inode(payload, tmp_path):
    _, _, record = payload
    target = next(item for item in record["files"] if item["name"] == "arrays.joblib")
    proc = _proc_fixture(tmp_path, record, maps=_map_line(target))
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10)
    assert result["complete"] is True
    assert result["references"] == [{"pid": 42, "kind": "mapping", "file": "arrays.joblib",
                                      "alias": "/tmp/renamed-by-bind-mount"}]


def test_scanner_matches_open_fd_by_identity(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record, fd_names=["manifest.json"])
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10)
    assert result["references"][0]["kind"] == "fd"
    assert result["references"][0]["file"] == "manifest.json"


def test_scanner_matches_payload_directory_open_fd(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record, fd_names=["__directory__"])
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10)
    assert result["references"] == [{"pid": 42, "kind": "fd", "file": "payload_directory", "fd": "__directory__"}]


def test_self_fd_enumerator_stays_open_while_its_entry_is_statted(payload, tmp_path):
    _, _, record = payload
    pid = str(os.getpid())
    proc = _proc_fixture(tmp_path, record, pid=pid)
    target = next(item for item in record["files"] if item["name"] == "arrays.joblib")
    open_state = {"active": False, "stat_seen_open": False}

    class Entries:
        def __enter__(self):
            open_state["active"] = True
            return self
        def __exit__(self, *_):
            open_state["active"] = False
        def __iter__(self):
            return iter([SimpleNamespace(name="900")])

    def fake_scandir(path):
        if path == str(proc / pid / "fd"):
            return Entries()
        return os.scandir(path)

    def fake_stat(path):
        if path == str(proc / pid / "fd" / "900"):
            open_state["stat_seen_open"] = open_state["active"]
            if not open_state["active"]:
                raise FileNotFoundError(path)
            return SimpleNamespace(st_dev=target["dev"], st_ino=target["ino"])
        return os.stat(path)

    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   scandir=fake_scandir, stat_path=fake_stat)
    assert open_state["stat_seen_open"] is True
    assert result["complete"] is True
    assert {"pid": os.getpid(), "kind": "fd", "file": "arrays.joblib", "fd": "900"} in result["references"]


def test_genuinely_vanished_external_fd_still_fails_closed(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)

    class Entries:
        def __enter__(self):
            return iter([SimpleNamespace(name="77")])
        def __exit__(self, *_):
            return False

    def fake_scandir(path):
        if path.endswith("/42/fd"):
            return Entries()
        return os.scandir(path)

    def fake_stat(path):
        if path.endswith("/42/fd/77"):
            raise FileNotFoundError(path)
        return os.stat(path)

    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   scandir=fake_scandir, stat_path=fake_stat)
    assert result["complete"] is False
    assert any("fd vanished" in message for message in result["errors"])


def test_unreadable_active_pid_fails_closed(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    def read(path):
        if path.endswith("/maps"):
            raise PermissionError("denied")
        return Path(path).read_text()
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   read_text=read)
    assert result["complete"] is False
    assert result["error_count"] >= 1


@pytest.mark.parametrize("race", ["vanished", "reused"])
def test_pid_vanishing_or_reuse_fails_closed(payload, tmp_path, race):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    original = inspection._proc_identity
    calls = 0
    def identity(path):
        nonlocal calls
        calls += 1
        if race == "vanished" and calls == 2:
            raise FileNotFoundError(path)
        if race == "reused" and calls == 2:
            return "changed-starttime", "S"
        return original(path)
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   proc_identity=identity)
    assert result["complete"] is False
    assert result["error_count"] >= 1


def test_final_identity_pass_catches_reuse_with_unchanged_pid_set(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    original = inspection._proc_identity
    calls = 0
    def identity(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            return "reused-starttime", "S"
        return original(path)
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   proc_identity=identity)
    assert result["complete"] is False
    assert any("final verification" in message for message in result["errors"])


def test_deadline_exhaustion_and_process_churn_fail_closed(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    ticks = iter([0, 2])
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=1,
                                   monotonic=lambda: next(ticks, 2))
    assert result["complete"] is False
    assert "deadline" in result["errors"][0]


def test_deadline_is_checked_inside_fd_loop(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    fd_dir = proc / "42" / "fd"
    for fd in ("1", "2"):
        os.symlink(str(Path(record["path"]) / "arrays.joblib"), fd_dir / fd)
    real_scandir = os.scandir
    ticks = iter([0, 0, 0, 2])
    class FdEntries:
        def __enter__(self):
            return iter([SimpleNamespace(name="1"), SimpleNamespace(name="2")])
        def __exit__(self, *_):
            return False
    def scandir(path):
        if path == str(fd_dir):
            return FdEntries()
        return real_scandir(path)
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=1,
                                   monotonic=lambda: next(ticks, 2), scandir=scandir)
    assert result["complete"] is False
    assert any("deadline" in item for item in result["errors"])


def test_proc_pid_set_changing_during_scan_fails_closed(payload, tmp_path):
    _, _, record = payload
    proc = _proc_fixture(tmp_path, record)
    real_listdir = os.listdir
    calls = 0
    def listdir(path):
        nonlocal calls
        if path == str(proc):
            calls += 1
            return ["42"] if calls == 1 else ["42", "43"]
        return real_listdir(path)
    result = inspection._proc_scan(record, proc_root=str(proc), deadline=inspection.time.monotonic() + 10,
                                   listdir=listdir)
    assert result["complete"] is False
    assert "process set changed" in result["errors"][0]


def test_candidate_rejects_foreign_uid_symlink_and_hardlink(payload, tmp_path):
    root, path, record = payload
    foreign = {**record, "uid": record["uid"] + 1}
    with pytest.raises(ValueError):
        inspection._validate_candidate(foreign, str(root))

    file_path = path / "arrays.joblib"
    saved = path.parent / "arrays.saved"
    file_path.rename(saved)
    file_path.symlink_to(saved)
    with pytest.raises(ValueError, match="single-link regular"):
        inspection._validate_candidate(record, str(root))
    file_path.unlink()
    saved.rename(file_path)

    alias = path.parent / "alias"
    os.link(file_path, alias)
    with pytest.raises(ValueError, match="single-link regular"):
        inspection._validate_candidate(record, str(root))


def test_production_candidate_validator_accepts_valid_candidate(payload, monkeypatch):
    root, _, record = payload
    monkeypatch.setattr(inspection, "_ROOT", str(root))
    monkeypatch.setattr(inspection.os, "geteuid", lambda: 0)
    monkeypatch.setattr(inspection.sys, "platform", "linux")
    assert inspection.validate_candidate(record) is None


def test_visibility_rejects_pid_namespace_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(inspection.sys, "platform", "linux")
    proc = tmp_path / "proc"
    for path in ("sys/kernel/ostype", "self/status", "1/status", "1/comm"):
        target = proc / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Linux" if path.endswith("ostype") else
                          "NSpid:\t1234\n" if path == "self/status" else
                          "NSpid:\t1\n" if path == "1/status" else "systemd\n")
    def fake_stat(path):
        if path.endswith("ns/mnt"):
            return SimpleNamespace(st_dev=1, st_ino=10)
        if path.endswith("ns/pid"):
            return SimpleNamespace(st_dev=1, st_ino=10 if "/self/" in path else 11)
        return os.stat("/")
    mounts = tmp_path / "mounts"
    mounts.write_text("proc /proc proc rw 0 0\n")
    with pytest.raises(RuntimeError, match="pid namespaces"):
        inspection._visibility_check(str(proc), stat_path=fake_stat,
                                     mounts_path=str(mounts), host_root="/")


@pytest.mark.parametrize("timeout", [True, float("inf"), float("nan")])
def test_public_inspector_rejects_invalid_timeout(timeout):
    result = inspection.inspect_payload_references({}, deadline_seconds=timeout)
    assert result["process_snapshot_complete"] is False
    assert result["error_count"] == 1
    assert "deadline_seconds" in result["errors"][0]


def test_public_result_always_denies_reclaim_authority():
    result = inspection.inspect_payload_references({"bad": "record"})
    assert result["no_reclaim_authority"] is True
    assert result["inspector_pid"] == os.getpid()
    assert result["process_snapshot_complete"] is False
    assert result["error_count"] == 1
