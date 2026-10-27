"""Bound the own-user feature worker partition without restarting model fits."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path("/home/imaopt/research-v2")
OWN_HOME = Path("/home/imaopt")
STORAGE_ROOT = Path("/home/imaopt/research-v2/storage-control")
MODEL_UNIT = "ima-v7-openrouter-recovery-20261007.service"
MODEL_CAMPAIGN = ROOT / "campaigns/agentic_v7_openrouter_recovery_20261007"
STATE = ROOT / "feature-exploration-budget.json"
DROPIN = OWN_HOME / ".config/systemd/user" / (MODEL_UNIT + ".d") / "95-feature-exploration-budget.conf"
TOTAL_BYTES = 100_000_000_000
FEATURE_BYTES = 8_000_000_000
MODEL_BYTES = TOTAL_BYTES - FEATURE_BYTES
METADATA_OPERATION_BYTES = 1_048_576
METADATA_OPERATION_INODES = 16
CONTROL_TARGETS = 2
ATOMIC_REPLACEMENT_TEMP_INODES = 1
BODY = f"# Owned by ima-feature-exploration-budget\n[Service]\nMemoryMax={MODEL_BYTES}\nCPUQuota=2400%\n"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # The unit invokes Python with -P, so make this release's low-level package
    # explicit instead of depending on cwd or importing the research controller.
    sys.path.insert(0, str(PROJECT_ROOT))

from ima import research_storage

StorageBlocked = research_storage.StorageBlocked


def require_owner_identity(uid, home):
    """Check identity values separately so tests need not patch process globals."""
    home = Path(home)
    if uid != 1001 or home != Path("/home/imaopt"):
        raise RuntimeError("Feature budget requires own imaopt user")


def require_owner():
    """Enforce the actual production process identity."""
    require_owner_identity(os.getuid(), Path.home())


def partition_safe(status, current_bytes):
    resources = status["resources"]
    committed = (resources["reserved_ram_gib"] + resources["resident_gib"] +
                 resources["emergency_gib"]) * 1024**3
    return max(committed, current_bytes + resources["emergency_gib"] * 1024**3) < MODEL_BYTES


def _fixed_show(args):
    allowed = {"ActiveState", "MainPID", "ControlGroup", "MemoryMax",
               "CPUQuotaPerSecUSec", "FreezerState"}
    return (len(args) == 5 and args[0] == "show" and args[1] == MODEL_UNIT and
            args[2] == "-p" and args[3] in allowed and args[4] == "--value")


def manager(*args):
    if not _fixed_show(args):
        raise RuntimeError("Refusing systemctl query outside the fixed model unit")
    return subprocess.check_output(["systemctl", "--user", *args], text=True).strip()


def _run_systemctl(*args, timeout=None):
    allowed = (args == ("daemon-reload",) or
               args in (("freeze", MODEL_UNIT), ("thaw", MODEL_UNIT)))
    if len(args) >= 4 and args[:3] == ("set-property", "--runtime", MODEL_UNIT):
        properties = args[3:]
        allowed = (len(properties) == 2 and
                   any(value.startswith("MemoryMax=") and
                       value.split("=", 1)[1].isdigit() and
                       0 < int(value.split("=", 1)[1]) <= TOTAL_BYTES
                       for value in properties) and
                   any(value.startswith("CPUQuota=") for value in properties))
    if not allowed:
        raise RuntimeError("Refusing systemctl mutation outside fixed feature-budget scope")
    command = ["systemctl", "--user", *args]
    options = {"check": True}
    if timeout is not None:
        options["timeout"] = timeout
    return subprocess.run(command, **options)


def _attach_cleanup_error(primary, cleanup_error, label):
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        add_note(f"{label} also failed: {cleanup_error!r}")


@contextmanager
def frozen_model():
    """Pause fits when active; prove an inactive main stays down while held."""
    active = manager("show", MODEL_UNIT, "-p", "ActiveState", "--value")
    if active in {"inactive", "failed"}:
        if manager("show", MODEL_UNIT, "-p", "MainPID", "--value") != "0":
            raise RuntimeError("Inactive main reports live PID; hold feature startup")
        if manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value"):
            raise RuntimeError("Inactive main still has a control group")
        primary = None
        try:
            yield
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                if manager("show", MODEL_UNIT, "-p", "ActiveState", "--value") not in {"inactive", "failed"}:
                    raise RuntimeError("Main started during feature budget partition")
                if manager("show", MODEL_UNIT, "-p", "MainPID", "--value") != "0":
                    raise RuntimeError("Main acquired a PID during feature budget partition")
                if manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value"):
                    raise RuntimeError("Main acquired a control group during feature budget partition")
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                _attach_cleanup_error(primary, cleanup_error, "inactive-main recheck")
        return

    if manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") != "running":
        raise RuntimeError("Model is already frozen by another owner")
    freeze_attempted = False
    primary = None
    try:
        freeze_attempted = True
        _run_systemctl("freeze", MODEL_UNIT, timeout=30)
        if manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") != "frozen":
            raise RuntimeError("Model admission pause was not acknowledged")
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if freeze_attempted:
            try:
                _run_systemctl("thaw", MODEL_UNIT, timeout=30)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                _attach_cleanup_error(primary, cleanup_error, "model thaw")


def _validate_mutation_authority():
    configured = os.environ.get("IMA_STORAGE_ROOT")
    if configured != str(STORAGE_ROOT):
        raise RuntimeError("Reserve/restore requires canonical IMA_STORAGE_ROOT")
    if STORAGE_ROOT.is_symlink() or not STORAGE_ROOT.is_dir() or STORAGE_ROOT.resolve() != STORAGE_ROOT:
        raise RuntimeError("Canonical storage authority must already exist as a real directory")


def _planned_directories():
    """Return missing parents beneath OWN_HOME without creating any metadata."""
    if OWN_HOME.resolve() != OWN_HOME or OWN_HOME.is_symlink() or not OWN_HOME.is_dir():
        raise RuntimeError("Expected the canonical own-user home directory")
    required = {STATE.parent, DROPIN.parent}
    missing = set()
    for directory in required:
        directory = Path(os.path.abspath(directory))
        if not directory.is_relative_to(OWN_HOME):
            raise RuntimeError("Feature-budget metadata escaped own-user home")
        cursor = directory
        while cursor != OWN_HOME and not cursor.exists():
            if cursor.is_symlink():
                raise RuntimeError("Symlink in feature-budget metadata path")
            missing.add(cursor)
            cursor = cursor.parent
        if cursor != OWN_HOME and not cursor.is_relative_to(OWN_HOME):
            raise RuntimeError("Feature-budget metadata parent escaped own-user home")
        while cursor != OWN_HOME:
            if cursor.is_symlink() or not cursor.is_dir() or cursor.stat().st_uid != os.getuid():
                raise RuntimeError("Foreign or non-directory feature-budget parent")
            cursor = cursor.parent
    return sorted(missing, key=lambda path: (len(path.parts), str(path)))


def _ensure_directories(directories):
    for directory in directories:
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_uid != os.getuid():
            raise RuntimeError("Unsafe feature-budget metadata directory")


def _json_bytes(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_existing_target(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("Symlink feature-budget metadata target")
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not path.is_file() or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise RuntimeError("Foreign or non-regular feature-budget metadata target")


def _atomic_owned_write(path, payload):
    """Publish one bounded file through the shared reservation and owned cleanup."""
    path = Path(path)
    _validate_existing_target(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    identity = None
    try:
        with research_storage.reserved_output_file(temporary, "xb") as stream:
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino, info.st_uid)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_existing_target(path)
        _validate_existing_target(temporary)
        staged = temporary.lstat()
        if (staged.st_dev, staged.st_ino, staged.st_uid) != identity:
            raise RuntimeError('Feature-budget staging identity changed before publication')
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException as primary:
        try:
            current = temporary.lstat()
            if (identity is not None and current.st_uid == os.getuid() and
                    (current.st_dev, current.st_ino, current.st_uid) == identity):
                temporary.unlink()
                _fsync_directory(temporary.parent)
        except FileNotFoundError:
            pass
        except BaseException as cleanup_error:
            _attach_cleanup_error(primary, cleanup_error, "temporary metadata cleanup")
        raise


def _atomic_owned_remove(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if path.is_symlink() or not path.is_file() or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise RuntimeError("Refusing to remove foreign feature-budget metadata")
    path.unlink()
    _fsync_directory(path.parent)


@contextmanager
def _admit_metadata(payloads, missing_directories):
    known_bytes = sum(len(payload) for payload in payloads)
    with research_storage.admitted_output(
            OWN_HOME,
            estimated_bytes=(known_bytes + METADATA_OPERATION_BYTES +
                             65536 * (len(missing_directories) + CONTROL_TARGETS)),
            stage="feature_budget_control_metadata",
            inodes=(CONTROL_TARGETS + ATOMIC_REPLACEMENT_TEMP_INODES +
                    len(missing_directories) + METADATA_OPERATION_INODES)):
        yield


def _assert_down():
    if manager("show", MODEL_UNIT, "-p", "ActiveState", "--value") not in {"inactive", "failed"}:
        raise RuntimeError("Main became active during feature budget admission")
    if manager("show", MODEL_UNIT, "-p", "MainPID", "--value") != "0":
        raise RuntimeError("Main acquired a PID during feature budget admission")
    if manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value"):
        raise RuntimeError("Main acquired a control group during feature budget admission")


def _assert_group(group, expected_pids, oom_before):
    if group is None:
        _assert_down()
        return
    members = set()
    for path in group.rglob("cgroup.procs"):
        members.update(path.read_text().split())
    members.update((group / "cgroup.procs").read_text().split())
    if members != expected_pids or not all(Path("/proc", pid).exists() for pid in members):
        raise RuntimeError("Model processes changed during feature budget partition")
    if (group / "memory.events").read_text() != oom_before:
        raise RuntimeError("Model memory events changed during feature budget partition")


def _prepare_layout():
    _validate_mutation_authority()
    return _planned_directories()


def _main_is_down():
    active = manager("show", MODEL_UNIT, "-p", "ActiveState", "--value")
    relative = manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value")
    if active in {"inactive", "failed"} and not relative:
        if manager("show", MODEL_UNIT, "-p", "MainPID", "--value") != "0":
            raise RuntimeError("Inactive main reports live PID")
        return True
    return False


def _reserve():
    _validate_existing_target(STATE)
    _validate_existing_target(DROPIN)
    if DROPIN.exists() and DROPIN.read_text() != BODY:
        raise RuntimeError("Unexpected owned budget dropin; do not overwrite")
    old = json.loads(STATE.read_text()) if STATE.exists() else None
    current_limit = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
    if not MODEL_BYTES - 4096 <= current_limit <= TOTAL_BYTES:
        raise RuntimeError("Unexpected model memory limit; no budget mutation")
    prior_limit = (old["previous_memory_max"] if old and old.get("status") in {"reserved", "reserving"}
                   else current_limit)
    prior_cpu = (old["previous_cpu_quota"] if old and old.get("status") in {"reserved", "reserving"}
                 else manager("show", MODEL_UNIT, "-p", "CPUQuotaPerSecUSec", "--value"))

    for _ in range(120):
        inactive = _main_is_down()
        if inactive:
            group = None
            current = 0
            status = {"resources": {"reserved_ram_gib": 0, "resident_gib": 0, "emergency_gib": 0},
                      "active_trials": [], "main_inactive": True}
            break
        status = json.loads((MODEL_CAMPAIGN / "status.json").read_text())
        relative = manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value")
        group = Path("/sys/fs/cgroup") / relative.lstrip("/")
        if not str(group).startswith("/sys/fs/cgroup/user.slice/user-1001.slice/"):
            raise RuntimeError("Not an own-user cgroup")
        current = int((group / "memory.current").read_text())
        if time.time() - datetime.fromisoformat(status["updated_at"]).timestamp() < 120 and partition_safe(status, current):
            break
        print("Waiting for safe feature memory partition; no fits cancelled", flush=True)
        time.sleep(5)
    else:
        raise TimeoutError("No safe budget partition; model fits preserved")

    with frozen_model():
        if group is None:
            _assert_down()
        else:
            status = json.loads((MODEL_CAMPAIGN / "status.json").read_text())
            if not partition_safe(status, int((group / "memory.current").read_text())):
                raise RuntimeError("Memory grew before admission pause; no limit changed")
        members = set() if group is None else {
            pid for path in group.rglob("cgroup.procs") for pid in path.read_text().split()}
        if group is not None:
            members.update((group / "cgroup.procs").read_text().split())
        oom_before = None if group is None else (group / "memory.events").read_text()
        receipt = {"status": "reserving", "previous_memory_max": prior_limit,
                   "previous_cpu_quota": prior_cpu, "model_bytes": MODEL_BYTES,
                   "feature_bytes": FEATURE_BYTES, "total_bytes": TOTAL_BYTES,
                   "main_pid_before": manager("show", MODEL_UNIT, "-p", "MainPID", "--value"),
                   "active_attempts_before": [row["attempt_id"] for row in status.get("active_trials", [])],
                   "created_at": datetime.now(timezone.utc).isoformat()}
        final_receipt = {**receipt, "status": "reserved", "observed_model_memory_max": MODEL_BYTES,
                         "preserved_pids": sorted(members), "admission_pause_acknowledged": True,
                         "memory_events_before": oom_before, "fits_cancelled": 0}
        directories = _prepare_layout()
        payloads = (_json_bytes(receipt), _json_bytes(final_receipt), BODY.encode("utf-8"))
        with _admit_metadata(payloads, directories):
            if group is None:
                _assert_down()
            _ensure_directories(directories)
            _atomic_owned_write(STATE, _json_bytes(receipt))
            if not DROPIN.exists():
                _atomic_owned_write(DROPIN, BODY.encode("utf-8"))
            elif DROPIN.read_text() != BODY:
                raise RuntimeError("Owned budget dropin changed before publication")
            _run_systemctl("daemon-reload")
            _run_systemctl("set-property", "--runtime", MODEL_UNIT,
                           f"MemoryMax={MODEL_BYTES}", "CPUQuota=2400%")
            observed = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
            if not MODEL_BYTES - 4096 <= observed <= MODEL_BYTES:
                raise RuntimeError("Model memory limit readback outside the fixed partition")
            if manager("show", MODEL_UNIT, "-p", "MainPID", "--value") != receipt["main_pid_before"]:
                raise RuntimeError("Model PID changed during feature budget partition")
            _assert_group(group, members, oom_before)
            final_receipt["observed_model_memory_max"] = observed
            _atomic_owned_write(STATE, _json_bytes(final_receipt))
    print(json.dumps(final_receipt), flush=True)


def _restore():
    _validate_existing_target(STATE)
    _validate_existing_target(DROPIN)
    if not STATE.exists():
        return
    receipt = json.loads(STATE.read_text())
    if receipt.get("status") == "restored":
        return
    current_limit = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
    if not (MODEL_BYTES - 4096 <= current_limit <= MODEL_BYTES or
            current_limit == receipt["previous_memory_max"]):
        raise RuntimeError("Model memory limit changed outside feature worker; refusing overwrite")
    if DROPIN.exists() and DROPIN.read_text() != BODY:
        raise RuntimeError("Owned budget dropin changed; refusing restore")
    restored = {**receipt, "status": "restored",
                "restored_at": datetime.now(timezone.utc).isoformat()}
    directories = _prepare_layout()
    payloads = (_json_bytes(restored), BODY.encode("utf-8"))
    with _admit_metadata(payloads, directories):
        if receipt.get("status") == "reserving" and manager(
                "show", MODEL_UNIT, "-p", "FreezerState", "--value") == "frozen":
            _run_systemctl("thaw", MODEL_UNIT, timeout=30)
            if manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") != "running":
                raise RuntimeError("Interrupted reservation freezer recovery was not acknowledged")
        with frozen_model():
            current_limit = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
            if not (MODEL_BYTES - 4096 <= current_limit <= MODEL_BYTES or
                    current_limit == receipt["previous_memory_max"]):
                raise RuntimeError("Model memory limit changed outside feature worker; refusing overwrite")
            previous = receipt["previous_cpu_quota"]
            cpu_quota = "infinity" if previous == "infinity" else f"{float(previous.rstrip('s')) * 100:g}%"
            _run_systemctl("set-property", "--runtime", MODEL_UNIT,
                           f"MemoryMax={receipt['previous_memory_max']}", f"CPUQuota={cpu_quota}")
            if DROPIN.exists():
                if DROPIN.read_text() != BODY:
                    raise RuntimeError("Owned budget dropin changed; refusing restore")
                _atomic_owned_remove(DROPIN)
                _run_systemctl("daemon-reload")
            _atomic_owned_write(STATE, _json_bytes(restored))


def main(action):
    require_owner()
    if action == "status":
        print(STATE.read_text() if STATE.exists() else "{}")
        return 0
    if action not in {"reserve", "restore"}:
        raise ValueError("Expected reserve, restore or status")
    _validate_mutation_authority()
    try:
        if action == "reserve":
            _reserve()
        else:
            _restore()
    except StorageBlocked as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2
    except OSError as exc:
        blocked = research_storage.as_storage_blocked(
            exc, stage="feature_budget_control_metadata", output=STATE)
        if blocked is None:
            raise
        print(str(blocked), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
