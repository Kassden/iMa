"""Point-in-time, read-only Linux process-reference inspection for cache payloads.

This is diagnostic evidence only.  It cannot establish ongoing quiescence or
authorize reclamation.
"""

from __future__ import annotations

import os
import math
import re
import stat
import sys
import time
from pathlib import Path

_ROOT = "/home/imaopt/research-v2/campaigns"
_UID = 1001
_FILES = {"arrays.joblib", "fitted.joblib", "manifest.json"}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_DEV = re.compile(r"([0-9a-fA-F]+):([0-9a-fA-F]+)\Z")
_VISIBILITY_LIMITS = ["Checks visible procfs namespace and PID 1 identities; external host-level filtering cannot be proven."]


def _record_identity(record: dict, *, root_path: str | None = None) -> tuple[dict, list[dict]]:
    """Validate an identity record against a supplied root (private for fixtures)."""
    root = Path(root_path if root_path is not None else record["root"])
    path = Path(record["path"])
    relative = path.relative_to(root) if path.is_relative_to(root) else None
    if not root.is_absolute() or not path.is_absolute() or relative is None or len(relative.parts) != 3 or relative.parts[1] != "fold-preparation-cache" or not _HEX.fullmatch(path.name):
        raise ValueError("payload path is outside the exact fold-cache shape")
    if str(root) != record["root"] or path != root.joinpath(*relative.parts):
        raise ValueError("payload path is not rooted at the recorded campaign root")
    if root_path is None and (str(root) != _ROOT or os.path.realpath(root) != _ROOT):
        raise ValueError("production inspection requires the canonical campaign root")
    file_records = record.get("files", [])
    if record.get("uid") != _UID or len(file_records) != 3 or {item.get("name") for item in file_records} != _FILES:
        raise ValueError("payload identity must contain exactly the expected uid and files")
    expected = {"root": str(root), "path": str(path), "uid": _UID,
                "dev": record.get("dev"), "ino": record.get("ino")}
    for item in record["files"]:
        if item.get("uid") != _UID or not all(isinstance(item.get(k), int) for k in ("dev", "ino", "size", "mtime_ns", "allocated_bytes")):
            raise ValueError("invalid payload file identity")
    return expected, record["files"]


def _same(st: os.stat_result, dev: int, ino: int) -> bool:
    return st.st_dev == dev and st.st_ino == ino


def _validate_candidate(record: dict, root_path: str, *, lstat=os.lstat) -> None:
    expected, files = _record_identity(record, root_path=root_path)
    root = Path(expected["root"])
    path = Path(expected["path"])
    # Every component from / through each payload file is inspected without following links.
    components = [Path("/")]
    current = Path("/")
    for part in root.parts[1:] + (path.relative_to(root)).parts:
        current = current / part
        components.append(current)
    stats = []
    for component in components:
        st = lstat(component)
        if stat.S_ISLNK(st.st_mode):
            raise ValueError("symlink in payload path")
        stats.append(st)
    root_st = lstat(root)
    if not stat.S_ISDIR(root_st.st_mode) or root_st.st_uid != _UID:
        raise ValueError("campaign root is not an owned directory")
    campaign_st = lstat(root / relative_campaign(record, root))
    cache_st = lstat(path.parent)
    for label, st in (("campaign", campaign_st), ("cache", cache_st)):
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != _UID or stat.S_IMODE(st.st_mode) & 0o022:
            raise ValueError(f"{label} ancestor has unsafe owner or directory mode")
    # Root, campaign, cache, and candidate must remain on one filesystem.
    target_st = lstat(path)
    if (not stat.S_ISDIR(target_st.st_mode) or target_st.st_uid != _UID
            or stat.S_IMODE(target_st.st_mode) & 0o022
            or not _same(target_st, expected["dev"], expected["ino"])):
        raise ValueError("payload directory identity or mode changed")
    # The approved root may itself be a mount. Reject crossings beneath it;
    # ancestors above that explicit boundary need not use its filesystem.
    if any(st.st_dev != root_st.st_dev for st in stats[len(root.parts)-1:]):
        raise ValueError("payload path crosses filesystems")
    entries = {p.name: p for p in path.iterdir()}
    if set(entries) != _FILES:
        raise ValueError("payload directory contents changed")
    for item in files:
        st = lstat(entries[item["name"]])
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_uid != _UID:
            raise ValueError("payload file is not an owned single-link regular file")
        if not _same(st, item["dev"], item["ino"]) or st.st_dev != root_st.st_dev:
            raise ValueError("payload file identity changed or crosses filesystems")
        if (st.st_size, st.st_mtime_ns, st.st_blocks * 512) != (item["size"], item["mtime_ns"], item["allocated_bytes"]):
            raise ValueError("payload file metadata changed")


def relative_campaign(record: dict, root: Path) -> str:
    """Return the campaign path component after the constrained shape check."""
    return Path(record["path"]).relative_to(root).parts[0]


def validate_candidate(record: dict) -> None:
    """Validate the production candidate; fixtures should exercise the private helper."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("candidate validation requires Linux")
    if os.geteuid() != 0:
        raise PermissionError("candidate validation requires effective uid 0")
    _record_identity(record)
    _validate_candidate(record, _ROOT)


def _proc_identity(pid_dir: str) -> tuple[str, str]:
    """Return starttime and state to detect PID reuse and classify vanished tasks."""
    raw = Path(pid_dir, "stat").read_text()
    tail = raw[raw.rfind(")") + 2:].split()
    if len(tail) < 20:
        raise ValueError("malformed proc stat")
    return tail[19], tail[0]


def _proc_scan(record: dict, *, proc_root: str, deadline: float,
               monotonic=time.monotonic, listdir=os.listdir, read_text=None,
               stat_path=os.stat, proc_identity=_proc_identity,
               scandir=os.scandir) -> dict:
    """Scan fixtureable procfs; no file contents are opened or read."""
    identities = {(f["dev"], f["ino"]): f["name"] for f in record["files"]}
    identities[(record["dev"], record["ino"])] = "payload_directory"
    refs, errors, ref_count, error_count = [], [], 0, 0
    initial_identities = {}

    def err(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 16:
            errors.append(message[:240])

    def check_deadline():
        if monotonic() >= deadline:
            raise TimeoutError("inspection deadline exceeded")

    try:
        names = listdir(proc_root)
    except OSError as exc:
        return {"complete": False, "references": [], "reference_count": 0,
                "errors": [f"proc enumeration: {exc}"[:240]], "error_count": 1}
    pids = sorted((n for n in names if n.isdigit()), key=int)
    for name in pids:
        if monotonic() >= deadline:
            err("inspection deadline exceeded")
            break
        base = os.path.join(proc_root, name)
        try:
            before, state = proc_identity(base)
            if state in {"Z", "X", "x"}:
                continue
            initial_identities[name] = before
            pidrefs = []
            try:
                map_path = os.path.join(base, "maps")
                if read_text is not None:
                    map_lines = iter(read_text(map_path).splitlines())
                else:
                    maps_stream = open(map_path, "r", encoding="ascii", errors="replace")
                    map_lines = maps_stream
                try:
                  for line in map_lines:
                    check_deadline()
                    fields = line.split(None, 5)
                    if len(fields) >= 5 and fields[4].isdigit():
                        devm = _DEV.fullmatch(fields[3])
                        if devm:
                            key = (os.makedev(int(devm[1], 16), int(devm[2], 16)), int(fields[4]))
                            if key in identities:
                                pidrefs.append({"pid": int(name), "kind": "mapping", "file": identities[key],
                                                "alias": (fields[5] if len(fields) > 5 else "")[:256]})
                finally:
                    if read_text is None:
                        maps_stream.close()
            except TimeoutError as exc:
                err(str(exc))
            except OSError as exc:
                err(f"pid {name} maps unreadable: {exc}")
            try:
                fd_path = os.path.join(base, "fd")
                # Keep the procfs directory handle open while inspecting entries.
                # For this process, that handle itself appears in /proc/self/fd;
                # closing it before stat would turn a valid entry into a false race.
                with scandir(fd_path) as entries:
                    for entry in entries:
                        fd = entry.name
                        check_deadline()
                        try:
                            st = stat_path(os.path.join(fd_path, fd))
                        except FileNotFoundError:
                            err(f"pid {name} fd vanished during scan")
                            continue
                        key = (st.st_dev, st.st_ino)
                        if key in identities:
                            pidrefs.append({"pid": int(name), "kind": "fd", "file": identities[key],
                                            "fd": fd})
            except TimeoutError as exc:
                err(str(exc))
            except OSError as exc:
                err(f"pid {name} fd table unreadable: {exc}")
            after, state_after = proc_identity(base)
            if before != after:
                err(f"pid {name} identity changed during scan")
            elif state_after not in {"Z", "X", "x"}:
                for ref in pidrefs:
                    ref_count += 1
                    if len(refs) < 128:
                        refs.append(ref)
        except FileNotFoundError:
            err(f"pid {name} vanished during scan")
        except PermissionError as exc:
            err(f"active pid {name} unreadable: {exc}")
        except OSError as exc:
            err(f"pid {name} scan failed: {exc}")
    try:
        final_pids = {n for n in listdir(proc_root) if n.isdigit()}
        if final_pids != set(pids):
            err("process set changed during inspection")
    except OSError as exc:
        err(f"final proc enumeration failed: {exc}")
    for name, initial in initial_identities.items():
        if monotonic() >= deadline:
            err("inspection deadline exceeded during final identity verification")
            break
        try:
            final, state = proc_identity(os.path.join(proc_root, name))
            if final != initial:
                err(f"pid {name} identity changed during final verification")
        except (FileNotFoundError, ProcessLookupError):
            err(f"pid {name} vanished during final verification")
        except OSError as exc:
            err(f"pid {name} final identity unreadable: {exc}")
    try:
        final_pids = {n for n in listdir(proc_root) if n.isdigit()}
        if final_pids != set(pids):
            err("process set changed during final identity verification")
    except OSError as exc:
        err(f"final proc enumeration failed: {exc}")
    if monotonic() >= deadline and not any("deadline" in message for message in errors):
        err("inspection deadline exceeded")
    complete = error_count == 0
    return {"complete": complete, "references": refs, "reference_count": ref_count,
            "errors": errors, "error_count": error_count}


def _visibility_check(proc_root: str = "/proc", *, stat_path=os.stat, read_text=None,
                      mounts_path="/proc/mounts", host_root="/") -> None:
    read_text = read_text or (lambda path: Path(path).read_text())
    if not sys.platform.startswith("linux") or read_text(os.path.join(proc_root, "sys/kernel/ostype")).strip() != "Linux":
        raise RuntimeError("Linux procfs unavailable")
    for namespace in ("mnt", "pid"):
        self_ns = stat_path(os.path.join(proc_root, f"self/ns/{namespace}"))
        init_ns = stat_path(os.path.join(proc_root, f"1/ns/{namespace}"))
        if (self_ns.st_dev, self_ns.st_ino) != (init_ns.st_dev, init_ns.st_ino):
            raise RuntimeError(f"inspector and PID 1 use different {namespace} namespaces")
    def nspid(path):
        for line in read_text(path).splitlines():
            if line.startswith("NSpid:"):
                return line.split()[1:]
        raise RuntimeError("NSpid unavailable")
    self_pids = nspid(os.path.join(proc_root, "self/status"))
    init_pids = nspid(os.path.join(proc_root, "1/status"))
    if len(self_pids) != 1 or not init_pids or init_pids[-1] != "1" or os.getpid() != int(self_pids[0]):
        raise RuntimeError("self NSpid does not establish single-entry host PID visibility")
    if read_text(os.path.join(proc_root, "1/comm")).strip() != "systemd":
        raise RuntimeError("PID 1 is not systemd")
    proc_root_stat, host_root_stat = stat_path(os.path.join(proc_root, "1/root")), stat_path(host_root)
    if (proc_root_stat.st_dev, proc_root_stat.st_ino) != (host_root_stat.st_dev, host_root_stat.st_ino):
        raise RuntimeError("PID 1 root filesystem differs from inspector root")
    for line in read_text(mounts_path).splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[2] == "proc" and re.search(r"(?:^|,)hidepid=[12](?:,|$)", fields[3]):
            raise RuntimeError("procfs hidepid restriction detected")


def inspect_payload_references(record, *, deadline_seconds=30):
    """Return bounded point-in-time references to an exact campaign cache payload."""
    captured_at = time.time()
    out = {"process_snapshot_complete": False, "references": [], "reference_count": 0,
           "errors": [], "error_count": 0, "captured_at": captured_at,
           "no_reclaim_authority": True, "inspector_pid": os.getpid(),
           "visibility_limits": list(_VISIBILITY_LIMITS)}
    try:
        if isinstance(deadline_seconds, bool) or not isinstance(deadline_seconds, (int, float)) or not math.isfinite(deadline_seconds):
            raise ValueError("deadline_seconds must be a finite number, not bool")
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        deadline = time.monotonic() + min(float(deadline_seconds), 30.0)
        if not isinstance(record, dict):
            raise ValueError("record must be a mapping")
        validate_candidate(record)
        _visibility_check()
        result = _proc_scan(record, proc_root="/proc", deadline=deadline)
        out.update(process_snapshot_complete=result["complete"], references=result["references"],
                   reference_count=result["reference_count"], errors=result["errors"],
                   error_count=result["error_count"])
    except Exception as exc:
        out["errors"] = [f"{type(exc).__name__}: {exc}"[:240]]
        out["error_count"] = 1
    return out
