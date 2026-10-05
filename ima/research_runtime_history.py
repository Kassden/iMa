"""Durable, own-campaign runtime observations; interruption is not an OOM diagnosis."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,199}\.service\Z")
_COUNTERS = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
_OOM_COUNTERS = {"oom", "oom_kill", "oom_group_kill"}
_SYSTEMD_EXES = {"/usr/lib/systemd/systemd", "/lib/systemd/systemd", "/usr/libexec/systemd/systemd"}


def _safe_unit(value):
    if value is not None and not _UNIT.fullmatch(value):
        raise ValueError("Expected one literal systemd service unit")
    return value


def read_own_unit_journal(unit, *, lines=200):
    """Bounded read only after systemd confirms the unit runs as this effective user."""
    _safe_unit(unit)
    if unit is None:
        return []
    try:
        current = os.geteuid()
        user_scope = False
        try:
            loaded = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=LoadState", "--value"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            user_scope = loaded == "loaded"
        except subprocess.SubprocessError:
            pass
        if user_scope:
            scope = ["--user", "--user-unit=" + unit]
        else:
            owner = subprocess.run(
                ["systemctl", "show", unit, "--property=User", "--value"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
            if not owner or (owner != str(current) and owner != pwd.getpwuid(current).pw_name):
                return []
            scope = ["--unit=" + unit]
        result = subprocess.run(
            ["journalctl", *scope, "--output=json", "--no-pager",
             "--lines=" + str(min(200, max(1, lines)))],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return []


def _cgroup_sample(raw):
    if not raw.get("supported"):
        return None
    events = raw.get("memory.events", {})
    if isinstance(events, str):
        events = dict(line.split() for line in events.splitlines() if line.strip())
    counters = {name: int(events.get(name, 0)) for name in _COUNTERS}
    current = int(raw.get("memory.current", 0))
    peak = int(raw.get("memory.peak", current))
    if min(current, peak, *counters.values()) < 0:
        raise ValueError("Negative cgroup observation")
    return {"path": str(raw.get("path", "unknown")),
            "boot_id": raw.get("boot_id"), "current_bytes": current,
            "peak_bytes": max(current, peak), "counters": counters,
            "limit_bytes": raw.get("memory.max")}


class RuntimeHistory:
    """One controller owns this file. All methods return JSON-serializable snapshots.

    startup interrupts previously observed active attempts exactly once. Additional
    interrupted_attempts can come from durable ledger recovery. observe replaces
    active IDs when supplied. Cgroup peaks are unit-wide, never per-fit samples.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.invocation_id = None
        self.unit = _safe_unit(os.environ.get("IMA_RESEARCH_SYSTEMD_UNIT") or None)

    def _read(self):
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            if payload["schema_version"] != 1:
                raise ValueError("Unsupported runtime-history version")
            return payload
        return {"schema_version": 1, "current_invocation_id": None,
                "invocations": [], "interrupted_attempts": [], "oom_facts": [],
                "cgroup_peak_bytes": 0, "cgroup_counter_totals": dict.fromkeys(_COUNTERS, 0),
                "cgroup_scopes": {}}

    def snapshot(self):
        result = self._read()
        result["restart_count"] = max(0, len(result["invocations"]) - 1)
        result["interruption_count"] = len(result["interrupted_attempts"])
        return result

    def _publish(self, payload):
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + "-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def startup(self, *, invocation_id=None, interrupted_attempts=(), cgroup=None,
                journal_entries=None, now=None):
        """Use systemd INVOCATION_ID when available, else one UUID per controller."""
        invocation_id = invocation_id or os.environ.get("INVOCATION_ID") or self.invocation_id or uuid.uuid4().hex
        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("Nonempty invocation ID required")
        return self._update(invocation_id, startup=True, interrupted_attempts=interrupted_attempts,
                            cgroup=cgroup, journal_entries=journal_entries, now=now)

    def observe(self, *, running_attempts=None, interrupted_attempts=(), attempt_measurements=None,
                cgroup=None, journal_entries=None, now=None):
        """attempt_measurements maps active IDs to per-fit JobMeasurement dicts.

        Unit-wide cgroup peaks must not be passed as per-fit private measurements.
        """
        if self.invocation_id is None:
            raise RuntimeError("Call startup before observe")
        return self._update(self.invocation_id, startup=False, running_attempts=running_attempts,
                            interrupted_attempts=interrupted_attempts, attempt_measurements=attempt_measurements, cgroup=cgroup,
                            journal_entries=journal_entries, now=now)

    def _update(self, invocation_id, *, startup, interrupted_attempts, cgroup,
                journal_entries, now, running_attempts=None, attempt_measurements=None):
        now = time.time() if now is None else float(now)
        if not math.isfinite(now) or now < 0:
            raise ValueError("Finite nonnegative observation time required")
        interrupted_attempts = tuple(interrupted_attempts)
        if running_attempts is not None:
            running_attempts = tuple(running_attempts)
            if any(not isinstance(attempt, str) or not attempt for attempt in running_attempts):
                raise ValueError("Nonempty attempt IDs required")
        # Observation imports are lazy so fixtures need neither psutil nor Linux.
        if cgroup is None:
            from .research_resources import observe_cgroup
            cgroup = observe_cgroup()
        sample = _cgroup_sample(cgroup)
        # Journal I/O belongs to the caller's bounded polling schedule.
        entries = () if journal_entries is None else journal_entries
        with open(str(self.path) + ".lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                payload = self._read()
                previous = next((row for row in payload["invocations"]
                                 if row["invocation_id"] == payload["current_invocation_id"]), None)
                invocation = next((row for row in payload["invocations"]
                                   if row["invocation_id"] == invocation_id), None)
                if invocation is None:
                    if not startup:
                        raise RuntimeError("Invocation missing from runtime history")
                    if previous:
                        self._interrupt(payload, previous["invocation_id"], previous["running_attempts"], now, "restart", previous.get("attempt_measurements", {}))
                        previous["running_attempts"] = []
                    invocation = {"invocation_id": invocation_id, "started_at": now,
                                  "previous_invocation_id": previous["invocation_id"] if previous else None,
                                  "last_observed_at": now, "running_attempts": [],
                                  "attempt_measurements": {}, "cgroup_peak_bytes": 0,
                                  "cgroup_first": None, "cgroup": None}
                    payload["invocations"].append(invocation)
                elif previous and previous is not invocation:
                    raise RuntimeError("Cannot observe an obsolete invocation")
                interrupted_owner = (invocation["previous_invocation_id"] or invocation_id) if startup else invocation_id
                payload["current_invocation_id"] = invocation_id
                invocation["last_observed_at"] = now
                if running_attempts is not None:
                    invocation["running_attempts"] = sorted(set(running_attempts))
                if attempt_measurements:
                    from .research_resources import JobMeasurement
                    for attempt, actual in attempt_measurements.items():
                        if attempt not in invocation["running_attempts"]:
                            raise ValueError("Measurement must belong to an active attempt")
                        measurement = JobMeasurement(**actual)
                        prior = invocation["attempt_measurements"].get(attempt)
                        if prior:
                            if prior["workload_fingerprint"] != measurement.workload_fingerprint:
                                raise ValueError("Active attempt workload cannot change")
                            actual = dict(actual, private_peak_bytes=max(prior["private_peak_bytes"], measurement.private_peak_bytes),
                                          wall_seconds=max(prior["wall_seconds"], measurement.wall_seconds))
                        invocation["attempt_measurements"][attempt] = dict(actual)
                self._interrupt(payload, interrupted_owner, interrupted_attempts, now,
                                "startup_recovery" if startup else "controller",
                                next((row for row in payload["invocations"] if row["invocation_id"] == interrupted_owner), invocation)["attempt_measurements"])
                interrupted = set(interrupted_attempts)
                invocation["running_attempts"] = [attempt for attempt in invocation["running_attempts"] if attempt not in interrupted]
                invocation["attempt_measurements"] = {attempt: actual for attempt, actual in invocation["attempt_measurements"].items()
                                                       if attempt in invocation["running_attempts"]}
                if sample is not None:
                    self._observe_cgroup(payload, invocation, sample, now)
                self._observe_journal(payload, entries)
                self._publish(payload)
                self.invocation_id = invocation_id
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return self.snapshot()

    @staticmethod
    def _interrupt(payload, invocation_id, attempts, now, source, measurements=None):
        known = {(row["invocation_id"], row["attempt_id"]) for row in payload["interrupted_attempts"]}
        for attempt in sorted(set(attempts)):
            if not isinstance(attempt, str) or not attempt:
                raise ValueError("Nonempty attempt IDs required")
            if (invocation_id, attempt) not in known:
                payload["interrupted_attempts"].append({"invocation_id": invocation_id,
                    "attempt_id": attempt, "observed_at": now, "source": source,
                    "actual": dict(measurements[attempt], failed=True, censored=True)
                    if measurements and attempt in measurements else None})

    @staticmethod
    def _observe_cgroup(payload, invocation, sample, now):
        invocation["cgroup"] = sample
        if invocation["cgroup_first"] is None:
            invocation["cgroup_first"] = sample
        invocation["cgroup_peak_bytes"] = max(invocation["cgroup_peak_bytes"], sample["peak_bytes"])
        payload["cgroup_peak_bytes"] = max(payload["cgroup_peak_bytes"], sample["peak_bytes"])
        scope_id = json.dumps([sample["boot_id"], sample["path"]])
        scope = payload["cgroup_scopes"].setdefault(scope_id, {"generation": 0, "counters": dict.fromkeys(_COUNTERS, 0)})
        if any(sample["counters"][name] < scope["counters"][name] for name in _COUNTERS):
            scope["generation"] += 1
            scope["counters"] = dict.fromkeys(_COUNTERS, 0)
        for name in _COUNTERS:
            before, after = scope["counters"][name], sample["counters"][name]
            if after > before:
                payload["cgroup_counter_totals"][name] += after - before
            if after > before and name in _OOM_COUNTERS:
                payload["oom_facts"].append({"fact_id": hashlib.sha256(
                    json.dumps([scope_id, scope["generation"], name, before, after]).encode()).hexdigest(),
                    "source": "cgroup_counter", "observed_invocation_id": invocation["invocation_id"],
                    "observed_at": now, "counter": name, "delta": after - before,
                    "before": before, "after": after, "path": sample["path"],
                    "peak_bytes": sample["peak_bytes"], "victim_attempt_id": None})
        scope["counters"] = sample["counters"]

    def _observe_journal(self, payload, entries):
        if self.unit is None:
            return
        known = {row["fact_id"] for row in payload["oom_facts"]}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Only a systemd manager can attest a unit result; application text
            # containing "oom" or a running ledger row is not such evidence.
            executable = entry.get("_EXE") or entry.get("EXE")
            manager = entry.get("_COMM") == "systemd" or executable in _SYSTEMD_EXES
            if not manager or (executable is not None and executable not in _SYSTEMD_EXES):
                continue
            system_manager = (str(entry.get("_PID")) == "1" and str(entry.get("_UID")) == "0"
                              and entry.get("UNIT") == self.unit)
            user_manager = (str(entry.get("_UID")) == str(os.geteuid())
                            and str(entry.get("_PID", "")).isdigit() and int(entry["_PID"]) > 1
                            and (entry.get("USER_UNIT") == self.unit or entry.get("_SYSTEMD_USER_UNIT") == self.unit))
            if not (system_manager or user_manager):
                continue
            message = entry.get("MESSAGE", "")
            if not isinstance(message, str):
                continue
            actual_oom = (entry.get("SERVICE_RESULT") == "oom-kill" or entry.get("RESULT") == "oom-kill"
                          or bool(re.search(r"(?:A process of this unit has been killed by the OOM killer\.|The kernel OOM killer killed some processes in this unit\.|Failed with result ['\"]?oom-kill['\"]?(?:\.|$))", message)))
            if not actual_oom:
                continue
            identity = entry.get("__CURSOR") or json.dumps(entry, sort_keys=True)
            fact_id = "journal:" + hashlib.sha256(str(identity).encode()).hexdigest()
            if fact_id not in known:
                payload["oom_facts"].append({"fact_id": fact_id, "source": "own_unit_journal",
                    "unit": self.unit, "timestamp_usec": entry.get("__REALTIME_TIMESTAMP"),
                    "invocation_id": entry.get("INVOCATION_ID") or entry.get("_SYSTEMD_INVOCATION_ID"),
                    "message": message, "victim_attempt_id": None})
                known.add(fact_id)
