"""Individually bounded research workers backed by Pebble's process supervision."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import signal
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, InvalidStateError, TimeoutError
from datetime import datetime, timezone
from pathlib import Path

from .research_resources import observe_process_tree


def read_runtime(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def _update_runtime(path, updates):
    """Serialize read/merge/write and durably publish before execution admission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path)+".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = read_runtime(path)
            changes = updates(state) if callable(updates) else updates
            state.update(changes)
            fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+"-")
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(state, stream, sort_keys=True, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return state
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _validate_timeout(seconds):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Positive finite attempt deadline required")


def remaining_deadline(path, seconds):
    _validate_timeout(seconds)
    state = read_runtime(path)
    deadline = state.get("deadline_at")
    if deadline:
        absolute = datetime.fromisoformat(deadline).timestamp()
        if state.get("progress_policy") and state.get("hard_deadline_at"):
            absolute = min(absolute, datetime.fromisoformat(state["hard_deadline_at"]).timestamp())
        return max(0., absolute-time.time())
    return float(seconds)


def extend_attempt_deadline(path, progress_path=None, *, now=None, task_nonce=None):
    """Atomically observe progress and receipt a bounded soft-deadline extension.

    Admission persists the policy and absolute hard cap. Receipts survive restart;
    resubmission cannot reset either cap or consume old optimizer evidence again.
    """
    from .probit_runtime_policy import normalize_progress_policy, observe_optimizer_progress
    now = time.time() if now is None else now
    progress_path = Path(progress_path) if progress_path else Path(path).parent/"progress.json"
    try:
        progress = read_runtime(progress_path)
    except (OSError, ValueError):
        progress = {}

    def update(state):
        if (not state.get("progress_policy") or state.get("status") != "running"
                or (task_nonce is not None and state.get("task_nonce") != task_nonce)):
            return {}
        policy = normalize_progress_policy(state["progress_policy"])
        changes = observe_optimizer_progress(state, progress, policy, now)
        evidence = changes.get("productive_progress", state.get("productive_progress"))
        deadline = datetime.fromisoformat(state["deadline_at"]).timestamp()
        hard = datetime.fromisoformat(state["hard_deadline_at"]).timestamp()
        receipts = state.get("runtime_extensions", [])
        consumed = receipts[-1]["progress_updated_at_epoch"] if receipts else 0
        if (evidence and len(receipts) < policy.max_extensions and now < hard
                and 0 <= now-evidence["updated_at_epoch"] <= policy.progress_max_age_seconds
                and evidence["updated_at_epoch"] > consumed
                and deadline-now <= policy.extension_window_seconds):
            new_deadline = min(hard, deadline+policy.extension_seconds)
            if new_deadline > max(deadline, now):
                changes.update(deadline_at=datetime.fromtimestamp(new_deadline, timezone.utc).isoformat(),
                    runtime_extensions=receipts+[{"extended_at_epoch": now,
                        "previous_deadline_at": state["deadline_at"],
                        "deadline_at": datetime.fromtimestamp(new_deadline, timezone.utc).isoformat(),
                        "progress_updated_at_epoch": evidence["updated_at_epoch"],
                        "reason": "productive_"+evidence["kind"], "progress": evidence,
                        "workload_fingerprint": policy.workload_fingerprint}])
        return changes

    return _update_runtime(path, update)


def _boot_identity():
    try:
        return "linux:"+Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        import psutil
        return "boot-time:"+str(psutil.boot_time())


def _process_identity(process):
    import psutil
    try:
        return {"pid":process.pid, "create_time":process.create_time(),
                "uid":process.uids().real, "process_group":os.getpgid(process.pid),
                "session_id":os.getsid(process.pid)}
    except (OSError, psutil.Error):
        return None


def _verified_process(identity):
    import psutil
    if not isinstance(identity, dict) or identity.get("uid") != os.getuid():
        return None
    pid, created = identity.get("pid"), identity.get("create_time")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return None
    if not isinstance(created, (int, float)) or not math.isfinite(created) or created <= 0:
        return None
    try:
        process = psutil.Process(pid)
        return process if _process_identity(process) == identity else None
    except psutil.Error:
        return None


def _descendant_identities(process):
    import psutil
    anchor = _process_identity(process)
    if anchor is None or _verified_process(anchor) is None:
        return []
    try:
        children = process.children(recursive=True)
    except psutil.Error:
        return []
    identities = [identity for child in children
                  if (identity := _process_identity(child)) and identity["uid"] == os.getuid()]
    return identities if _verified_process(anchor) is not None else []


def _isolated_group_identities(process):
    """A live birth/nonce-verified session leader also proves its orphaned members."""
    import psutil
    identities = {identity["pid"]:identity for identity in _descendant_identities(process)}
    for member in psutil.process_iter():
        try:
            if member.pid != process.pid and os.getpgid(member.pid) == process.pid and os.getsid(member.pid) == process.pid:
                identity = _process_identity(member)
                if identity and identity["uid"] == os.getuid():
                    identities[member.pid] = identity
        except (OSError,psutil.Error):
            continue
    return list(identities.values())


def _signal_verified(identity, sig):
    """Pin Linux process identity through signaling; psutil guards reuse elsewhere."""
    import psutil
    fd = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            try:
                fd = os.pidfd_open(identity["pid"])
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS, errno.EINVAL}:
                    return
        process = _verified_process(identity)
        if process is None:
            return
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            process.send_signal(sig)
    except (OSError, psutil.Error):
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _task_guard_path(identity, boot_id):
    key = hashlib.sha256(json.dumps([boot_id, identity], sort_keys=True).encode()).hexdigest()
    return Path(tempfile.gettempdir())/f"ima-worker-{os.getuid()}-{key}.json"


def _owned_root_action(state, action, *, active_only=False):
    """Lock worker assignment while checking task ownership and acting on its root."""
    identity = state["process_identity"]
    nonce = state.get("task_nonce")
    if not isinstance(nonce,str) or not nonce:
        return None
    guard = _task_guard_path(identity,state["boot_id"])
    if not guard.is_file():
        return None
    with open(str(guard)+".lock", "a+b") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        try:
            assignment = read_runtime(guard)
            if assignment.get("task_nonce") != nonce or (active_only and assignment.get("status") != "running"):
                return None
            process = _verified_process(identity)
            if process is None:
                return None
            result = action(process)
            return result if _verified_process(identity) is not None else None
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def _timed_call(function, args, kwargs, runtime_path, seconds):
    import psutil
    path = Path(runtime_path)
    previous = read_runtime(path)
    started = datetime.now(timezone.utc)
    if remaining_deadline(path, seconds) <= 0:
        raise TimeoutError("Persisted attempt deadline expired before execution")
    if os.name == "posix" and os.getpgrp() != os.getpid():
        os.setsid()
    process = psutil.Process()
    identity = _process_identity(process)
    if identity is None or identity["process_group"] != process.pid or identity["session_id"] != process.pid:
        raise RuntimeError("Attempt process isolation could not be proved")
    state = {"schema_version":1,"pid":os.getpid(),"process_group":os.getpgrp(),
             "started_at":previous.get("started_at",started.isoformat()),
             "resumed_at":started.isoformat(),
             "status":"running","stage":"training",
             "private_peak_bytes":previous.get("private_peak_bytes",0),
             "boot_id":_boot_identity(),"process_identity":identity,
             "children":_descendant_identities(process),
             "task_nonce":previous["task_nonce"]}
    guard = _task_guard_path(identity,state["boot_id"])
    _update_runtime(guard,{"task_nonce":state["task_nonce"],"status":"running"})
    _update_runtime(path, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            sample = observe_process_tree()
            children = {child["pid"]:child for child in state["children"]}
            children.update((child["pid"],child) for child in _descendant_identities(process))
            state.update(updated_at=datetime.now(timezone.utc).isoformat(),
                         private_current_bytes=sample.get("private_bytes",0),
                         private_peak_bytes=max(state["private_peak_bytes"],sample.get("private_bytes",0)),
                         cpu_seconds=sample.get("cpu_seconds",0),children=list(children.values()))
            _update_runtime(path, lambda current: {key:value for key,value in state.items() if key != "status"}
                            if current.get("task_nonce") == state["task_nonce"] else {})
            stop.wait(2)

    thread = threading.Thread(target=heartbeat,daemon=True)
    thread.start()
    try:
        if remaining_deadline(path, seconds) <= 0:
            raise TimeoutError("Persisted attempt deadline expired before execution")
        result = function(*args,**kwargs)
        if remaining_deadline(path, seconds) <= 0:
            raise TimeoutError("Persisted attempt deadline expired during execution")
        state["status"] = "finished"
        return result
    except TimeoutError:
        state["status"] = "timed_out"
        raise
    except BaseException:
        state["status"] = "failed"
        raise
    finally:
        stop.set()
        thread.join()
        children = {child["pid"]:child for child in state["children"]}
        children.update((child["pid"],child) for child in _descendant_identities(process))
        state["children"] = list(children.values())
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _update_runtime(guard,{"task_nonce":state["task_nonce"],"status":"idle"})
        _update_runtime(path, lambda current: (dict(state, status=current["status"])
                        if current.get("status") in {"timed_out", "cancelled"} else state)
                        if current.get("task_nonce") == state["task_nonce"] else {})


def cleanup_attempt_group(path):
    """Signal only birth-verified processes from the recorded attempt lineage."""
    state = read_runtime(path)
    pid, group = state.get("pid"),state.get("process_group")
    if os.name != "posix" or not isinstance(pid,int) or pid != group or group == os.getpgrp():
        return
    import psutil
    if state.get("boot_id") != _boot_identity():
        return
    leader = state.get("process_identity")
    if not isinstance(leader,dict) or leader.get("pid") != pid or leader.get("process_group") != group or leader.get("session_id") != group:
        return
    identities = {identity["pid"]:identity for identity in state.get("children",[])
                  if isinstance(identity,dict) and isinstance(identity.get("pid"),int)}
    identities.update((child["pid"],child) for child in
                      (_owned_root_action(state,_isolated_group_identities) or []))
    kill_root = state.get("status") not in {"finished","failed"}
    if kill_root:
        identities[pid] = leader

    def send(identity, sig):
        if identity["pid"] == pid:
            _owned_root_action(state,lambda process: _signal_verified(identity,sig),active_only=True)
        else:
            _signal_verified(identity,sig)

    signaled = set()
    until = time.monotonic()+1
    while True:
        alive = []
        for identity in list(identities.values()):
            if identity["pid"] == pid:
                process = _owned_root_action(state,lambda process: process,active_only=True)
            else:
                process = _verified_process(identity)
            if process is not None:
                children = (_owned_root_action(state,_isolated_group_identities,active_only=True) or []) if identity["pid"] == pid else _descendant_identities(process)
                identities.update((child["pid"],child) for child in children)
                try:
                    if process.status() != psutil.STATUS_ZOMBIE:
                        alive.append(process)
                except psutil.Error:
                    pass
        for identity in list(identities.values()):
            key = (identity["pid"],identity.get("create_time"))
            if key not in signaled:
                send(identity,signal.SIGTERM)
                signaled.add(key)
        if not alive or time.monotonic() >= until:
            break
        time.sleep(.05)
    killed = []
    for identity in identities.values():
        process = (_owned_root_action(state,lambda process: process,active_only=True)
                   if identity["pid"] == pid else _verified_process(identity))
        if process is not None:
            killed.append(process)
        send(identity,signal.SIGKILL)
    psutil.wait_procs(killed,timeout=1)


class BoundedFitExecutor:
    def __init__(self,max_workers,*,max_tasks_per_child=1):
        from pebble import ProcessPool
        self.pool = ProcessPool(max_workers=max_workers,max_tasks=max_tasks_per_child or 0,
                                context=multiprocessing.get_context("spawn"))
        self._timers = set()
        self._timer_lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self,typ,value,tb):
        try:
            if typ:
                self.pool.stop()
            else:
                self.pool.close()
            self.pool.join()
        finally:
            with self._timer_lock:
                timers = list(self._timers)
            for timer in timers:
                timer.cancel()
                timer.join(timeout=2)

    def submit(self,function,*args,runtime_path,timeout,progress_policy=None,progress_path=None,**kwargs):
        _validate_timeout(timeout)
        from .probit_runtime_policy import normalize_progress_policy
        policy = normalize_progress_policy(progress_policy) if progress_policy is not None else None
        now = datetime.now(timezone.utc)

        def admission(state):
            nonlocal policy
            if state.get("progress_policy"):
                persisted = normalize_progress_policy(state["progress_policy"])
                if policy is None or persisted.workload_fingerprint != policy.workload_fingerprint:
                    raise ValueError("Persisted progress policy cannot change workload on restart")
                # New ledger measurements may change the next attempt's budget,
                # but this attempt keeps its admitted limits and used extensions.
                policy = persisted
            allowance = max(timeout, policy.initial_seconds) if policy else timeout
            deadline = state.get("deadline_at") or datetime.fromtimestamp(now.timestamp()+allowance,timezone.utc).isoformat()
            changes = {"schema_version":1,"started_at":state.get("started_at",now.isoformat()),
                    "deadline_at":deadline,
                    "status":"queued","submitted_at":now.isoformat(),"task_nonce":uuid.uuid4().hex}
            if policy:
                if timeout > policy.hard_seconds and not state.get("deadline_at"):
                    raise ValueError("Requested timeout exceeds policy hard cap")
                changes.update(progress_policy=policy.to_dict(),
                    hard_deadline_at=state.get("hard_deadline_at") or datetime.fromtimestamp(
                        datetime.fromisoformat(deadline).timestamp()+policy.hard_seconds-allowance, timezone.utc).isoformat(),
                    runtime_extensions=state.get("runtime_extensions", []),
                    progress_not_before_epoch=now.timestamp(), optimizer_observation=None, productive_progress=None)
            return changes

        admitted = _update_runtime(runtime_path,admission)
        remaining = remaining_deadline(runtime_path,timeout)
        if not remaining:
            _update_runtime(runtime_path,{"status":"timed_out"})
            future = Future()
            future.set_exception(TimeoutError("Persisted attempt deadline expired"))
            return future
        hard_remaining = max(0., datetime.fromisoformat(admitted["hard_deadline_at"]).timestamp()-time.time()) if policy else remaining
        raw = self.pool.schedule(_timed_call,
            args=(function,args,kwargs,str(runtime_path),remaining),timeout=hard_remaining)
        future = Future()
        completion_lock = threading.Lock()

        def expire(force=False):
            try:
                with completion_lock:
                    if future.done():
                        return
                    if policy and not force:
                        extend_attempt_deadline(runtime_path, progress_path, task_nonce=admitted["task_nonce"])
                        left = remaining_deadline(runtime_path,timeout)
                        if left > 0:
                            start_timer(min(left, 2., policy.extension_window_seconds/2))
                            return
                    try:
                        future.set_exception(TimeoutError("Persisted attempt deadline expired"))
                    except InvalidStateError:
                        return
                # Pebble cancellation terminates its owning worker, including running tasks.
                raw.cancel()
                _update_runtime(runtime_path, lambda state: {"status":"timed_out"}
                                if state.get("task_nonce") == admitted["task_nonce"]
                                and state.get("status") != "cancelled" else {})
            finally:
                with self._timer_lock:
                    self._timers.discard(threading.current_thread())

        timer = None

        def start_timer(delay):
            nonlocal timer
            timer = threading.Timer(delay,expire)
            timer.daemon = True
            with self._timer_lock:
                self._timers.add(timer)
            timer.start()

        def finish(completed):
            try:
                result = completed.result()
                if remaining_deadline(runtime_path,timeout) <= 0:
                    expire(force=True)
                else:
                    with completion_lock:
                        if not future.done():
                            future.set_result(result)
            except BaseException as exc:
                if policy and isinstance(exc, TimeoutError):
                    _update_runtime(runtime_path, lambda state: {"status":"timed_out"}
                                    if state.get("task_nonce") == admitted["task_nonce"]
                                    and state.get("status") != "cancelled" else {})
                if not future.done():
                    try:
                        future.set_exception(exc)
                    except InvalidStateError:
                        pass

        def stop_timer(completed):
            if timer is not None:
                timer.cancel()
                with self._timer_lock:
                    self._timers.discard(timer)
            if completed.cancelled():
                raw.cancel()
                if policy:
                    _update_runtime(runtime_path, lambda state: {"status":"cancelled"}
                                    if state.get("task_nonce") == admitted["task_nonce"] else {})

        raw.add_done_callback(finish)
        future.add_done_callback(stop_timer)
        with completion_lock:
            if not future.done():
                start_timer(min(remaining, 2., policy.extension_window_seconds/2) if policy else remaining)
        return future


def active_attempt_summary(directory, rows):
    now = time.time()
    result = []
    for row in rows:
        if row["status"] not in {"running","reserved"}:
            continue
        path = Path(directory)/"trials"/row["attempt_id"]
        runtime = read_runtime(path/"runtime.json")
        progress = read_runtime(path/"progress.json")
        started = runtime.get("started_at")
        recipe = row["payload"]["recipe"]
        result.append({"attempt_id":row["attempt_id"],"status":row["status"],
            "program_id":row["payload"]["program_id"],"model_family":recipe["model"]["kind"],
            "target":recipe["target"]["kind"],"elapsed_seconds":max(0,now-datetime.fromisoformat(started).timestamp()) if started else None,
            "deadline_at":runtime.get("deadline_at"),"runtime":runtime,"progress":progress})
    return result
