"""Own-service startup memory policy; deliberately independent of model code."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def apply_backoff(config_path, state_path, stop_path, events, unit):
    config = json.loads(config_path.read_text())
    workers = config["max_concurrent_trials"]
    if type(workers) is not int or workers < 1:
        raise ValueError("Invalid worker ceiling")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    matching = [e for e in events if e.get("USER_UNIT") == unit
                and e.get("UNIT_RESULT") == "oom-kill"]
    latest = max(matching, key=lambda e: int(e["__REALTIME_TIMESTAMP"]), default=None)
    if state.get("pending"):
        target = state["target_workers"]
    elif latest and int(latest["__REALTIME_TIMESTAMP"]) > int(state.get("event_time", 0)):
        target = max(1, workers - 2)
        state = {"cursor": latest["__CURSOR"],
                 "event_time": latest["__REALTIME_TIMESTAMP"],
                 "previous_workers": workers, "target_workers": target,
                 "paused": workers == 1, "pending": True}
        atomic_json(state_path, state)
    else:
        return {"changed": False, "workers": workers}
    if state.get("paused"):
        try:
            with stop_path.open("x") as stream:
                stream.write("V5 paused: own-service OOM at one worker; operator review required.\n")
        except FileExistsError:
            pass
        raise RuntimeError("Memory backoff exhausted; refusing automatic restart")
    # A replayed pending transaction may only lower the current ceiling.
    config["max_concurrent_trials"] = min(workers, target)
    atomic_json(config_path, config)
    state["pending"] = False
    atomic_json(state_path, state)
    return {"changed": workers != config["max_concurrent_trials"], **state}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--unit", default="ima-discovery-v5-supervisor.service",
                        choices=["ima-discovery-v5-supervisor.service",
                                 "ima-v5-memory-guard-canary.service"])
    args = parser.parse_args()
    ops = args.campaign / "ops"
    with (ops / "memory-backoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = subprocess.run(
            ["journalctl", "--user", "-u", args.unit, "UNIT_RESULT=oom-kill", "-n", "50",
             "-o", "json", "--no-pager"], check=True, text=True, capture_output=True)
        events = [json.loads(line) for line in output.stdout.splitlines() if line.strip()]
        print(json.dumps(apply_backoff(ops / "openrouter-config.json",
                                     ops / "memory-backoff-state.json",
                                     args.campaign / "STOP", events, args.unit)))


if __name__ == "__main__":
    main()
