"""Own-user transactional hard-budget partition; never restart model workers."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager


ROOT = Path("/home/imaopt/research-v2")
MODEL_UNIT = "ima-v7-openrouter-recovery-20261007.service"
MODEL_CAMPAIGN = ROOT/"campaigns/agentic_v7_openrouter_recovery_20261007"
STATE = ROOT/"feature-exploration-budget.json"
DROPIN = Path("/home/imaopt/.config/systemd/user")/(MODEL_UNIT+".d")/"95-feature-exploration-budget.conf"
TOTAL_BYTES = 100_000_000_000
FEATURE_BYTES = 8_000_000_000
MODEL_BYTES = TOTAL_BYTES-FEATURE_BYTES
BODY = f"# Owned by ima-feature-exploration-budget\n[Service]\nMemoryMax={MODEL_BYTES}\nCPUQuota=2400%\n"


def partition_safe(status, current_bytes):
    resources = status["resources"]
    committed = (resources["reserved_ram_gib"]+resources["resident_gib"]+resources["emergency_gib"])*1024**3
    return max(committed, current_bytes+resources["emergency_gib"]*1024**3) < MODEL_BYTES


def manager(*args):
    return subprocess.check_output(["systemctl", "--user", *args], text=True).strip()


def save(value):
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(STATE)


@contextmanager
def frozen_model():
    """Pause admission and existing fits briefly; never terminate them."""
    if manager('show',MODEL_UNIT,'-p','ActiveState','--value') in {'inactive','failed'}:
        if manager('show',MODEL_UNIT,'-p','MainPID','--value')!='0':
            raise RuntimeError('Inactive main reports live PID; hold feature startup')
        yield
        if manager('show',MODEL_UNIT,'-p','ActiveState','--value') not in {'inactive','failed'}:
            raise RuntimeError('Main started during feature budget partition')
        return
    if manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") != "running":
        raise RuntimeError("Model is already frozen by another owner")
    try:
        subprocess.run(["systemctl", "--user", "freeze", MODEL_UNIT], check=True, timeout=30)
        if manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") != "frozen":
            raise RuntimeError("Model admission pause was not acknowledged")
        yield
    finally:
        subprocess.run(["systemctl", "--user", "thaw", MODEL_UNIT], check=True, timeout=30)


def main(action):
    if os.getuid() != 1001 or Path.home() != Path("/home/imaopt"):
        raise RuntimeError("Feature budget requires own imaopt user")
    if action == "reserve":
        if DROPIN.exists() and DROPIN.read_text() != BODY:
            raise RuntimeError("Unexpected owned budget dropin; do not overwrite")
        old = json.loads(STATE.read_text()) if STATE.exists() else None
        current_limit = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
        if not MODEL_BYTES-4096 <= current_limit <= TOTAL_BYTES:
            raise RuntimeError("Unexpected model memory limit; no budget mutation")
        prior_limit = old["previous_memory_max"] if old and old.get("status") in {"reserved", "reserving"} else current_limit
        prior_cpu = old["previous_cpu_quota"] if old and old.get("status") in {"reserved", "reserving"} else manager(
            "show", MODEL_UNIT, "-p", "CPUQuotaPerSecUSec", "--value")
        for _ in range(120):
            active=manager('show',MODEL_UNIT,'-p','ActiveState','--value')
            relative = manager("show", MODEL_UNIT, "-p", "ControlGroup", "--value")
            if active in {'inactive','failed'} and not relative:
                if manager('show',MODEL_UNIT,'-p','MainPID','--value')!='0':
                    raise RuntimeError('Inactive main reports live PID')
                # Reserve its future 92GB limit, not a second independent 100GB.
                # Historical status is not evidence of live memory ownership.
                group=None; current=0
                status={'resources':{'reserved_ram_gib':0,'resident_gib':0,'emergency_gib':0},
                        'active_trials':[],'main_inactive':True}
                break
            status = json.loads((MODEL_CAMPAIGN/"status.json").read_text())
            group = Path("/sys/fs/cgroup")/relative.lstrip("/")
            if not str(group).startswith("/sys/fs/cgroup/user.slice/user-1001.slice/"):
                raise RuntimeError("Not an own-user cgroup")
            current = int((group/"memory.current").read_text())
            if time.time()-datetime.fromisoformat(status["updated_at"]).timestamp() < 120 and partition_safe(status, current):
                break
            print("Waiting for safe feature memory partition; no fits cancelled", flush=True)
            time.sleep(5)
        else:
            raise TimeoutError("No safe budget partition; model fits preserved")
        receipt = {"status": "reserving", "previous_memory_max": prior_limit,
                   "previous_cpu_quota": prior_cpu, "model_bytes": MODEL_BYTES,
                   "feature_bytes": FEATURE_BYTES, "total_bytes": TOTAL_BYTES,
                   "main_pid_before": manager("show", MODEL_UNIT, "-p", "MainPID", "--value"),
                   "active_attempts_before": [row["attempt_id"] for row in status.get("active_trials", [])],
                   "created_at": datetime.now(timezone.utc).isoformat()}
        save(receipt)
        with frozen_model():
            # Recheck after the acknowledged pause closes the admission race.
            status = status if group is None else json.loads((MODEL_CAMPAIGN/"status.json").read_text())
            if not partition_safe(status,0 if group is None else int((group/"memory.current").read_text())):
                raise RuntimeError("Memory grew before admission pause; no limit changed")
            members = set() if group is None else {pid for path in group.rglob("cgroup.procs") for pid in path.read_text().split()}
            if group is not None: members.update((group/"cgroup.procs").read_text().split())
            oom_before = None if group is None else (group/"memory.events").read_text()
            DROPIN.parent.mkdir(parents=True, exist_ok=True)
            DROPIN.write_text(BODY)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "set-property", "--runtime", MODEL_UNIT,
                            f"MemoryMax={MODEL_BYTES}", "CPUQuota=2400%"], check=True)
            observed = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
            assert MODEL_BYTES-4096 <= observed <= MODEL_BYTES
            assert manager("show", MODEL_UNIT, "-p", "MainPID", "--value") == receipt["main_pid_before"]
            assert all(Path("/proc", pid).exists() for pid in members)
            assert group is None or (group/"memory.events").read_text() == oom_before
            receipt.update(status="reserved", observed_model_memory_max=observed,
                           preserved_pids=sorted(members), admission_pause_acknowledged=True,
                           memory_events_before=oom_before, fits_cancelled=0)
        save(receipt)
        print(json.dumps(receipt), flush=True)
    elif action == "restore":
        if not STATE.exists():
            return
        receipt = json.loads(STATE.read_text())
        if receipt["status"] == "reserving" and manager("show", MODEL_UNIT, "-p", "FreezerState", "--value") == "frozen":
            subprocess.run(["systemctl", "--user", "thaw", MODEL_UNIT], check=True, timeout=30)
        if receipt["status"] == "restored":
            return
        current_limit = int(manager("show", MODEL_UNIT, "-p", "MemoryMax", "--value"))
        if not (MODEL_BYTES-4096 <= current_limit <= MODEL_BYTES or current_limit == receipt["previous_memory_max"]):
            raise RuntimeError("Model memory limit changed outside feature worker; refusing overwrite")
        if DROPIN.exists():
            if DROPIN.read_text() != BODY:
                raise RuntimeError("Owned budget dropin changed; refusing restore")
            DROPIN.unlink()
        previous = receipt["previous_cpu_quota"]
        cpu_quota = "infinity" if previous == "infinity" else str(float(previous.rstrip("s"))*100)+"%"
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "set-property", "--runtime", MODEL_UNIT,
                        f"MemoryMax={receipt['previous_memory_max']}", f"CPUQuota={cpu_quota}"], check=True)
        receipt.update(status="restored", restored_at=datetime.now(timezone.utc).isoformat())
        save(receipt)
    elif action == "status":
        print(STATE.read_text() if STATE.exists() else "{}")
    else:
        raise ValueError("Expected reserve, restore or status")


if __name__ == "__main__":
    main(sys.argv[1])
