"""Resource observation and admission control for research workers."""

from __future__ import annotations

import os
import shutil
import hashlib
import fcntl
import json
import math
import resource
import threading
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - exercised on minimal remote canaries.
    psutil = None


GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_percent: float
    load_percent: float
    memory_available_gib: float
    memory_total_gib: float
    disk_free_gib: float
    cpu_count: int


@dataclass(frozen=True)
class AdmissionPolicy:
    reserve_memory_gib: float | None = None
    reserve_memory_fraction: float = 0.20
    estimated_peak_trial_rss_gib: float = 2.0
    configured_ceiling: int = 32
    disk_pause_threshold_gib: float = 10.0
    cpu_admission_ceiling_percent: float = 95.0


def observe_resources(path: str = ".") -> ResourceSnapshot:
    cpu_count = os.cpu_count() or 1
    load1, _, _ = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    if psutil is not None:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(path)
        cpu_percent = float(psutil.cpu_percent(interval=0.0))
        available = float(memory.available / GIB)
        total = float(memory.total / GIB)
        free = float(disk.free / GIB)
    else:
        available, total = _memory_from_proc()
        free = float(shutil.disk_usage(path).free / GIB)
        cpu_percent = 0.0
    return ResourceSnapshot(
        cpu_percent=cpu_percent,
        load_percent=float(load1 / cpu_count * 100.0),
        memory_available_gib=available,
        memory_total_gib=total,
        disk_free_gib=free,
        cpu_count=int(cpu_count),
    )


def admission_slots(
    snapshot: ResourceSnapshot,
    policy: AdmissionPolicy | None = None,
    *,
    requested: int | None = None,
) -> int:
    policy = policy or AdmissionPolicy(
        reserve_memory_gib=(
            float(os.environ["IMA_RESEARCH_RESERVE_MEMORY_GIB"])
            if "IMA_RESEARCH_RESERVE_MEMORY_GIB" in os.environ else None
        ),
        estimated_peak_trial_rss_gib=float(
            os.environ.get("IMA_RESEARCH_TRIAL_RSS_GIB", "2")
        ),
        configured_ceiling=int(os.environ.get("IMA_RESEARCH_MAX_WORKERS", "32")),
        disk_pause_threshold_gib=float(
            os.environ.get("IMA_RESEARCH_DISK_PAUSE_GIB", "10")
        ),
        cpu_admission_ceiling_percent=float(
            os.environ.get("IMA_RESEARCH_CPU_CEILING_PERCENT", "95")
        ),
    )
    if snapshot.disk_free_gib < policy.disk_pause_threshold_gib:
        return 0
    if snapshot.cpu_percent >= policy.cpu_admission_ceiling_percent:
        return 1
    reserve = policy.reserve_memory_gib
    if reserve is None:
        reserve = max(16.0, snapshot.memory_total_gib * policy.reserve_memory_fraction)
    memory_slots = int(max(0.0, snapshot.memory_available_gib - reserve) / policy.estimated_peak_trial_rss_gib)
    cpu_slots = max(1, snapshot.cpu_count - 2 if snapshot.cpu_count > 4 else snapshot.cpu_count)
    slots = max(0, min(cpu_slots, memory_slots, policy.configured_ceiling))
    if requested is not None:
        slots = min(slots, requested)
    return max(0, slots)


def resource_report(snapshot: ResourceSnapshot, slots: int) -> dict[str, float | int]:
    return {
        "cpu_percent": round(snapshot.cpu_percent, 2),
        "load_percent": round(snapshot.load_percent, 2),
        "memory_available_gib": round(snapshot.memory_available_gib, 2),
        "memory_total_gib": round(snapshot.memory_total_gib, 2),
        "disk_free_gib": round(snapshot.disk_free_gib, 2),
        "cpu_count": snapshot.cpu_count,
        "admission_slots": slots,
    }


def _memory_from_proc() -> tuple[float, float]:
    values: dict[str, float] = {}
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = float(raw.strip().split()[0]) * 1024 / GIB
    except OSError:
        pass
    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", total)
    return available, total


class JobWorkload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    stage: Literal["feature_generation","selection","preparation","fit","graph_component","simulation"]
    family: str
    rows: int = Field(ge=1)
    generated_features: int = Field(ge=0)
    selected_features: int = Field(ge=0)
    dtype: Literal["float32","float64"] = "float64"
    sparsity: float = Field(default=0,ge=0,le=1)
    categorical_cardinalities: tuple[int,...] = ()
    folds: int = Field(default=1,ge=1)
    native_threads: int = Field(default=1,ge=1)
    native_thread_settings: dict[str,int] = Field(default_factory=dict)
    search_settings: dict[str,Any] = Field(default_factory=dict)
    cached_artifact_ids: tuple[str,...] = ()
    shared_artifacts: dict[str,int] = Field(default_factory=dict)
    implementation_revision: str
    dependency_versions: dict[str,str]

    @model_validator(mode="after")
    def complete(self):
        if not self.family or not self.implementation_revision or not self.dependency_versions:
            raise ValueError("Complete workload implementation and dependencies required")
        if any(n < 0 for n in (*self.categorical_cardinalities,*self.shared_artifacts.values())):
            raise ValueError("Negative workload dimensions")
        if any(n < 1 for n in self.native_thread_settings.values()):
            raise ValueError("Native thread counts must be positive")
        if self.native_thread_settings and max(self.native_thread_settings.values()) > self.native_threads:
            raise ValueError("Native library threads exceed reserved threads")
        return self

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class JobMeasurement:
    workload_fingerprint: str
    private_peak_bytes: int
    wall_seconds: float
    failed: bool = False
    censored: bool = False
    memory_metric: str = "uss"

    def __post_init__(self):
        if self.private_peak_bytes < 0 or not math.isfinite(self.wall_seconds) or self.wall_seconds < 0:
            raise ValueError("Invalid current-job measurement")


@dataclass(frozen=True)
class JobEstimate:
    schema_version: int
    workload_fingerprint: str
    stage: str
    family: str
    cpu_threads: int
    private_peak_bytes: int
    shared_artifacts: dict[str,int]
    wall_seconds: float
    preparation_seconds: float
    fit_seconds: float
    disk_bytes: int
    sample_count: int
    confidence: str
    quantile: float
    margin: float

    def __post_init__(self):
        if self.schema_version != 1 or self.cpu_threads < 1 or self.private_peak_bytes <= 0 or self.disk_bytes < 0 or self.sample_count < 0:
            raise ValueError("Invalid job-estimate dimensions/version")
        if any(not math.isfinite(value) or value < 0 for value in (self.wall_seconds,self.preparation_seconds,self.fit_seconds)):
            raise ValueError("Invalid job-estimate time")
        if any(not name or size < 0 for name,size in self.shared_artifacts.items()):
            raise ValueError("Invalid shared-artifact reservation")
        if not 0 < self.quantile <= 1 or not math.isfinite(self.margin) or self.margin < 1:
            raise ValueError("Invalid job-estimate safety policy")

    @property
    def ram_gib(self):
        return (self.private_peak_bytes + sum(self.shared_artifacts.values())) / GIB

    @property
    def private_ram_gib(self):
        return self.private_peak_bytes / GIB

    def to_dict(self):
        return asdict(self)

    def comparison(self, actual: JobMeasurement):
        if actual.workload_fingerprint != self.workload_fingerprint:
            raise ValueError("Estimate/actual workload mismatch")
        return {"estimate":self.to_dict(),"actual":asdict(actual),"private_error_bytes":actual.private_peak_bytes-self.private_peak_bytes,"wall_error_seconds":actual.wall_seconds-self.wall_seconds}


def _family_cost(workload):
    """Cold heuristics, not benchmark claims. Settings are part of the exact key.

    Probit settings: quadrature_order, heteroscedastic, mean_parameters,
    variance_parameters, parameter_dimension, max_iter (or iterations/
    iteration_bucket), race_count,
    max_runners, gradient_mode. Missing shape uses conservative defaults.
    """
    settings = workload.search_settings
    def positive(name, default):
        value = settings.get(name)
        value = default if value is None else value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite cost setting required: " + name)
        return float(value)

    iterations = positive("max_iter", positive("iterations", positive("iteration_bucket", 1) * 200))
    folds = workload.folds
    family = workload.family.removeprefix("graph:")
    if family == "gaussian_probit":
        heteroscedastic = settings.get("heteroscedastic")
        heteroscedastic = False if heteroscedastic is None else heteroscedastic
        if not isinstance(heteroscedastic, bool):
            raise ValueError("heteroscedastic must be boolean")
        mean = positive("mean_parameters", max(1, workload.selected_features) + 1)
        variance = positive("variance_parameters", mean) if heteroscedastic else 0
        dimension = max(mean + variance, positive("parameter_dimension", mean + variance))
        quadrature = positive("quadrature_order", 48)
        runners = positive("max_runners", 14)
        races = positive("race_count", max(1, workload.rows / runners))
        gradient = settings.get("gradient_mode", "numerical")
        if gradient not in {"numerical", "analytic"}:
            raise ValueError("Unknown probit gradient cost mode")
        # Gaussian winner integration is quadratic in runners. Numerical
        # differences add one evaluation per parameter, including variance.
        evaluations = dimension + 1 if gradient == "numerical" else max(2, dimension / 16)
        wall = 3600 * max(1, races / 1000) * max(1, runners / 14) ** 2
        wall *= max(1, quadrature / 48) * max(1, iterations / 200) * folds
        wall *= max(1, evaluations / 17) * (2 if heteroscedastic else 1)
        memory = 2 * GIB + workload.rows * int(math.ceil(dimension)) * 8 * 12
        memory += math.ceil(runners * runners * quadrature * 8 * 8)
        return memory, wall
    if family in {"boosted", "catboost_classifier", "catboost_regressor"}:
        return 0, 600 * folds * max(1, iterations / 200) * max(1, positive("depth", 6) / 6) ** 2
    return 0, 600 * folds * max(1, iterations / 200)


def estimate_job(workload: JobWorkload, samples=(), *, quantile=.95, margin=1.25, cold_private_bytes=None, cold_wall_seconds=None) -> JobEstimate:
    """Exact workload samples only; failures/OOM bounds never disappear in a quantile."""
    if not 0 < quantile <= 1 or not math.isfinite(margin) or margin < 1:
        raise ValueError("Invalid high-quantile safety margin")
    if not isinstance(workload,JobWorkload):
        workload = JobWorkload.model_validate(workload)
    fingerprint = workload.fingerprint()
    matching = [s for s in samples if s.workload_fingerprint == fingerprint]
    widths = max(workload.generated_features,workload.selected_features,1) + sum(workload.categorical_cardinalities)
    dense_bytes = workload.rows * widths * (4 if workload.dtype == "float32" else 8)
    multipliers = {"feature_generation":12,"selection":8,"preparation":4,"fit":8,"graph_component":8,"simulation":6}
    cold = max(512*1024**2,dense_bytes*multipliers[workload.stage]) if cold_private_bytes is None else cold_private_bytes
    wall = {"feature_generation":300.,"selection":600.,"preparation":120.,"fit":600.,"graph_component":600.,"simulation":120.}[workload.stage] if cold_wall_seconds is None else cold_wall_seconds
    if workload.stage in {"fit", "graph_component"}:
        family_private, family_wall = _family_cost(workload)
        if cold_private_bytes is None:
            cold = max(cold, family_private)
        if cold_wall_seconds is None:
            wall = max(wall, family_wall)
    if cold <= 0 or not math.isfinite(wall) or wall <= 0:
        raise ValueError("Conservative cold estimate must be positive")
    # Failed samples remain bounds, never evidence for lowering cold estimates.
    representative = [s for s in matching if not s.censored and not s.failed]
    measured = len(representative) >= 5
    if measured:
        cold = float(np.quantile([s.private_peak_bytes for s in representative],quantile))
        wall = float(np.quantile([s.wall_seconds for s in representative],quantile))
    bounds = [s for s in matching if s.censored or s.failed]
    if bounds:
        cold = max(cold,max(s.private_peak_bytes for s in bounds))
        wall = max(wall,max(s.wall_seconds for s in bounds))
    # Until representative coverage, keep both cold and measured lower bounds.
    if matching and not measured:
        cold = max(cold,max(s.private_peak_bytes for s in matching))
        wall = max(wall,max(s.wall_seconds for s in matching))
    seconds = wall * margin
    preparation = workload.stage in {"preparation","feature_generation","selection"}
    return JobEstimate(1,fingerprint,workload.stage,workload.family,workload.native_threads,math.ceil(cold*margin),dict(workload.shared_artifacts),seconds,seconds if preparation else 0.,0. if preparation else seconds,math.ceil(dense_bytes*2),len(matching),"measured_exact_workload" if measured else "conservative_cold",quantile,margin)


class JobEstimator:
    """Own-campaign estimate ledger; locked atomic publication preserves failed samples."""
    def __init__(self, ledger_path):
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True,exist_ok=True)

    @property
    def samples(self):
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text())
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported job-estimate ledger version")
        return [JobMeasurement(**row["actual"]) for row in payload["samples"]]

    def estimate(self, workload, **kwargs):
        return estimate_job(workload,self.samples,**kwargs)

    def record_censored_history(self, workload, snapshot):
        """Import known per-fit interruption bounds once, never unit-wide peaks."""
        workload = workload if isinstance(workload, JobWorkload) else JobWorkload.model_validate(workload)
        records = []
        for row in snapshot["interrupted_attempts"]:
            actual = row.get("actual")
            if actual and actual["workload_fingerprint"] == workload.fingerprint():
                identity = json.dumps([row["invocation_id"], row["attempt_id"]], separators=(",", ":"))
                records.append(self.record(workload, JobMeasurement(**dict(actual, failed=True, censored=True)),
                                           sample_id="interrupted:" + identity))
        return records

    def record(self, workload, actual, *, sample_id=None):
        workload = workload if isinstance(workload,JobWorkload) else JobWorkload.model_validate(workload)
        if isinstance(actual,dict):
            if not actual.get("supported",True):
                raise ValueError("Unsupported measurement cannot train an estimator")
            actual = JobMeasurement(workload.fingerprint(),actual["private_peak_bytes"],actual["wall_seconds"],actual.get("failed",False),actual.get("censored",False),actual.get("memory_metric","rss_fallback"))
        if actual.workload_fingerprint != workload.fingerprint():
            raise ValueError("Measurement workload identity mismatch")
        with open(str(self.path)+".lock","a+b") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                payload = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version":1,"samples":[]}
                if payload["schema_version"] != 1:
                    raise ValueError("Unsupported job-estimate ledger version")
                if sample_id is not None:
                    previous = next((row for row in payload["samples"] if row.get("sample_id") == sample_id), None)
                    if previous is not None:
                        if previous["actual"] != asdict(actual):
                            raise ValueError("Idempotent estimate sample identity reused with different measurement")
                        return previous
                before = estimate_job(workload,[JobMeasurement(**row["actual"]) for row in payload["samples"]])
                record = {"workload":workload.model_dump(mode="json"),"actual":asdict(actual),"comparison":before.comparison(actual)}
                if sample_id is not None:
                    record["sample_id"] = sample_id
                payload["samples"].append(record)
                fd,temporary = tempfile.mkstemp(dir=self.path.parent,prefix=self.path.name+"-")
                try:
                    with os.fdopen(fd,"w") as stream:
                        json.dump(payload,stream,sort_keys=True,allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary,self.path)
                    directory = os.open(self.path.parent,os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                return record
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)


def observe_process_tree(pid=None):
    if psutil is None:
        return {"supported":False,"reason":"psutil unavailable"}
    try:
        root = psutil.Process(pid or os.getpid())
        processes = [root,*root.children(recursive=True)]
    except psutil.Error as exc:
        return {"supported":False,"reason":f"process tree unavailable: {type(exc).__name__}"}
    totals = {"rss_bytes":0,"uss_bytes":0,"pss_bytes":0,"cpu_seconds":0.,"read_bytes":0,"write_bytes":0,"processes":0,"threads":0}
    has_uss, has_pss, has_io = True, True, True
    for process in processes:
        try:
            try:
                memory = process.memory_full_info()
            except psutil.AccessDenied:
                memory = process.memory_info()
            totals["rss_bytes"] += memory.rss
            totals["uss_bytes"] += getattr(memory,"uss",0)
            totals["pss_bytes"] += getattr(memory,"pss",0)
            has_uss &= hasattr(memory,"uss")
            has_pss &= hasattr(memory,"pss")
            cpu = process.cpu_times()
            totals["cpu_seconds"] += cpu.user+cpu.system
            totals["threads"] += process.num_threads()
            totals["processes"] += 1
            try:
                io = process.io_counters()
                totals["read_bytes"] += io.read_bytes
                totals["write_bytes"] += io.write_bytes
            except (AttributeError,psutil.Error):
                has_io = False
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    totals.update(supported=totals["processes"]>0,memory_metric="uss" if has_uss else "rss_fallback",private_bytes=totals["uss_bytes"] if has_uss else totals["rss_bytes"],pss_supported=has_pss,io_supported=has_io)
    return totals


def observe_cgroup():
    """Read own cgroup v2 limits; no writes or assumptions about host policy."""
    try:
        relative = next(line.split(":",2)[2].strip() for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        result = {"supported":True,"path":str(root)}
        for filename in ("memory.current","memory.peak","memory.max","memory.events"):
            try:
                raw = (root/filename).read_text().strip()
            except FileNotFoundError:
                if filename == "memory.peak":
                    continue
                raise
            result[filename] = raw if filename == "memory.events" or raw == "max" else int(raw)
        try:
            result["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError:
            pass
        return result
    except (OSError,StopIteration,ValueError):
        return {"supported":False,"reason":"own cgroup v2 memory interface unavailable"}


class JobMonitor:
    """Sample this job's process tree; cgroup lifetime peak remains separately labelled."""
    def __init__(self, *, pid=None, interval=.05):
        if interval <= 0:
            raise ValueError("Positive sampling interval required")
        self.pid, self.interval = pid, interval
        self.samples = []
        self._stop = threading.Event()

    def _sample(self):
        self.samples.append({**observe_process_tree(self.pid),"cgroup":observe_cgroup()})

    def __enter__(self):
        self.started = time.monotonic()
        self.usage_start = resource.getrusage(resource.RUSAGE_SELF)
        self.child_usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        self._sample()
        def sample():
            while not self._stop.wait(self.interval):
                self._sample()
        self.thread = threading.Thread(target=sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self._stop.set()
        self.thread.join()
        self._sample()
        self.wall_seconds = time.monotonic()-self.started
        self.failed = exc[0] is not None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.page_faults = {"minor":max(0,usage.ru_minflt-self.usage_start.ru_minflt)+max(0,child_usage.ru_minflt-self.child_usage_start.ru_minflt),"major":max(0,usage.ru_majflt-self.usage_start.ru_majflt)+max(0,child_usage.ru_majflt-self.child_usage_start.ru_majflt),"scope":"Current process and children terminated during job; counter deltas"}

    def report(self):
        valid = [s for s in self.samples if s.get("supported")]
        if not valid:
            return {"supported":False,"wall_seconds":self.wall_seconds}
        first,last = valid[0],valid[-1]
        peak = max(s["private_bytes"] for s in valid)
        io_supported = all(s["io_supported"] for s in valid)
        return {"supported":True,"failed":self.failed,"wall_seconds":self.wall_seconds,"sample_count":len(valid),"sampling_interval_seconds":self.interval,"memory_metric":first["memory_metric"],"private_peak_bytes":peak,"private_baseline_bytes":first["private_bytes"],"private_growth_peak_bytes":max(0,peak-first["private_bytes"]),"rss_peak_bytes":max(s["rss_bytes"] for s in valid),"pss_peak_bytes":max(s["pss_bytes"] for s in valid) if all(s["pss_supported"] for s in valid) else None,"cpu_seconds":max(0,last["cpu_seconds"]-first["cpu_seconds"]),"io_supported":io_supported,"read_bytes":max(0,last["read_bytes"]-first["read_bytes"]) if io_supported else None,"write_bytes":max(0,last["write_bytes"]-first["write_bytes"]) if io_supported else None,"page_faults":self.page_faults,"cgroup_current_peak_bytes":max((s["cgroup"].get("memory.current",0) for s in valid),default=0),"cgroup":last["cgroup"],"limitation":"Sampled peaks can miss allocations shorter than the sampling interval; process-tree private residency includes worker baseline"}

    def measurement(self, workload, *, failed=False, censored=False):
        report = self.report()
        if not report["supported"]:
            raise RuntimeError("Current-job process measurement unavailable")
        return JobMeasurement(workload.fingerprint(),report["private_peak_bytes"],report["wall_seconds"],failed,censored,report["memory_metric"])


class ProgressiveCapacity:
    """Explicit cap ladder; representative folds and sustained measured headroom gate rises."""
    def __init__(self, caps=(2,4,8,12,16,26), *, sustain_seconds=30, emergency_gib=2,
                 representative_baseline=None):
        if not caps or tuple(sorted(set(caps))) != tuple(caps) or caps[0] < 1 or sustain_seconds < 30 or emergency_gib < 0:
            raise ValueError("Ordered positive caps and at least 30 seconds of headroom required")
        self.caps = tuple(caps)
        self.index = 0
        self.sustain_seconds = sustain_seconds
        self.emergency_gib = emergency_gib
        self.since = None
        self.fold_baseline = 0
        self.paused = False
        self._evidence_baseline = None
        if representative_baseline is not None:
            self.reset(representative_baseline=representative_baseline)

    def reset(self, *, representative_baseline):
        """Restart at the first cap; count only successes above this startup count.

        Pass the durable successful sample count at startup. An isolated cheap
        canary finishing before the first headroom sample is then eligible.
        Each rise consumes the current count, so evidence cannot fund two rises.
        """
        if isinstance(representative_baseline, bool) or not isinstance(representative_baseline, int) or representative_baseline < 0:
            raise ValueError("Nonnegative representative sample baseline required")
        self.index = 0
        self.since = None
        self.paused = False
        self.fold_baseline = representative_baseline
        self._evidence_baseline = representative_baseline

    @property
    def cap(self):
        return 0 if self.paused else self.caps[self.index]

    def observe(self, *, headroom_gib, private_per_fit_gib, representative_folds, now=None):
        now = time.monotonic() if now is None else now
        if not math.isfinite(headroom_gib) or not math.isfinite(private_per_fit_gib) or private_per_fit_gib <= 0 or representative_folds < 0:
            raise ValueError("Measured finite headroom and representative cost required")
        self.paused = headroom_gib <= self.emergency_gib
        if self.paused:
            self.since = None
            return {"cap":0,"reason":"critical_headroom","changed":False}
        if self.index == len(self.caps)-1:
            return {"cap":self.cap,"reason":"configured_ceiling","changed":False}
        needed = (self.caps[self.index+1]-self.caps[self.index])*private_per_fit_gib+self.emergency_gib
        if headroom_gib < needed:
            self.since = None
            return {"cap":self.cap,"reason":"increment_headroom","changed":False,"needed_gib":needed}
        if self.since is None:
            self.since = now
            self.fold_baseline = representative_folds if self._evidence_baseline is None else self._evidence_baseline
        if now-self.since >= self.sustain_seconds and representative_folds > self.fold_baseline:
            self.index += 1
            self.since = None
            if self._evidence_baseline is not None:
                self._evidence_baseline = representative_folds
            return {"cap":self.cap,"reason":"sustained_representative_headroom","changed":True}
        return {"cap":self.cap,"reason":"await_sustained_headroom_and_new_fold","changed":False}
