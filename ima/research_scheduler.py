"""Resource reservations for continuous single-host dispatch."""
import math
import time
from dataclasses import dataclass, field

from .research_resources import GIB, JobEstimate


@dataclass(frozen=True)
class ResourceRequest:
    cpu_threads: int = 1
    ram_gib: float = 4
    shared_artifacts: dict[str,int] = field(default_factory=dict)

    def __post_init__(self):
        if self.cpu_threads < 1 or not math.isfinite(self.ram_gib) or self.ram_gib <= 0 or any(v < 0 for v in self.shared_artifacts.values()):
            raise ValueError("Invalid worker resource request")


class ResourceAdmission:
    def __init__(self, max_jobs, cpu_threads, ram_gib, *, resident_gib=0, emergency_gib=0, max_preparations=None, max_fits=None, max_expensive_fits=None, aging_seconds=30):
        if min(max_jobs,cpu_threads,ram_gib) <= 0:
            raise ValueError("Resource budgets must be positive")
        self.max_jobs, self.cpu_threads, self.ram_gib = max_jobs,cpu_threads,ram_gib
        self.active = {}
        self.resident_gib, self.emergency_gib = resident_gib, emergency_gib
        self.max_preparations = max_preparations
        self.max_fits = max_fits
        self.max_expensive_fits = max_expensive_fits
        self.aging_seconds = aging_seconds
        self.memory_headroom_gib = None
        self.cgroup_headroom_gib = None
        self.disk_headroom_bytes = None
        self.deferred_since = {}
        self.last_blockers = {}
        if min(resident_gib,emergency_gib,aging_seconds) < 0:
            raise ValueError("Invalid residency/headroom/aging policy")
        if any(value is not None and value < 1 for value in (max_preparations,max_fits,max_expensive_fits)):
            raise ValueError("Positive separate preparation/fit limits required")

    @staticmethod
    def _private(request):
        return request.private_peak_bytes/GIB if isinstance(request,JobEstimate) else request.ram_gib

    @staticmethod
    def _shared(requests):
        shared = {}
        for request in requests:
            for identity,size in request.shared_artifacts.items():
                if identity in shared and shared[identity] != size:
                    raise ValueError("Shared artifact identity has conflicting size")
                shared[identity] = size
        return shared

    def set_headroom(self, *, memory_available_gib=None, cgroup_available_gib=None, disk_available_bytes=None):
        self.memory_headroom_gib = memory_available_gib
        self.cgroup_headroom_gib = cgroup_available_gib
        self.disk_headroom_bytes = disk_available_bytes

    def blockers(self, request):
        requests = [*self.active.values(),request]
        shared_before = sum(self._shared(self.active.values()).values())/GIB
        shared_after = sum(self._shared(requests).values())/GIB
        incremental = self._private(request) + shared_after-shared_before
        reasons = []
        if len(self.active) >= self.max_jobs:
            reasons.append("max_jobs")
        if sum(r.cpu_threads for r in requests) > self.cpu_threads:
            reasons.append("cpu_threads")
        if sum(self._private(r) for r in requests)+shared_after+self.resident_gib+self.emergency_gib > self.ram_gib:
            reasons.append("configured_memory")
        for label,headroom in (("host_memory",self.memory_headroom_gib),("cgroup_memory",self.cgroup_headroom_gib)):
            if headroom is not None and incremental+self.emergency_gib > headroom:
                reasons.append(label)
        if self.disk_headroom_bytes is not None and isinstance(request,JobEstimate) and request.disk_bytes > self.disk_headroom_bytes:
            reasons.append("disk")
        preparation = lambda r:isinstance(r,JobEstimate) and r.stage in {"preparation","feature_generation","selection"}
        fit = lambda r:not preparation(r) and not (isinstance(r,JobEstimate) and r.stage == "simulation")
        if self.max_preparations is not None and preparation(request) and sum(preparation(r) for r in self.active.values()) >= self.max_preparations:
            reasons.append("max_preparations")
        if self.max_fits is not None and fit(request) and sum(fit(r) for r in self.active.values()) >= self.max_fits:
            reasons.append("max_fits")
        if self.max_expensive_fits is not None and expensive_fit(request) and sum(expensive_fit(r) for r in self.active.values()) >= self.max_expensive_fits:
            reasons.append("max_expensive_fits")
        if expensive_fit(request) and request.confidence!="measured_exact_workload" and any(expensive_fit(r) for r in self.active.values()):
            reasons.append("cold_expensive_family")
        return reasons

    def admits(self, request):
        return not self.blockers(request)

    def reserve(self, key, request):
        if key in self.active or not self.admits(request):
            raise ValueError("Resource reservation exceeds policy or duplicates work")
        self.active[key] = request
        self.deferred_since.pop(key,None)
        self.last_blockers.pop(key,None)
        # Deduct new reservations until the caller refreshes observed headroom.
        shared_before = sum(self._shared([r for k,r in self.active.items() if k != key]).values())/GIB
        incremental = self._private(request)+sum(self._shared(self.active.values()).values())/GIB-shared_before
        if self.memory_headroom_gib is not None:
            self.memory_headroom_gib -= incremental
        if self.cgroup_headroom_gib is not None:
            self.cgroup_headroom_gib -= incremental
        if self.disk_headroom_bytes is not None and isinstance(request,JobEstimate):
            self.disk_headroom_bytes -= request.disk_bytes

    def release(self, key):
        del self.active[key]
        # Real host/cgroup headroom must be refreshed, never inferred from a completed fit.

    def snapshot(self):
        private = sum(self._private(r) for r in self.active.values())
        shared = sum(self._shared(self.active.values()).values())/GIB
        result = {"running_jobs":len(self.active),"reserved_cpu_threads":sum(r.cpu_threads for r in self.active.values()),"reserved_ram_gib":private+shared,"max_jobs":self.max_jobs,"cpu_budget":self.cpu_threads,"ram_budget_gib":self.ram_gib}
        result.update(max_fits=self.max_fits,max_expensive_fits=self.max_expensive_fits)
        if shared or any(isinstance(r,JobEstimate) for r in self.active.values()) or self.resident_gib or self.emergency_gib:
            result.update(private_ram_gib=private,shared_ram_gib=shared,resident_gib=self.resident_gib,emergency_gib=self.emergency_gib,admission_blockers=dict(self.last_blockers))
        return result

    def peek_feasible(self, pending, *, now=None):
        """Return the next (key,request) without reserving; pending order encodes lane fairness.

        Backfill while a large task waits. After aging_seconds stop admitting newer
        work until that task can fit after active reservations drain. Impossible jobs
        remain explicitly blocked and do not prevent feasible backfill.
        """
        now = time.monotonic() if now is None else now
        pending = list(pending.items()) if isinstance(pending,dict) else list(pending)
        keys = {key for key,_ in pending}
        self.deferred_since = {key:value for key,value in self.deferred_since.items() if key in keys}
        self.last_blockers = {}
        aged = []
        feasible = []
        for key,request in pending:
            if key in self.active:
                continue
            reasons = self.blockers(request)
            if reasons:
                self.last_blockers[key] = reasons
                since = self.deferred_since.setdefault(key,now)
                standalone = self._private(request)+sum(request.shared_artifacts.values())/GIB+self.resident_gib+self.emergency_gib
                possible = request.cpu_threads <= self.cpu_threads and standalone <= self.ram_gib
                if possible and now-since >= self.aging_seconds and not {"max_expensive_fits","cold_expensive_family"}.intersection(reasons):
                    aged.append((since,key,request))
            else:
                feasible.append((key,request))
        if aged:
            _,key,request = min(aged,key=lambda item:item[0])
            self.last_blockers["aging_reservation"] = [key]
            return (key,request) if self.admits(request) else None
        return feasible[0] if feasible else None


def memory_budget_gib(memory_budget_gb_decimal):
    if not math.isfinite(memory_budget_gb_decimal) or memory_budget_gb_decimal <= 0:
        raise ValueError("Positive decimal GB budget required")
    return memory_budget_gb_decimal * 1_000_000_000 / GIB


def expensive_fit(request):
    return isinstance(request,JobEstimate) and request.stage in {"fit","graph_component"} and (
        "gaussian_probit" in request.family or request.family.startswith("graph:"))


def fair_program_order(program_ids, reservations):
    """Durable round robin within a lane; new probes cannot starve behind a large budget."""
    last = {row["payload"].get("program_id"):index for index,row in enumerate(reservations)}
    return sorted(program_ids,key=lambda pid:(last.get(pid,-1),program_ids.index(pid)))
