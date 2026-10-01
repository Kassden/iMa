"""Resource reservations for continuous single-host dispatch."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceRequest:
    cpu_threads: int = 1
    ram_gib: float = 4

    def __post_init__(self):
        if self.cpu_threads < 1 or self.ram_gib <= 0:
            raise ValueError("Invalid worker resource request")


class ResourceAdmission:
    def __init__(self, max_jobs, cpu_threads, ram_gib):
        if min(max_jobs,cpu_threads,ram_gib) <= 0:
            raise ValueError("Resource budgets must be positive")
        self.max_jobs, self.cpu_threads, self.ram_gib = max_jobs,cpu_threads,ram_gib
        self.active = {}

    def admits(self, request):
        return len(self.active) < self.max_jobs and sum(r.cpu_threads for r in self.active.values()) + request.cpu_threads <= self.cpu_threads and sum(r.ram_gib for r in self.active.values()) + request.ram_gib <= self.ram_gib

    def reserve(self, key, request):
        if key in self.active or not self.admits(request):
            raise ValueError("Resource reservation exceeds policy or duplicates work")
        self.active[key] = request

    def release(self, key):
        del self.active[key]

    def snapshot(self):
        return {"running_jobs":len(self.active),"reserved_cpu_threads":sum(r.cpu_threads for r in self.active.values()),"reserved_ram_gib":sum(r.ram_gib for r in self.active.values()),"max_jobs":self.max_jobs,"cpu_budget":self.cpu_threads,"ram_budget_gib":self.ram_gib}


def fair_program_order(program_ids, reservations):
    """Durable round robin within a lane; new probes cannot starve behind a large budget."""
    last = {row["payload"].get("program_id"):index for index,row in enumerate(reservations)}
    return sorted(program_ids,key=lambda pid:(last.get(pid,-1),program_ids.index(pid)))
