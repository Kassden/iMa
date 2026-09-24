"""Resource observation and admission control for research workers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import psutil


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
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(path)
    return ResourceSnapshot(
        cpu_percent=float(psutil.cpu_percent(interval=0.0)),
        load_percent=float(load1 / cpu_count * 100.0),
        memory_available_gib=float(memory.available / GIB),
        memory_total_gib=float(memory.total / GIB),
        disk_free_gib=float(disk.free / GIB),
        cpu_count=int(cpu_count),
    )


def admission_slots(
    snapshot: ResourceSnapshot,
    policy: AdmissionPolicy | None = None,
    *,
    requested: int | None = None,
) -> int:
    policy = policy or AdmissionPolicy()
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
