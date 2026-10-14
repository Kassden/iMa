"""Restore qualified slot capacity; individual jobs still reserve real resources."""
from __future__ import annotations

VERSION = 1
MIN_SUCCESSFUL_MEASUREMENTS = 5


def qualified_capacity_class(base):
    class QualifiedCapacity(base):
        throughput_policy_version = VERSION

        def reset(self, *, representative_baseline):
            super().reset(representative_baseline=representative_baseline)
            self.qualified_resume = representative_baseline >= MIN_SUCCESSFUL_MEASUREMENTS
            if self.qualified_resume:
                self.index = len(self.caps) - 1

        def observe(self, **kwargs):
            result = super().observe(**kwargs)
            if getattr(self, "qualified_resume", False):
                result = dict(result, policy_version=VERSION,
                              qualified_measurements=self._evidence_baseline,
                              admission_basis="per_job_cpu_private_shared_host_cgroup_disk")
                if result["reason"] == "configured_ceiling":
                    result["reason"] = "qualified_resume_per_job_admission"
            return result

    return QualifiedCapacity


def install_throughput_policy():
    from . import research_resources as resources
    from . import research_expansion as expansion
    from . import research_scheduler as scheduler
    base = resources.ProgressiveCapacity
    if getattr(base, "throughput_policy_version", None) != VERSION:
        resources.ProgressiveCapacity = qualified_capacity_class(base)
    # The controller imports ProgressiveCapacity inside its entry point. A
    # pre-bound import would silently bypass the policy, so fail closed on drift.
    if "ProgressiveCapacity" in expansion.run_expansion_campaign.__globals__:
        raise RuntimeError("Throughput policy requires controller-local capacity import")
    assert expansion.run_expansion_campaign.__code__.co_names.count("ProgressiveCapacity") == 1
    return {"policy_version": VERSION, "minimum_successful_measurements": MIN_SUCCESSFUL_MEASUREMENTS,
            "resource_admission": scheduler.ResourceAdmission.__module__,
            "estimator": resources.JobEstimator.__module__, "active_fits_cancelled": 0,
            "estimate_policy_changed": False, "scientific_admission_changed": False}
