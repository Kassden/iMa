import unittest
from unittest.mock import patch

from ima.campaign_throughput_policy import qualified_capacity_class, install_throughput_policy
from ima.research_resources import ProgressiveCapacity, JobWorkload, estimate_job, GIB
from ima.research_scheduler import ResourceAdmission


class CampaignThroughputTests(unittest.TestCase):
    def test_resume_uses_slots_not_global_largest_job(self):
        capacity = qualified_capacity_class(ProgressiveCapacity)(representative_baseline=771, emergency_gib=8)
        result = capacity.observe(headroom_gib=35, private_per_fit_gib=56,
                                  representative_folds=771, now=0)
        self.assertEqual(result["cap"], 26)
        self.assertEqual(result["reason"], "qualified_resume_per_job_admission")
        self.assertEqual(capacity.observe(headroom_gib=7, private_per_fit_gib=56,
                                         representative_folds=771, now=1)["cap"], 0)
        self.assertEqual(capacity.observe(headroom_gib=35, private_per_fit_gib=56,
                                         representative_folds=771, now=2)["cap"], 26)

    def test_unqualified_campaign_keeps_progressive_evidence(self):
        for count in (0, 4):
            capacity = qualified_capacity_class(ProgressiveCapacity)(representative_baseline=count)
            self.assertEqual(capacity.cap, 2)
            self.assertEqual(capacity.observe(headroom_gib=100, private_per_fit_gib=2,
                                             representative_folds=count, now=0)["cap"], 2)
            self.assertEqual(capacity.observe(headroom_gib=100, private_per_fit_gib=2,
                                             representative_folds=count + 1, now=30)["cap"], 4)
        capacity.reset(representative_baseline=5)
        self.assertEqual(capacity.cap, 26)

    def estimate(self, gib, threads=2):
        work = JobWorkload(stage="fit", family="graph:benter_conditional_logit", rows=100,
                           generated_features=0, selected_features=4, native_threads=threads,
                           implementation_revision="test", dependency_versions={"numpy": "test"})
        return estimate_job(work, cold_private_bytes=gib * GIB, margin=1)

    def test_large_fit_does_not_serialize_independent_fits(self):
        capacity = qualified_capacity_class(ProgressiveCapacity)(representative_baseline=771)
        admission = ResourceAdmission(29, 24, 93.13, resident_gib=6, emergency_gib=8,
                                      max_fits=capacity.cap, max_expensive_fits=capacity.cap - 1)
        admission.reserve("long", self.estimate(52))
        admission.set_headroom(memory_available_gib=35, cgroup_available_gib=35,
                               disk_available_bytes=100 * GIB)
        for index in range(3):
            self.assertTrue(admission.admits(self.estimate(6)))
            admission.reserve(f"fast-{index}", self.estimate(6))
        self.assertIn("cold_expensive_family", admission.blockers(self.estimate(6)))
        self.assertIn("configured_memory", admission.blockers(self.estimate(30)))
        self.assertIn("cpu_threads", admission.blockers(self.estimate(1, threads=24)))
        admission.observe_private_peak("long", 60 * GIB)
        self.assertEqual(admission.active["long"].private_peak_bytes, 60 * GIB)
        admission.release("fast-0")
        self.assertIn("host_memory", admission.blockers(self.estimate(6)))
        # Releasing a reservation does not prove its process pages are already
        # returned. A fresh host/cgroup sample makes that capacity usable.
        admission.set_headroom(memory_available_gib=15, cgroup_available_gib=15,
                               disk_available_bytes=100 * GIB)
        self.assertTrue(admission.admits(self.estimate(6)))
        admission.set_headroom(cgroup_available_gib=1)
        self.assertIn("cgroup_memory", admission.blockers(self.estimate(6)))

    def test_install_reuses_admission_and_estimator(self):
        from ima import research_resources as resources, research_scheduler as scheduler
        original = resources.ProgressiveCapacity
        admission, estimator = scheduler.ResourceAdmission, resources.JobEstimator
        with patch.object(resources, "ProgressiveCapacity", original):
            receipt = install_throughput_policy()
            installed = resources.ProgressiveCapacity
            install_throughput_policy()
            self.assertIs(resources.ProgressiveCapacity, installed)
            self.assertIs(scheduler.ResourceAdmission, admission)
            self.assertIs(resources.JobEstimator, estimator)
            self.assertFalse(receipt["estimate_policy_changed"])


if __name__ == "__main__":
    unittest.main()
