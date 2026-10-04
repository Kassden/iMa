import unittest
import tempfile
from pathlib import Path

from ima.research_resources import GIB, JobWorkload, estimate_job, JobMeasurement, JobMonitor, JobEstimator, ProgressiveCapacity
from ima.research_scheduler import ResourceAdmission, ResourceRequest, memory_budget_gib


def workload(**extra):
    return JobWorkload(stage="fit",family="ridge",rows=1000,generated_features=100,selected_features=64,native_threads=2,implementation_revision="v1",dependency_versions={"sklearn":"test"},**extra)


class EstimateTests(unittest.TestCase):
    def test_exact_stage_keys_and_failed_lower_bounds(self):
        work = workload()
        samples = [JobMeasurement(work.fingerprint(),GIB,float(i)) for i in range(1,7)]
        estimate = estimate_job(work,samples)
        self.assertEqual(6,estimate.sample_count)
        self.assertEqual("measured_exact_workload",estimate.confidence)
        censored = JobMeasurement(work.fingerprint(),4*GIB,20.,failed=True,censored=True)
        larger = estimate_job(work,[*samples,censored])
        self.assertGreaterEqual(larger.private_peak_bytes,5*GIB)
        changed = work.model_copy(update={"rows":1001})
        self.assertEqual(0,estimate_job(changed,samples).sample_count)
        self.assertNotEqual(work.fingerprint(),work.model_copy(update={"stage":"selection"}).fingerprint())
        self.assertGreater(larger.comparison(censored)["actual"]["private_peak_bytes"],0)

    def test_current_job_measurement(self):
        with JobMonitor(interval=.01) as monitor:
            data = bytearray(2*1024**2)
            data[0] = 1
        report = monitor.report()
        self.assertTrue(report["supported"])
        self.assertGreaterEqual(report["sample_count"],2)
        self.assertGreater(report["private_peak_bytes"],0)
        self.assertNotIn("ru_maxrss",report)

    def test_durable_estimator_restart(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"samples.json"
            estimator = JobEstimator(path)
            work = workload()
            estimator.record(work,JobMeasurement(work.fingerprint(),GIB,5.))
            restarted = JobEstimator(path)
            self.assertEqual(1,restarted.estimate(work).sample_count)
            self.assertEqual(1,len(restarted.samples))

    def test_progressive_capacity_requires_30s_and_new_representative_fold(self):
        capacity = ProgressiveCapacity((2,4,8))
        def observe(now,folds=0,headroom=20):
            return capacity.observe(headroom_gib=headroom,private_per_fit_gib=2,representative_folds=folds,now=now)
        self.assertEqual(2,observe(0)["cap"])
        self.assertEqual(2,observe(29,1)["cap"])
        self.assertEqual(4,observe(30,1)["cap"])
        self.assertEqual(4,observe(31,1)["cap"])
        self.assertEqual(4,observe(61,1)["cap"])
        self.assertEqual(8,observe(62,2)["cap"])
        self.assertEqual(0,observe(63,2,headroom=1)["cap"])


class SchedulerTests(unittest.TestCase):
    def test_legacy_default_snapshot(self):
        admission = ResourceAdmission(2,4,8)
        admission.reserve("a",ResourceRequest(2,4))
        self.assertTrue(admission.admits(ResourceRequest(2,4)))
        admission.reserve("b",ResourceRequest(2,4))
        self.assertFalse(admission.admits(ResourceRequest()))
        admission.release("b")
        self.assertEqual(4,admission.snapshot()["reserved_ram_gib"])

    def test_shared_once_and_private_perfit(self):
        estimate = estimate_job(workload(shared_artifacts={"fold":2*GIB}),cold_private_bytes=GIB,margin=1)
        admission = ResourceAdmission(3,6,4)
        admission.reserve("one",estimate)
        admission.reserve("two",estimate)
        self.assertEqual(4,admission.snapshot()["reserved_ram_gib"])
        self.assertEqual(2,admission.snapshot()["shared_ram_gib"])
        self.assertFalse(admission.admits(estimate))
        admission.release("one")
        self.assertEqual(3,admission.snapshot()["reserved_ram_gib"])

    def test_backfill_then_age_reservation(self):
        admission = ResourceAdmission(4,4,8,aging_seconds=10)
        admission.reserve("active",ResourceRequest(2,4))
        pending = [("large",ResourceRequest(4,8)),("small",ResourceRequest(1,1))]
        self.assertEqual("small",admission.peek_feasible(pending,now=0)[0])
        self.assertIsNone(admission.peek_feasible(pending,now=11))
        admission.release("active")
        self.assertEqual("large",admission.peek_feasible(pending,now=12)[0])
        self.assertEqual("small",admission.peek_feasible([("impossible",ResourceRequest(9,50)),pending[1]],now=100)[0])

    def test_real_headroom_and_decimal_gb(self):
        self.assertAlmostEqual(93.13225746,memory_budget_gib(100),places=7)
        admission = ResourceAdmission(2,4,20,emergency_gib=1)
        admission.set_headroom(memory_available_gib=6,cgroup_available_gib=4)
        self.assertIn("cgroup_memory",admission.blockers(ResourceRequest(1,4)))
        admission.reserve("one",ResourceRequest(1,2))
        self.assertFalse(admission.admits(ResourceRequest(1,2)))

    def test_preparation_limit_independent(self):
        preparation = estimate_job(workload().model_copy(update={"stage":"preparation"}),cold_private_bytes=GIB,margin=1)
        admission = ResourceAdmission(4,8,10,max_preparations=1)
        admission.reserve("prepare",preparation)
        self.assertIn("max_preparations",admission.blockers(preparation))
        self.assertTrue(admission.admits(estimate_job(workload(),cold_private_bytes=GIB,margin=1)))
