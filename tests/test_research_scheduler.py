import unittest
import tempfile
from pathlib import Path

from ima.research_resources import GIB, JobWorkload, estimate_job, JobMeasurement, JobMonitor, JobEstimator, ProgressiveCapacity
from ima.research_scheduler import ResourceAdmission, ResourceRequest, expensive_fit, memory_budget_gib


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
    def test_expensive_fit_classification_excludes_preparation_and_cheap_families(self):
        for family, stage, expected in (
            ("gaussian_probit", "fit", True),
            ("graph:benter_conditional_logit", "fit", True),
            ("gaussian_probit", "graph_component", True),
            ("gaussian_probit", "preparation", False),
            ("gaussian_probit", "selection", False),
            ("gaussian_probit", "simulation", False),
            ("benter_conditional_logit", "fit", False),
            ("boosted", "fit", False),
        ):
            with self.subTest(family=family, stage=stage):
                estimate = estimate_job(workload().model_copy(update={"family": family, "stage": stage}))
                self.assertEqual(expected, expensive_fit(estimate))
        self.assertFalse(expensive_fit(ResourceRequest()))

    def test_expensive_fit_cap_allows_cheap_backfill_even_after_aging(self):
        admission = ResourceAdmission(4, 8, 16, max_fits=2, max_expensive_fits=1, aging_seconds=10)
        work = workload().model_copy(update={"family": "gaussian_probit"})
        samples = [JobMeasurement(work.fingerprint(), GIB, 5.) for _ in range(5)]
        heavy = estimate_job(work, samples,
                             cold_private_bytes=GIB, margin=1)
        cheap = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        admission.reserve("slow-probit", heavy)
        pending = [("second-probit", heavy), ("cheap", cheap)]
        for now in (0, 11):
            self.assertEqual(("cheap", cheap), admission.peek_feasible(pending, now=now))
            self.assertEqual(["max_expensive_fits"], admission.last_blockers["second-probit"])
            self.assertNotIn("aging_reservation", admission.last_blockers)
        admission.reserve("cheap", cheap)
        self.assertEqual(2, admission.snapshot()["running_jobs"])
        self.assertIn("max_fits", admission.blockers(cheap))
        admission.release("slow-probit")
        self.assertEqual(("second-probit", heavy), admission.peek_feasible(pending, now=12))
        admission.reserve("second-probit", heavy)
        self.assertEqual({"cheap", "second-probit"}, set(admission.active))

    def test_cold_probit_remains_single_at_cap_eight_and_allows_cheap_backfill(self):
        work = workload().model_copy(update={"family": "gaussian_probit"})
        cold = estimate_job(work, cold_private_bytes=GIB, margin=1)
        cheap = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        self.assertNotEqual("measured_exact_workload", cold.confidence)
        for active_family in ("gaussian_probit", "graph:ridge"):
            with self.subTest(active_family=active_family):
                admission = ResourceAdmission(8, 16, 16, max_fits=8,
                                              max_expensive_fits=7, aging_seconds=10)
                active = estimate_job(work.model_copy(update={"family": active_family}),
                                      cold_private_bytes=GIB, margin=1)
                admission.reserve("active-expensive", active)
                self.assertEqual(["cold_expensive_family"], admission.blockers(cold))
                with self.assertRaisesRegex(ValueError, "exceeds policy"):
                    admission.reserve("second-cold-probit", cold)
                pending = [("second-cold-probit", cold), ("cheap", cheap)]
                for now in (0, 11):
                    self.assertEqual(("cheap", cheap), admission.peek_feasible(pending, now=now))
                    self.assertEqual(["cold_expensive_family"], admission.last_blockers["second-cold-probit"])
                    self.assertNotIn("aging_reservation", admission.last_blockers)
                admission.reserve("cheap", cheap)
                self.assertEqual(1, sum(expensive_fit(request) for request in admission.active.values()))
                admission.release("active-expensive")
                self.assertEqual(("second-cold-probit", cold), admission.peek_feasible(pending, now=12))
                admission.reserve("second-cold-probit", cold)
                self.assertEqual({"cheap", "second-cold-probit"}, set(admission.active))

    def test_measured_expensive_family_can_expand_and_preserves_cheap_slot(self):
        admission = ResourceAdmission(8, 16, 16, max_fits=8,
                                      max_expensive_fits=7, aging_seconds=10)
        work = workload().model_copy(update={"family": "gaussian_probit"})
        samples = [JobMeasurement(work.fingerprint(), GIB, 5.) for _ in range(5)]
        measured = estimate_job(work, samples, margin=1)
        cold = estimate_job(work, cold_private_bytes=GIB, margin=1)
        cheap = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        self.assertEqual("measured_exact_workload", measured.confidence)
        admission.reserve("measured-one", measured)
        self.assertEqual([], admission.blockers(measured))
        admission.reserve("measured-two", measured)
        self.assertEqual(2, admission.snapshot()["running_jobs"])
        pending = [("cold-probit", cold), ("cheap", cheap)]
        for now in (0, 11):
            self.assertEqual(("cheap", cheap), admission.peek_feasible(pending, now=now))
            self.assertEqual(["cold_expensive_family"], admission.last_blockers["cold-probit"])
        for number in range(3, 8):
            admission.reserve(f"measured-{number}", measured)
        self.assertEqual(["max_expensive_fits"], admission.blockers(measured))
        pending = [("eighth-expensive", measured), ("cheap", cheap)]
        for now in (12, 23):
            self.assertEqual(("cheap", cheap), admission.peek_feasible(pending, now=now))
            self.assertEqual(["max_expensive_fits"], admission.last_blockers["eighth-expensive"])
            self.assertNotIn("aging_reservation", admission.last_blockers)
        admission.reserve("cheap", cheap)
        self.assertEqual(8, admission.snapshot()["running_jobs"])
        self.assertEqual(7, sum(expensive_fit(request) for request in admission.active.values()))

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
