import unittest
import tempfile
from pathlib import Path

from ima.research_resources import GIB, JobWorkload, estimate_job, JobMeasurement, JobMonitor, JobEstimator, ProgressiveCapacity
from ima.research_scheduler import ResourceAdmission, ResourceRequest, expensive_fit, memory_budget_gib


def workload(**extra):
    return JobWorkload(stage="fit",family="ridge",rows=1000,generated_features=100,selected_features=64,native_threads=2,implementation_revision="v1",dependency_versions={"sklearn":"test"},**extra)


class EstimateTests(unittest.TestCase):
    def test_observed_peak_is_a_monotone_reservation_floor(self):
        work = workload()
        estimate = estimate_job(work, cold_private_bytes=GIB, margin=1)
        resources = ResourceAdmission(4, 8, 12, emergency_gib=2)
        resources.reserve('prep', estimate)
        resources.observe_private_peak('prep', 8*GIB)
        resources.observe_private_peak('prep', GIB)
        self.assertEqual(8, resources.snapshot()['reserved_ram_gib'])
        self.assertEqual(estimate.confidence, resources.active['prep'].confidence)
        candidate = estimate_job(work, cold_private_bytes=3*GIB, margin=1)
        self.assertIn('configured_memory', resources.blockers(candidate))
        for key, value in [('unknown', 1), ('prep', -1), ('prep', True)]:
            with self.assertRaises(ValueError):
                resources.observe_private_peak(key, value)

    def test_peak_growth_debits_cached_headroom_until_fresh_sample(self):
        estimate = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        resources = ResourceAdmission(4, 8, 10)
        resources.set_headroom(memory_available_gib=4, cgroup_available_gib=4)
        resources.reserve("fit", estimate)
        self.assertEqual((3, 3), (resources.memory_headroom_gib, resources.cgroup_headroom_gib))

        resources.observe_private_peak("fit", 3 * GIB)
        candidate = ResourceRequest(ram_gib=2)
        self.assertEqual((1, 1), (resources.memory_headroom_gib, resources.cgroup_headroom_gib))
        self.assertIn("host_memory", resources.blockers(candidate))
        self.assertIn("cgroup_memory", resources.blockers(candidate))
        self.assertNotIn("configured_memory", resources.blockers(candidate))

        # A smaller observation preserves both the reservation floor and its debit.
        resources.observe_private_peak("fit", 2 * GIB)
        self.assertEqual(3, resources.active["fit"].private_peak_bytes / GIB)
        self.assertEqual((1, 1), (resources.memory_headroom_gib, resources.cgroup_headroom_gib))
        self.assertIn("host_memory", resources.blockers(candidate))
        self.assertIn("cgroup_memory", resources.blockers(candidate))

        # A new host/cgroup sample replaces cached deductions and reflects current availability.
        resources.set_headroom(memory_available_gib=4, cgroup_available_gib=4)
        self.assertTrue(resources.admits(candidate))

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
    def test_default_four_cold_probes_and_release_backfill(self):
        admission = ResourceAdmission(8, 16, 32, aging_seconds=10)
        cold = estimate_job(workload().model_copy(update={"family": "graph:ridge"}),
                            cold_private_bytes=GIB, margin=1)
        cheap = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        for number in range(4):
            self.assertTrue(admission.admits(cold))
            admission.reserve(f"cold-{number}", cold)
            self.assertEqual(number + 1, admission.snapshot()["running_cold_expensive_fits"])
        self.assertEqual(4, admission.snapshot()["max_cold_expensive_fits"])
        self.assertEqual(["cold_expensive_family"], admission.blockers(cold))
        with self.assertRaisesRegex(ValueError, "exceeds policy"):
            admission.reserve("fifth", cold)
        pending = [("fifth", cold), ("cheap", cheap)]
        for now in (0, 11):
            self.assertEqual(("cheap", cheap), admission.peek_feasible(pending, now=now))
            self.assertNotIn("aging_reservation", admission.last_blockers)
        admission.release("cold-0")
        self.assertEqual(3, admission.snapshot()["running_cold_expensive_fits"])
        self.assertEqual(("fifth", cold), admission.peek_feasible(pending, now=12))

    def test_measured_fits_do_not_consume_explicit_single_cold_quota(self):
        admission = ResourceAdmission(8, 16, 32, max_cold_expensive_fits=1)
        work = workload().model_copy(update={"family": "gaussian_probit"})
        measured = estimate_job(work, [JobMeasurement(work.fingerprint(), GIB, 5.)] * 5, margin=1)
        cold = estimate_job(work, cold_private_bytes=GIB, margin=1)
        admission.reserve("warm", measured)
        self.assertEqual(0, admission.snapshot()["running_cold_expensive_fits"])
        admission.reserve("cold", cold)
        admission.reserve("warm-two", measured)
        self.assertEqual(1, admission.snapshot()["running_cold_expensive_fits"])
        self.assertEqual(["cold_expensive_family"], admission.blockers(cold))

    def test_cold_probe_cap_must_be_positive_integer(self):
        for cap in (0, -1, None, True, 1.5, float("inf")):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                ResourceAdmission(8, 16, 32, max_cold_expensive_fits=cap)

    def test_parallel_cold_probes_preserve_resource_guards(self):
        cold = estimate_job(workload().model_copy(update={"family": "graph:ridge"}),
                            cold_private_bytes=GIB, margin=1)
        for blocker, policy, headroom in (
            ("configured_memory", {"ram_gib": 1}, {}),
            ("host_memory", {}, {"memory_available_gib": 1}),
            ("cgroup_memory", {}, {"cgroup_available_gib": 1}),
            ("cpu_threads", {"cpu_threads": 2}, {}),
            ("disk", {}, {"disk_available_bytes": cold.disk_bytes}),
            ("max_fits", {"max_fits": 1}, {}),
            ("max_expensive_fits", {"max_expensive_fits": 1}, {}),
            ("max_jobs", {"max_jobs": 1}, {}),
        ):
            with self.subTest(blocker=blocker):
                admission = ResourceAdmission(**dict({"max_jobs": 8, "cpu_threads": 16, "ram_gib": 32}, **policy))
                admission.set_headroom(**headroom)
                admission.reserve("first", cold)
                self.assertIn(blocker, admission.blockers(cold))
                self.assertNotIn("cold_expensive_family", admission.blockers(cold))
                with self.assertRaisesRegex(ValueError, "exceeds policy"):
                    admission.reserve("second", cold)
                if headroom:
                    admission.release("first")
                    self.assertIn(blocker, admission.blockers(cold))

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

    def test_explicit_single_cold_probe_allows_cheap_backfill(self):
        work = workload().model_copy(update={"family": "gaussian_probit"})
        cold = estimate_job(work, cold_private_bytes=GIB, margin=1)
        cheap = estimate_job(workload(), cold_private_bytes=GIB, margin=1)
        self.assertNotEqual("measured_exact_workload", cold.confidence)
        for active_family in ("gaussian_probit", "graph:ridge"):
            with self.subTest(active_family=active_family):
                admission = ResourceAdmission(8, 16, 16, max_fits=8,
                                              max_expensive_fits=7, max_cold_expensive_fits=1, aging_seconds=10)
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
            self.assertEqual(("cold-probit", cold), admission.peek_feasible(pending, now=now))
            self.assertNotIn("cold-probit", admission.last_blockers)
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
