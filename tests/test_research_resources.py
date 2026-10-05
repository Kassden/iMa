import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from ima.research_resources import (
    AdmissionPolicy,
    ResourceSnapshot,
    admission_slots,
    resource_report,
    GIB,
    JobEstimator,
    JobMeasurement,
    JobWorkload,
    ProgressiveCapacity,
    estimate_job,
    observe_cgroup,
)


class ResearchResourceTests(unittest.TestCase):
    def test_admission_uses_cpu_memory_and_requested_ceiling(self):
        snapshot = ResourceSnapshot(
            cpu_percent=30.0,
            load_percent=50.0,
            memory_available_gib=100.0,
            memory_total_gib=128.0,
            disk_free_gib=100.0,
            cpu_count=28,
        )
        slots = admission_slots(
            snapshot,
            AdmissionPolicy(estimated_peak_trial_rss_gib=4.0, configured_ceiling=32),
            requested=40,
        )
        self.assertEqual(18, slots)

    def test_disk_pressure_pauses_new_admission(self):
        snapshot = ResourceSnapshot(10.0, 10.0, 100.0, 128.0, 2.0, 28)
        self.assertEqual(0, admission_slots(snapshot))

    def test_cpu_pressure_leaves_single_slot_for_no_spike_kill_policy(self):
        snapshot = ResourceSnapshot(99.0, 50.0, 100.0, 128.0, 100.0, 28)
        self.assertEqual(1, admission_slots(snapshot))

    def test_report_expresses_load_as_percent(self):
        snapshot = ResourceSnapshot(24.123, 85.456, 80.0, 128.0, 99.0, 28)
        report = resource_report(snapshot, 12)
        self.assertEqual(85.46, report["load_percent"])
        self.assertEqual(24.12, report["cpu_percent"])
        self.assertEqual(12, report["admission_slots"])


def workload(family="gaussian_probit", **settings):
    return JobWorkload(stage="fit", family=family, rows=14000, generated_features=0,
                       selected_features=32, folds=3, implementation_revision="test",
                       dependency_versions={"test": "1"}, search_settings=settings)


class FamilyEstimateTests(unittest.TestCase):
    def test_cold_probit_exceeds_legacy_750_seconds_and_parameter_costs_scale(self):
        base = estimate_job(workload(mean_parameters=33), margin=1)
        self.assertEqual("conservative_cold", base.confidence)
        self.assertGreater(base.wall_seconds, 750)
        self.assertGreaterEqual(base.private_peak_bytes, 2 * GIB)
        for settings in ({"quadrature_order": 96}, {"heteroscedastic": True},
                         {"mean_parameters": 66}, {"parameter_dimension": 100}, {"max_iter": 400},
                         {"max_runners": 28}, {"race_count": 2000}):
            with self.subTest(settings=settings):
                heavier = estimate_job(workload(**dict({"mean_parameters": 33}, **settings)), margin=1)
                self.assertGreater(heavier.wall_seconds, base.wall_seconds)
                self.assertNotEqual(heavier.workload_fingerprint, base.workload_fingerprint)
        low = estimate_job(workload(heteroscedastic=True, variance_parameters=33), margin=1)
        high = estimate_job(workload(heteroscedastic=True, variance_parameters=66), margin=1)
        self.assertGreater(high.wall_seconds, low.wall_seconds)
        self.assertGreater(high.private_peak_bytes, low.private_peak_bytes)

    def test_explicit_analytic_mode_and_fold_cost_have_distinct_exact_keys(self):
        numeric = workload(gradient_mode="numerical")
        analytic = workload(gradient_mode="analytic")
        self.assertGreater(estimate_job(numeric).wall_seconds, estimate_job(analytic).wall_seconds)
        one = analytic.model_copy(update={"folds": 1})
        self.assertGreater(estimate_job(analytic).wall_seconds, estimate_job(one).wall_seconds)

    def test_parent_null_parameter_defaults_are_conservative(self):
        work = workload(depth=None, quadrature_order=None, heteroscedastic=None, iteration_bucket=1)
        estimate = estimate_job(work)
        self.assertGreater(estimate.wall_seconds, 750)
        self.assertEqual("conservative_cold", estimate.confidence)

    def test_success_coverage_can_lower_cold_but_failed_or_censored_bounds_stay(self):
        work = workload()
        success = [JobMeasurement(work.fingerprint(), GIB, 10) for _ in range(5)]
        failed = JobMeasurement(work.fingerprint(), 4 * GIB, 20000, failed=True)
        censored = JobMeasurement(work.fingerprint(), 6 * GIB, 40000, failed=True, censored=True)
        measured = estimate_job(work, success, margin=1)
        self.assertEqual("measured_exact_workload", measured.confidence)
        self.assertEqual(10, measured.wall_seconds)
        bounds = estimate_job(work, [*success, failed, censored], margin=1)
        self.assertEqual(6 * GIB, bounds.private_peak_bytes)
        self.assertEqual(40000, bounds.wall_seconds)
        failures = estimate_job(work, [failed] * 5, margin=1)
        self.assertEqual("conservative_cold", failures.confidence)
        self.assertGreaterEqual(failures.wall_seconds, estimate_job(work, margin=1).wall_seconds)

    def test_changed_search_settings_never_reuse_exact_samples(self):
        work = workload(quadrature_order=48, heteroscedastic=False, mean_parameters=33)
        samples = [JobMeasurement(work.fingerprint(), GIB, 10)] * 5
        for setting in ({"quadrature_order": 96}, {"heteroscedastic": True}, {"mean_parameters": 66}):
            changed = work.model_copy(update={"search_settings": dict(work.search_settings, **setting)})
            self.assertEqual(0, estimate_job(changed, samples).sample_count)

    def test_preparation_growth_and_durable_failure_bounds(self):
        work = workload("ridge").model_copy(update={"stage": "feature_generation"})
        generated = work.model_copy(update={"generated_features": 2000})
        self.assertGreater(estimate_job(generated).private_peak_bytes, estimate_job(work).private_peak_bytes)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "samples.json"
            estimator = JobEstimator(path)
            estimator.record(work, JobMeasurement(work.fingerprint(), 12 * GIB, 2000, True, True))
            restart = JobEstimator(path).estimate(work, margin=1)
            self.assertEqual(12 * GIB, restart.private_peak_bytes)
            self.assertEqual(2000, restart.preparation_seconds)
            self.assertEqual("conservative_cold", restart.confidence)

    def test_boosted_cost_scales_but_shared_identity_is_not_duplicated_in_private_charge(self):
        base = workload("boosted", depth=6, max_iter=200)
        deep = workload("boosted", depth=12, max_iter=400)
        self.assertGreater(estimate_job(deep).wall_seconds, estimate_job(base).wall_seconds)
        shared = base.model_copy(update={"shared_artifacts": {"immutable-fold": GIB}})
        self.assertEqual(estimate_job(base).private_peak_bytes, estimate_job(shared).private_peak_bytes)
        self.assertEqual({"immutable-fold": GIB}, estimate_job(shared).shared_artifacts)

    def test_invalid_cost_fields_are_rejected(self):
        for settings in ({"quadrature_order": 0}, {"max_iter": float("inf")},
                         {"mean_parameters": -2}, {"heteroscedastic": "true"},
                         {"gradient_mode": "guess"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                estimate_job(workload(**settings))


class RecoveryCapacityTests(unittest.TestCase):
    def observe(self, capacity, now, folds, headroom=100):
        return capacity.observe(headroom_gib=headroom, private_per_fit_gib=2,
                                representative_folds=folds, now=now)

    def test_post_restart_canary_before_first_sample_counts_once(self):
        capacity = ProgressiveCapacity(representative_baseline=5)
        self.assertEqual(2, self.observe(capacity, 0, 6)["cap"])
        self.assertEqual(2, self.observe(capacity, 29, 6)["cap"])
        self.assertEqual(4, self.observe(capacity, 30, 6)["cap"])
        self.assertEqual(4, self.observe(capacity, 31, 6)["cap"])
        self.assertEqual(4, self.observe(capacity, 61, 6)["cap"])
        self.assertEqual(8, self.observe(capacity, 62, 7)["cap"])

    def test_old_representative_samples_cannot_advance_and_reset_returns_cap_two(self):
        capacity = ProgressiveCapacity(representative_baseline=5)
        self.observe(capacity, 0, 5)
        self.assertEqual(2, self.observe(capacity, 100, 5)["cap"])
        self.assertEqual(4, self.observe(capacity, 101, 6)["cap"])
        capacity.reset(representative_baseline=6)
        self.assertEqual(2, capacity.cap)
        self.observe(capacity, 102, 6)
        self.assertEqual(2, self.observe(capacity, 132, 6)["cap"])

    def test_headroom_drop_restarts_thirty_seconds_without_consuming_canary(self):
        capacity = ProgressiveCapacity(representative_baseline=0)
        self.observe(capacity, 0, 1)
        self.assertEqual("increment_headroom", self.observe(capacity, 20, 1, headroom=5)["reason"])
        self.observe(capacity, 21, 1)
        self.assertEqual(2, self.observe(capacity, 50, 1)["cap"])
        self.assertEqual(4, self.observe(capacity, 51, 1)["cap"])
        self.assertEqual(0, self.observe(capacity, 52, 1, headroom=1)["cap"])

    def test_default_ladder_never_jumps_or_reuses_coverage(self):
        capacity = ProgressiveCapacity(representative_baseline=0)
        now = 0
        for expected, count in zip((4, 8, 12, 16, 26), range(1, 6)):
            self.observe(capacity, now, count)
            self.assertEqual(expected, self.observe(capacity, now + 30, count)["cap"])
            now += 31
        self.assertEqual(26, self.observe(capacity, now + 100, 1000)["cap"])

    def test_invalid_reset_threshold(self):
        for baseline in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                ProgressiveCapacity(representative_baseline=baseline)


class OwnCgroupTests(unittest.TestCase):
    def test_missing_peak_file_keeps_actual_counters_available(self):
        values = {"/proc/self/cgroup": "0::/own\n", "/sys/fs/cgroup/own/memory.current": "100",
                  "/sys/fs/cgroup/own/memory.max": "100000000000", "/sys/fs/cgroup/own/memory.events": "oom 1\noom_kill 1",
                  "/proc/sys/kernel/random/boot_id": "boot"}
        def read(path, *args, **kwargs):
            if str(path) not in values:
                raise FileNotFoundError(str(path))
            return values[str(path)]
        with patch.object(Path, "read_text", read):
            sample = observe_cgroup()
        self.assertTrue(sample["supported"])
        self.assertEqual(100, sample["memory.current"])
        self.assertEqual("boot", sample["boot_id"])
        self.assertNotIn("memory.peak", sample)


if __name__ == "__main__":
    unittest.main()
