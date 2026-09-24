import unittest

from ima.research_resources import (
    AdmissionPolicy,
    ResourceSnapshot,
    admission_slots,
    resource_report,
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


if __name__ == "__main__":
    unittest.main()
