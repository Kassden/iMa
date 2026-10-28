import unittest
from unittest.mock import patch

from ima.research_planning import planning_reason, observe_material_result


class PlanningTests(unittest.TestCase):
    def args(self, **changes):
        return dict(now=1000, last_plan=100, terminal_since_plan=0, terminal_threshold=32,
                    remaining_trials=100, low_watermark=8, active_jobs=2,
                    available_program_slots=20, checkpoint_seconds=300) | changes

    def test_timer_does_not_call_with_unchanged_useful_backlog(self):
        for now in (1000, 10000, 172800):
            self.assertIsNone(planning_reason(**self.args(now=now)))

    def test_material_champion_batch_and_exhaustion_can_plan(self):
        self.assertEqual('startup_review', planning_reason(**self.args(last_plan=0)))
        self.assertEqual('material_result_change', planning_reason(**self.args(material_change=True)))
        self.assertEqual('completed_trial_batch', planning_reason(**self.args(terminal_since_plan=32)))
        self.assertEqual('queue_refill_after_results', planning_reason(**self.args(remaining_trials=2,terminal_since_plan=1)))
        self.assertEqual('idle_queue_review', planning_reason(**self.args(remaining_trials=0,active_jobs=0)))

    def test_pending_long_fits_do_not_trigger_empty_queue_timer_calls(self):
        self.assertIsNone(planning_reason(**self.args(remaining_trials=0)))
        self.assertIsNone(planning_reason(**self.args(now=200, remaining_trials=0,active_jobs=0)))

    def test_cooldown_full_backlog_and_invalid_inputs(self):
        self.assertIsNone(planning_reason(**self.args(now=101,material_change=True)))
        self.assertEqual('idle_queue_review', planning_reason(**self.args(
            remaining_trials=100,active_jobs=0,available_program_slots=0)))
        for changes in ({'remaining_trials':-1}, {'now':float('nan')}, {'terminal_threshold':0}):
            with self.assertRaises(ValueError):
                planning_reason(**self.args(**changes))

    def test_blocked_legacy_lane_refills_without_waiting_full_checkpoint(self):
        self.assertEqual('required_lane_refill', planning_reason(**self.args(
            now=106, lane_refill_required=True)))
        self.assertIsNone(planning_reason(**self.args(now=104, lane_refill_required=True)))

    def test_champions_only_compare_with_matching_contracts(self):
        best = {'contract-a': 2.15}
        with patch('ima.research_telemetry.comparison_key', return_value='contract-a'):
            self.assertFalse(observe_material_result({'status':'completed','objective_value':2.29}, {}, best))
            self.assertTrue(observe_material_result({'status':'completed','objective_value':2.14}, {}, best))
            self.assertFalse(observe_material_result({'status':'completed','objective_value':float('nan')}, {}, best))
        with patch('ima.research_telemetry.comparison_key', return_value='contract-b'):
            self.assertTrue(observe_material_result({'status':'completed','objective_value':2.29}, {}, best))
        self.assertEqual({'contract-a':2.14,'contract-b':2.29}, best)
        self.assertTrue(observe_material_result({'status':'failed'}, {}, best))
