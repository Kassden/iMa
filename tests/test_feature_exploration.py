import json
import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from ima.feature_exploration import ExplorationConfig, ExplorationQueue, exploration_lock
from ima.feature_definitions import FeatureDefinition


def definition(name="load_ratio"):
    return FeatureDefinition(name=name, unit="1", input_refs=("actual_weight", "declared_weight"),
        expression_ast={"op": "divide", "args": [{"op": "ref", "ref": "actual_weight"},
                                                 {"op": "ref", "ref": "declared_weight"}]})


class FeatureQueueTests(unittest.TestCase):
    def test_live_budget_preflight_and_aggregate_limit(self):
        path = Path(__file__).resolve().parents[1]/"deploy/systemd/ima-feature-exploration-budget.py"
        spec = importlib.util.spec_from_file_location("feature_budget_test", path)
        budget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(budget)
        assert budget.MODEL_BYTES+budget.FEATURE_BYTES == 100_000_000_000
        status = {"resources": {"reserved_ram_gib": 30, "resident_gib": 6, "emergency_gib": 8}}
        assert budget.partition_safe(status, 30_000_000_000)
        assert not budget.partition_safe(status, 90_000_000_000)
        status["resources"]["reserved_ram_gib"] = 90
        assert not budget.partition_safe(status, 1_000_000_000)
        with patch.object(budget, "manager", side_effect=["active", "running", "frozen"]), patch.object(budget.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with budget.frozen_model():
                    raise RuntimeError("synthetic")
            self.assertEqual(run.call_args_list[0].args[0][-2:], ["freeze", budget.MODEL_UNIT])
            self.assertEqual(run.call_args_list[-1].args[0][-2:], ["thaw", budget.MODEL_UNIT])

    def test_inactive_main_needs_no_cgroup_or_freeze_and_rejects_racing_start(self):
        path = Path(__file__).resolve().parents[1]/"deploy/systemd/ima-feature-exploration-budget.py"
        spec = importlib.util.spec_from_file_location("feature_budget_inactive_test", path)
        budget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(budget)
        unit = {'ActiveState':'inactive', 'MainPID':'0', 'ControlGroup':''}
        def manager(*args):
            return unit[args[args.index('-p')+1]]
        with patch.object(budget, "manager", side_effect=manager), \
             patch.object(budget.subprocess, "run") as run:
            with budget.frozen_model():
                pass
            run.assert_not_called()
        with patch.object(budget, "manager", side_effect=manager), \
             patch.object(budget.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Main started"):
                with budget.frozen_model():
                    unit['ActiveState'] = 'active'
            run.assert_not_called()

    def test_restart_deduplication_and_context_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"queue.sqlite"
            queue = ExplorationQueue(path, "context-a")
            identifier = queue.enqueue([definition()])
            self.assertEqual(identifier, queue.enqueue([definition("same_formula_another_name")]))
            queue.update(identifier, "running")
            recovered = ExplorationQueue(path, "context-a")
            recovered.recover()
            self.assertEqual(recovered.rows()[0]["status"], "queued")
            self.assertEqual(len(recovered.rows()), 1)
            other = ExplorationQueue(Path(directory)/"other.sqlite", "context-b")
            self.assertNotEqual(identifier, other.enqueue([definition()]))

    def test_suggested_work_priority_and_single_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = ExplorationQueue(Path(directory)/"queue.sqlite")
            queue.enqueue([definition()])
            # A second valid formula, independent of its display name.
            other = FeatureDefinition(name="difference", unit="kg", input_refs=("actual_weight", "declared_weight"),
                expression_ast={"op": "subtract", "args": [{"op": "ref", "ref": "actual_weight"},
                                                           {"op": "ref", "ref": "declared_weight"}]})
            queue.enqueue([other], "orchestrator")
            self.assertEqual(queue.rows()[0]["source"], "orchestrator")
            with exploration_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another feature exploration"):
                    with exploration_lock(directory):
                        pass

    def test_invalid_batches_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = ExplorationQueue(root/"queue.sqlite")
            for definitions in ([], [definition(), definition("alias")]):
                with self.assertRaises(ValueError):
                    queue.enqueue(definitions)
            for changes in ({"refill_size": 0}, {"native_threads": True}, {"fit_timeout": float("inf")},
                            {"model_l2": 0}, {"disk_reserve_bytes": -1}):
                with self.assertRaises(ValueError):
                    ExplorationConfig(root, root/"features.parquet", root/"registry", **changes)


if __name__ == "__main__":
    unittest.main()
