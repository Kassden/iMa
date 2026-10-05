import hashlib
import json
import socket
import tempfile
import unittest
import warnings
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_v6_reliability_canary as canary
from tests.test_research_executor_v6 import dataset


class ReliabilityCanaryTests(unittest.TestCase):
    def test_existing_output_directory_file_and_dangling_symlink_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "canary"
            existing.mkdir()
            sentinel = existing / "receipt.json"
            sentinel.write_text("immutable existing canary")
            file = root / "existing-file"
            file.write_text("immutable file")
            link = root / "dangling-output"
            link.symlink_to(root / "missing-target")
            with patch.object(canary, "BoundedFitExecutor") as executor:
                for output in (existing, file, link):
                    with self.subTest(output=output), self.assertRaises(FileExistsError):
                        canary.run_canary(root / "missing-dataset", output, "revision")
                executor.assert_not_called()
            self.assertEqual("immutable existing canary", sentinel.read_text())
            self.assertEqual("immutable file", file.read_text())
            self.assertTrue(link.is_symlink())
            self.assertFalse((root / "missing-target").exists())

    def test_real_sequential_models_prewarm_once_readonly_and_finite_objectives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = dataset(root, races=260)
            original = source.read_bytes()
            output = root / "canary"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                receipt = canary.run_canary(source, output, "scientific-canary-test")
            stored = json.loads((output / "receipt.json").read_text())
            self.assertEqual("completed", receipt["status"], receipt)
            self.assertTrue(stored["completed"])
            self.assertEqual(hashlib.sha256(original).hexdigest(), stored["dataset_hash"])
            self.assertEqual(original, source.read_bytes())
            self.assertEqual(1, stored["max_workers"])
            self.assertEqual(600, stored["per_fit_timeout_seconds"])
            self.assertFalse(stored["tracking_enabled"])
            self.assertEqual(canary.PROTOCOL, stored["protocol_parameters"])
            self.assertEqual(3, len(stored["preparation_dependency_ids"]))
            self.assertEqual(1, len(set(stored["preparation_dependency_ids"])))
            self.assertEqual(1, len(stored["preparation"]["artifacts"]))
            self.assertEqual("miss", stored["preparation"]["artifacts"]["fold-001"]["cache_status"])
            self.assertEqual(1, len(list((output / "fold-preparation-cache").glob("*/manifest.json"))))
            prepared = stored["preparation"]["readonly_artifacts"]["fold-001"]
            self.assertTrue(prepared["read_only"])
            self.assertEqual([row[0] for row in canary.MODELS], [row["model_family"] for row in stored["results"]])
            for row in stored["results"]:
                self.assertTrue(row["completed"])
                self.assertTrue(row["objective_finite"])
                self.assertEqual("model", row["objective_source"])
                self.assertEqual("development_fundamental_race_log_loss", row["objective_name"])
                self.assertEqual("none", row["recipe"]["blend"]["kind"])
                self.assertEqual("baseline-v1", row["recipe"]["feature_schema"])
                self.assertEqual(3, row["recipe"]["schema_version"])
                self.assertEqual("hit", row["preparation"]["fold-001"]["cache_status"])
                self.assertEqual(prepared["artifact_id"], row["preparation"]["fold-001"]["artifact_id"])
                self.assertEqual(prepared, row["readonly_artifacts"]["fold-001"])
                self.assertTrue(row["resources"]["supported"])
                self.assertGreater(row["resources"]["private_peak_bytes"], 0)
                self.assertGreater(row["duration_seconds"], 0)
                self.assertGreater(row["supervised_wall_seconds"], 0)
                self.assertEqual("finished", row["runtime"]["status"])
                self.assertTrue(Path(row["result_path"]).is_file())
            self.assertEqual(stored["results"][0]["lineage"]["protocol_id"], stored["results"][2]["lineage"]["protocol_id"])
            with self.assertRaises(FileExistsError):
                canary.run_canary(source, output, "another-revision")
            self.assertEqual(stored, json.loads((output / "receipt.json").read_text()))

    def test_dependency_mismatch_fails_before_prewarm_or_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = dataset(root)
            with patch.object(canary, "preparation_dependency_id", side_effect=["a", "a", "b"]), \
                 patch.object(canary, "BoundedFitExecutor") as executor:
                receipt = canary.run_canary(source, root / "canary", "revision")
                executor.assert_not_called()
            self.assertFalse(receipt["completed"])
            self.assertEqual("failed", receipt["status"])
            self.assertIn("share one preparation dependency", receipt["error"])
            self.assertEqual([], receipt["results"])

    def test_timeout_is_a_failure_receipt_with_attempt_scoped_cleanup(self):
        request = Mock(output_dir=Path("canary/trials/timeout"))
        future = Future()
        future.set_exception(TimeoutError("fixture deadline"))
        executor = Mock()
        executor.submit.return_value = future
        with patch.object(canary, "cleanup_attempt_group") as cleanup, \
             patch.object(canary, "read_runtime", return_value={"status": "running", "private_peak_bytes": 123}):
            result = canary._bounded_task(executor, request)
        self.assertEqual("timeout", result["status"])
        self.assertIn("fixture deadline", result["error"])
        self.assertEqual(123, result["resources"]["private_peak_bytes"])
        self.assertTrue(result["resources"]["censored"])
        self.assertFalse(result["resources"]["supported"])
        cleanup.assert_called_once_with(request.output_dir / "runtime.json")
        self.assertEqual(600, executor.submit.call_args.kwargs["timeout"])

    def test_nonfinite_objective_never_claims_completion_or_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source, _ = dataset(Path(directory))
            recipe = canary.PipelineRecipe(schema_version=3, blend={"kind": "none"})
            request = canary.RecipeExecutionRequest("attempt", "proposal", 0, recipe, source, Path(directory),
                                                    dataset_hash="hash", code_revision="revision")
            payload = {"result": {"status": "completed", "objective_value": float("nan"),
                       "metrics": {"objective_source": "model"},
                       "lineage": {"dataset_hash": "hash", "code_revision": "revision"}},
                       "supervised_wall_seconds": 1, "runtime": {}, "resources": {}}
            result = canary._fit_receipt(request, payload, {}, "dependency")
            self.assertFalse(result["completed"])
            self.assertFalse(result["objective_finite"])
            self.assertIsNone(result["objective_value"])
            self.assertEqual("failed", result["status"])
            json.dumps(result, allow_nan=False)

    def test_worker_rejects_network_calls(self):
        def accidental_network(_):
            socket.create_connection(("example.invalid", 443))
        with patch.object(canary, "execute_recipe", side_effect=accidental_network):
            with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
                canary._measured_task(Mock(), artifacts={})

    def test_scientific_failure_retains_measured_resources(self):
        result = Mock()
        result.serializable.return_value = {"status": "failed", "error": "fixture convergence failure"}
        with patch.object(canary, "execute_recipe", return_value=result):
            payload = canary._measured_task(Mock(), artifacts={})
        self.assertEqual("failed", payload["result"]["status"])
        self.assertTrue(payload["resources"]["supported"])
        self.assertTrue(payload["resources"]["failed"])

    def test_cli_camelcase_dataset_path_and_existing_output_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("sys.stderr"):
                with self.assertRaises(SystemExit) as error:
                    canary.main(["--datasetPath", "missing", "--output", directory, "--code-revision", "revision"])
            self.assertEqual(2, error.exception.code)


if __name__ == "__main__":
    unittest.main()
