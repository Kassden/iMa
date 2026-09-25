import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ima.research_evidence import (
    metric_value,
    mlflow_sample_readback,
    read_jsonl,
    summarize_campaign,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class ResearchEvidenceTests(unittest.TestCase):
    def test_metric_value_reads_nested_numeric_paths(self):
        self.assertEqual(2.0, metric_value({"a": {"b": 2}}, "a.b"))
        self.assertIsNone(metric_value({"a": {"b": True}}, "a.b"))
        self.assertIsNone(metric_value({"a": {}}, "a.b"))

    def test_read_jsonl_reports_corrupt_rows_without_dropping_valid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"ok": 1}\nnot-json\n[]\n', encoding="utf-8")
            result = read_jsonl(path)
        self.assertEqual([{"ok": 1}], result.rows)
        self.assertEqual(2, len(result.errors))
        self.assertEqual(2, result.errors[0]["line"])
        self.assertEqual(3, result.errors[1]["line"])

    def test_summarizes_campaign_counts_and_selected_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "trials.jsonl", [
                {
                    "trial_id": "trial-0001",
                    "run_id": "a",
                    "status": "completed",
                    "metrics": {
                        "kind": "logit",
                        "feature_schema": "baseline-v1",
                        "fundamental_weight": 0.0,
                        "market_weight": 1.0,
                        "test_blended": {"race_log_loss": 2.2},
                        "test_fundamental": {"race_log_loss": 2.7},
                        "mlflow_run_id": "run-a",
                    },
                },
                {
                    "trial_id": "trial-0002",
                    "run_id": "b",
                    "status": "failed",
                    "error": "boom",
                },
                {
                    "trial_id": "trial-0003",
                    "run_id": "c",
                    "status": "completed",
                    "metrics": {
                        "kind": "boosted",
                        "feature_schema": "rich-v1",
                        "fundamental_weight": 0.4,
                        "market_weight": 0.8,
                        "test_blended": {"race_log_loss": 2.0},
                        "test_fundamental": {"race_log_loss": 2.1},
                        "mlflow_run_id": "run-c",
                    },
                },
            ])
            write_jsonl(root / "decisions.jsonl", [
                {"decision": "continue", "next_action": "run_next_trial"}
            ])
            summary = summarize_campaign(root)

        self.assertEqual(3, summary["trial_rows"])
        self.assertEqual(2, summary["completed_trials"])
        self.assertEqual({"completed": 2, "failed": 1}, summary["statuses"])
        self.assertEqual({"boosted": 1, "logit": 1}, summary["families"])
        self.assertEqual({"baseline-v1": 1, "rich-v1": 1}, summary["schemas"])
        self.assertEqual(1, summary["fundamental_weight"]["zero_count"])
        self.assertEqual("legacy_dev", summary["primary_metric"]["label"])
        self.assertEqual("c", summary["selected_trials"]["best_primary"]["run_id"])
        self.assertEqual("a", summary["selected_trials"]["first_completed"]["run_id"])
        self.assertEqual("c", summary["selected_trials"]["latest_completed"]["run_id"])
        self.assertEqual("continue", summary["latest_decision"]["decision"])

    def test_mlflow_sample_readback_uses_supplied_client(self):
        class FakeClient:
            def get_run(self, run_id):
                data = SimpleNamespace(metrics={"m": 1.0}, params={"p": "x"})
                info = SimpleNamespace(status="FINISHED", experiment_id="1")
                return SimpleNamespace(info=info, data=data)

            def list_artifacts(self, run_id):
                return [SimpleNamespace(path="context.json"), SimpleNamespace(path="models")]

        payload = mlflow_sample_readback(
            {"best": {"mlflow_run_id": "abc"}},
            "http://mlflow.local",
            client=FakeClient(),
        )
        self.assertTrue(payload["enabled"])
        self.assertEqual("FINISHED", payload["samples"]["best"]["status"])
        self.assertEqual(["context.json", "models"], payload["samples"]["best"]["artifact_paths"])

    def test_cli_writes_summary_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign"
            campaign.mkdir()
            write_jsonl(campaign / "trials.jsonl", [
                {
                    "trial_id": "trial-0001",
                    "run_id": "demo",
                    "status": "completed",
                    "metrics": {
                        "kind": "logit",
                        "feature_schema": "baseline-v1",
                        "test_blended": {"race_log_loss": 2.0},
                    },
                }
            ])
            output = root / "summary.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.summarize_optimizer_campaign",
                    str(campaign),
                    "--output",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual("", completed.stderr)
            self.assertEqual(0, completed.returncode)
            summary = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(1, summary["completed_trials"])
        self.assertIn('"completed_trials": 1', completed.stdout)


if __name__ == "__main__":
    unittest.main()
