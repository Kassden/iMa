import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


class AgenticOptimizerE2ETests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        frame = pd.read_csv("tests/fixtures/research_races.csv")
        frame["horse_rating"] = 60 + frame["horse_no"]
        frame["horse_age"] = 4 + (frame["horse_no"] % 3)
        frame["field_size"] = frame.groupby("race_id")["race_id"].transform("size")
        dataset = root / "dataset.csv"
        frame.to_csv(dataset, index=False)
        protocol = root / "protocol.json"
        protocol.write_text(json.dumps({
            "min_train_races": 2,
            "calibration_races": 1,
            "score_races": 1,
            "max_folds": 1,
        }), encoding="utf-8")
        config = root / "config.json"
        config.write_text(json.dumps({
            "schema_version": 1,
            "policy": "agentic",
            "planner_mode": "fixture",
            "dataset_path": str(dataset),
            "protocol_path": str(protocol),
            "max_trials": 9,
            "proposal_batch_size": 3,
            "max_concurrent_trials": 2,
            "replan_every_terminal_trials": 3,
        }), encoding="utf-8")
        return config

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["IMA_RESEARCH_DISK_PAUSE_GIB"] = "0"
        environment["IMA_RESEARCH_RESERVE_MEMORY_GIB"] = "0"
        environment["IMA_RESEARCH_CPU_CEILING_PERCENT"] = "101"
        return subprocess.run(
            [sys.executable, "-m", "scripts.optimize", *arguments],
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )

    def test_terminal_agentic_run_executes_two_feedback_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            campaign = root / "campaign"
            run = self.run_cli(
                "run", "--campaign", str(campaign), "--config", str(config)
            )
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertIn("Mode: complete", run.stdout)

            trials = [
                json.loads(line)
                for line in (campaign / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            decisions = [
                json.loads(line)
                for line in (campaign / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(9, len(trials))
            self.assertEqual(["bootstrap", "fixture", "fixture"], [
                decision["source"] for decision in decisions
            ])
            self.assertEqual(3, decisions[1]["completed_trial_count"])
            self.assertEqual(6, decisions[2]["completed_trial_count"])
            cycle_one_ids = {trial["attempt_id"] for trial in trials[:6]}
            self.assertTrue(
                set(decisions[2]["evidence_trial_ids"]).issubset(cycle_one_ids)
            )
            executed_hashes = {trial["recipe_hash"] for trial in trials}
            for suggestion in decisions[2]["suggestions"]:
                self.assertIn(suggestion["recipe_hash"], executed_hashes)

            status = self.run_cli("status", "--campaign", str(campaign))
            self.assertEqual(0, status.returncode, status.stderr)
            status_payload = json.loads(status.stdout)
            self.assertEqual(9, status_payload["ledger"]["completed"])
            self.assertFalse(status_payload["controller_online"])

    def test_dry_run_does_not_create_study_or_call_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            campaign = root / "preview"
            preview = self.run_cli(
                "run", "--campaign", str(campaign), "--config", str(config), "--dry-run"
            )
            self.assertEqual(0, preview.returncode, preview.stderr)
            self.assertFalse((campaign / "search" / "optuna-journal.log").exists())
            payload = json.loads((campaign / "dry-run.json").read_text(encoding="utf-8"))
            self.assertEqual("dry_run", payload["mode"])
            self.assertEqual(0, payload["resolved_concurrency"])

    def test_terminal_campaign_exposes_run_metrics_and_cycle_trace_in_mlflow(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            values = json.loads(config.read_text(encoding="utf-8"))
            values.update({
                "max_trials": 1,
                "proposal_batch_size": 1,
                "max_concurrent_trials": 1,
                "mlflow_tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
            })
            config.write_text(json.dumps(values), encoding="utf-8")
            campaign = root / "tracked-campaign"
            run = self.run_cli(
                "run", "--campaign", str(campaign), "--config", str(config)
            )
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertIn("Mode: complete", run.stdout)

            trace_files = list((campaign / "traces").glob("cycle-*.json"))
            self.assertEqual(1, len(trace_files))
            linkage = json.loads(trace_files[0].read_text(encoding="utf-8"))
            with mock.patch.dict("os.environ", {}, clear=False):
                mlflow.set_tracking_uri(values["mlflow_tracking_uri"])
                trace = mlflow.get_trace(linkage["trace_id"], flush=True)
                self.assertEqual("optimizer-cycle-0000", trace.info.tags["mlflow.traceName"])
                experiment = mlflow.get_experiment_by_name("ima-agentic-v2")
                runs = mlflow.search_runs([experiment.experiment_id])
                self.assertEqual(1, len(runs))
                self.assertIn(
                    "metrics.summary.selected.race_log_loss.mean", runs.columns
                )


if __name__ == "__main__":
    unittest.main()
