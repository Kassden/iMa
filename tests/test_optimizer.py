import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ima.experiments import ExperimentSpec, experiment_specs
from ima import openrouter_orchestrator
from ima.openrouter_orchestrator import OpenRouterConfig, OpenRouterError
from ima.openrouter_orchestrator import _proposal_payload_from_response
from ima.optimizer import (
    CampaignConfig,
    ExperimentProposal,
    _run_trial_worker,
    completed_run_ids,
    dry_run,
    local_proposals,
    openrouter_proposals,
    resolved_concurrency,
    validate_proposal_batch,
    voting_rank,
)


class OptimizerTests(unittest.TestCase):
    def test_campaign_config_validates_budget_and_openrouter_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                CampaignConfig(Path(directory), max_trials=0).validate()
            with self.assertRaises(ValueError):
                CampaignConfig(Path(directory), max_trials=1, proposal_batch_size=2).validate()
            with self.assertRaises(ValueError):
                CampaignConfig(Path(directory), policy="local", openrouter_batch=True).validate()
            CampaignConfig(
                Path(directory),
                max_trials=2,
                proposal_batch_size=2,
                max_concurrent_trials=2,
                service_tier="flex",
            ).validate()
            CampaignConfig(
                Path(directory),
                max_trials=2,
                proposal_batch_size=2,
                max_concurrent_trials="auto",
                spec_profile="long",
            ).validate()

    def test_proposal_rejects_forbidden_leakage_terms(self):
        proposal = ExperimentProposal(
            trial_id="trial-0001",
            hypothesis="Use final_odds to improve the model",
            changed_surface="feature_family",
            spec=ExperimentSpec("demo", "logit", {"C": 0.5}),
        )
        with self.assertRaises(ValueError):
            proposal.validate()

    def test_batch_rejects_duplicate_specs(self):
        proposals = [
            ExperimentProposal("trial-0001", "a", "hyperparameters", ExperimentSpec("same", "logit", {})),
            ExperimentProposal("trial-0002", "b", "hyperparameters", ExperimentSpec("same", "logit", {})),
        ]
        with self.assertRaises(ValueError):
            validate_proposal_batch(proposals)

    def test_local_policy_skips_completed_runs_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trials.jsonl").write_text(
                json.dumps({"run_id": "logit-c005-balanced", "status": "completed"}) + "\n",
                encoding="utf-8",
            )
            config = CampaignConfig(root, max_trials=2, proposal_batch_size=2)
            proposals = local_proposals(config)
            self.assertNotEqual("logit-c005-balanced", proposals[0].spec.run_id)
            self.assertIn("logit-c005-balanced", completed_run_ids(root))

    def test_local_policy_stops_at_max_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "trials.jsonl").write_text(
                json.dumps({"run_id": "logit-c005-balanced", "status": "completed"}) + "\n",
                encoding="utf-8",
            )
            config = CampaignConfig(root, max_trials=1, proposal_batch_size=1)
            self.assertEqual([], local_proposals(config))

    def test_local_policy_accepts_unlimited_campaign_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CampaignConfig(
                Path(directory),
                max_trials=None,
                proposal_batch_size=40,
                max_concurrent_trials="auto",
                spec_profile="long",
            )
            proposals = local_proposals(config)
            self.assertEqual(40, len(proposals))
            self.assertGreaterEqual(resolved_concurrency(config, len(proposals)), 1)

    def test_long_profile_provides_unattended_experiment_catalogue(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CampaignConfig(
                Path(directory),
                max_trials=1000,
                proposal_batch_size=32,
                max_concurrent_trials="auto",
                spec_profile="long",
            )
            proposals = local_proposals(config)
            self.assertEqual(32, len(proposals))
            self.assertGreaterEqual(resolved_concurrency(config, len(proposals)), 1)
            self.assertTrue(any("long" in proposal.spec.run_id for proposal in proposals))

    def test_adaptive_profile_drains_long_profile_first(self):
        with tempfile.TemporaryDirectory() as directory:
            config = CampaignConfig(
                Path(directory),
                max_trials=None,
                proposal_batch_size=4,
                spec_profile="adaptive",
            )
            proposals = local_proposals(config)
            self.assertEqual(
                [spec.run_id for spec in experiment_specs("long")[:4]],
                [proposal.spec.run_id for proposal in proposals],
            )

    def test_adaptive_profile_generates_deterministic_specs_after_long_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for spec in experiment_specs("long"):
                rows.append({
                    "run_id": spec.run_id,
                    "status": "completed",
                    "metrics": {
                        "run_id": spec.run_id,
                        "kind": spec.kind,
                        "parameters": spec.parameters,
                        "test_blended": {"race_log_loss": 2.2, "top_pick_win_rate": 0.2},
                        "test_fundamental": {"race_log_loss": 2.4},
                    },
                })
            rows[4]["metrics"]["test_blended"]["race_log_loss"] = 2.0
            rows[4]["metrics"]["test_fundamental"]["race_log_loss"] = 2.1
            (root / "trials.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            config = CampaignConfig(
                root,
                max_trials=None,
                proposal_batch_size=5,
                spec_profile="adaptive",
            )
            first = local_proposals(config)
            second = local_proposals(config)
            self.assertEqual([proposal.spec.run_id for proposal in first], [proposal.spec.run_id for proposal in second])
            self.assertTrue(all(proposal.spec.run_id.startswith("adaptive-g0001-") for proposal in first))
            self.assertTrue(any(proposal.spec.parameters.get("C") != 2.0 for proposal in first))

    def test_dry_run_writes_campaign_and_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = dry_run(CampaignConfig(root, max_trials=2, proposal_batch_size=2))
            self.assertEqual("dry_run", payload["mode"])
            self.assertEqual(2, len(payload["proposals"]))
            self.assertTrue((root / "campaign.json").exists())
            written = json.loads((root / "dry-run.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["proposals"][0]["trial_id"], written["proposals"][0]["trial_id"])

    def test_trial_worker_uses_mlflow_environment_config(self):
        captured = {}

        def fake_run_experiments(frame, output_dir, template_path, specs, mlflow_config=None):
            captured["mlflow_config"] = mlflow_config
            return {
                "runs": [{
                    "run_id": specs[0].run_id,
                    "test_blended": {"race_log_loss": 2.0},
                    "test_fundamental": {"race_log_loss": 2.1},
                    "incremental_pseudo_r2": 0.01,
                }]
            }

        proposal = ExperimentProposal(
            "trial-0001",
            "Check env tracking config.",
            "hyperparameters",
            ExperimentSpec("demo-logit", "logit", {"C": 1.0}),
        )
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://mlflow.local:5000"}):
                with mock.patch("ima.data.build_full_history_dataset", return_value=object()):
                    with mock.patch("ima.experiments.run_experiments", fake_run_experiments):
                        result = _run_trial_worker(
                            proposal.serializable(),
                            directory,
                            "docs/model-results/dashboard-template.html",
                        )
        self.assertEqual("completed", result["status"])
        self.assertTrue(captured["mlflow_config"].enabled)
        self.assertEqual("http://mlflow.local:5000", captured["mlflow_config"].tracking_uri)

    def test_voting_ranker_combines_multiple_metrics(self):
        winner, votes = voting_rank([
            {
                "run_id": "a",
                "test_blended": {
                    "race_log_loss": 2.0,
                    "top_pick_win_rate": 0.30,
                    "winner_top3_rate": 0.55,
                },
                "test_fundamental": {"race_log_loss": 2.1, "top_pick_win_rate": 0.29},
                "incremental_pseudo_r2": 0.01,
            },
            {
                "run_id": "b",
                "test_blended": {
                    "race_log_loss": 1.9,
                    "top_pick_win_rate": 0.28,
                    "winner_top3_rate": 0.58,
                },
                "test_fundamental": {"race_log_loss": 2.0, "top_pick_win_rate": 0.31},
                "incremental_pseudo_r2": 0.02,
            },
        ])
        self.assertEqual("b", winner)
        self.assertGreaterEqual(len(votes), 4)

    def test_openrouter_requires_api_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(OpenRouterError):
                OpenRouterConfig.from_env(model="openai/test")

    def test_openrouter_response_is_schema_checked_into_proposals(self):
        def fake_post(url, payload, config):
            self.assertIn("/v1/chat/completions", url)
            self.assertEqual("flex", payload["service_tier"])
            return {
                "service_tier": "flex",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "proposals": [
                                    {
                                        "run_id": "logit-c005-balanced",
                                        "hypothesis": "Check lower regularization on balanced baseline.",
                                        "changed_surface": "hyperparameters",
                                    }
                                ]
                            })
                        }
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            config = CampaignConfig(
                Path(directory),
                policy="openrouter",
                service_tier="flex",
                model="openai/test",
            )
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}):
                with mock.patch.object(openrouter_orchestrator, "_post_json", fake_post):
                    proposals, planner = openrouter_proposals(config)
            self.assertEqual("logit-c005-balanced", proposals[0].spec.run_id)
            self.assertEqual("flex", planner["service_tier"])

    def test_openrouter_parser_accepts_bare_proposal_list(self):
        payload = _proposal_payload_from_response({
            "choices": [
                {
                    "message": {
                        "content": json.dumps([
                            {
                                "run_id": "logit-c005-balanced",
                                "hypothesis": "Try baseline.",
                                "changed_surface": "hyperparameters",
                            }
                        ])
                    }
                }
            ]
        })
        self.assertEqual("logit-c005-balanced", payload["proposals"][0]["run_id"])

    def test_openrouter_batch_dry_run_persists_submission(self):
        def fake_post(url, payload, config):
            self.assertIn("/v1/batches", url)
            self.assertEqual("/v1/chat/completions", payload["endpoint"])
            self.assertEqual("flex", payload["requests"][0]["body"]["service_tier"])
            return {"id": "batch_123", "status": "validating"}

        with tempfile.TemporaryDirectory() as directory:
            config = CampaignConfig(
                Path(directory),
                policy="openrouter",
                openrouter_batch=True,
                service_tier="flex",
                model="openai/test",
            )
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}):
                with mock.patch.object(openrouter_orchestrator, "_post_json", fake_post):
                    from ima.optimizer import dry_run

                    payload = dry_run(config)
            self.assertEqual("openrouter_batch_submitted", payload["mode"])
            self.assertTrue((Path(directory) / "openrouter-batch.json").exists())


if __name__ == "__main__":
    unittest.main()
