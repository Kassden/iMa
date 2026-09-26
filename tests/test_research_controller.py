import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ima.optimizer import CampaignConfig
from ima.research_controller import (
    DatasetFeatureProfile,
    _code_revision,
    _fixture_proposals,
    _trace_completed_cycle,
    campaign_status,
    request_campaign_stop,
    run_research_campaign,
)
from ima.research_resources import ResourceSnapshot
from ima.research_specs import PipelineRecipe
from ima.research_store import ResearchLedger


class ResearchControllerTests(unittest.TestCase):
    def setUp(self):
        healthy = ResourceSnapshot(10.0, 10.0, 100.0, 128.0, 100.0, 28)
        self.resource_patch = mock.patch(
            "ima.research_controller.observe_resources", return_value=healthy
        )
        self.resource_patch.start()

    def tearDown(self):
        self.resource_patch.stop()

    def fixture(self, root: Path) -> tuple[Path, Path]:
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
        return dataset, protocol

    def test_deployment_revision_can_be_explicit_without_git_metadata(self):
        with mock.patch.dict("os.environ", {"IMA_CODE_REVISION": "release-abc"}):
            self.assertEqual("release-abc", _code_revision())

    def test_executable_campaign_requires_explicit_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self.fixture(root)
            config = CampaignConfig(
                campaign_dir=root / "campaign",
                policy="agentic",
                planner_mode="local",
                dataset_path=dataset,
            )
            with self.assertRaisesRegex(ValueError, "explicit protocol_path"):
                run_research_campaign(config)

    def config(self, campaign: Path, dataset: Path, protocol: Path, max_trials: int):
        return CampaignConfig(
            campaign_dir=campaign,
            policy="agentic",
            planner_mode="fixture",
            dataset_path=dataset,
            protocol_path=protocol,
            max_trials=max_trials,
            proposal_batch_size=min(3, max_trials),
            max_concurrent_trials=2,
            replan_every_terminal_trials=3,
        )

    def test_run_trains_then_replans_from_completed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            payload = run_research_campaign(
                self.config(campaign, dataset, protocol, max_trials=6)
            )
            self.assertEqual("complete", payload["mode"])
            self.assertEqual(6, payload["ledger"]["completed"])
            decisions = [
                json.loads(line) for line in (campaign / "decisions.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual("bootstrap", decisions[0]["source"])
            self.assertEqual("fixture", decisions[1]["source"])
            self.assertEqual(3, len(decisions[1]["evidence_trial_ids"]))
            for proposal in decisions[1]["proposals"]:
                self.assertTrue(proposal["parent_trial_ids"])
                self.assertEqual([decisions[1]["evidence_id"]], proposal["evidence_ids"])
            self.assertEqual(6, payload["search"]["completed"])

    def test_mlflow_tracking_writes_model_run_and_cycle_trace(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = self.config(campaign, dataset, protocol, max_trials=1)
            config = CampaignConfig(**{
                **config.__dict__,
                "max_concurrent_trials": 1,
                "mlflow_tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
            })
            with mock.patch.dict("os.environ", {}, clear=False):
                payload = run_research_campaign(config)
                self.assertEqual("complete", payload["mode"])
                traces = list((campaign / "traces").glob("cycle-*.json"))
                self.assertEqual(1, len(traces))
                linkage = json.loads(traces[0].read_text(encoding="utf-8"))
                trace = mlflow.get_trace(linkage["trace_id"], flush=True)
                self.assertIsNotNone(trace)
                self.assertEqual("unavailable", linkage["cost_status"])
                experiment = mlflow.get_experiment_by_name("ima-agentic-v2")
                runs = mlflow.search_runs([experiment.experiment_id])
                self.assertEqual(1, len(runs))
                self.assertIn(
                    "metrics.summary.selected.race_log_loss.mean", runs.iloc[0]
                )

    def test_trace_failure_is_reported_without_raising(self):
        config = CampaignConfig(
            campaign_dir=Path("campaign"),
            mlflow_tracking_uri="http://mlflow.invalid",
        )
        with mock.patch(
            "ima.research_controller.log_optimizer_cycle_trace",
            side_effect=RuntimeError("tracking unavailable"),
        ):
            error = _trace_completed_cycle(
                config,
                Path("campaign"),
                {"cycle": 7, "source": "fixture"},
                [],
            )
        self.assertIn("cycle-7: trace: RuntimeError: tracking unavailable", error)

    def test_resume_preserves_first_cycle_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            first = run_research_campaign(
                self.config(campaign, dataset, protocol, max_trials=3)
            )
            second = run_research_campaign(
                self.config(campaign, dataset, protocol, max_trials=6)
            )
            self.assertEqual(3, first["ledger"]["completed"])
            self.assertEqual(6, second["ledger"]["completed"])
            lines = (campaign / "trials.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(6, len(lines))
            self.assertEqual(6, len({json.loads(line)["attempt_id"] for line in lines}))
            decisions = [
                json.loads(line) for line in (campaign / "decisions.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([0, 1], [decision["cycle"] for decision in decisions])
            first_cycle = json.loads(
                (campaign / "decisions" / "cycle-0000.json").read_text(encoding="utf-8")
            )
            second_cycle = json.loads(
                (campaign / "decisions" / "cycle-0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual("bootstrap", first_cycle["source"])
            self.assertEqual("fixture", second_cycle["source"])

    def test_counterfactual_evidence_changes_fixture_recipe(self):
        base = {
            "evidence_id": "evidence-a",
            "completed_trials": [{
                "attempt_id": "attempt-parent",
                "objective_value": 1.2,
                "mean_selected_minus_market": -0.1,
            }],
        }
        revised = json.loads(json.dumps(base))
        revised["evidence_id"] = "evidence-b"
        revised["completed_trials"][0]["mean_selected_minus_market"] = 0.1
        first = _fixture_proposals(base, 1, 1)[0]
        second = _fixture_proposals(revised, 1, 1)[0]
        self.assertNotEqual(first.recipe.recipe_hash(), second.recipe.recipe_hash())
        self.assertNotEqual(first.changed_axes, second.changed_axes)

    def test_missing_transform_column_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            frame = pd.read_csv(dataset).drop(columns=["horse_rating"])
            frame.to_csv(dataset, index=False)
            profile = DatasetFeatureProfile(dataset, json.loads(protocol.read_text()))
            recipe = PipelineRecipe(transforms=({
                "kind": "race_relative_rank",
                "parameters": {"columns": ["horse_rating"]},
            },))
            self.assertIn("horse_rating", profile.admission_error(recipe))
            evidence = {
                "evidence_id": "evidence-missing-rating",
                "completed_trials": [{
                    "attempt_id": "attempt-parent",
                    "target_kind": "win_probability",
                    "objective_value": 1.2,
                    "mean_selected_minus_market": 0.1,
                }],
                "feature_profile": profile.summary,
            }
            fallback = _fixture_proposals(evidence, 1, 1)[0]
            self.assertFalse(fallback.recipe.transforms)

    def test_status_and_stop_use_campaign_state(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            campaign.mkdir()
            marker = request_campaign_stop(campaign)
            self.assertTrue(marker.exists())
            status = campaign_status(campaign)
            self.assertEqual(str(campaign), status["campaign_dir"])

    def test_stop_marker_is_consumed_and_campaign_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            campaign.mkdir()
            request_campaign_stop(campaign)
            config = self.config(campaign, dataset, protocol, max_trials=1)
            config = CampaignConfig(**{**config.__dict__, "max_concurrent_trials": 1})
            stopped = run_research_campaign(
                config
            )
            self.assertEqual("stopped", stopped["mode"])
            self.assertFalse((campaign / "STOP").exists())
            resumed = run_research_campaign(
                config
            )
            self.assertEqual("complete", resumed["mode"])
            self.assertEqual(1, resumed["ledger"]["completed"])

    def test_resume_rejects_changed_dataset_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = self.config(campaign, dataset, protocol, max_trials=1)
            config = CampaignConfig(**{**config.__dict__, "max_concurrent_trials": 1})
            run_research_campaign(config)
            frame = pd.read_csv(dataset)
            frame.loc[0, "horse_rating"] += 1
            frame.to_csv(dataset, index=False)
            with self.assertRaisesRegex(RuntimeError, "scientific identity changed"):
                run_research_campaign(
                    CampaignConfig(**{**config.__dict__, "max_trials": 2})
                )

    def test_search_progresses_beyond_six_seed_recipes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = self.config(campaign, dataset, protocol, max_trials=7)
            config = CampaignConfig(**{
                **config.__dict__,
                "planner_mode": "local",
            })
            payload = run_research_campaign(config)
            self.assertEqual(7, payload["ledger"]["completed"])
            self.assertGreaterEqual(payload["search"]["trials"], 7)

    def test_openrouter_planner_is_called_with_completed_evidence(self):
        captured = {}

        def fake_choose(evidence, count, config):
            captured["evidence"] = evidence
            captured["status"] = json.loads(
                (root / "campaign" / "status.json").read_text(encoding="utf-8")
            )
            proposals = []
            parents = [row["attempt_id"] for row in evidence["completed_trials"]]
            for index in range(count):
                proposals.append({
                    "proposal_id": f"remote-{index}",
                    "parent_trial_ids": parents,
                    "evidence_ids": [evidence["evidence_id"]],
                    "hypothesis": "Vary regularization from the cited development evidence.",
                    "changed_axes": ["hyperparameters"],
                    "recipe": {
                        "schema_version": 2,
                        "target": {"kind": "win_probability", "parameters": {}},
                        "feature_schema": "baseline-v1",
                        "drop_feature_families": [],
                        "transforms": [],
                        "train_window": "all_history",
                        "model": {"kind": "logit", "parameters": {"C": 0.07 + index * 0.01}},
                        "calibration": {"kind": "temperature", "parameters": {}},
                        "blend": {"kind": "market_softmax", "parameters": {}},
                        "seed": 42,
                    },
                    "expected_observation": "Comparable development objective changes.",
                    "falsification_rule": "Reject when the objective does not improve.",
                    "max_trials": 1,
                    "max_wall_seconds": 1200,
                })
            return {
                "raw_response": {"usage": {"cost": 0.01}},
                "proposals": proposals,
                "service_tier": "flex",
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            config = CampaignConfig(
                campaign_dir=root / "campaign",
                policy="agentic",
                planner_mode="openrouter",
                model="provider/test-model",
                service_tier="flex",
                max_total_cost_usd=1.0,
                dataset_path=dataset,
                protocol_path=protocol,
                max_trials=4,
                proposal_batch_size=2,
                max_concurrent_trials=1,
                replan_every_terminal_trials=2,
            )
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}), mock.patch(
                "ima.research_controller.choose_research_proposals",
                side_effect=fake_choose,
            ):
                payload = run_research_campaign(config)
            self.assertEqual(4, payload["ledger"]["completed"])
            self.assertEqual(2, len(captured["evidence"]["completed_trials"]))
            self.assertEqual("provider_planning", captured["status"]["status"])
            planner = json.loads(
                (config.campaign_dir / "planner" / "cycle-0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual("provider/test-model", planner["requested_model"])
            decision = json.loads(
                (config.campaign_dir / "decisions" / "cycle-0001.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual("provider/test-model", decision["planner_model"])
            self.assertEqual(0.01, decision["planner_usage"]["total_cost_usd"])
            self.assertEqual("reported", decision["planner_usage"]["cost_status"])

    def test_planner_budget_runs_across_cycles_without_early_replanning(self):
        def propose(evidence, count, config):
            return {
                "raw_response": {"usage": {"cost": 0.01}},
                "service_tier": "flex",
                "proposals": [{
                    "proposal_id": "five-trial-direction",
                    "parent_trial_ids": [row["attempt_id"] for row in evidence["completed_trials"]],
                    "evidence_ids": [evidence["evidence_id"]],
                    "hypothesis": "Explore winner regularization across five trials.",
                    "changed_axes": ["hyperparameters"],
                    "recipe": PipelineRecipe().canonical_payload(),
                    "search_space": {"C": {"kind": "float", "low": 0.01, "high": 2.0}},
                    "expected_observation": "Lower development loss.",
                    "falsification_rule": "No improvement after five trials.",
                    "max_trials": 5,
                }],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            config = CampaignConfig(**{
                **self.config(root / "campaign", dataset, protocol, max_trials=7).__dict__,
                "planner_mode": "openrouter",
                "model": "provider/test-model",
                "service_tier": "flex",
                "max_total_cost_usd": 1.0,
                "proposal_batch_size": 2,
                "replan_every_terminal_trials": 2,
            })
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}), mock.patch(
                "ima.research_controller.choose_research_proposals", side_effect=propose
            ) as planner:
                payload = run_research_campaign(config)
            self.assertEqual(7, payload["ledger"]["completed"])
            self.assertEqual(1, planner.call_count)
            decisions = [json.loads(line) for line in (
                config.campaign_dir / "decisions.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                ["bootstrap", "openrouter", "approved_space_optuna", "approved_space_optuna"],
                [decision["source"] for decision in decisions],
            )
            self.assertEqual([2, 2, 1], [len(decision["suggestions"]) for decision in decisions[1:]])

    def test_spend_cap_pauses_without_dispatching_provider_or_new_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            first = self.config(campaign, dataset, protocol, max_trials=3)
            first = CampaignConfig(**{**first.__dict__, "planner_mode": "local"})
            run_research_campaign(first)
            planner_dir = campaign / "planner"
            planner_dir.mkdir()
            (planner_dir / "cycle-0000.json").write_text(json.dumps({
                "response": {"raw_response": {"usage": {"cost": 1.0}}},
            }), encoding="utf-8")
            resumed = CampaignConfig(**{
                **first.__dict__,
                "planner_mode": "openrouter",
                "model": "provider/test-model",
                "service_tier": "flex",
                "max_total_cost_usd": 0.5,
                "max_trials": 4,
            })
            with mock.patch(
                "ima.research_controller.choose_research_proposals"
            ) as provider:
                payload = run_research_campaign(resumed)
            self.assertEqual("paused_spend", payload["mode"])
            self.assertEqual(1.0, payload["planner_spend_usd"])
            self.assertEqual(3, payload["search"]["trials"])
            provider.assert_not_called()

    def test_duplicate_provider_recipe_is_rejected_and_refilled_locally(self):
        def duplicate_choose(evidence, count, config):
            parent_ids = [row["attempt_id"] for row in evidence["completed_trials"]]
            return {
                "raw_response": {"usage": {"cost": 0.01}},
                "service_tier": "flex",
                "proposals": [{
                    "proposal_id": "duplicate-seed",
                    "parent_trial_ids": parent_ids,
                    "evidence_ids": [evidence["evidence_id"]],
                    "hypothesis": "Retry the baseline control.",
                    "changed_axes": ["hyperparameters"],
                    "recipe": evidence["best_by_target"]["win_probability"][0]["recipe"],
                    "expected_observation": "The control remains stable.",
                    "falsification_rule": "Reject when it duplicates prior work.",
                }],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = CampaignConfig(
                campaign_dir=campaign,
                policy="agentic",
                planner_mode="openrouter",
                model="provider/test-model",
                service_tier="flex",
                max_total_cost_usd=1.0,
                dataset_path=dataset,
                protocol_path=protocol,
                max_trials=4,
                proposal_batch_size=2,
                max_concurrent_trials=1,
                replan_every_terminal_trials=2,
            )
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}), mock.patch(
                "ima.research_controller.choose_research_proposals",
                side_effect=duplicate_choose,
            ):
                payload = run_research_campaign(config)
            self.assertEqual("complete", payload["mode"])
            decision = json.loads(
                (campaign / "decisions" / "cycle-0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual("duplicate_recipe", decision["rejected_proposals"][0]["reason"])
            self.assertEqual(2, len(decision["local_refill_trial_ids"]))

    def test_invalid_provider_lineage_is_rejected_and_refilled_locally(self):
        def invalid_lineage_choose(evidence, count, config):
            parent_id = evidence["completed_trials"][0]["attempt_id"]
            proposals = []
            for index, (evidence_ids, parent_ids) in enumerate((
                (["stale-evidence"], [parent_id]),
                ([evidence["evidence_id"]], []),
                ([evidence["evidence_id"]], ["attempt-unknown"]),
            )):
                proposals.append({
                    "proposal_id": f"invalid-lineage-{index}",
                    "parent_trial_ids": parent_ids,
                    "evidence_ids": evidence_ids,
                    "hypothesis": "Exercise planner lineage validation.",
                    "changed_axes": ["hyperparameters"],
                    "recipe": {
                        "schema_version": 2,
                        "target": {"kind": "win_probability", "parameters": {}},
                        "feature_schema": "baseline-v1",
                        "drop_feature_families": [],
                        "transforms": [],
                        "train_window": "all_history",
                        "model": {
                            "kind": "logit",
                            "parameters": {"C": 0.071 + index * 0.001},
                        },
                        "calibration": {"kind": "temperature", "parameters": {}},
                        "blend": {"kind": "market_softmax", "parameters": {}},
                        "seed": 42,
                    },
                    "expected_observation": "The invalid proposal is rejected.",
                    "falsification_rule": "Fail if invalid lineage executes.",
                })
            return {
                "raw_response": {"usage": {"cost": 0.01}},
                "service_tier": "flex",
                "proposals": proposals,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = CampaignConfig(
                campaign_dir=campaign,
                policy="agentic",
                planner_mode="openrouter",
                model="provider/test-model",
                service_tier="flex",
                max_total_cost_usd=1.0,
                dataset_path=dataset,
                protocol_path=protocol,
                max_trials=6,
                proposal_batch_size=3,
                max_concurrent_trials=1,
                replan_every_terminal_trials=3,
            )
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}), mock.patch(
                "ima.research_controller.choose_research_proposals",
                side_effect=invalid_lineage_choose,
            ):
                payload = run_research_campaign(config)
            self.assertEqual("complete", payload["mode"])
            decision = json.loads(
                (campaign / "decisions" / "cycle-0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                ["stale_evidence_id", "missing_parent_trials", "unknown_parent_trials"],
                [item["reason"] for item in decision["rejected_proposals"]],
            )
            self.assertEqual(
                ["attempt-unknown"],
                decision["rejected_proposals"][2]["unknown_parent_trial_ids"],
            )
            self.assertEqual(3, len(decision["local_refill_trial_ids"]))

    def test_zero_resource_slots_pause_before_search_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            no_disk = ResourceSnapshot(10.0, 10.0, 100.0, 128.0, 1.0, 28)
            with mock.patch("ima.research_controller.observe_resources", return_value=no_disk):
                payload = run_research_campaign(
                    self.config(campaign, dataset, protocol, max_trials=3)
                )
            self.assertEqual("paused_admission", payload["mode"])
            self.assertEqual(0, payload["search"]["trials"])
            self.assertEqual(0, payload["resources"]["admission_slots"])

    def test_repeated_training_failures_trip_circuit_breaker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            frame = pd.read_csv(dataset).drop(columns=["target_probability"])
            frame.to_csv(dataset, index=False)
            config = self.config(root / "campaign", dataset, protocol, max_trials=9)
            config = CampaignConfig(**{
                **config.__dict__,
                "max_consecutive_failed_trials": 3,
            })
            payload = run_research_campaign(config)
            self.assertEqual("blocked_failures", payload["mode"])
            self.assertEqual(3, payload["ledger"]["failed"])
            self.assertEqual(3, payload["search"]["trials"])

    def test_completed_trial_gets_durable_mlflow_linkage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, protocol = self.fixture(root)
            campaign = root / "campaign"
            config = self.config(campaign, dataset, protocol, max_trials=1)
            config = CampaignConfig(**{
                **config.__dict__,
                "mlflow_tracking_uri": f"sqlite:///{root / 'mlflow.db'}",
                "max_concurrent_trials": 1,
            })
            payload = run_research_campaign(config)
            self.assertEqual(1, payload["ledger"]["completed"])
            self.assertEqual(0, payload["ledger"]["pending_tracking"])
            terminal = ResearchLedger(campaign / "ledger.sqlite").terminal_results()
            linkage = json.loads(terminal[0]["remote_id"])
            self.assertIn("run_id", linkage)
            self.assertIn("model_version", linkage)


if __name__ == "__main__":
    unittest.main()
