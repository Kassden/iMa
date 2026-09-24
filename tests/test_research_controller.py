import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from ima.optimizer import CampaignConfig
from ima.research_controller import (
    _fixture_proposals,
    campaign_status,
    request_campaign_stop,
    run_research_campaign,
)
from ima.research_resources import ResourceSnapshot
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

    def test_status_and_stop_use_campaign_state(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory) / "campaign"
            campaign.mkdir()
            marker = request_campaign_stop(campaign)
            self.assertTrue(marker.exists())
            status = campaign_status(campaign)
            self.assertEqual(str(campaign), status["campaign_dir"])

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
            planner = json.loads(
                (config.campaign_dir / "planner" / "cycle-0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual("provider/test-model", planner["requested_model"])

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
