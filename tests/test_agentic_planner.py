import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from ima.openrouter_orchestrator import (
    OpenRouterConfig,
    OpenRouterError,
    agentic_planner_messages,
    choose_research_proposals,
    research_proposals_from_response,
)
from ima.research_evidence import evidence_bundle_from_campaign


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class AgenticPlannerTests(unittest.TestCase):
    def test_evidence_bundle_is_redacted_and_messages_exclude_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "trials.jsonl", [{
                "trial_id": "trial-0001",
                "run_id": "baseline",
                "status": "completed",
                "metrics": {
                    "kind": "logit",
                    "feature_schema": "baseline-v1",
                    "test_blended": {"race_log_loss": 2.1, "top_pick_win_rate": 0.3},
                    "test_fundamental": {"race_log_loss": 2.3},
                },
            }])
            bundle = evidence_bundle_from_campaign(root)
        payload = json.dumps(bundle, sort_keys=True)
        self.assertIn("evidence_id", bundle)
        self.assertNotIn("horse_id", payload)
        self.assertNotIn("target_win", payload)
        messages = agentic_planner_messages(bundle, 1)
        user_content = messages[1]["content"]
        self.assertIn("propose_agentic_research_recipes", user_content)
        self.assertNotIn("raw_runner_rows", user_content)
        self.assertNotIn('"model.kind"', user_content)
        self.assertNotIn('"target.kind"', user_content)
        schema = json.loads(user_content)["required_recipe_object_shape"]
        self.assertIsInstance(schema["feature_schema"], str)
        self.assertIsInstance(schema["train_window"], str)
        compatibility = json.loads(user_content)["target_recipe_compatibility"]
        self.assertEqual("pairwise_ranker", compatibility["ranking_strength"]["model"])
        self.assertEqual("none", compatibility["placing_top_k"]["blend"])

    def test_research_proposal_response_validates_typed_recipe(self):
        response = {
            "choices": [{
                "message": {"content": json.dumps({"proposals": [{
                    "proposal_id": "proposal-1",
                    "parent_trial_ids": ["trial-0001"],
                    "evidence_ids": ["evidence-1"],
                    "hypothesis": "Try richer schema with lower regularization.",
                    "changed_axes": ["feature_schema", "hyperparameters"],
                    "recipe": {
                        "schema_version": 2,
                        "target": {"kind": "win_probability"},
                        "feature_schema": "benter-rich-v1",
                        "drop_feature_families": [],
                        "transforms": [],
                        "train_window": "all_history",
                        "model": {"kind": "logit", "parameters": {"C": 0.1}},
                        "calibration": {"kind": "temperature"},
                        "blend": {"kind": "market_softmax"},
                        "seed": 42,
                    },
                    "expected_observation": "Lower development loss.",
                    "falsification_rule": "Reject if paired score loss worsens.",
                    "max_trials": 1,
                }]})}
            }]
        }
        proposals = research_proposals_from_response(response)
        self.assertEqual("benter-rich-v1", proposals[0].recipe.feature_schema)

    def test_forbidden_agentic_proposal_terms_are_rejected(self):
        response = {
            "choices": [{
                "message": {"content": json.dumps({"proposals": [{
                    "proposal_id": "proposal-1",
                    "hypothesis": "Use final_odds.",
                    "changed_axes": ["hyperparameters"],
                    "recipe": {"schema_version": 2},
                    "expected_observation": "Win.",
                    "falsification_rule": "None.",
                }]})}
            }]
        }
        with self.assertRaises(ValidationError):
            research_proposals_from_response(response)

    def test_choose_research_proposals_uses_flex_request_and_validates_payload(self):
        bundle = {"schema_version": "evidence-bundle-v1", "evidence_id": "e1"}

        def fake_post(url, payload, config):
            self.assertIn("/v1/chat/completions", url)
            self.assertEqual("flex", payload["service_tier"])
            return {
                "service_tier": "flex",
                "choices": [{
                    "message": {"content": json.dumps({"proposals": [{
                        "proposal_id": "proposal-1",
                        "evidence_ids": ["e1"],
                        "hypothesis": "Try boosted baseline.",
                        "changed_axes": ["model_family"],
                        "recipe": {
                            "schema_version": 2,
                            "model": {"kind": "boosted", "parameters": {"learning_rate": 0.06}},
                        },
                        "expected_observation": "Different bias variance.",
                        "falsification_rule": "Reject if score loss worsens.",
                    }]})}
                }],
            }

        with mock.patch("ima.openrouter_orchestrator._post_json", fake_post):
            result = choose_research_proposals(
                bundle,
                1,
                OpenRouterConfig("key", "openai/test", service_tier="flex"),
            )
        self.assertEqual("flex", result["service_tier"])
        self.assertEqual("boosted", result["proposals"][0]["recipe"]["model"]["kind"])

    def test_choose_wraps_malformed_provider_recipe_as_bounded_error(self):
        response = {
            "choices": [{"message": {"content": json.dumps({"proposals": [{
                "proposal_id": "bad-dotted-shape",
                "parent_trial_ids": ["attempt-1"],
                "evidence_ids": ["e1"],
                "hypothesis": "Try a different regularization value.",
                "changed_axes": ["hyperparameters"],
                "recipe": {
                    "schema_version": 2,
                    "model.kind": "logit",
                    "target.kind": "win_probability",
                },
                "expected_observation": "Development loss changes.",
                "falsification_rule": "Reject if loss does not improve.",
            }]})}}],
        }
        with mock.patch("ima.openrouter_orchestrator._post_json", return_value=response):
            with self.assertRaisesRegex(OpenRouterError, "invalid research proposals"):
                choose_research_proposals(
                    {"evidence_id": "e1"}, 1,
                    OpenRouterConfig("key", "openai/test", service_tier="flex"),
                )


if __name__ == "__main__":
    unittest.main()
