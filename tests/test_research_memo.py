import hashlib
import json
import unittest
from unittest import mock

from ima.research_memo import MEMO_PATH, research_memo_context
from ima.openrouter_orchestrator import OpenRouterConfig, OpenRouterError, choose_research_decision


class ResearchMemoTests(unittest.TestCase):
    def test_full_memo_and_checksum_are_delivered(self):
        memo = research_memo_context()
        self.assertEqual(MEMO_PATH.read_text(), memo["content"])
        self.assertEqual(hashlib.sha256(MEMO_PATH.read_bytes()).hexdigest(), memo["sha256"])
        decision = {"decision_id": "D1", "evidence_id": "E1", "trial_budget": 0,
                    "review_reason": "Replay historical controls on current data",
                    "research_memo_sha256": memo["sha256"]}
        response = {"choices": [{"message": {"content": json.dumps(decision)}}],
                    "usage": {"cost": .001}}
        with mock.patch("ima.openrouter_orchestrator._post_json", return_value=response) as post:
            result = choose_research_decision(
                {"decision_id": "D1", "evidence_id": "E1", "capabilities": {"research_memo": memo}},
                {"trial_ceiling": 260, "max_new_programs": 12}, OpenRouterConfig("test", "test/model"))
        submitted = json.loads(post.call_args.args[1]["messages"][1]["content"])
        self.assertEqual(memo, submitted["evidence"]["capabilities"]["research_memo"])
        self.assertEqual(memo["sha256"], submitted["required_identity"]["research_memo_sha256"])
        self.assertEqual("D1", submitted["required_identity"]["decision_id"])
        self.assertTrue(submitted["output_budget"]["reasoning_shares_budget"])
        self.assertIn("nodes[].output", submitted["graph_contract"]["market_output"])
        self.assertEqual(memo["sha256"], result["decision"]["research_memo_sha256"])

    def test_missing_memo_acknowledgement_is_not_accepted(self):
        response = {"choices": [{"message": {"content": json.dumps({
            "decision_id": "D1", "evidence_id": "E1", "trial_budget": 0, "review_reason": "Review"})}}],
            "usage": {"cost": .001}}
        with mock.patch("ima.openrouter_orchestrator._post_json", return_value=response):
            with self.assertRaisesRegex(OpenRouterError, "memo checksum"):
                choose_research_decision(
                    {"decision_id": "D1", "evidence_id": "E1", "capabilities": {"research_memo": research_memo_context()}},
                    {"trial_ceiling": 260, "max_new_programs": 12}, OpenRouterConfig("test", "test/model"))
