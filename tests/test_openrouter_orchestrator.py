import unittest
import socket
from unittest import mock

from ima.openrouter_orchestrator import (
    OpenRouterConfig,
    OpenRouterError,
    batch_is_terminal,
    get_batch,
    normalize_openrouter_usage,
    submit_proposal_batch,
)


class OpenRouterOrchestratorTests(unittest.TestCase):
    def test_submit_batch_uses_current_v1_endpoint(self):
        captured = {}

        def fake_post(url, payload, config):
            captured["url"] = url
            captured["payload"] = payload
            return {"id": "batch_123", "status": "validating"}

        with mock.patch("ima.openrouter_orchestrator._post_json", fake_post):
            batch = submit_proposal_batch(
                [],
                1,
                OpenRouterConfig("key", "openai/test", service_tier="flex"),
            )
        self.assertEqual("batch_123", batch["id"])
        self.assertTrue(captured["url"].endswith("/v1/batches"))
        self.assertEqual("/v1/chat/completions", captured["payload"]["endpoint"])

    def test_get_batch_uses_v1_batch_id_endpoint(self):
        def fake_get(url, config):
            self.assertTrue(url.endswith("/v1/batches/batch_123"))
            return {"id": "batch_123", "status": "completed", "results": []}

        with mock.patch("ima.openrouter_orchestrator._get_json", fake_get):
            batch = get_batch("batch_123", OpenRouterConfig("key", "openai/test"))
        self.assertTrue(batch_is_terminal(batch))

    def test_get_batch_requires_id(self):
        with self.assertRaises(OpenRouterError):
            get_batch("", OpenRouterConfig("key", "openai/test"))

    def test_terminal_statuses_match_batch_contract(self):
        self.assertTrue(batch_is_terminal({"status": "completed"}))
        self.assertTrue(batch_is_terminal({"status": "failed"}))
        self.assertTrue(batch_is_terminal({"status": "expired"}))
        self.assertTrue(batch_is_terminal({"status": "cancelled"}))
        self.assertFalse(batch_is_terminal({"status": "in_progress"}))

    def test_transport_timeout_is_wrapped(self):
        with mock.patch(
            "urllib.request.urlopen", side_effect=socket.timeout("slow flex response")
        ):
            with self.assertRaisesRegex(OpenRouterError, "slow flex response"):
                get_batch("batch_123", OpenRouterConfig("key", "openai/test"))

    def test_usage_normalization_preserves_reported_tokens_and_exact_cost(self):
        usage = normalize_openrouter_usage({"usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "cost": 0.0042,
        }})
        self.assertEqual(120, usage["input_tokens"])
        self.assertEqual(30, usage["output_tokens"])
        self.assertEqual(150, usage["total_tokens"])
        self.assertEqual(0.0042, usage["total_cost_usd"])
        self.assertEqual("reported", usage["cost_status"])

    def test_usage_normalization_supports_responses_names_and_missing_cost(self):
        usage = normalize_openrouter_usage({"usage": {
            "input_tokens": 10,
            "output_tokens": 5,
        }})
        self.assertEqual(15, usage["total_tokens"])
        self.assertIsNone(usage["total_cost_usd"])
        self.assertEqual("unavailable", usage["cost_status"])

    def test_usage_normalization_rejects_malformed_or_negative_values(self):
        usage = normalize_openrouter_usage({"usage": {
            "prompt_tokens": "not-a-number",
            "completion_tokens": -2,
            "cost": -1,
        }})
        self.assertIsNone(usage["input_tokens"])
        self.assertIsNone(usage["output_tokens"])
        self.assertIsNone(usage["total_cost_usd"])
        self.assertEqual("unavailable", usage["cost_status"])


if __name__ == "__main__":
    unittest.main()
