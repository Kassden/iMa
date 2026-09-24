import unittest
from unittest import mock

from ima.openrouter_orchestrator import (
    OpenRouterConfig,
    OpenRouterError,
    batch_is_terminal,
    get_batch,
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


if __name__ == "__main__":
    unittest.main()
