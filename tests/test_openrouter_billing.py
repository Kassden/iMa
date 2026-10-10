import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import httpx

from ima.openrouter_billing import _publish, billing_receipt, reconcile


class BillingRecoveryTests(unittest.TestCase):
    def terminal_error(self):
        return {"id": "gen-ok", "total_cost": 0.00932526, "cancelled": False,
                "finish_reason": None, "native_finish_reason": None,
                "generation_time": 298295, "native_tokens_prompt": 76866,
                "native_tokens_completion": 13374, "upstream_id": "chatcmpl-aborted",
                "provider_responses": [{"id": "chatcmpl-aborted", "status": 499}]}

    def test_aborted_provider_stream_charge_without_finish_reason(self):
        data = self.terminal_error()
        receipt = billing_receipt({"data": data}, "gen-ok", {}, "rev")
        self.assertEqual(receipt["billing_finalization_basis"], "provider_terminal_error")
        self.assertTrue(receipt["billing_only"])
        self.assertEqual(receipt["response"]["usage"]["cost"], 0.00932526)
        self.assertEqual(receipt["response"]["usage"]["total_tokens"], 90240)
        self.assertNotIn("choices", receipt["response"])

    def test_provider_error_finalization_rejects_ambiguous_or_inflight_metadata(self):
        variants = [
            {"provider_responses": [{"id": "chatcmpl-other", "status": 499}]},
            {"provider_responses": [{"id": "chatcmpl-aborted", "status": 200}]},
            {"provider_responses": [{"id": "chatcmpl-aborted", "status": 102}]},
            {"provider_responses": [{"id": "chatcmpl-aborted", "status": "499"}]},
            {"provider_responses": [{"id": "chatcmpl-aborted", "status": 499}, None]},
            {"provider_responses": [{"id": "chatcmpl-aborted", "status": 499},
                                    {"id": "chatcmpl-active", "status": 200}]},
            {"generation_time": None}, {"generation_time": float("nan")},
            {"generation_time": -1}, {"generation_time": True},
            {"generation_time": 10**400},
            {"native_tokens_completion": None}, {"native_tokens_prompt": True},
            {"upstream_id": None}, {"total_cost": None},
        ]
        for changes in variants:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                billing_receipt({"data": dict(self.terminal_error(), **changes)}, "gen-ok", {}, "rev")

    def test_terminal_error_get_only_recovery_unblocks_spend_idempotently(self):
        from ima.research_expansion import _planner_spend
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            (root / "decisions").mkdir()
            path = root / "planner-transport/D000198-01.json"
            path.write_text(json.dumps({"evidence_id": "e1", "events": [
                {"phase": "response_headers", "status_code": 200, "generation_id": "gen-ok"},
                {"phase": "request_failed", "exception_type": "TimeoutError"}]}))
            decision = root / "decisions/D000198.json"
            decision.write_text(json.dumps({"planner_status": "failed", "planner_usage": {}}))
            original = (path.read_bytes(), decision.read_bytes())
            def handler(request):
                self.assertEqual(request.method, "GET")
                return httpx.Response(200, json={"data": self.terminal_error()})
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                self.assertTrue(_planner_spend(root)["spend_unknown"])
                report = reconcile(root, "secret", revision="rev", client=client)
                self.assertEqual(report["recovered"], ["D000198-01.json"])
                self.assertFalse(_planner_spend(root)["spend_unknown"])
                self.assertEqual(_planner_spend(root)["total_cost_usd"], 0.00932526)
                self.assertEqual(reconcile(root, "secret", revision="rev", client=client)["requests"], 0)
                self.assertEqual(original, (path.read_bytes(), decision.read_bytes()))

    def test_atomic_publication_never_overwrites(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            self.assertTrue(_publish(path, {"first": True}))
            original = path.read_bytes()
            self.assertFalse(_publish(path, {"second": True}))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temporary).glob(".billing-*")), [])

    def test_timeout_and_wrong_provider_identity_remain_unknown(self):
        for failure in ("timeout", "identity", "missing_charge", "404"):
            with self.subTest(failure=failure), TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                (root / "planner-transport").mkdir()
                (root / "planner-transport/D000001-01.json").write_text(json.dumps({"events": [{"generation_id": "gen-ok"}, {"phase": "request_failed"}]}))
                def handler(request):
                    if failure == "timeout":
                        raise httpx.ReadTimeout("secret must not be logged", request=request)
                    if failure == "404":
                        return httpx.Response(404)
                    data = {"id": "gen-wrong" if failure == "identity" else "gen-ok", "total_cost": None if failure == "missing_charge" else 0.1}
                    return httpx.Response(200, json={"data": data})
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    report = reconcile(root, "secret", revision="rev", client=client)
                self.assertEqual(report["unresolved"], ["D000001-01.json"])
                self.assertFalse((root / "planner-calls/D000001-01.json").exists())
                self.assertNotIn("secret", json.dumps(report))

    def test_invalid_provider_cost_and_identity(self):
        for cost in (None, True, -1, float("nan"), float("inf"), "0.01", 10**400):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                billing_receipt({"data": {"id": "gen-ok", "total_cost": cost}}, "gen-ok", {}, "rev")
        with self.assertRaises(ValueError):
            billing_receipt({"data": {"id": "gen-other", "total_cost": 0.1}}, "gen-ok", {}, "rev")
        for finish_reason in (None, True, 1, ["stop"], {"state": "running"}, " "):
            with self.subTest(finish_reason=finish_reason), self.assertRaises(ValueError):
                billing_receipt({"data": {"id": "gen-ok", "total_cost": 0.1, "cancelled": False, "finish_reason": finish_reason}}, "gen-ok", {}, "rev")

    def test_get_only_idempotent_accounting(self):
        from ima.research_expansion import _planner_spend
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            (root / "decisions").mkdir()
            transport = {"evidence_id": "e1", "events": [
                {"phase": "response_headers", "status_code": 200, "generation_id": "gen-ok"},
                {"phase": "request_failed", "error_type": "TimeoutError"},
            ]}
            (root / "planner-transport/D000009-01.json").write_text(json.dumps(transport))
            (root / "decisions/D000009.json").write_text(json.dumps({"planner_status": "failed", "planner_usage": {}}))
            original = (root / "decisions/D000009.json").read_bytes()
            self.assertTrue(_planner_spend(root)["spend_unknown"])
            requests = []
            def handler(request):
                requests.append(request)
                self.assertEqual(request.method, "GET")
                self.assertEqual(request.url.params["id"], "gen-ok")
                return httpx.Response(200, json={"data": {"id": "gen-ok", "total_cost": 0.00851584, "cancelled": True, "native_tokens_prompt": 100, "native_tokens_completion": 20}})
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                report = reconcile(root, "secret", revision="rev", client=client)
                self.assertEqual(report["recovered"], ["D000009-01.json"])
                self.assertFalse(_planner_spend(root)["spend_unknown"])
                self.assertEqual(_planner_spend(root)["total_cost_usd"], 0.00851584)
                receipt = (root / "planner-calls/D000009-01.json").read_bytes()
                reconcile(root, "secret", revision="rev", client=client)
                self.assertEqual(len(requests), 1)
                self.assertEqual((root / "planner-calls/D000009-01.json").read_bytes(), receipt)
                self.assertEqual((root / "decisions/D000009.json").read_bytes(), original)
                self.assertNotIn(b"secret", receipt)

    def test_failure_bounds_and_existing_receipts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            (root / "planner-calls").mkdir()
            for number in range(1, 6):
                (root / f"planner-transport/D{number:06d}-01.json").write_text(json.dumps({"events": [{"generation_id": f"gen-{number}"}, {"phase": "request_failed"}]}))
            existing = root / "planner-calls/D000001-01.json"
            existing.write_text('{"response": {"usage": {}}}')
            original = existing.read_bytes()
            with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429))) as client:
                report = reconcile(root, "secret", revision="rev", client=client)
            self.assertEqual(report["requests"], 3)
            self.assertEqual(len(report["unresolved"]), 3)
            self.assertEqual(existing.read_bytes(), original)

    def test_cursor_prevents_starvation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            for number in range(1, 5):
                (root / f"planner-transport/D{number:06d}-01.json").write_text(json.dumps({"events": [{"generation_id": f"gen-{number}"}, {"phase": "request_failed"}]}))
            def handler(request):
                identity = request.url.params["id"]
                return httpx.Response(200, json={"data": {"id": identity, "total_cost": 0.1, "cancelled": True}}) if identity == "gen-4" else httpx.Response(404)
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                first = reconcile(root, "secret", revision="rev", client=client)
                second = reconcile(root, "secret", revision="rev", client=client)
            self.assertEqual(first["requests"], 3)
            self.assertEqual(second["recovered"], ["D000004-01.json"])

    def test_duplicate_ids_and_bad_shape(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            for number, payload in enumerate(([], {"events": [None]}, {"events": [{"generation_id": "gen-duplicate"}, {"phase": "request_failed"}]}, {"events": [{"generation_id": "gen-duplicate"}, {"phase": "request_failed"}]}), 1):
                (root / f"planner-transport/D{number:06d}-01.json").write_text(json.dumps(payload))
            with httpx.Client(transport=httpx.MockTransport(lambda request: self.fail("No unsafe lookup"))) as client:
                report = reconcile(root, "secret", revision="rev", client=client)
            self.assertEqual(len(report["unresolved"]), 4)

    def test_symlink_and_ambiguous_ids_fail_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            (root / "planner-transport/D000001-01.json").write_text(json.dumps({"events": [{"generation_id": "gen-a"}, {"generation_id": "gen-b"}]}))
            (root / "planner-transport/D000002-01.json").symlink_to(root / "planner-transport/D000001-01.json")
            with httpx.Client(transport=httpx.MockTransport(lambda request: self.fail("No request expected"))) as client:
                report = reconcile(root, "secret", revision="rev", client=client)
            self.assertEqual(len(report["unresolved"]), 2)
            (root / "planner-calls").rmdir()
            (root / "planner-calls").symlink_to(root / "planner-transport")
            with self.assertRaises(ValueError):
                reconcile(root, "secret", revision="rev")

    def test_inflight_generation_is_not_reconciled(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "planner-transport").mkdir()
            (root / "planner-transport/D000001-01.json").write_text(json.dumps({"events": [{"phase": "response_headers", "generation_id": "gen-ok"}]}))
            with httpx.Client(transport=httpx.MockTransport(lambda request: self.fail("No in-flight billing lookup"))) as client:
                report = reconcile(root, "secret", revision="rev", client=client)
            self.assertEqual(report["requests"], 0)
