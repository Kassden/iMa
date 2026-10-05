import asyncio
import copy
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import httpcore
import httpx

from ima.openrouter_transport import (
    classify_transport_attempt,
    classify_transport_attempts,
    post_json_bounded,
)


def events(*phases):
    return [{"phase": phase} for phase in phases]


def unsent():
    return events("request_started", "connection.connect_tcp.started",
                  "connection.connect_tcp.failed", "request_failed")


class TransportClassificationTests(unittest.TestCase):
    def assertClassification(self, expected, result):
        self.assertEqual({"classification", "reason"}, set(result))
        self.assertEqual(expected, result["classification"])
        self.assertIsInstance(result["reason"], str)

    def test_complete_pre_send_dns_and_tcp_failures_are_unsent(self):
        for operation in ("connection.connect_tcp",):
            for exception in ("ConnectError", "ConnectTimeout", "gaierror", "TimeoutError"):
                with self.subTest(operation=operation, exception=exception):
                    attempt = events("request_started", operation + ".started",
                                     operation + ".failed", "request_failed")
                    attempt[2]["exception_type"] = exception
                    self.assertClassification("not_dispatched", classify_transport_attempt(attempt))

    def test_request_started_and_exception_type_alone_are_unknown(self):
        for phases in (("request_started",), ("request_started", "request_failed")):
            with self.subTest(phases=phases):
                attempt = events(*phases)
                attempt[-1]["exception_type"] = "ConnectError"
                self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_resolved_dns_followed_by_tcp_failure_is_unsent(self):
        attempt = events("request_started", "connection.resolve_dns.started",
                         "connection.resolve_dns.complete", "connection.connect_tcp.started",
                         "connection.connect_tcp.failed", "request_failed")
        self.assertClassification("not_dispatched", classify_transport_attempt(attempt))

    def test_dns_failure_without_complete_tcp_failure_is_unknown(self):
        attempt = events("request_started", "connection.resolve_dns.started",
                         "connection.resolve_dns.failed", "request_failed")
        self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_later_successful_resolution_cannot_hide_incomplete_retry(self):
        attempt = [*unsent()[:-1], *events("connection.resolve_dns.started",
                                         "connection.resolve_dns.complete", "request_failed")]
        self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_dict_contract_has_explicit_proof_reason(self):
        self.assertEqual({"classification": "not_dispatched",
                          "reason": "complete_pre_send_connect_failure"},
                         classify_transport_attempt(unsent()))

    def test_send_and_response_evidence_always_wins(self):
        for protocol in ("http11", "http2"):
            for operation in ("send_request_headers", "send_request_body",
                              "receive_response_headers", "receive_response_body"):
                for state in ("started", "complete", "failed"):
                    phase = f"{protocol}.{operation}.{state}"
                    with self.subTest(phase=phase):
                        attempt = [None, *events(phase), *unsent()]
                        self.assertClassification("possibly_dispatched",
                                                  classify_transport_attempt(attempt))
        for phase in ("response_headers", "response_complete"):
            self.assertClassification("possibly_dispatched", classify_transport_attempt(events(phase)))

    def test_truncated_connect_failure_is_unknown(self):
        attempt = unsent()
        for truncated in (attempt[1:], attempt[:-1], [attempt[0], *attempt[2:]]):
            with self.subTest(truncated=truncated):
                self.assertClassification("unknown", classify_transport_attempt(truncated))

    def test_tls_and_unresolved_connect_timeouts_are_unknown(self):
        for phases in (
            ("request_started", "connection.connect_tcp.started", "request_failed"),
            ("request_started", "connection.connect_tcp.started",
             "connection.connect_tcp.complete", "connection.start_tls.started",
             "connection.start_tls.failed", "request_failed"),
            ("request_started", "connection.start_tls.started", "request_failed"),
            ("request_started", "connection.connect_tcp.started",
             "connection.connect_tcp.complete", "request_failed"),
        ):
            with self.subTest(phases=phases):
                self.assertClassification("unknown", classify_transport_attempt(events(*phases)))

    def test_missing_or_malformed_events_are_unknown(self):
        for attempt in (None, [], {}, "request_started", 42, [None], [{}],
                        [{"phase": None}], [{"phase": []}], [*unsent(), None]):
            with self.subTest(attempt=attempt):
                self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_unrecognized_or_duplicate_lifecycle_cannot_prove_unsent(self):
        for extra in ("connection.connect_tcp.started", "connection.connect_tcp.failed",
                      "request_started", "request_failed", "unexpected.failed"):
            with self.subTest(extra=extra):
                attempt = [*unsent()[:-1], {"phase": extra}, unsent()[-1]]
                self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_failed_connect_followed_by_incomplete_retry_is_unknown(self):
        attempt = [*unsent()[:-1], *events("connection.connect_tcp.started", "request_failed")]
        self.assertClassification("unknown", classify_transport_attempt(attempt))

    def test_prior_sent_retry_followed_by_dns_cannot_be_unsent(self):
        sent = events("request_started", "http11.send_request_headers.started", "request_failed")
        self.assertClassification("possibly_dispatched", classify_transport_attempt([*sent, *unsent()]))
        for attempts in ([sent, unsent()], [unsent(), sent], [[], sent, unsent()]):
            self.assertClassification("unknown", classify_transport_attempts(attempts))

    def test_all_attempts_must_be_proven_unsent(self):
        self.assertClassification("not_dispatched", classify_transport_attempts([unsent(), unsent()]))
        for attempts in (None, [], {}, 42, "invalid", [unsent(), []], [None, unsent()],
                         [unsent(), events("request_started")], [unsent(), {}]):
            with self.subTest(attempts=attempts):
                self.assertClassification("unknown", classify_transport_attempts(attempts))

    def test_replay_is_pure_and_does_not_trust_saved_classification(self):
        attempt = [*unsent()[:-1], {"phase": "transport_classification",
                                   "classification": "possibly_dispatched"}, unsent()[-1]]
        original = copy.deepcopy(attempt)
        self.assertClassification("not_dispatched", classify_transport_attempt(iter(attempt)))
        self.assertEqual(original, attempt)
        self.assertClassification("not_dispatched", classify_transport_attempts(iter([attempt])))
        forged = events("request_started", "transport_classification", "request_failed")
        forged[1]["classification"] = "not_dispatched"
        self.assertClassification("unknown", classify_transport_attempt(forged))


class MemoryStream:
    """Drive real HTTPX/httpcore tracing without opening a network connection."""

    def __init__(self, requests, *, delay=0, status=200, body=None, failure=None):
        self.requests = requests
        self.delay = delay
        self.failure = failure
        self.body = json.dumps(body if body is not None else {"usage": {"cost": .01}}).encode()
        self.headers = (
            f"HTTP/1.1 {status} Fixture\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(self.body)}\r\n"
            "X-Generation-Id: gen-test-header\r\n"
            "X-Request-Id: req-test-header\r\n\r\n"
        ).encode()

    async def read(self, max_bytes, timeout=None):
        if self.headers:
            headers, self.headers = self.headers, b""
            return headers
        await asyncio.sleep(self.delay)
        body, self.body = self.body, b""
        return body

    async def write(self, buffer, timeout=None):
        if self.failure == "send":
            raise httpcore.WriteTimeout("private-message-secret")
        if buffer and not buffer.startswith(b"POST "):
            self.requests.append(json.loads(buffer))

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        if self.failure == "tls":
            raise httpcore.ConnectError("private-message-secret")
        return self

    async def aclose(self):
        pass

    def get_extra_info(self, info):
        return None


class MemoryBackend:
    def __init__(self, stream, failure):
        self.stream = stream
        self.failure = failure
        self.connect_count = 0

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connect_count += 1
        if self.failure == "dns":
            raise httpcore.ConnectError("private-message-secret: DNS resolution failed")
        if self.failure == "connect_timeout":
            raise httpcore.ConnectTimeout("private-message-secret")
        return self.stream


@contextmanager
def memory_endpoint(*, delay=0, status=200, body=None, failure=None):
    requests = []
    stream = MemoryStream(requests, delay=delay, status=status, body=body, failure=failure)
    backend = MemoryBackend(stream, failure)
    transport = httpx.AsyncHTTPTransport()
    transport._pool._network_backend = backend
    client_type = httpx.AsyncClient

    def client(**kwargs):
        return client_type(**kwargs, transport=transport, trust_env=False)

    with mock.patch("httpx.AsyncClient", side_effect=client), \
            mock.patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), \
            mock.patch("socket.getaddrinfo", side_effect=AssertionError("DNS forbidden")):
        yield "https://fixture.invalid/private-url-secret", requests, backend


class TransportClassificationReceiptTests(unittest.TestCase):
    def config(self, receipts, responses, deadline=1):
        return SimpleNamespace(api_key="private-key-secret", timeout_seconds=1,
                               absolute_deadline_seconds=deadline,
                               transport_observer=receipts.append,
                               response_observer=responses.append)

    def assert_classification(self, receipts, expected, terminal):
        self.assertEqual(terminal, receipts[-1]["phase"])
        self.assertEqual("transport_classification", receipts[-2]["phase"])
        self.assertEqual(expected, receipts[-2]["classification"])
        self.assertEqual(expected, classify_transport_attempt(receipts)["classification"])
        self.assertEqual(receipts[-2]["reason"], classify_transport_attempt(receipts)["reason"])
        self.assertEqual(1, sum(e["phase"] == "transport_classification" for e in receipts))
        for event in receipts:
            self.assertEqual(1, event["schema_version"])
            self.assertGreaterEqual(event["elapsed_seconds"], 0)
        serialized = json.dumps(receipts)
        for secret in ("private-key-secret", "private-payload-secret", "private-url-secret",
                       "private-message-secret", "Authorization", "Bearer", '"cost"', '"usage"'):
            self.assertNotIn(secret, serialized)

    def test_physical_connect_failures_are_persisted_as_unsent(self):
        for failure, exception in (("dns", httpx.ConnectError),
                                   ("connect_timeout", httpx.ConnectTimeout)):
            with self.subTest(failure=failure):
                receipts, responses = [], []
                with memory_endpoint(failure=failure) as (url, requests, backend):
                    with self.assertRaises(exception):
                        post_json_bounded(url, {"prompt": "private-payload-secret"},
                                          self.config(receipts, responses))
                self.assertEqual(1, backend.connect_count)
                self.assertEqual([], requests)
                self.assertEqual([], responses)
                self.assert_classification(receipts, "not_dispatched", "request_failed")

    def test_physical_tls_failure_is_unknown(self):
        receipts, responses = [], []
        with memory_endpoint(failure="tls") as (url, requests, backend):
            with self.assertRaises(httpx.ConnectError):
                post_json_bounded(url, {}, self.config(receipts, responses))
        self.assertEqual(1, backend.connect_count)
        self.assertEqual([], requests)
        self.assert_classification(receipts, "unknown", "request_failed")

    def test_physical_header_send_timeout_is_possibly_dispatched(self):
        receipts, responses = [], []
        with memory_endpoint(failure="send") as (url, _, backend):
            with self.assertRaises(httpx.WriteTimeout):
                post_json_bounded(url, {}, self.config(receipts, responses))
        self.assertEqual(1, backend.connect_count)
        self.assert_classification(receipts, "possibly_dispatched", "request_failed")

    def test_body_deadline_retains_identity_and_dispatch_uncertainty(self):
        receipts, responses = [], []
        with memory_endpoint(delay=1) as (url, requests, _):
            with self.assertRaises(TimeoutError):
                post_json_bounded(url, {}, self.config(receipts, responses, deadline=.2))
        self.assertEqual([{}], requests)
        self.assertEqual([], responses)
        headers = next(e for e in receipts if e["phase"] == "response_headers")
        self.assertEqual("gen-test-header", headers["generation_id"])
        self.assert_classification(receipts, "possibly_dispatched", "request_failed")

    def test_success_preserves_result_observer_and_terminal_event(self):
        receipts, responses = [], []
        with memory_endpoint() as (url, requests, _):
            result = post_json_bounded(url, {"prompt": "private-payload-secret"},
                                       self.config(receipts, responses))
        self.assertEqual([{"prompt": "private-payload-secret"}], requests)
        self.assertEqual([result], responses)
        self.assert_classification(receipts, "possibly_dispatched", "response_complete")

    def test_response_observer_failure_still_persists_sent_classification(self):
        receipts = []
        config = self.config(receipts, [])
        config.response_observer = mock.Mock(side_effect=ValueError("Unknown reported cost"))
        with memory_endpoint() as (url, _, _):
            with self.assertRaisesRegex(ValueError, "Unknown reported cost"):
                post_json_bounded(url, {}, config)
        self.assert_classification(receipts, "possibly_dispatched", "request_failed")

    def test_repeated_calls_retain_prior_sent_evidence(self):
        attempts = []
        for failure, exception in (("send", httpx.WriteTimeout), ("dns", httpx.ConnectError)):
            receipts = []
            with memory_endpoint(failure=failure) as (url, _, _):
                with self.assertRaises(exception):
                    post_json_bounded(url, {}, self.config(receipts, []))
            attempts.append(receipts)
        self.assertEqual("unknown", classify_transport_attempts(attempts)["classification"])

    def test_client_setup_failure_is_unknown_without_request_start(self):
        receipts = []
        with mock.patch("httpx.AsyncClient", side_effect=ValueError("private-message-secret")):
            with self.assertRaises(ValueError):
                post_json_bounded("https://fixture.invalid", {}, self.config(receipts, []))
        self.assertNotIn("request_started", [event["phase"] for event in receipts])
        self.assert_classification(receipts, "unknown", "request_failed")

    def test_transport_observer_is_optional(self):
        config = self.config([], [])
        del config.transport_observer
        with memory_endpoint() as (url, _, _):
            self.assertEqual({"usage": {"cost": .01}}, post_json_bounded(url, {}, config))

    def test_observer_mutation_cannot_change_internal_classification(self):
        classifications = []
        config = self.config([], [])

        def observer(event):
            if event["phase"] == "transport_classification":
                classifications.append(event["classification"])
            event["phase"] = "corrupted"

        config.transport_observer = observer
        with memory_endpoint(failure="dns") as (url, _, _):
            with self.assertRaises(httpx.ConnectError):
                post_json_bounded(url, {}, config)
        self.assertEqual(["not_dispatched"], classifications)

    def test_existing_transport_tests_with_network_free_endpoint(self):
        from tests import test_openrouter_transport as existing

        @contextmanager
        def endpoint(**kwargs):
            with memory_endpoint(**kwargs) as (url, requests, _):
                yield url, requests

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(existing.OpenRouterTransportTests)
        result = unittest.TestResult()
        with mock.patch.object(existing, "endpoint", endpoint):
            suite.run(result)
        self.assertEqual(4, result.testsRun)
        self.assertEqual([], result.errors)
        self.assertEqual([], result.failures)


if __name__ == "__main__":
    unittest.main()
