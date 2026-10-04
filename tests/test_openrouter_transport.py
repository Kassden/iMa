import json
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from ima.openrouter_transport import post_json_bounded


@contextmanager
def endpoint(*, delay=0, status=200, body=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            encoded = json.dumps(body if body is not None else {"usage": {"cost": .01}}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("X-Generation-Id", "gen-test-header")
            self.send_header("X-Request-Id", "req-test-header")
            self.end_headers()
            self.wfile.flush()
            time.sleep(delay)
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class OpenRouterTransportTests(unittest.TestCase):
    def config(self, events, responses, *, deadline=5):
        return SimpleNamespace(api_key="secret-not-in-receipts", timeout_seconds=5,
                               absolute_deadline_seconds=deadline,
                               transport_observer=events.append,
                               response_observer=responses.append)

    def test_headers_and_cost_are_observed_before_success(self):
        events, responses = [], []
        with endpoint() as (url, requests):
            result = post_json_bounded(url, {"model": "fixture"}, self.config(events, responses))
        self.assertEqual([{"model": "fixture"}], requests)
        self.assertEqual([result], responses)
        headers = next(event for event in events if event["phase"] == "response_headers")
        self.assertEqual("gen-test-header", headers["generation_id"])
        self.assertEqual("req-test-header", headers["request_id"])
        self.assertEqual(200, headers["status_code"])
        self.assertEqual("response_complete", events[-1]["phase"])
        self.assertNotIn("secret-not-in-receipts", json.dumps(events))

    def test_body_timeout_retains_generation_id_without_inventing_cost(self):
        events, responses = [], []
        with endpoint(delay=3) as (url, requests):
            with self.assertRaises(TimeoutError):
                post_json_bounded(url, {"model": "fixture"},
                                  self.config(events, responses, deadline=2))
        headers = next(event for event in events if event["phase"] == "response_headers")
        self.assertEqual("gen-test-header", headers["generation_id"])
        self.assertEqual("request_failed", events[-1]["phase"])
        self.assertEqual("TimeoutError", events[-1]["exception_type"])
        self.assertEqual([], responses)
        self.assertEqual(1, len(requests))
        self.assertNotIn("cost", json.dumps(events))

    def test_http_failure_keeps_identity_and_does_not_retry(self):
        import httpx
        events, responses = [], []
        with endpoint(status=503) as (url, requests):
            with self.assertRaises(httpx.HTTPStatusError):
                post_json_bounded(url, {}, self.config(events, responses))
        headers = next(event for event in events if event["phase"] == "response_headers")
        self.assertEqual(503, headers["status_code"])
        self.assertEqual("gen-test-header", headers["generation_id"])
        self.assertEqual([], responses)
        self.assertEqual(1, len(requests))

    def test_non_object_json_is_not_accepted_as_a_planner_response(self):
        events, responses = [], []
        with endpoint(body=[1, 2]) as (url, _):
            with self.assertRaisesRegex(ValueError, "JSON object"):
                post_json_bounded(url, {}, self.config(events, responses))
        self.assertEqual([], responses)


if __name__ == "__main__":
    unittest.main()
