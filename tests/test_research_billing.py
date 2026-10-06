import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ima import research_controller as core
from ima.research_expansion import _planner_cost, _planner_spend, _unsent_failure_streak


def unsent_events():
    return [{"phase": phase} for phase in (
        "request_started", "connection.connect_tcp.started",
        "connection.connect_tcp.failed", "request_failed",
    )]


def sent_events():
    return [{"phase": phase} for phase in (
        "request_started", "http11.send_request_headers.started", "request_failed",
    )]


class ResearchBillingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target in ("socket.socket.connect", "socket.getaddrinfo"):
            guard = patch(target, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def write(self, folder, identifier, payload):
        core._write_json_atomic(self.root/folder/(identifier+".json"), payload)

    def decision(self, identifier="D000001", *, status="failed", usage=None):
        self.write("decisions", identifier, {"planner_status": status,
                                            "planner_usage": usage if usage is not None else {}})

    def transport(self, identifier="D000001-01", events=None):
        self.write("planner-transport", identifier,
                   {"events": unsent_events() if events is None else events})

    def receipt(self, identifier="D000001-01", cost=.125):
        usage = {"cost": cost} if cost is not None else {"prompt_tokens": 10}
        self.write("planner-calls", identifier, {"response": {"usage": usage}})

    def response(self, identifier="D000001", usage=None):
        self.write("planner-responses", identifier, {"response": {"usage": usage or {}}})

    def assert_unknown(self, subtotal, identifier="D000001"):
        spend = _planner_spend(self.root)
        self.assertTrue(spend["spend_unknown"], spend)
        self.assertIsNone(spend["total_cost_usd"])
        self.assertAlmostEqual(subtotal, spend["known_spend_usd"])
        self.assertIn(identifier, spend["unresolved_decision_ids"])
        self.assertNotIn(identifier, spend["not_dispatched_decision_ids"])
        self.assertEqual(0, _unsent_failure_streak(self.root))
        return spend

    def test_empty_campaign_has_no_unresolved_spend(self):
        self.assertEqual(0, _planner_cost(self.root))
        self.assertFalse(_planner_spend(self.root)["spend_unknown"])
        self.assertEqual(0, _unsent_failure_streak(self.root))

    def test_all_complete_contiguous_unsent_attempts_are_exempt(self):
        self.decision()
        self.transport()
        self.transport("D000001-02")
        spend = _planner_spend(self.root)
        self.assertFalse(spend["spend_unknown"])
        self.assertEqual(0, spend["total_cost_usd"])
        self.assertEqual(["D000001"], spend["not_dispatched_decision_ids"])
        self.assertEqual(1, _unsent_failure_streak(self.root))

    def test_missing_prefix_attempt_is_unknown(self):
        self.decision()
        self.transport("D000001-02")
        self.assert_unknown(0)

    def test_missing_middle_attempt_is_unknown(self):
        self.decision()
        self.transport()
        self.transport("D000001-03")
        self.assert_unknown(0)

    def test_recorded_attempt_count_rejects_missing_final_receipt(self):
        self.write("decisions", "D000001", {
            "planner_status": "failed", "planner_usage": {},
            "transport_attempts": [{"events": unsent_events()}, {"events": sent_events()}],
        })
        self.transport()
        self.assert_unknown(0)

    def test_recorded_attempt_count_rejects_missing_all_physical_evidence(self):
        self.write("decisions", "D000001", {
            "planner_status": "accepted", "planner_usage": {"total_cost_usd": .125},
            "transport_attempts": [{"events": sent_events()}],
        })
        self.assert_unknown(.125)

    def test_numeric_receipt_cannot_cover_missing_middle_attempt(self):
        self.decision(usage={"total_cost_usd": .125})
        self.receipt()
        self.transport("D000001-03")
        self.assert_unknown(.125)

    def test_missing_prefix_physical_receipt_is_unknown(self):
        self.receipt("D000001-02")
        self.assert_unknown(.125)

    def test_zero_numbered_or_malformed_attempt_is_unknown(self):
        self.decision()
        for identifier in ("D000001-00", "D000001-unknown", "D000001-999999999"):
            with self.subTest(identifier=identifier):
                self.transport(identifier)
                self.assert_unknown(0)

    def test_duplicate_numeric_aliases_do_not_prove_coverage(self):
        self.decision()
        self.transport()
        self.transport("D000001-1")
        self.assert_unknown(0)

    def test_incomplete_connect_failure_remains_unknown(self):
        self.decision()
        for attempt in ([], unsent_events()[:-1], unsent_events()[1:], [{"phase": "request_started"}]):
            with self.subTest(attempt=attempt):
                self.transport(events=attempt)
                self.assert_unknown(0)

    def test_missing_events_key_remains_unknown(self):
        self.decision()
        self.write("planner-transport", "D000001-01", {})
        self.assert_unknown(0)

    def test_failed_numeric_aggregate_without_transport_is_only_subtotal(self):
        for cost in (.125, 0):
            with self.subTest(cost=cost):
                self.decision(usage={"total_cost_usd": cost})
                self.assert_unknown(cost)

    def test_failed_physical_cost_without_transport_is_only_subtotal(self):
        self.decision(usage={"total_cost_usd": .125})
        self.receipt()
        self.assert_unknown(.125)

    def test_accepted_legacy_numeric_aggregate_remains_replayable(self):
        self.decision(status="accepted", usage={"total_cost_usd": .125})
        self.assertEqual(.125, _planner_cost(self.root))
        self.assertFalse(_planner_spend(self.root)["spend_unknown"])

    def test_legacy_receipt_precedence_preserves_point_four_and_point_fifty_five(self):
        self.receipt(cost=.1)
        self.response(usage={"total_cost_usd": .2})
        self.assertAlmostEqual(.1, _planner_cost(self.root))
        self.write("decisions", "D000001", {"planner_usage": {"total_cost_usd": .2}})
        self.response("D000002", usage={"total_cost_usd": .3})
        self.assertAlmostEqual(.4, _planner_cost(self.root))
        self.receipt("D000001-02", cost=.15)
        self.assertAlmostEqual(.55, _planner_cost(self.root))
        self.receipt("D000001-03", cost=None)
        spend = self.assert_unknown(.55)
        self.assertEqual(["D000001"], spend["unresolved_decision_ids"])

    def test_failed_decision_with_full_reported_receipt_coverage_is_known(self):
        self.decision(usage={"total_cost_usd": .125})
        self.receipt()
        self.transport(events=sent_events())
        self.assertEqual(.125, _planner_cost(self.root))
        self.assertEqual(0, _unsent_failure_streak(self.root))

    def test_paid_prefix_and_unsent_sibling_with_full_coverage_are_known(self):
        self.decision(usage={"total_cost_usd": .125})
        self.receipt()
        self.transport(events=sent_events())
        self.transport("D000001-02")
        spend = _planner_spend(self.root)
        self.assertEqual(.125, spend["total_cost_usd"])
        self.assertEqual([], spend["not_dispatched_decision_ids"])
        self.assertEqual(0, _unsent_failure_streak(self.root))

    def test_sent_or_ambiguous_sibling_without_receipt_freezes(self):
        self.decision(usage={"total_cost_usd": .125})
        self.receipt()
        for attempt in (sent_events(), [], [{"phase": "request_started"}]):
            with self.subTest(attempt=attempt):
                self.transport("D000001-02", events=attempt)
                self.transport("D000001-03")
                self.assert_unknown(.125)

    def test_conflicting_reported_response_is_retained_and_frozen(self):
        self.decision()
        self.transport()
        self.response(usage={"total_cost_usd": .2})
        self.assert_unknown(.2)

    def test_unpriced_response_prevents_unsent_exemption(self):
        self.decision()
        self.transport()
        self.response()
        self.assert_unknown(0)

    def test_zero_cost_response_still_conflicts_with_unsent_proof(self):
        self.decision()
        self.transport()
        self.response(usage={"total_cost_usd": 0})
        self.assert_unknown(0)

    def test_numeric_decision_usage_conflicts_with_unsent_proof(self):
        self.decision(usage={"total_cost_usd": .125})
        self.transport()
        self.assert_unknown(.125)

    def test_invalid_reported_cost_cannot_turn_into_unsent_exemption(self):
        self.transport()
        for cost in (True, -1, "invalid", float("nan"), float("inf")):
            with self.subTest(cost=cost):
                self.decision(usage={"total_cost_usd": cost})
                self.assert_unknown(0)
        self.decision(usage={"cost_status": "reported"})
        self.assert_unknown(0)

    def test_physical_response_conflicts_with_unsent_proof(self):
        self.decision()
        self.transport()
        self.receipt()
        self.assert_unknown(.125)

    def test_conflicting_physical_response_remains_unknown_among_paid_siblings(self):
        self.decision()
        self.transport()
        self.receipt()
        self.transport("D000001-02", events=sent_events())
        self.receipt("D000001-02", cost=.075)
        self.assert_unknown(.2)

    def test_missing_physical_cost_cannot_be_replaced_by_aggregate(self):
        self.decision(usage={"total_cost_usd": .125})
        self.transport()
        self.receipt(cost=None)
        self.assert_unknown(0)

    def test_duplicate_aggregates_are_not_double_charged(self):
        self.decision(status="accepted", usage={"total_cost_usd": .3})
        self.response(usage={"total_cost_usd": .3})
        self.assertEqual(.3, _planner_cost(self.root))

    def test_conflicting_aggregate_values_keep_subtotal_but_freeze(self):
        self.decision(status="accepted", usage={"total_cost_usd": .1})
        self.response(usage={"total_cost_usd": .2})
        self.assert_unknown(.2)

    def test_fixture_cost_status_is_exempt(self):
        self.decision(usage={"cost_status": "fixture"})
        self.assertEqual(0, _planner_cost(self.root))
        self.transport("D000001-03", events=[])
        self.assertEqual(0, _planner_cost(self.root))
        self.assertEqual(0, _unsent_failure_streak(self.root))

    def test_fixture_marker_cannot_discard_actual_paid_receipt(self):
        self.decision(usage={"cost_status": "fixture"})
        self.receipt()
        self.assert_unknown(.125)

    def test_three_consecutive_failures_require_identical_spend_and_streak_proof(self):
        for number in range(1, 4):
            identifier = f"D{number:06d}"
            self.decision(identifier)
            self.transport(identifier+"-01")
        self.assertEqual(3, _unsent_failure_streak(self.root))
        self.transport("D000002-03")
        self.assertEqual(1, _unsent_failure_streak(self.root))
        spend = _planner_spend(self.root)
        self.assertEqual(["D000002"], spend["unresolved_decision_ids"])
        self.assertEqual(["D000001", "D000003"], spend["not_dispatched_decision_ids"])
        self.decision("D000004", status="accepted", usage={"total_cost_usd": .1})
        self.assertEqual(0, _unsent_failure_streak(self.root))

    def test_receipts_remain_immutable_during_replay(self):
        self.decision()
        self.transport()
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        self.assertEqual(0, _planner_cost(self.root))
        self.assertEqual(1, _unsent_failure_streak(self.root))
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*.json")})


if __name__ == "__main__":
    unittest.main()
