import copy
import json
import unittest

from ima.research_summary import reported_cost, summary, summary_json, summary_markdown, summary_preview
from ima.research_telemetry import champions, comparison_key
from tests.test_research_telemetry import terminal


def failure_snapshot():
    return {
        "campaign_id": "v6-fixture", "trace_role": "decision", "decision_id": "D000017",
        "created_at": "2026-10-05T03:12:21+08:00", "invocation_id": "new-invocation",
        "code_revision": "pinned-revision", "evidence_id": "input-17", "output_evidence_id": "output-17",
        "input_terminal_watermark": 100, "output_terminal_watermark": 102,
        "planner_status": "failed", "error": "DNS ConnectError before TCP/TLS/HTTP",
        "planner_spend_unknown": True, "transport_phase": "resolution", "planner_usage": {},
        "planner_spend": {"known_spend_usd": 0.477406735, "total_cost_usd": None,
                          "spend_unknown": True, "unknown_receipt_ids": ["D000017"]},
        "counts": {"completed": 87, "failed": 15, "reserved": 5, "running": 2},
        "resources": {"max_jobs": 2, "fit_ceiling": 26, "cpu_budget": 24, "ram_budget_gib": 93.13225746154785},
        "host_resources": {"cgroup": {"memory.events": {"oom_kill": 0}}},
        "invocation_history": [{"invocation_id": "old-invocation", "oom_kill": 1, "peak_bytes": 93100000000}],
        "restart_count": 1,
        "active_trials": [
            {"attempt_id": "slow-1", "status": "running", "recipe": {"model": {"kind": "performance_probit"}},
             "fold": 1, "stage": "training", "started_at": "2026-10-04T19:00:00Z", "deadline_at": "2026-10-04T19:10:00Z"},
            {"attempt_id": "slow-2", "status": "running", "model_family": "performance_probit", "fold": 1,
             "elapsed_seconds": 528, "max_wall_seconds": 600},
        ],
    }


class ResearchSummaryTests(unittest.TestCase):
    def test_d17_failure_is_distinct_from_delivery_and_keeps_unknown_billing(self):
        snapshot = failure_snapshot() | {"delivery_status": "delivered"}
        before = copy.deepcopy(snapshot)
        value = summary(snapshot)
        self.assertEqual(snapshot, before)
        self.assertEqual(value, summary(snapshot))
        self.assertEqual(value["operation"]["status"], "failed")
        self.assertTrue(value["operation"]["failed"])
        self.assertEqual(value["operation"]["delivery_status"], "delivered")
        self.assertEqual(value["billing"]["known_subtotal_usd"], 0.477406735)
        self.assertIsNone(value["billing"]["total_cost_usd"])
        self.assertIsNone(value["billing"]["receipt_cost_usd"])
        self.assertEqual(value["billing"]["unknown_receipt_ids"], ["D000017"])
        self.assertEqual(value["identity"]["observed_at_utc"], "2026-10-04T19:12:21Z")
        preview = summary_preview(value)
        self.assertLess(preview.index("health="), preview.index("operation="))
        self.assertLess(preview.index("blocker="), preview.index("known subtotal"))
        for text in ("failed", "DNS", "resolution", "historical OOM", "total USD unknown", "performance_probit", "ceiling=26"):
            self.assertIn(text, preview)

    def test_active_fits_are_not_inferred_from_terminal_file(self):
        value = summary(failure_snapshot())
        trials = value["execution"]["active_trials"]
        self.assertEqual(trials[0]["elapsed_seconds"], 741)
        self.assertEqual(trials[0]["fold"], 1)
        self.assertEqual(trials[0]["deadline_at"], "2026-10-04T19:10:00Z")
        self.assertIsNone(trials[1]["deadline_at"])
        self.assertEqual(value["execution"]["active_trial_count"], 7)
        self.assertFalse(value["execution"]["active_details_complete"])
        value = summary({"counts": {"terminal": 2, "completed": 2}})
        self.assertIsNone(value["execution"]["active_trials"])
        self.assertIsNone(value["execution"]["active_trial_count"])
        self.assertNotIn("running=0", summary_preview(value))

    def test_history_survives_current_zero_oom_counter(self):
        value = summary(failure_snapshot())
        self.assertTrue(value["execution"]["historical_oom"])
        self.assertEqual(value["execution"]["invocation_history"][0]["peak_bytes"], 93100000000)
        self.assertEqual(value["execution"]["restart_count"], 1)
        self.assertEqual(value["execution"]["host_resources"]["cgroup"]["memory.events"]["oom_kill"], 0)

    def test_chosen_allocated_held_continuing_and_actual_rationale(self):
        snapshot = {"decision_id": "D1", "planner_status": "accepted", "trial_budget": 10,
                    "allocated_trials": 8, "unallocated_trials": 2, "continuing_trials": 40,
                    "programs": [{"hypothesis": "Try a wider quadrature", "reason": "Actual planner reason"}],
                    "review_reason": "Exact review | rationale\nsecond line", "unallocated_reason": "Hold for evidence",
                    "rejected_fields": ["unsafe_field"], "research_memo_sha256": "exact-ack", "next_action": "await fits"}
        value = summary(snapshot)
        self.assertEqual({key: value["decision"]["budgets"][key] for key in ("chosen", "allocated", "held", "continuing")},
                         {"chosen": 10, "allocated": 8, "held": 2, "continuing": 40})
        self.assertEqual(value["decision"]["programs"], snapshot["programs"])
        self.assertEqual(value["decision"]["rationale"]["review_reason"], snapshot["review_reason"])
        self.assertEqual(value["decision"]["memo_ack"], "exact-ack")
        self.assertEqual(value["decision"]["rejected_fields"], ["unsafe_field"])
        self.assertIsNone(summary({"decision_id": "failed"})["decision"]["budgets"]["chosen"])

    def test_champion_delta_only_with_identical_comparison_key(self):
        prior = terminal("prior", 2.166148682757164)
        current = terminal("current", 2.1678962151517527, model="boosted")
        other = terminal("other-target", 0.01)
        other["payload"]["recipe"]["target"] = {"kind": "finish_time"}
        rows = champions([current, other])["champions_by_contract"]
        value = summary({"champions_by_contract": rows, "prior_champions_by_contract": champions([prior])["champions_by_contract"]})
        key = comparison_key(current["result"], current["payload"]["recipe"])
        row = next(row for row in value["champions"] if row["comparison_key"] == key)
        self.assertGreater(row["absolute_delta"], 0)
        self.assertAlmostEqual(row["relative_delta"], row["absolute_delta"] / prior["result"]["objective_value"])
        self.assertEqual(row["model_family"], "boosted")
        self.assertEqual(row["identity"]["evaluation_population_id"], "population-1")
        other_row = next(row for row in value["champions"] if row["comparison_key"] != key)
        self.assertIsNone(other_row["prior_value"])
        self.assertIsNone(other_row["absolute_delta"])
        self.assertEqual(len(value["champions"]), 2)

    def test_mismatched_embedded_contract_and_zero_prior_never_create_bad_delta(self):
        current = {"objective_value": 1, "comparison_key": "other"}
        prior = {"objective_value": 0, "comparison_key": "key"}
        value = summary({"champions_by_contract": {"key": current}, "prior_champions_by_contract": {"key": prior}})
        self.assertIsNone(value["champions"][0]["absolute_delta"])
        current["comparison_key"] = "key"
        value = summary({"champions_by_contract": {"key": current}, "prior_champions_by_contract": {"key": prior}})
        self.assertEqual(value["champions"][0]["absolute_delta"], 1)
        self.assertIsNone(value["champions"][0]["relative_delta"])

    def test_existing_planner_reference_list_is_grouped_by_contract(self):
        prior = terminal("prior", 2.166148682757164)
        worse = terminal("prior-family", 2.3, model="boosted")
        unrelated = terminal("other-population", 0.1)
        unrelated["result"]["lineage"]["evaluation_population_id"] = "other-population"
        current = terminal("current", 2.1678962151517527, model="boosted")
        references = list(champions([prior, worse, unrelated])["champions_by_family"].values())
        value = summary({"champions_by_contract": champions([current])["champions_by_contract"],
                         "planner_evidence": {"references": references}})
        row, = value["champions"]
        self.assertEqual(row["prior"]["attempt_id"], "prior")
        self.assertGreater(row["absolute_delta"], 0)

    def test_renderers_share_one_object_and_long_errors_survive_in_full(self):
        error = "DNS failure\n" + "diagnostic | " * 1000
        value = summary(failure_snapshot() | {"error": error})
        self.assertEqual(json.loads(summary_json(value)), value)
        markdown = summary_markdown(value)
        self.assertIn("diagnostic &#124;", markdown)
        self.assertIn(json.dumps(error), markdown)
        preview = summary_preview(value)
        self.assertIn("...", preview)
        self.assertIn("known subtotal USD 0.477406735", preview)
        self.assertNotIn("best None", preview)
        self.assertLess(len(preview), 1500)

    def test_native_receipts_only_not_assumed_zero_or_known_subtotal(self):
        for status, cost in (("fixture", 9), ("unavailable", 0), ("estimated", 1), ("reported", -1),
                             ("reported", float("nan")), ("reported", True), ("reported", "1")):
            self.assertIsNone(reported_cost({"cost_status": status, "total_cost_usd": cost}))
        self.assertEqual(reported_cost({"cost_status": "reported", "total_cost_usd": 0}), 0)
        value = summary({"decision_id": "D1", "planner_usage": {"known_cost_usd": 0.5}})
        self.assertEqual(value["billing"]["known_subtotal_usd"], 0.5)
        self.assertIsNone(value["billing"]["total_cost_usd"])

    def test_missing_time_caps_champions_and_cost_are_unknown(self):
        value = summary({})
        self.assertIsNone(value["identity"]["observed_at_utc"])
        self.assertIsNone(value["execution"]["caps"]["effective_fit_cap"])
        self.assertIsNone(value["billing"]["known_subtotal_usd"])
        self.assertIn("no comparable completed result yet", summary_preview(value))
        self.assertNotIn("best None", summary_markdown(value))
        value = summary({"created_at": "2026-10-05T03:12:21"})
        self.assertIsNone(value["identity"]["observed_at_utc"])

    def test_controller_spend_and_parent_runtime_history_fields(self):
        spend = {"known_spend_usd": 0.477406735, "total_cost_usd": None, "spend_unknown": True,
                 "unresolved_decision_ids": ["D000016"], "not_dispatched_decision_ids": ["D000017"]}
        value = summary({"decision_id": "D000017", "planner_usage": {"cost_status": "not_dispatched"},
                         "planner_spend": spend, "runtime_history": {
                             "invocations": [{"invocation_id": "old", "oom_count": 1}],
                             "oom_facts": [{"source": "journal", "time": "original-time"}],
                             "restart_count": 1, "current_invocation_id": "new"}})
        self.assertEqual(value["billing"]["unknown_receipt_ids"], ["D000016"])
        self.assertEqual(value["billing"]["not_dispatched_decision_ids"], ["D000017"])
        self.assertIsNone(value["billing"]["total_cost_usd"])
        self.assertTrue(value["billing"]["total_unresolved"])
        self.assertEqual(value["identity"]["invocation_id"], "new")
        self.assertEqual(value["execution"]["restart_count"], 1)
        self.assertEqual(value["execution"]["oom_history"][0]["source"], "journal")
        self.assertTrue(value["execution"]["historical_oom"])
        spend.update(total_cost_usd=0.477406735, spend_unknown=False, unresolved_decision_ids=[])
        value = summary({"decision_id": "D000017", "planner_usage": {"cost_status": "not_dispatched"}, "planner_spend": spend})
        self.assertFalse(value["billing"]["total_unresolved"])
        self.assertEqual(value["billing"]["total_cost_usd"], 0.477406735)
        self.assertIsNone(value["billing"]["receipt_cost_usd"])
        value = summary({"decision_id": "D17", "planner_usage": {"cost_status": "not_dispatched"}})
        self.assertFalse(value["billing"]["total_unresolved"])
        self.assertIsNone(value["billing"]["receipt_cost_usd"])
        self.assertIsNone(value["billing"]["known_subtotal_usd"])

    def test_hundreds_of_blockers_do_not_hide_failure_or_cost_preview(self):
        snapshot = failure_snapshot()
        snapshot["resources"]["admission_blockers"] = {f"attempt-{i}": ["max_fits", "ram_budget"] for i in range(300)}
        value = summary(snapshot)
        preview = summary_preview(value)
        self.assertIn("operation=failed; blocker=DNS", preview)
        self.assertIn("known subtotal USD 0.477406735", preview)
        self.assertNotIn("attempt-299", preview)
        self.assertEqual(len(value["operation"]["blockers"]), 302)

    def test_worker_progress_supplies_fold_and_stage(self):
        value = summary({"active_trials": [{"attempt_id": "slow", "status": "running", "model_family": "performance_probit",
                                            "progress": {"fold": 1, "stage": "training", "iteration": 30}}]})
        self.assertEqual(value["execution"]["active_trials"][0]["fold"], 1)
        self.assertEqual(value["execution"]["active_trials"][0]["stage"], "training")

    def test_input_and_output_watermarks_and_details_stay_separate(self):
        value = summary({"evidence_id": "original", "input_terminal_watermark": 1,
                         "output_evidence_id": "fresh", "output_terminal_watermark": 3,
                         "planner_evidence": {"counts": {"completed": 1}, "remaining_program_capacity": {"old": 40}},
                         "output_snapshot": {"counts": {"completed": 3}}, "counts": {"completed": 3}})
        self.assertEqual(value["execution"]["counts"], {"completed": 3})
        self.assertEqual(value["identity"]["input_terminal_watermark"], 1)
        self.assertEqual(value["identity"]["output_terminal_watermark"], 3)
        self.assertEqual(value["decision"]["budgets"]["remaining_program_capacity"], {"old": 40})


if __name__ == "__main__":
    unittest.main()
