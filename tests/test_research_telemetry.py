import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ima.research_telemetry import champions, comparison_key, drain_trace_outbox, evidence_snapshot, log_snapshot, next_trace_number, _serialize_trace


def recipe(model="benter_conditional_logit", **target_parameters):
    return {
        "target": {"kind": "win_probability", "parameters": target_parameters},
        "model": {"kind": model, "parameters": {"l2": 0.1}},
        "feature_schema": "benter-rich-v1",
    }


def result(score=2.1, **changes):
    value = {
        "status": "completed",
        "target_kind": "win_probability",
        "objective_name": "fundamental_log_loss",
        "objective_value": score,
        "metrics": {"metric_contract_version": "v6"},
        "lineage": {
            "protocol_id": "protocol-1",
            "dataset_hash": "dataset-1",
            "evaluation_population_id": "population-1",
            "probability_basis": "fundamental",
            "availability_policy": "verified-point-in-time",
        },
    }
    return value | changes


def terminal(identifier, score, model="benter_conditional_logit", **changes):
    row = {
        "attempt_id": identifier,
        "status": "completed",
        "payload": {"program_id": "program-" + identifier, "recipe": recipe(model)},
        "result": result(score),
    }
    return row | changes


class ComparisonKeyTests(unittest.TestCase):
    def test_key_is_stable_under_dictionary_order_and_does_not_mutate_inputs(self):
        original_result, original_recipe = result(), recipe(alpha=1, beta=2)
        before = copy.deepcopy((original_result, original_recipe))
        reordered_recipe = copy.deepcopy(original_recipe)
        reordered_recipe["target"]["parameters"] = {"beta": 2, "alpha": 1}
        self.assertEqual(
            comparison_key(original_result, original_recipe),
            comparison_key(original_result, reordered_recipe),
        )
        self.assertIsInstance(comparison_key(original_result, original_recipe), str)
        self.assertEqual((original_result, original_recipe), before)

    def test_target_parameters_separate_comparisons(self):
        self.assertNotEqual(
            comparison_key(result(), recipe(top_k=2)),
            comparison_key(result(), recipe(top_k=3)),
        )

    def test_same_named_protocol_with_different_content_is_not_comparable(self):
        first, second = result(), result()
        first["lineage"]["protocol_hash"] = "hash-1"
        second["lineage"]["protocol_hash"] = "hash-2"
        self.assertNotEqual(comparison_key(first, recipe()), comparison_key(second, recipe()))

    def test_each_scoring_contract_dimension_separates_comparisons(self):
        baseline = result()
        key = comparison_key(baseline, recipe())
        changes = {
            "protocol_id": "protocol-2",
            "evaluation_population_id": "population-2",
            "probability_basis": "market-blended",
            "availability_policy": "assumed-retrospective",
        }
        for name, value in changes.items():
            other = copy.deepcopy(baseline)
            other["lineage"][name] = value
            with self.subTest(dimension=name):
                self.assertNotEqual(key, comparison_key(other, recipe()))

    def test_objective_and_metric_version_separate_comparisons(self):
        key = comparison_key(result(), recipe())
        for other in (
            result(objective_name="race_rmse"),
            result(metrics={"metric_contract_version": "different"}),
        ):
            with self.subTest(result=other):
                self.assertNotEqual(key, comparison_key(other, recipe()))

    def test_different_feature_datasets_can_compare_on_same_score_population(self):
        first, second = result(), result()
        second["lineage"]["dataset_hash"] = "new-features-same-runners"
        self.assertEqual(comparison_key(first, recipe()), comparison_key(second, recipe()))

    def test_estimator_and_objective_value_are_not_comparison_boundaries(self):
        self.assertEqual(
            comparison_key(result(2.2), recipe()),
            comparison_key(result(2.0), recipe("boosted")),
        )

    def test_physical_and_latent_target_units_are_not_comparable(self):
        physical, latent = result(), result()
        physical["lineage"]["target_unit"] = "m/s"
        latent["lineage"]["target_unit"] = "latent-strength"
        self.assertNotEqual(comparison_key(physical, recipe()), comparison_key(latent, recipe()))


class ChampionTests(unittest.TestCase):
    def test_current_best_retains_its_actual_recipe(self):
        old = terminal("old", 2.2)
        new = terminal("new", 2.0, model="boosted")
        before = copy.deepcopy((old, new))
        grouped = champions([old, new])["champions_by_contract"]
        key = comparison_key(new["result"], new["payload"]["recipe"])
        self.assertIn(key, grouped)
        self.assertEqual(grouped[key]["attempt_id"], "new")
        self.assertEqual(grouped[key]["recipe"], new["payload"]["recipe"])
        self.assertEqual((old, new), before)

    def test_champion_is_fresh_when_a_completion_is_appended(self):
        old, new = terminal("old", 2.2), terminal("new", 2.0)
        key = comparison_key(old["result"], old["payload"]["recipe"])
        self.assertEqual(champions([old])["champions_by_contract"][key]["attempt_id"], "old")
        self.assertEqual(champions([old, new])["champions_by_contract"][key]["attempt_id"], "new")

    def test_incompatible_population_does_not_displace_champion(self):
        first, other = terminal("first", 2.2), terminal("other", 0.1)
        other["result"]["lineage"]["evaluation_population_id"] = "other-population"
        grouped = champions([first, other])["champions_by_contract"]
        first_key = comparison_key(first["result"], first["payload"]["recipe"])
        other_key = comparison_key(other["result"], other["payload"]["recipe"])
        self.assertNotEqual(first_key, other_key)
        self.assertEqual(grouped[first_key]["attempt_id"], "first")
        self.assertEqual(grouped[other_key]["attempt_id"], "other")

    def test_failed_pending_and_nonfinite_scores_cannot_be_champions(self):
        good = terminal("good", 2.2)
        invalid = []
        for status in ("failed", "reserved", "running"):
            invalid.append(terminal(status, 0.01, status=status))
        for identifier, score in (("nan", float("nan")), ("inf", float("inf")), ("negative-inf", float("-inf")), ("missing", None)):
            invalid.append(terminal(identifier, score))
        key = comparison_key(good["result"], good["payload"]["recipe"])
        grouped = champions(invalid + [good])["champions_by_contract"]
        self.assertEqual(grouped[key]["attempt_id"], "good")
        self.assertEqual({row["attempt_id"] for row in grouped.values()}, {"good"})

    def test_input_order_does_not_change_unique_best(self):
        rows = [terminal("third", 2.3), terminal("first", 2.0), terminal("second", 2.1)]
        key = comparison_key(rows[0]["result"], rows[0]["payload"]["recipe"])
        self.assertEqual(champions(rows)["champions_by_contract"][key]["attempt_id"], "first")
        self.assertEqual(champions(list(reversed(rows)))["champions_by_contract"][key]["attempt_id"], "first")

    def test_equal_score_champions_are_deterministic_under_replay_order(self):
        rows = [terminal("z", 2.0), terminal("a", 2.0)]
        self.assertEqual(champions(rows), champions(list(reversed(rows))))

    def test_family_champions_retain_constituents_and_contract_champion(self):
        rows = [terminal("benter", 2.2), terminal("boosted", 2.0, model="boosted")]
        view = champions(rows)
        self.assertEqual(len(view["champions_by_contract"]), 1)
        self.assertEqual(len(view["champions_by_family"]), 2)
        self.assertEqual(
            {row["attempt_id"] for row in view["champions_by_family"].values()},
            {"benter", "boosted"},
        )
        self.assertEqual(view["terminal_watermark"], 2)

    def test_missing_protocol_identity_does_not_group_unrelated_attempts(self):
        first, second = terminal("first", 2.2), terminal("second", 0.1)
        for row in (first, second):
            del row["result"]["lineage"]["protocol_id"]
        grouped = champions([first, second])["champions_by_contract"]
        self.assertEqual(len(grouped), 2)


class SnapshotCostTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.campaign = Path(self.directory.name) / "isolated-v6"
        self.config = SimpleNamespace(
            research_policy="expansion_v6", mlflow_tracking_uri="http://tracking.invalid"
        )
        self.mlflow = MagicMock()
        self.mlflow.set_experiment.return_value.experiment_id = "experiment-1"
        self.span = self.mlflow.start_span.return_value.__enter__.return_value
        self.span.trace_id = "trace-1"
        self.tracking_patch = patch("ima.mlflow_tracking._mlflow", return_value=self.mlflow)
        self.tracking_patch.start()
        self.addCleanup(self.tracking_patch.stop)
        serialization = patch("ima.research_telemetry._serialize_trace", return_value={"serialized": "trace-1"})
        serialization.start()
        self.addCleanup(serialization.stop)

    def payload(self, cost=0.02):
        return {
            "decision_id": "decision-1", "evidence_id": "evidence-1",
            "trial_budget": 1, "allocated_trials": 1,
            "planner_usage": {
                "total_cost_usd": cost,
                "cost_status": "reported" if cost is not None else "unavailable",
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            },
        }

    def test_paid_decision_native_cost_is_written_once_and_link_reused(self):
        payload = self.payload()
        first = log_snapshot(self.campaign, payload, self.config, role="decision", number=1)
        second = log_snapshot(self.campaign, payload, self.config, role="decision", number=1)
        self.assertEqual(first, second)
        self.assertEqual(first["total_cost_usd"], 0.02)
        self.assertEqual(self.mlflow.start_span.call_count, 1)
        self.span.set_attribute.assert_any_call("mlflow.llm.cost", {"total_cost": 0.02})
        self.span.set_attribute.assert_any_call(
            "mlflow.chat.tokenUsage", {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        )

    def test_execution_snapshot_does_not_recharge_origin_decision(self):
        link = log_snapshot(self.campaign, self.payload(), self.config, role="execution", number=1)
        self.assertIsNone(link["total_cost_usd"])
        self.assertFalse(any(
            call.args[0] == "mlflow.llm.cost" for call in self.span.set_attribute.call_args_list
        ))

    def test_betting_trace_is_chain_without_recharging_planner_tokens_or_cost(self):
        link = log_snapshot(self.campaign,self.payload(),self.config,role="betting",number=1)
        self.assertEqual("CHAIN",self.mlflow.start_span.call_args.kwargs["span_type"])
        self.assertIsNone(link["total_cost_usd"])
        self.assertFalse(any(call.args[0] in {"mlflow.llm.cost","mlflow.chat.tokenUsage"}
                             for call in self.span.set_attribute.call_args_list))

    def paper_payload(self):
        import hashlib
        action = "a"*24
        report = {"request_id":"paper-study-1","evidence_id":"evidence-1","paper_only":True,
                  "executable_evidence":False,"coverage":{"evaluated_races":2,"requested_races":2,"available_races":20},
                  "model_ids":[{"attempt_id":"current-1"}],"probability_basis":"fundamental",
                  "request":{"quote_mode":"fair_price"}}
        path = self.campaign.resolve()/"paper-actions"/(action+".json")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"action_id":action,"report":report}))
        return {"action_id":action,"request_id":"paper-study-1","evidence_id":"evidence-1",
                "completion":{"status":"completed","report_path":str(path),
                              "report_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}}

    def test_paper_preview_uses_receipted_report_not_trial_champions(self):
        payload = self.paper_payload() | {"planner_usage":{"total_cost_usd":9,"total_tokens":999}}
        log_snapshot(self.campaign,payload,self.config,role="betting",number=1)
        preview = self.mlflow.update_current_trace.call_args.kwargs["response_preview"]
        for text in ("paper-study-1","status=completed","races=2/2 requested","available=20",
                     "models=1","basis=fundamental","quote_mode=fair_price","paper only"):
            self.assertIn(text,preview)
        for text in ("counts unavailable","no comparable","Planner $","best="):
            self.assertNotIn(text,preview)
        self.assertFalse(any(call.args[0] in {"mlflow.llm.cost","mlflow.chat.tokenUsage"}
                             for call in self.span.set_attribute.call_args_list))

    def test_failed_paper_preview_does_not_fabricate_coverage(self):
        payload = {"request_id":"unsupported-paper","completion":{"status":"failed","error":"No compatible races"}}
        log_snapshot(self.campaign,payload,self.config,role="betting",number=1)
        preview = self.mlflow.update_current_trace.call_args.kwargs["response_preview"]
        self.assertIn("unsupported-paper; status=failed; No compatible races",preview)
        self.assertNotIn("races=0",preview)
        self.assertNotIn("no comparable",preview)

    def test_paper_preview_rejects_tampered_report_before_remote_span(self):
        payload = self.paper_payload()
        Path(payload["completion"]["report_path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            log_snapshot(self.campaign,payload,self.config,role="betting",number=1)
        self.mlflow.start_span.assert_not_called()

    def test_unreported_paid_cost_stays_unknown(self):
        link = log_snapshot(self.campaign, self.payload(None), self.config, role="decision", number=1)
        self.assertIsNone(link["total_cost_usd"])
        self.assertFalse(any(
            call.args[0] == "mlflow.llm.cost" for call in self.span.set_attribute.call_args_list
        ))

    def test_fixture_decision_is_chain_without_synthetic_llm_cost_or_tokens(self):
        payload = self.payload()
        payload["planner_usage"]["cost_status"] = "fixture"
        link = log_snapshot(self.campaign, payload, self.config, role="decision", number=1)
        self.assertEqual(self.mlflow.start_span.call_args.kwargs["span_type"], "CHAIN")
        self.assertIsNone(link["total_cost_usd"])
        self.assertFalse(any(call.args[0] in {"mlflow.llm.cost", "mlflow.chat.tokenUsage"}
                             for call in self.span.set_attribute.call_args_list))

    def test_paid_planner_decision_is_llm_even_when_cost_is_unreported(self):
        log_snapshot(self.campaign, self.payload(None), self.config, role="decision", number=1)
        self.assertEqual(self.mlflow.start_span.call_args.kwargs["span_type"], "LLM")

    def test_d17_domain_failure_delivers_with_shared_summary_and_failure_tags(self):
        from ima.research_summary import summary_preview
        from tests.test_research_summary import failure_snapshot
        payload = failure_snapshot()
        before = copy.deepcopy(payload)
        receipt = log_snapshot(self.campaign, payload, self.config, role="decision", number=17)
        self.assertIsNone(receipt["total_cost_usd"])
        self.assertEqual(payload, before)
        emitted = self.span.set_outputs.call_args.args[0]
        written = emitted["summary"]
        self.assertEqual(written["operation"]["status"], "failed")
        self.assertEqual(written["operation"]["delivery_status"], "pending")
        self.assertEqual(emitted["planner_usage"], {})
        self.assertEqual(self.span.set_status.call_args.args[0].status_code.value, "ERROR")
        arguments = self.mlflow.update_current_trace.call_args.kwargs
        self.assertEqual(arguments["response_preview"], summary_preview(written))
        tags = arguments["tags"]
        self.assertEqual(tags["ima.operation_status"], "failed")
        self.assertIn("DNS", tags["ima.blocker"])
        self.assertEqual(tags["ima.model_families"], "performance_probit")
        self.assertEqual(tags["ima.active_trial_count"], "7")
        self.assertEqual(tags["ima.completed_count"], "87")
        self.assertEqual(tags["ima.summary_schema_version"], "1")
        self.assertEqual(tags["mlflow.traceName"], "isolated-v6 | decision D000017")
        artifacts = self.campaign / "traces" / "decision-000017"
        self.assertEqual(json.loads((artifacts / "summary.json").read_text()), written)
        self.assertEqual((artifacts / "summary.md").read_text(), emitted["summary_markdown"])
        row, = self.outbox_rows()
        self.assertEqual(row["delivered"], 1)
        self.assertEqual(json.loads(row["payload"]), payload)
        self.assertFalse(any(call.args[0] == "mlflow.llm.cost" for call in self.span.set_attribute.call_args_list))
        self.assertEqual(receipt, log_snapshot(self.campaign, payload, self.config, role="decision", number=17))
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_unreported_numeric_cost_never_becomes_native_usd(self):
        for status in ("estimated", "unavailable", "not_dispatched"):
            with self.subTest(status=status):
                self.span.reset_mock()
                payload = self.payload() | {"planner_usage": {"cost_status": status, "total_cost_usd": 99,
                                                             "input_tokens": 5}}
                receipt = log_snapshot(self.campaign, payload, self.config, role="decision",
                                       number={"estimated": 1, "unavailable": 2, "not_dispatched": 3}[status])
                self.assertIsNone(receipt["total_cost_usd"])
                emitted = self.span.set_outputs.call_args.args[0]
                self.assertEqual(emitted["planner_usage"], payload["planner_usage"])
                self.assertIsNone(emitted["summary"]["billing"]["total_cost_usd"])
                self.assertFalse(any(call.args[0] == "mlflow.llm.cost" for call in self.span.set_attribute.call_args_list))
                self.span.set_attribute.assert_any_call("mlflow.chat.tokenUsage", {"input_tokens": 5})

    def test_proven_unsent_call_preserves_absent_receipt_without_native_usd(self):
        payload = self.payload(None) | {
            "planner_status": "failed", "error": "DNS failure", "transport_phase": "resolution",
            "planner_usage": {"cost_status": "not_dispatched"},
            "planner_spend": {"known_spend_usd": 0.477406735, "total_cost_usd": 0.477406735,
                              "spend_unknown": False, "unresolved_decision_ids": [],
                              "not_dispatched_decision_ids": ["decision-1"]},
        }
        receipt = log_snapshot(self.campaign, payload, self.config, role="decision", number=1)
        self.assertIsNone(receipt["total_cost_usd"])
        emitted = self.span.set_outputs.call_args.args[0]
        self.assertEqual(emitted["planner_usage"], {"cost_status": "not_dispatched"})
        self.assertFalse(emitted["summary"]["billing"]["total_unresolved"])
        self.assertEqual(emitted["summary"]["billing"]["total_cost_usd"], 0.477406735)
        self.assertFalse(any(call.args[0] in {"mlflow.llm.cost", "mlflow.chat.tokenUsage"}
                             for call in self.span.set_attribute.call_args_list))

    def test_failed_planner_readback_outage_keeps_failure_and_reuses_identity(self):
        payload = self.payload() | {"planner_status": "failed", "error": "Invalid planner decision"}
        with patch("ima.research_telemetry._verify_or_restore_trace", side_effect=RuntimeError("readback offline")):
            with self.assertRaisesRegex(RuntimeError, "readback offline"):
                log_snapshot(self.campaign, payload, self.config, role="decision", number=1)
        row, = self.outbox_rows()
        self.assertEqual(row["delivered"], 0)
        self.assertIsNotNone(row["receipt"])
        artifacts = self.campaign / "traces" / "decision-000001"
        original_json = (artifacts / "summary.json").read_bytes()
        original_markdown = (artifacts / "summary.md").read_bytes()
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertEqual((artifacts / "summary.json").read_bytes(), original_json)
        self.assertEqual((artifacts / "summary.md").read_bytes(), original_markdown)
        self.assertEqual(self.mlflow.start_span.call_count, 1)
        self.assertEqual(len([call for call in self.span.set_attribute.call_args_list
                              if call.args[0] == "mlflow.llm.cost"]), 1)
        self.assertEqual(self.outbox_rows()[0]["delivered"], 1)
        self.assertEqual(self.span.set_outputs.call_args.args[0]["summary"]["operation"]["status"], "failed")

    def test_old_delivered_trace_is_not_rewritten_to_new_summary(self):
        receipt = log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        artifacts = self.campaign / "traces" / "decision-000001"
        (artifacts / "summary.json").write_text('{"historical": true}')
        self.assertEqual(receipt, log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1))
        self.assertEqual(json.loads((artifacts / "summary.json").read_text()), {"historical": True})
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def outbox_rows(self):
        with sqlite3.connect(self.campaign / "trace-outbox.sqlite") as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM traces ORDER BY number")]

    def test_failed_logging_is_durable_and_drain_retries_without_replanning(self):
        self.mlflow.set_experiment.side_effect = RuntimeError("tracking offline")
        with self.assertRaisesRegex(RuntimeError, "tracking offline"):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        row, = self.outbox_rows()
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["delivered"], 0)
        self.assertEqual(json.loads(row["payload"]), self.payload())
        self.assertTrue((self.campaign / "traces" / "decision-000001" / "summary.md").exists())
        self.mlflow.set_experiment.side_effect = None
        view = drain_trace_outbox(self.campaign, self.config)
        self.assertEqual(view, {"delivered": 1, "pending": 0, "errors": []})
        row, = self.outbox_rows()
        self.assertEqual(row["attempts"], 2)
        self.assertIsNone(row["last_error"])
        self.assertEqual(self.mlflow.set_experiment.call_count, 2)
        self.assertEqual(self.mlflow.start_span.call_count, 1)
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["delivered"], 0)
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_actual_noop_span_stays_pending_without_receipt_and_retries(self):
        from mlflow.entities import NoOpSpan
        self.mlflow.start_span.return_value.__enter__.return_value = NoOpSpan()
        payload = self.payload()
        with self.assertRaisesRegex(RuntimeError,"NoOpSpan; no recording trace"):
            log_snapshot(self.campaign,payload,self.config,role="decision",number=1)
        row, = self.outbox_rows()
        self.assertEqual(0,row["delivered"])
        self.assertIsNone(row["receipt"])
        self.assertIsNone(row["trace_json"])
        self.assertEqual(payload,json.loads(row["payload"]))
        self.assertFalse((self.campaign/"traces"/"decision-000001.json").exists())
        self.mlflow.update_current_trace.assert_not_called()
        self.mlflow.start_span.return_value.__enter__.return_value = self.span
        self.assertEqual({"delivered":1,"pending":0,"errors":[]},drain_trace_outbox(self.campaign,self.config))
        self.assertEqual(2,self.outbox_rows()[0]["attempts"])
        self.assertEqual(1,len([call for call in self.span.set_attribute.call_args_list if call.args[0]=="mlflow.llm.cost"]))

    def test_direct_serializer_rejects_actual_noop_before_inherited_context_access(self):
        from mlflow.entities import NoOpSpan
        with self.assertRaisesRegex(RuntimeError,"Delivery remains pending"):
            _serialize_trace(NoOpSpan(),"1",{"ima.operation_status":"failed"},"test","test",self.campaign,{})

    def test_flush_failure_reuses_persisted_remote_identity_and_cost(self):
        self.mlflow.flush_trace_async_logging.side_effect = [RuntimeError("flush offline"), None]
        with self.assertRaisesRegex(RuntimeError, "flush offline"):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        row, = self.outbox_rows()
        self.assertEqual(json.loads(row["receipt"])["trace_id"], "trace-1")
        self.assertEqual(row["delivered"], 0)
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertEqual(self.mlflow.start_span.call_count, 1)
        cost_calls = [call for call in self.span.set_attribute.call_args_list
                      if call.args[0] == "mlflow.llm.cost"]
        self.assertEqual(len(cost_calls), 1)
        link = json.loads((self.campaign / "traces" / "decision-000001.json").read_text())
        self.assertEqual(link["trace_id"], "trace-1")

    def test_span_exit_failure_has_durable_identity_before_any_cost_bearing_export(self):
        def verify_receipt_before_cost(key, value):
            if key == "mlflow.llm.cost":
                row, = self.outbox_rows()
                self.assertEqual(json.loads(row["receipt"])["trace_id"], "trace-1")
                self.assertIsNotNone(row["trace_json"])
        self.span.set_attribute.side_effect = verify_receipt_before_cost
        self.mlflow.start_span.return_value.__exit__.side_effect = RuntimeError("export interrupted")
        with self.assertRaisesRegex(RuntimeError, "export interrupted"):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        self.mlflow.start_span.return_value.__exit__.side_effect = None
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_delivered_receipt_publication_failure_retries_without_new_span(self):
        with patch("ima.mlflow_tracking._write_json_atomic", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        self.assertEqual(self.outbox_rows()[0]["delivered"], 0)
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertTrue((self.campaign / "traces" / "decision-000001.json").exists())
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_repeated_failures_increment_attempts_and_remain_retryable(self):
        self.mlflow.set_experiment.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        report = drain_trace_outbox(self.campaign, self.config)
        self.assertEqual((report["delivered"], report["pending"]), (0, 1))
        self.assertEqual(report["errors"][0]["number"], 1)
        self.assertEqual(self.outbox_rows()[0]["attempts"], 2)

    def test_drain_limit_and_one_failure_do_not_discard_other_summaries(self):
        self.mlflow.set_experiment.side_effect = RuntimeError("offline")
        for number in (1, 2):
            with self.assertRaises(RuntimeError):
                log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=number)
        self.mlflow.set_experiment.side_effect = None
        self.assertEqual(drain_trace_outbox(self.campaign, self.config, limit=1)["pending"], 1)
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertEqual(self.mlflow.start_span.call_count, 2)

    def test_conflicting_canonical_summary_fails_closed(self):
        log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        with self.assertRaisesRegex(ValueError, "Conflicting replay"):
            log_snapshot(self.campaign, self.payload(0.5), self.config, role="decision", number=1)
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_restart_sequence_includes_failed_entries_and_legacy_published_links(self):
        self.assertEqual(next_trace_number(self.campaign, "execution"), 1)
        self.mlflow.set_experiment.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            log_snapshot(self.campaign, self.payload(), self.config, role="execution", number=4)
        self.assertEqual(next_trace_number(self.campaign, "execution"), 5)
        self.assertEqual(next_trace_number(self.campaign, "decision"), 1)
        directory = self.campaign / "traces"
        directory.mkdir(exist_ok=True)
        (directory / "execution-000010.json").write_text("{}")
        self.assertEqual(next_trace_number(self.campaign, "execution"), 11)

    def test_pre_outbox_published_receipt_is_adopted_without_recharging_cost(self):
        directory = self.campaign / "traces"
        directory.mkdir(parents=True)
        receipt = {"trace_id": "historic", "experiment_id": "experiment-1",
                   "role": "decision", "number": 1, "total_cost_usd": 0.02}
        (directory / "decision-000001.json").write_text(json.dumps(receipt))
        self.assertEqual(log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1), receipt)
        self.assertEqual(self.outbox_rows()[0]["delivered"], 1)
        self.mlflow.start_span.assert_not_called()

    def test_changed_destination_cannot_replay_pending_trace(self):
        self.mlflow.set_experiment.side_effect = RuntimeError("offline")
        with self.assertRaises(RuntimeError):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        changed = SimpleNamespace(**vars(self.config))
        changed.mlflow_tracking_uri = "http://other.invalid"
        report = drain_trace_outbox(self.campaign, changed)
        self.assertEqual(report["pending"], 1)
        self.assertIn("destination changed", report["errors"][0]["error"])
        self.assertEqual(self.mlflow.set_experiment.call_count, 1)

    def test_planner_input_is_exact_and_output_has_distinct_identity_and_watermark(self):
        evidence = {"decision_id": "decision-1", "evidence_id": "input-1",
                    "terminal_watermark": 2, "actual_planner_context": {"remaining": 9}}
        output = evidence_snapshot([terminal("new", 1.0)])
        before = copy.deepcopy((evidence, output))
        link = log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1,
                            planner_evidence=evidence, output_snapshot=output)
        self.span.set_inputs.assert_called_once_with(evidence)
        emitted = self.span.set_outputs.call_args.args[0]
        self.assertEqual(emitted["evidence_id"], "input-1")
        self.assertEqual(emitted["input_terminal_watermark"], 2)
        self.assertEqual(emitted["output_terminal_watermark"], 1)
        self.assertEqual(link["output_evidence_id"], output["evidence_id"])
        self.assertEqual((evidence, output), before)

    def test_legacy_merged_caller_recovers_original_input_from_durable_evidence(self):
        evidence = {"decision_id": "decision-1", "evidence_id": "original",
                    "terminal_watermark": 0, "queue": {"remaining": 3}}
        directory = self.campaign / "evidence"
        directory.mkdir(parents=True)
        (directory / "decision-1.json").write_text(json.dumps(evidence))
        fresh = evidence_snapshot([terminal("new", 1.0)])
        log_snapshot(self.campaign, self.payload() | fresh, self.config, role="decision", number=1)
        self.span.set_inputs.assert_called_once_with(evidence)
        emitted = self.span.set_outputs.call_args.args[0]
        self.assertEqual(emitted["evidence_id"], "original")
        self.assertEqual(emitted["output_evidence_id"], fresh["evidence_id"])

    def test_wrong_decision_input_rejected_before_remote_logging(self):
        with self.assertRaisesRegex(ValueError, "different decision"):
            log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1,
                         planner_evidence={"decision_id": "other", "evidence_id": "input"})
        self.mlflow.start_span.assert_not_called()

    def test_snapshot_identity_ignores_clock_but_tracks_new_results(self):
        with patch("ima.research_telemetry.utc_now", side_effect=["time-1", "time-2", "time-3"]):
            first = evidence_snapshot([terminal("one", 2.0)])
            second = evidence_snapshot([terminal("one", 2.0)])
            third = evidence_snapshot([terminal("one", 2.0), terminal("two", 1.0)])
        self.assertEqual(first["evidence_id"], second["evidence_id"])
        self.assertNotEqual(first["evidence_id"], third["evidence_id"])
        link = log_snapshot(self.campaign, first, self.config, role="execution", number=1)
        self.assertEqual(link, log_snapshot(self.campaign, second, self.config, role="execution", number=1))
        self.assertEqual(self.mlflow.start_span.call_count, 1)

    def test_decision_snapshot_counts_terminal_results_without_fabricating_active_counts(self):
        fresh = evidence_snapshot([terminal("good", 2.0), terminal("bad", None, status="failed"),
                                   terminal("pruned", None, status="pruned")])
        log_snapshot(self.campaign, self.payload() | fresh, self.config, role="decision", number=1)
        preview = self.mlflow.update_current_trace.call_args.kwargs["response_preview"]
        self.assertIn("completed=1", preview)
        self.assertIn("failed=1", preview)
        self.assertIn("pruned=1", preview)
        self.assertIn("terminal=3", preview)
        self.assertNotIn("running=0", preview)

    def test_overview_includes_every_supplied_ledger_count(self):
        counts = {"completed": 5, "running": 0, "reserved": 1, "failed": 2,
                  "pending_tracking": 3, "pending_tells": 4}
        payload = evidence_snapshot([terminal("good", 2.0)], counts=counts)
        log_snapshot(self.campaign, payload, self.config, role="execution", number=1)
        preview = self.mlflow.update_current_trace.call_args.kwargs["response_preview"]
        for key, value in counts.items():
            self.assertIn(f"{key}={value}", preview)
        self.assertNotIn("budget 0", preview)

    def test_missing_counts_are_unknown_not_a_false_zero(self):
        log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1)
        self.assertIn("counts unavailable", self.mlflow.update_current_trace.call_args.kwargs["response_preview"])

    def test_disabled_tracking_does_not_enqueue_or_import_remote_client(self):
        self.config.mlflow_tracking_uri = None
        self.assertIsNone(log_snapshot(self.campaign, self.payload(), self.config, role="decision", number=1))
        self.assertEqual(drain_trace_outbox(self.campaign, self.config)["pending"], 0)
        self.assertFalse((self.campaign / "trace-outbox.sqlite").exists())
        self.mlflow.start_span.assert_not_called()


class LocalMLflowReadbackTests(unittest.TestCase):
    def test_blocked_final_execution_backend_and_durable_status_are_error(self):
        import mlflow
        from mlflow.entities import TraceState, SpanStatusCode
        from ima.research_controller import _tracking_names
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6", mlflow_tracking_uri=f"sqlite:///{root / 'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            client.create_experiment(_tracking_names(config.research_policy)[0], artifact_location=(root / "artifacts").as_uri())
            campaign = root / "campaign"
            try:
                for number, status in enumerate(("blocked_tracking", "blocked_failures", "blocked_paper"), 1):
                    with self.subTest(status=status):
                        payload = {"status": status, "evidence_id": f"final-{number}",
                                   "counts": {"completed": 1, "failed": 1}, "active_trials": []}
                        receipt = log_snapshot(campaign, payload, config, role="execution", number=number)
                        trace = client.get_trace(receipt["trace_id"])
                        span = trace.data.spans[0]
                        self.assertEqual(trace.info.state, TraceState.ERROR)
                        self.assertEqual(span.status.status_code, SpanStatusCode.ERROR)
                        self.assertEqual(span.status.description, status)
                        self.assertEqual(trace.info.tags["ima.operation_status"], status)
                        self.assertTrue(span.outputs["summary"]["operation"]["failed"])
                        self.assertIn(f"operation={status}", trace.info.response_preview)
                        self.assertIsNone(receipt["total_cost_usd"])
                        self.assertNotIn("mlflow.trace.cost", trace.info.trace_metadata)
                        self.assertNotIn("mlflow.llm.cost", span.attributes)
                        with sqlite3.connect(campaign / "trace-outbox.sqlite") as conn:
                            delivered, body = conn.execute("SELECT delivered,trace_json FROM traces WHERE role='execution' AND number=?", (number,)).fetchone()
                        body = json.loads(body)
                        self.assertEqual(delivered, 1)
                        self.assertEqual(body["info"]["state"], "ERROR")
                        self.assertEqual(body["data"]["spans"][0]["status"]["code"], "STATUS_CODE_ERROR")
                        with patch.object(mlflow, "start_span", side_effect=AssertionError("must reuse delivered trace")):
                            self.assertEqual(log_snapshot(campaign, payload, config, role="execution", number=number), receipt)
                self.assertEqual(drain_trace_outbox(campaign, config), {"delivered": 0, "pending": 0, "errors": []})
            finally:
                mlflow.set_tracking_uri(previous_uri)

    def test_serializer_uses_shared_blocked_status_when_live_span_is_unset(self):
        import mlflow
        from ima.research_controller import _tracking_names
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = f"sqlite:///{root / 'tracking.sqlite'}"
            name = _tracking_names("expansion_v6")[0]
            client = mlflow.MlflowClient(tracking_uri=tracking_uri)
            experiment_id = client.create_experiment(name, artifact_location=(root / "artifacts").as_uri())
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(name)
                for status in ("blocked_tracking", "blocked_failures", "blocked_paper"):
                    with self.subTest(status=status), mlflow.start_span(name=f"serializer-{status}") as span:
                        body = _serialize_trace(span, experiment_id, {"ima.operation_status": status},
                                                status, status, root / "campaign", {})
                        self.assertEqual(body["info"]["state"], "ERROR")
                        self.assertEqual(body["data"]["spans"][0]["status"]["code"], "STATUS_CODE_ERROR")
            finally:
                mlflow.set_tracking_uri(previous_uri)

    def test_failed_decision_backend_and_serialized_restore_remain_error(self):
        import mlflow
        from mlflow.entities import TraceState, SpanStatusCode
        from ima.research_controller import _tracking_names
        from tests.test_research_summary import failure_snapshot
        previous_uri = mlflow.get_tracking_uri()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6", mlflow_tracking_uri=f"sqlite:///{root / 'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            client.create_experiment(_tracking_names(config.research_policy)[0], artifact_location=(root / "artifacts").as_uri())
            campaign = root / "campaign"
            payload = failure_snapshot()
            try:
                with patch("ima.research_telemetry._verify_or_restore_trace", side_effect=RuntimeError("offline readback")):
                    with self.assertRaisesRegex(RuntimeError, "offline readback"):
                        log_snapshot(campaign, payload, config, role="decision", number=17)
                with sqlite3.connect(campaign / "trace-outbox.sqlite") as conn:
                    receipt, body = conn.execute("SELECT receipt,trace_json FROM traces").fetchone()
                receipt, body = json.loads(receipt), json.loads(body)
                self.assertEqual(body["info"]["state"], "ERROR")
                self.assertEqual(body["data"]["spans"][0]["status"]["code"], "STATUS_CODE_ERROR")
                self.assertNotIn("mlflow.trace.cost", body["info"]["trace_metadata"])
                original = client.get_trace(receipt["trace_id"])
                self.assertEqual(original.info.state, TraceState.ERROR)
                client.delete_traces(experiment_id=receipt["experiment_id"], trace_ids=[receipt["trace_id"]])
                with patch.object(mlflow, "start_span", side_effect=AssertionError("must reuse trace")):
                    self.assertEqual(drain_trace_outbox(campaign, config), {"delivered": 1, "pending": 0, "errors": []})
                trace = client.get_trace(receipt["trace_id"])
                self.assertEqual(trace.info.state, TraceState.ERROR)
                self.assertEqual(trace.data.spans[0].status.status_code, SpanStatusCode.ERROR)
                self.assertIn("DNS", trace.data.spans[0].status.description)
                self.assertEqual(trace.info.tags["ima.operation_status"], "failed")
                self.assertEqual(trace.data.spans[0].outputs["planner_usage"], {})
                self.assertEqual(trace.data.spans[0].outputs["summary"]["billing"]["known_subtotal_usd"], 0.477406735)
                self.assertEqual(trace.info.tags["ima.evidence_id"], "input-17")
                self.assertEqual(trace.info.tags["ima.output_evidence_id"], "output-17")
                self.assertEqual(drain_trace_outbox(campaign, config)["delivered"], 0)
            finally:
                mlflow.set_tracking_uri(previous_uri)

    def test_required_tracing_preserves_disabled_and_sampling_policies(self):
        import mlflow
        from ima.research_telemetry import initialize_required_tracing
        config = SimpleNamespace(research_policy="expansion_v6", mlflow_tracking_uri="sqlite:///unused.sqlite")
        from mlflow.tracing.provider import is_tracing_enabled
        previously_enabled = is_tracing_enabled()
        try:
            with patch("ima.mlflow_tracking._mlflow") as backend:
                with mlflow.tracing.context(enabled=False):
                    with self.assertRaisesRegex(RuntimeError, "explicitly disabled"):
                        initialize_required_tracing(config)
                with patch.dict("os.environ", {"MLFLOW_TRACE_SAMPLING_RATIO": "0.5"}):
                    with self.assertRaisesRegex(RuntimeError, "sampling policy was not changed"):
                        initialize_required_tracing(config)
                    self.assertEqual("0.5", __import__("os").environ["MLFLOW_TRACE_SAMPLING_RATIO"])
                mlflow.tracing.disable()
                with self.assertRaisesRegex(RuntimeError, "provider is disabled"):
                    initialize_required_tracing(config)
                self.assertFalse(is_tracing_enabled())
                backend.assert_not_called()
        finally:
            if previously_enabled:
                mlflow.tracing.enable()

    def test_actual_model_disable_overlap_and_locked_five_trace_readback(self):
        import mlflow
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from mlflow.entities import NoOpSpan
        from mlflow.tracing.provider import trace_disabled, provider, is_tracing_enabled
        from ima.research_telemetry import initialize_required_tracing, tracking_operation
        previous_uri = mlflow.get_tracking_uri()
        previously_enabled = is_tracing_enabled()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6", mlflow_tracking_uri=f"sqlite:///{root/'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            client.create_experiment("ima-agentic-v6-research", artifact_location=(root/"artifacts").as_uri())
            entered, release, trace_started = threading.Event(), threading.Event(), threading.Event()

            @trace_disabled
            def held_model_operation():
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("Test model operation was not released")

            def locked_model_operation():
                with tracking_operation():
                    held_model_operation()

            def emit(number):
                trace_started.set()
                return log_snapshot(root/"campaign", {"decision_id":f"D{number:06d}",
                                    "evidence_id":f"evidence-{number}",
                                    "planner_usage":{"cost_status":"fixture"}}, config,
                                    role="decision", number=number)
            try:
                initialized = initialize_required_tracing(config)
                initial_provider = provider.get()
                self.assertEqual(initialized, initialize_required_tracing(config))
                self.assertIs(initial_provider, provider.get())
                with ThreadPoolExecutor(max_workers=2) as workers:
                    # Reproduce the SDK's real global swap without the application lock.
                    model = workers.submit(held_model_operation)
                    try:
                        self.assertTrue(entered.wait(5))
                        with mlflow.start_span(name="disabled-overlap-proof") as span:
                            self.assertIsInstance(span, NoOpSpan)
                    finally:
                        release.set()
                    model.result(timeout=10)
                    entered.clear()
                    release.clear()
                    model = workers.submit(locked_model_operation)
                    try:
                        self.assertTrue(entered.wait(5))
                        trace = workers.submit(emit, 1)
                        self.assertTrue(trace_started.wait(5))
                        self.assertFalse(trace.done())
                    finally:
                        release.set()
                    model.result(timeout=10)
                    receipts = [trace.result(timeout=20)]
                    receipts.extend(emit(number) for number in range(2, 6))
                self.assertEqual({"delivered":0,"pending":0,"errors":[]},
                                 drain_trace_outbox(root/"campaign",config))
                for number, receipt in enumerate(receipts, 1):
                    actual = client.get_trace(receipt["trace_id"])
                    self.assertEqual(f"evidence-{number}", actual.info.tags["ima.evidence_id"])
                    self.assertEqual(1, len(actual.data.spans))
                with sqlite3.connect(root/"campaign"/"trace-outbox.sqlite") as conn:
                    self.assertEqual((5,5,0), conn.execute(
                        "SELECT count(*),sum(delivered),sum(last_error IS NOT NULL) FROM traces").fetchone())
            finally:
                release.set()
                mlflow.set_tracking_uri(previous_uri)
                if previously_enabled:
                    mlflow.tracing.enable()
                else:
                    mlflow.tracing.disable()

    def test_disabled_actual_tracing_context_preserves_pending_then_readback_recovers(self):
        import mlflow
        from mlflow.tracing.provider import is_tracing_enabled
        previous_uri = mlflow.get_tracking_uri()
        previously_enabled = is_tracing_enabled()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6",mlflow_tracking_uri=f"sqlite:///{root/'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            client.create_experiment("ima-agentic-v6-research",artifact_location=(root/"artifacts").as_uri())
            campaign = root/"private-noop-recovery"
            payload = {"decision_id":"D000001","evidence_id":"exact-evidence","planner_usage":{"cost_status":"fixture"}}
            try:
                with mlflow.tracing.context(enabled=False):
                    with self.assertRaisesRegex(RuntimeError,"NoOpSpan; no recording trace"):
                        log_snapshot(campaign,payload,config,role="decision",number=1)
                with sqlite3.connect(campaign/"trace-outbox.sqlite") as conn:
                    delivered,receipt,body = conn.execute("SELECT delivered,receipt,trace_json FROM traces").fetchone()
                self.assertEqual((0,None,None),(delivered,receipt,body))
                report = drain_trace_outbox(campaign,config)
                self.assertEqual({"delivered":1,"pending":0,"errors":[]},report)
                receipt = json.loads((campaign/"traces"/"decision-000001.json").read_text())
                trace = client.get_trace(receipt["trace_id"])
                self.assertIsNotNone(trace)
                self.assertEqual("exact-evidence",trace.info.tags["ima.evidence_id"])
                self.assertEqual(1,len(trace.data.spans))
                self.assertEqual(0,drain_trace_outbox(campaign,config)["delivered"])
            finally:
                mlflow.set_tracking_uri(previous_uri)
                if not previously_enabled:
                    mlflow.tracing.disable()
                else:
                    mlflow.tracing.enable()

    def test_actual_v6_package_logs_lineage_manifest_reports_and_long_recipe_descriptor(self):
        import hashlib
        import mlflow
        from ima.mlflow_tracking import MLflowConfig, log_research_package_version
        from ima.research_model_package import ResearchModelPackage
        from ima.research_specs import PipelineRecipe
        from tests.test_mlflow_tracking import DummyProbabilityModel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MLflowConfig(enabled=True, register_models=False,
                                  experiment_name="isolated-v6-package-canary",
                                  tracking_uri=f"sqlite:///{root / 'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.tracking_uri)
            experiment_id = client.create_experiment(config.experiment_name,
                                                     artifact_location=(root / "artifacts").as_uri())
            actual_recipe = PipelineRecipe(schema_version=3,
                                           extra_numeric_features=tuple(f"measurement_{i}" for i in range(1000)))
            package = ResearchModelPackage(DummyProbabilityModel(), actual_recipe,
                                          protocol_id="protocol-1", code_revision="revision-1")
            trial = root / "trial"
            package_dir = package.save(trial / "package")
            dataset = root / "data" / "dataset.csv"
            dataset.parent.mkdir()
            dataset.write_text("race_id,horse_no\nr1,1\nr1,2\n")
            (dataset.parent / "manifest.json").write_text(json.dumps({"dataset_id": "dataset-v6"}))
            reports = ("preparation-canary.json", "formula-report.json", "graph-fit-report.json")
            for name in reports:
                (trial / name).write_text(json.dumps({"fixture_report": name}))
            lineage = {"dataset_hash": "a" * 64, "dataset_id": "dataset-v6",
                       "evaluation_population_id": "score-runners-1", "availability_policy": "point-in-time",
                       "probability_basis": "fundamental", "target_unit": "probability",
                       "predictor_catalog_id": "catalog-1", "protocol_id": "protocol-1"}
            actual_result = {"status": "completed", "recipe_hash": actual_recipe.recipe_hash(),
                             "target_kind": "win_probability", "objective_name": "fundamental_log_loss",
                             "objective_value": 2.0, "duration_seconds": 0.1, "lineage": lineage,
                             "metrics": {"metric_contract_version": 6}, "artifacts": {}}
            with patch("ima.mlflow_tracking._mlflow", return_value=mlflow):
                link = log_research_package_version(package_dir, config, attempt_id="v6-canary-1",
                                                    result=actual_result, dataset_path=dataset)
                self.assertEqual(link, log_research_package_version(package_dir, config, attempt_id="v6-canary-1",
                                                                    result=actual_result, dataset_path=dataset))
            run = client.get_run(link["run_id"])
            for name, value in lineage.items():
                self.assertEqual(run.data.tags[f"ima.{name}"], value)
                if name not in {"dataset_hash", "protocol_id"}:
                    self.assertEqual(run.data.params[name], value)
            encoded = json.dumps(list(actual_recipe.extra_numeric_features), sort_keys=True, separators=(",", ":"))
            descriptor = run.data.params["recipe.extra_numeric_features"]
            self.assertIn(hashlib.sha256(encoded.encode()).hexdigest(), descriptor)
            self.assertIn("see recipe.json", descriptor)
            self.assertLess(len(descriptor), 5000)
            self.assertTrue(set(reports) <= {Path(item.path).name for item in client.list_artifacts(link["run_id"], "analysis")})
            artifact = client.download_artifacts(link["run_id"], "dataset/manifest.json", str(root / "download"))
            self.assertEqual(json.loads(Path(artifact).read_text())["dataset_id"], "dataset-v6")
            self.assertEqual(len(client.search_runs([experiment_id])), 1)

    def test_actual_fixture_trace_is_chain_with_exact_input_and_no_llm_statistics(self):
        import mlflow
        from ima.research_controller import _tracking_names

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6",
                                     mlflow_tracking_uri=f"sqlite:///{root / 'tracking.sqlite'}")
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            experiment_id = client.create_experiment(_tracking_names(config.research_policy)[0],
                                                     artifact_location=(root / "artifacts").as_uri())
            evidence = evidence_snapshot([], decision_id="D000001", queue={"remaining": 5})
            output = evidence_snapshot([terminal("completed", 2.0)])
            payload = {"decision_id": "D000001", "evidence_id": evidence["evidence_id"],
                       "trial_budget": 5, "allocated_trials": 5,
                       "planner_usage": {"cost_status": "fixture"}}
            with patch("ima.mlflow_tracking._mlflow", return_value=mlflow):
                link = log_snapshot(root / "campaign", payload, config, role="decision", number=1,
                                    planner_evidence=evidence, output_snapshot=output)
                self.assertEqual(link, log_snapshot(root / "campaign", payload, config, role="decision", number=1,
                                                   planner_evidence=evidence, output_snapshot=output))
                trace = client.get_trace(link["trace_id"])
                self.assertEqual(trace.data.spans[0].span_type, "CHAIN")
                self.assertEqual(trace.data.spans[0].inputs, evidence)
                self.assertEqual(trace.info.tags["ima.evidence_id"], evidence["evidence_id"])
                self.assertEqual(trace.info.tags["ima.output_evidence_id"], output["evidence_id"])
                self.assertEqual(trace.info.tags["ima.planner_cost_status"], "fixture")
                self.assertNotIn("mlflow.trace.cost", trace.info.trace_metadata)
                self.assertNotIn("mlflow.trace.tokenUsage", trace.info.trace_metadata)
                self.assertNotIn("mlflow.llm.cost", trace.data.spans[0].attributes)
                self.assertNotIn("mlflow.chat.tokenUsage", trace.data.spans[0].attributes)
                self.assertEqual(len(client.search_traces([experiment_id])), 1)

    def test_serialized_trace_restores_same_id_without_in_memory_queue(self):
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SimpleNamespace(research_policy="expansion_v6",
                                     mlflow_tracking_uri=f"sqlite:///{root / 'tracking.sqlite'}")
            campaign = root / "campaign"
            from ima.research_controller import _tracking_names
            client = mlflow.MlflowClient(tracking_uri=config.mlflow_tracking_uri)
            client.create_experiment(_tracking_names(config.research_policy)[0],
                                     artifact_location=(root / "artifacts").as_uri())
            payload = {"decision_id": "D000001", "evidence_id": "input-1",
                       "planner_usage": {"total_cost_usd": 0.012, "cost_status": "reported",
                                         "input_tokens": 7, "output_tokens": 3, "total_tokens": 10}}
            with patch("ima.mlflow_tracking._mlflow", return_value=mlflow):
                with patch("ima.research_telemetry._verify_or_restore_trace", side_effect=RuntimeError("readback interrupted")):
                    with self.assertRaisesRegex(RuntimeError, "readback interrupted"):
                        log_snapshot(campaign, payload, config, role="decision", number=1)
                with sqlite3.connect(campaign / "trace-outbox.sqlite") as conn:
                    receipt, body = conn.execute("SELECT receipt,trace_json FROM traces").fetchone()
                receipt, body = json.loads(receipt), json.loads(body)
                self.assertEqual(body["info"]["trace_id"], receipt["trace_id"])
                client.delete_traces(experiment_id=receipt["experiment_id"], trace_ids=[receipt["trace_id"]])
                with patch.object(mlflow, "start_span", side_effect=AssertionError("must not create another span")):
                    report = drain_trace_outbox(campaign, config)
                self.assertEqual(report, {"delivered": 1, "pending": 0, "errors": []})
                trace = client.get_trace(receipt["trace_id"])
                self.assertEqual(trace.info.trace_id, receipt["trace_id"])
                self.assertEqual(trace.data.spans[0].attributes["mlflow.llm.cost"], {"total_cost": 0.012})
                self.assertEqual(trace.data.spans[0].attributes["mlflow.chat.tokenUsage"],
                                 {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
                self.assertEqual(json.loads(trace.info.trace_metadata["mlflow.trace.cost"]), {"total_cost": 0.012})


if __name__ == "__main__":
    unittest.main()
