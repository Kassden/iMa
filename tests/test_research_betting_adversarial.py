"""Independent adversarial checks of frozen, offline paper-action inputs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pydantic import ValidationError

from ima.research_betting import evaluate_paper_research, validate_paper_research
from ima.research_expansion import (
    DecisionStore, PlannerDecision, _preflight_betting_requests, _publish_paper_report,
    _paper_worker, _recover_paper_report,
)


class PaperAdversarialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.campaign = Path(temporary.name).resolve()
        self.frame = pd.DataFrame({
            "fold_id": ["f1"] * 4, "race_id": ["r1"] * 4,
            "horse_no": ["1", "2", "3", "4"], "date": ["2024-01-01"] * 4,
            "target_win": [1, 0, 0, 0], "model_probability": [.4, .3, .2, .1],
            "selected_probability": [.1, .1, .1, .7],
            "market_probability": [.1, .1, .1, .7],
        })
        self.rows = [self.make_attempt("a1"), self.make_attempt("a2")]
        self.request = {"request_id": "paper-adversarial", "evidence_id": "e1",
                        "attempt_ids": ["a1"], "pools": ["WIN"],
                        "max_races": 1, "simulations": 1000}

    def make_attempt(self, attempt):
        directory = self.campaign / "trials" / attempt
        directory.mkdir(parents=True, exist_ok=True)
        predictions = directory / "predictions.csv"
        self.frame.to_csv(predictions, index=False)
        protocol = directory / "protocol.json"
        protocol.write_bytes(b'{"protocol_id":"p1"}')
        keys = ["fold_id", "race_id", "horse_no", "date"]
        population = self.frame.sort_values(keys, kind="stable")[keys + ["target_win"]]
        population_hash = hashlib.sha256(population.to_json(
            orient="records", date_format="iso").encode()).hexdigest()
        return {"attempt_id": attempt, "status": "completed", "result": {
            "attempt_id": attempt, "status": "completed", "target_kind": "win_probability",
            "recipe_hash": "recipe-" + attempt,
            "artifacts": {"predictions": str(predictions), "protocol": str(protocol)},
            "lineage": {"dataset_id": "dataset1", "dataset_hash": "dataset-sha",
                        "protocol_id": "p1", "protocol_hash": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                        "evaluation_population_hash": population_hash,
                        "evaluation_population_id": "evaluated-" + population_hash,
                        "availability_policy": "strict_before_meeting",
                        "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        "code_revision": "fixture", "environment_hash": "fixture"}}}

    def evaluate(self, **changes):
        return evaluate_paper_research(self.request | changes, campaign_dir=self.campaign,
                                       terminal_results=self.rows)

    def preflight(self, **changes):
        return validate_paper_research(self.request | changes, campaign_dir=self.campaign,
                                       terminal_results=self.rows)

    def rewrite_predictions(self, frame, attempt="a1", *, freeze=False):
        row = next(row for row in self.rows if row["attempt_id"] == attempt)
        path = Path(row["result"]["artifacts"]["predictions"])
        frame.to_csv(path, index=False)
        if freeze:
            row["result"]["lineage"]["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_probability_only_edit_cannot_rebind_frozen_prediction_sha(self):
        frame = self.frame.copy()
        frame["model_probability"] = [.1, .2, .3, .4]
        self.rewrite_predictions(frame)
        for loader in (self.preflight, self.evaluate):
            with self.subTest(loader=loader.__name__), self.assertRaises(ValueError):
                loader()

    def test_all_declared_comparison_axes_must_match(self):
        original = copy.deepcopy(self.rows)
        for field in ("dataset_id", "dataset_hash", "protocol_id", "protocol_hash",
                      "evaluation_population_id", "evaluation_population_hash",
                      "availability_policy", "declared_evaluation_population_id", "comparison_contract_id"):
            with self.subTest(field=field):
                self.rows = copy.deepcopy(original)
                self.rows[1]["result"]["lineage"][field] = "different"
                with self.assertRaises(ValueError):
                    self.preflight(attempt_ids=["a1", "a2"])

    def test_place_marginals_and_other_targets_never_become_joint_orders(self):
        for target in ("placing_top_k", "ranking", "finish_time", "speed"):
            with self.subTest(target=target):
                self.rows[0]["result"]["target_kind"] = target
                with self.assertRaises(ValueError):
                    self.evaluate(pools=["QIN"])

    def test_shifted_runner_fold_date_or_label_rejected(self):
        for column, value in (("horse_no", "99"), ("fold_id", "f2"),
                              ("date", "2024-01-02"), ("target_win", 0)):
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame.loc[0, column] = value
                self.rewrite_predictions(frame, freeze=True)
                with self.assertRaises(ValueError):
                    self.preflight()

    def test_row_order_not_runner_alignment_changes_ensemble(self):
        self.rewrite_predictions(self.frame.iloc[::-1], "a2", freeze=True)
        report = self.evaluate(attempt_ids=["a1", "a2"], model_weights=[1, 3])
        self.assertAlmostEqual(report["races"][0]["reports"][0]["probability"], .4)

    def test_invalid_win_vectors_and_duplicate_keys_fail_closed(self):
        for probabilities in ([.4, .3, .2, .2], [1.1, -.1, 0, 0],
                              [float("nan"), .3, .2, .1], [float("inf"), 0, 0, 0]):
            with self.subTest(probabilities=probabilities):
                frame = self.frame.copy()
                frame["model_probability"] = probabilities
                self.rewrite_predictions(frame, freeze=True)
                with self.assertRaises(ValueError):
                    self.preflight()
        self.rewrite_predictions(pd.concat([self.frame, self.frame.iloc[:1]]), freeze=True)
        with self.assertRaises(ValueError):
            self.preflight()

    def test_fundamental_never_reads_final_market_column(self):
        frame = self.frame.copy()
        frame["selected_probability"] = float("nan")
        frame["market_probability"] = float("inf")
        self.rewrite_predictions(frame, freeze=True)
        report = self.evaluate()
        ticket = report["races"][0]["reports"][0]
        self.assertEqual(ticket["ticket"]["runners"], ["1"])
        self.assertAlmostEqual(ticket["probability"], .4)
        self.assertFalse(report["ex_post_market_tainted"])
        with self.assertRaises(ValueError):
            self.evaluate(probability_basis="blended")

    def test_blended_without_quotes_stays_tainted_and_has_no_pnl(self):
        report = self.evaluate(probability_basis="blended", bankroll=1000, kelly_policy={})
        self.assertTrue(report["ex_post_market_tainted"])
        ticket = report["races"][0]["reports"][0]
        self.assertEqual(ticket["ticket"]["runners"], ["4"])
        self.assertIsNone(ticket["expected_profit"])
        self.assertIsNone(ticket["decimal_return"])
        self.assertFalse(ticket["executable_evidence"])
        self.assertIsNone(report["races"][0]["sizing"])
        self.assertIsNone(report["scenario_payout_contract"])

    def test_explicit_small_field_place_counts_remain_hypothetical(self):
        for count in (4, 5, 6, 7):
            with self.subTest(count=count):
                self.frame = pd.DataFrame({
                    "fold_id": ["f1"] * count, "race_id": ["r1"] * count,
                    "horse_no": [str(i + 1) for i in range(count)],
                    "date": ["2024-01-01"] * count, "target_win": [1] + [0] * (count - 1),
                    "model_probability": [1 / count] * count})
                self.rows = [self.make_attempt("a1")]
                places = 2 if count <= 6 else 3
                race = self.evaluate(pools=["PLACE"], paid_places=places)["races"][0]
                self.assertAlmostEqual(race["reports"][0]["probability"], places / count)
                self.assertEqual(race["rules"]["paid_places"], places)
                self.assertFalse(race["rules"]["verified"])

    def test_currencies_and_unobserved_quote_injections_rejected(self):
        for changes in ({"currency": "USD"}, {"currency": "CNY"},
                        {"quotes": {"WIN": 5}}, {"exchange_rate": 7.8},
                        {"quote_mode": "quoted"}, {"quote_mode": "ex_post"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.evaluate(**changes)

    def test_scenario_ev_is_not_realized_or_executable_evidence(self):
        report = self.evaluate(quote_mode="scenario", scenario_decimal_returns={"WIN": 4}, bankroll=1000)
        race = report["races"][0]
        ticket = race["reports"][0]
        self.assertEqual(ticket["mode"], "scenario")
        self.assertAlmostEqual(ticket["expected_profit"], 6)
        self.assertFalse(ticket["executable_evidence"])
        self.assertFalse(report["executable_evidence"])
        self.assertFalse(race["rules"]["verified"])
        self.assertEqual(report["scenario_payout_contract"]["currency"], "HKD")
        self.assertIn("no observed quote", report["scenario_payout_contract"]["source"])
        self.assertEqual(race["sizing"]["mode"], "hypothetical_scenario")
        self.assertFalse(race["sizing"]["executable_evidence"])
        self.assertIsNone(race["joint_outcome_metrics"])

    def test_missing_fundamental_column_does_not_fall_back_to_blend(self):
        self.rewrite_predictions(self.frame.drop(columns="model_probability"), freeze=True)
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_cross_attempt_artifact_and_symlink_rejected(self):
        original = self.rows[0]["result"]["artifacts"]["predictions"]
        self.rows[0]["result"]["artifacts"]["predictions"] = self.rows[1]["result"]["artifacts"]["predictions"]
        with self.assertRaises(ValueError):
            self.preflight()
        self.rows[0]["result"]["artifacts"]["predictions"] = original
        path = Path(original)
        path.unlink()
        path.symlink_to(self.rows[1]["result"]["artifacts"]["predictions"])
        with self.assertRaises(ValueError):
            self.preflight()
        for attempt in ("../a1", "/tmp/a1", "a1/../../a2"):
            with self.subTest(attempt=attempt), self.assertRaises(ValidationError):
                self.preflight(attempt_ids=[attempt])

    def test_preflight_is_load_only_and_sha_matches_exact_bytes(self):
        with patch("ima.research_betting._race_report", side_effect=AssertionError("must not simulate")):
            result = self.preflight()
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["available_races"], 1)
        self.assertEqual(result["model_ids"][0]["prediction_sha256"],
                         self.rows[0]["result"]["lineage"]["prediction_sha256"])

    def test_worker_input_race_rejected_before_publication(self):
        preflight = self.preflight()
        frame = self.frame.copy()
        frame["model_probability"] = [.1, .2, .3, .4]
        self.rewrite_predictions(frame, freeze=True)
        report = self.evaluate()
        action = {"action_id": "a" * 24, "decision_id": "d1",
                  "request": report["request"], "preflight": preflight}
        store = Mock()
        with self.assertRaisesRegex(ValueError, "changed after decision preflight"):
            _publish_paper_report(store, self.campaign, action, report)
        store.receipt.assert_not_called()
        self.assertFalse((self.campaign / "paper-actions").exists())

    def test_stale_evidence_and_request_rebinding_fail_before_receipts(self):
        with sqlite3.connect(self.campaign / "ledger.sqlite") as connection:
            connection.execute("CREATE TABLE attempts (attempt_id TEXT, status TEXT, payload_json TEXT, result_json TEXT)")
            for row in self.rows:
                connection.execute("INSERT INTO attempts VALUES (?,?,?,?)", (
                    row["attempt_id"], row["status"], "{}", json.dumps(row["result"])))
        store = DecisionStore(self.campaign / "decisions.sqlite")
        config = SimpleNamespace(campaign_dir=self.campaign)
        evidence = {"completed_trial_index": [{"attempt_id": "a1"}]}
        decision = PlannerDecision(decision_id="d1", evidence_id="e1", trial_budget=0,
                                   betting_requests=[self.request])
        store.record(decision)
        action = _preflight_betting_requests(decision, store, config, evidence)[0]
        store.receipt("d1", "betting:" + action["action_id"], action)
        store.complete("d1")
        self.assertEqual(action, _preflight_betting_requests(decision, store, config, evidence)[0])
        rebound = decision.model_copy(update={"decision_id": "d2"})
        store.record(rebound)
        with self.assertRaisesRegex(ValueError, "already bound"):
            _preflight_betting_requests(rebound, store, config, evidence)
        stale = PlannerDecision(decision_id="d3", evidence_id="e2", trial_budget=0,
                                betting_requests=[self.request | {"request_id": "stale"}])
        store.record(stale)
        with self.assertRaisesRegex(ValueError, "exact evidence"):
            _preflight_betting_requests(stale, store, config, evidence)
        self.assertEqual(1, len(store.betting_actions()))

    def test_real_worker_never_calls_provider_or_modifies_source(self):
        before = {path.relative_to(self.campaign): path.read_bytes()
                  for path in self.campaign.rglob("*") if path.is_file()}
        with patch.dict(os.environ), patch("ima.openrouter_orchestrator._post_json",
                                          side_effect=AssertionError("paper work must not call provider")) as provider:
            result = _paper_worker(self.request, self.campaign, self.rows)
        self.assertTrue(result["success"], result)
        self.assertTrue(result["report"]["paper_only"])
        provider.assert_not_called()
        after = {path.relative_to(self.campaign): path.read_bytes()
                 for path in self.campaign.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_published_report_recovery_is_idempotent_without_simulation(self):
        report = self.evaluate()
        store = DecisionStore(self.campaign / "decisions.sqlite")
        decision = PlannerDecision(decision_id="d1", evidence_id="e1", trial_budget=0,
                                   betting_requests=[self.request])
        store.record(decision)
        action = {"action_id": "c" * 24, "decision_id": "d1", "request": report["request"],
                  "preflight": self.preflight()}
        first = _publish_paper_report(store, self.campaign, action, report)
        path = Path(first["report_path"])
        original_bytes = path.read_bytes()
        reopened = DecisionStore(store.path)
        with patch("ima.research_betting.evaluate_paper_research",
                   side_effect=AssertionError("completed paper work must not rerun")) as simulate:
            self.assertEqual(first, _recover_paper_report(reopened, self.campaign, action))
        simulate.assert_not_called()
        self.assertEqual(original_bytes, path.read_bytes())
        changed = copy.deepcopy(report)
        changed["races"][0]["reports"][0]["probability"] = .99
        with self.assertRaisesRegex(ValueError, "Conflicting immutable paper report replay"):
            _publish_paper_report(reopened, self.campaign, action, changed)
        self.assertEqual(original_bytes, path.read_bytes())

    def test_report_rebinding_cannot_publish_or_complete(self):
        report = self.evaluate()
        action = {"action_id": "d" * 24, "decision_id": "d1", "request": report["request"],
                  "preflight": self.preflight()}
        for field, value in (("request_id", "other"), ("evidence_id", "other"),
                             ("paper_only", False), ("executable_evidence", True)):
            with self.subTest(field=field):
                store = Mock()
                changed = report | {field: value}
                with self.assertRaises(ValueError):
                    _publish_paper_report(store, self.campaign, action, changed)
                store.receipt.assert_not_called()
        self.assertFalse((self.campaign / "paper-actions").exists())


if __name__ == "__main__":
    unittest.main()
