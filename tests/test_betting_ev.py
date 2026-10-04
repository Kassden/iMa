from dataclasses import replace
from datetime import timedelta
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ima.betting_contracts import Ticket
from ima.betting_ev import (conditional_choice_scenarios, evaluate_ticket, joint_return_matrix,
                            scenarios_from_orders, scenarios_from_rank_distribution, ticket_probabilities)
from ima.betting_settlement import Finish
from tests.test_betting_contracts import AT, START, quote, rules


def payload():
    return {"race_id": "r1", "decision_at": AT.isoformat(), "race_start": START.isoformat(),
            "runners": ["1", "2", "3", "4"], "strengths": [.2, .3, .3, .2],
            "rules": {"version": "synthetic-v1", "source": "synthetic test fixture",
                      "currency": "HKD", "effective_from": (AT-timedelta(days=1)).isoformat(),
                      "effective_to": (AT+timedelta(days=1)).isoformat(), "paid_places": 2,
                      "qpl_places": 3, "minimum_stake": "1", "stake_increment": "1",
                      "verified": True, "scratch_policy": "refund_selected"},
            "tickets": [{"pool": "WIN", "runners": ["1"], "quote": {
                "value": "60", "convention": "dividend", "unit_stake": "10", "currency": "HKD",
                "quoted_at": (AT-timedelta(seconds=5)).isoformat(),
                "available_at": (AT-timedelta(seconds=4)).isoformat(), "source_hash": "fixture",
                "rule_version": "synthetic-v1", "status": "pre_race"}}]}


class BettingEVTests(unittest.TestCase):
    def report(self, p=.2, q=None, mode="quoted", r=None):
        return evaluate_ticket(Ticket("r1", "WIN", ("1",)), p, 10, "HKD", r or rules(),
                               at=AT, race_start=START, quote=q, quote_mode=mode)

    def test_ev_hkd10_and_no_second_takeout(self):
        report = self.report(q=quote(value=60, convention="dividend"))
        self.assertAlmostEqual(report.expected_profit, 2)
        self.assertEqual(report.fair_decimal_return, 5)
        self.assertTrue(report.executable_evidence)

    def test_missing_stale_final_and_scenario_quotes(self):
        for q in [None, quote(status="final"), quote(close_status="unknown"),
                  quote(available_at=AT+timedelta(seconds=1)),
                  quote(quoted_at=AT-timedelta(hours=1), available_at=AT-timedelta(hours=1))]:
            report = self.report(q=q)
            self.assertEqual(report.mode, "fair_price")
            self.assertIsNone(report.expected_profit)
            self.assertFalse(report.executable_evidence)
        for mode, status in [("ex_post", "final"), ("scenario", "scenario")]:
            report = self.report(q=quote(status=status), mode=mode)
            self.assertEqual(report.mode, mode)
            self.assertFalse(report.executable_evidence)
        self.assertIsNone(self.report(q=quote(), mode="fair_price").decimal_return)
        self.assertFalse(self.report(q=quote(), r=rules(verified=False)).executable_evidence)

    def test_wrong_units_currency_and_probability(self):
        with self.assertRaises(ValueError):
            self.report(q=quote(currency="USD"))
        for p in [-.01, 1.01, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.report(p=p)
        self.assertIsNone(self.report(p=0).fair_decimal_return)
        self.assertEqual(self.report(p=1).fair_decimal_return, 1)

    def test_all_pools_coherent_symmetric_field(self):
        ids = ["1", "2", "3", "4"]
        s = conditional_choice_scenarios(ids, [1]*4, depth=4)
        r = rules()
        expected = {"WIN": .25, "PLACE": .5, "QIN": 1/6, "QPL": .5,
                    "FORECAST": 1/12, "TRI": .25, "TIERCE": 1/24,
                    "FIRST4": 1, "QUARTET": 1/24}
        from ima.betting_contracts import POOL_SIZE, UNORDERED
        for pool, value in expected.items():
            orders = itertools.combinations(ids, POOL_SIZE[pool]) if pool in UNORDERED else itertools.permutations(ids, POOL_SIZE[pool])
            tickets = [Ticket("r1", pool, order) for order in orders]
            p = ticket_probabilities(tickets, s, r)
            for probability in p:
                self.assertAlmostEqual(probability, value)
            if pool not in {"PLACE", "QPL"}:
                self.assertAlmostEqual(sum(p), 1)
            elif pool == "PLACE":
                self.assertAlmostEqual(sum(p), r.paid_places)
            else:
                self.assertAlmostEqual(sum(p), 3)

    def test_joint_correlation_and_all_lose_scenarios(self):
        s = conditional_choice_scenarios(["1", "2", "3", "4"], [1]*4, depth=3)
        tickets = [Ticket("r1", "WIN", ("1",)), Ticket("r1", "PLACE", ("1",)),
                   Ticket("r1", "QIN", ("1", "2"))]
        matrix = joint_return_matrix(tickets, s, rules(), [6, 3, 8])
        self.assertTrue(np.any(np.all(matrix == 0, axis=1)))
        self.assertTrue(np.any(np.all(matrix[:, :2] > 0, axis=1)))
        self.assertFalse(np.any((matrix[:, 0] > 0) & (matrix[:, 1] == 0)))

    def test_extreme_strengths_and_zero_handling(self):
        s = conditional_choice_scenarios(["1", "2", "3"], [1e300, 1e290, 1e289], depth=3)
        self.assertAlmostEqual(sum(s.probabilities), 1)
        s = conditional_choice_scenarios(["1", "2"], [1, 0], depth=1)
        self.assertEqual(ticket_probabilities([Ticket("r1", "WIN", ("1",))], s, rules()), (1,))
        self.assertEqual(ticket_probabilities([Ticket("r1", "WIN", ("2",))], s, rules()), (0,))
        with self.assertRaises(ValueError):
            conditional_choice_scenarios(["1", "2"], [1, 0], depth=2)
        with self.assertRaises(ValueError):
            conditional_choice_scenarios(["1", "2", "3"], [1]*3, depth=3, max_orders=5)

    def test_coverage_and_prefix_fail_closed(self):
        with self.assertRaises(ValueError):
            scenarios_from_orders([("1",)], [.9], assumptions="test", source_id="test")
        s = scenarios_from_orders([("1",)], [1], assumptions="test", source_id="test")
        with self.assertRaises(ValueError):
            ticket_probabilities([Ticket("r1", "WIN", ("2",))], s, rules())
        with self.assertRaises(ValueError):
            ticket_probabilities([Ticket("r1", "PLACE", ("1",))], s, rules())
        ties = replace(s, finishes=(Finish((("1", "2"),)),))
        with self.assertRaises(ValueError):
            joint_return_matrix([Ticket("r1", "WIN", ("1",))], ties, rules(), [6])

    def test_rank_distribution_integration(self):
        from ima.rank_distributions import plackett_luce_distribution
        d = plackett_luce_distribution([.5, .3, .2], ["1", "2", "3"])
        s = scenarios_from_rank_distribution(d, source_id="model-package:test")
        p = ticket_probabilities([Ticket("r1", "QIN", ("1", "2"))], s, rules())[0]
        self.assertAlmostEqual(p, d.quinella_probability(("1", "2")))
        self.assertIn("not identified", s.assumptions)

    def test_real_distribution_package_reload_and_runner_protocol(self):
        import pandas as pd
        from ima.betting_ev import scenarios_from_package_distribution
        from ima.performance_distributions import SharedResidualPerformanceModel
        from ima.research_model_package import ResearchModelPackage, load_research_package
        from ima.research_specs import PipelineRecipe
        train = pd.DataFrame({"x": [0,1,2,3], "speed": [16,17,18,19], "date": ["2020-01-01"]*4})
        calibration = pd.DataFrame({"x": [0,1,2,3], "speed": [16.2,17.3,17.9,19.1], "date": ["2020-02-01"]*4})
        model = SharedResidualPerformanceModel(("x",)).fit(train, calibration, label_column="speed")
        recipe = PipelineRecipe(schema_version=3, model={"kind":"ridge_regressor"},
            target={"kind":"adjusted_finish_time_or_speed"},
            calibration={"kind":"none"}, blend={"kind":"none"},
            performance_distribution={"kind":"shared_residual"})
        package = ResearchModelPackage(model, recipe, protocol_id="distribution-protocol-test", code_revision="fixture")
        frame = pd.DataFrame({"x": [0,1,2], "race_id": ["r1"]*3, "horse_id": ["h1","h2","h3"],
                              "horse_no": [1,2,3], "field_size": [3]*3, "date": ["2020-03-01"]*3})
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_research_package(package.save(Path(tmp)/"package"))
            direct = package.predict_distribution(frame)
            restored = loaded.predict_distribution(frame)
            np.testing.assert_allclose(direct.location, restored.location, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(direct.scale, restored.scale, rtol=1e-10, atol=1e-12)
            s = scenarios_from_package_distribution(loaded, frame, min_draws=4096, max_draws=8192, tolerance=.04)
            self.assertEqual(s.metadata["protocol_id"], "distribution-protocol-test")
            self.assertEqual(s.metadata["package_id"], loaded.manifest().package_id)
            self.assertAlmostEqual(sum(s.probabilities), 1)
            self.assertEqual(s.metadata["runner_ids"], ("1","2","3"))
            with self.assertRaises(ValueError):
                scenarios_from_package_distribution(loaded, frame.iloc[:2])
            with self.assertRaises(ValueError):
                scenarios_from_package_distribution(loaded, frame.assign(date="2020-01-01"))

    def test_deadheat_scenario_dividends_and_refund_matrix(self):
        from ima.betting_ev import JointScenarios
        t = Ticket("r1", "WIN", ("1",))
        s = JointScenarios((Finish((("1", "2"), ("3",))),
                            Finish((("2",), ("3",)), scratches=("1",)),
                            Finish((("2",), ("1",), ("3",)))), (.2, .1, .7), "synthetic tie/refund", "fixture")
        q = quote(t, status="scenario", value=3)
        gross = joint_return_matrix([t], s, rules(), [6], scenario_dividends=[{t:q},{},{}])
        np.testing.assert_array_equal(gross, [[3],[1],[0]])
        with self.assertRaises(ValueError):
            joint_return_matrix([t], s, rules(), [6], scenario_dividends=[{}, {}, {}])
        with self.assertRaises(ValueError):
            joint_return_matrix([t], s, rules(), [6], scenario_dividends=[{t:quote(t,status="final")},{},{}])

    def test_unrelated_scratch_does_not_excuse_short_prefix(self):
        from ima.betting_ev import JointScenarios
        s = JointScenarios((Finish((("1",),), scratches=("4",)),), (1,), "fixture", "fixture")
        with self.assertRaises(ValueError):
            ticket_probabilities([Ticket("r1", "PLACE", ("1",))], s, rules())

    def test_cli_reproducible_quotes_and_missing_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.json"
            path.write_text(json.dumps(payload()))
            command = [sys.executable, "-m", "scripts.evaluate_betting_ev", "--input", str(path),
                       "--bankroll", "1000", "--portfolio"]
            first = subprocess.run(command, capture_output=True, text=True, check=True)
            second = subprocess.run(command, capture_output=True, text=True, check=True)
            self.assertEqual(first.stdout, second.stdout)
            output = json.loads(first.stdout)
            self.assertAlmostEqual(output["reports"][0]["expected_profit"], 2)
            self.assertTrue(output["sizing"]["success"])
            data = payload()
            del data["tickets"][0]["quote"]
            path.write_text(json.dumps(data))
            output = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            self.assertIsNone(output["sizing"])
            self.assertEqual(output["reports"][0]["mode"], "fair_price")

    def test_cli_all_requested_pools(self):
        from scripts.evaluate_betting_ev import evaluate_payload
        from ima.betting_contracts import POOL_SIZE
        data = payload()
        data["tickets"] = [{"pool": pool, "runners": [str(i+1) for i in range(size)]}
                           for pool, size in POOL_SIZE.items()]
        output = evaluate_payload(data)
        self.assertEqual(len(output["reports"]), 9)
        self.assertTrue(all(r["mode"] == "fair_price" for r in output["reports"]))
