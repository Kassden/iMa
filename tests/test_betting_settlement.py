from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from ima.betting_contracts import Ticket
from ima.betting_settlement import Finish, settle_ticket, ticket_wins
from tests.test_betting_contracts import AT, START, quote, rules
from tests.test_betting_ev import payload


class BettingSettlementTests(unittest.TestCase):
    def settle(self, ticket, finish, dividends=None, r=None):
        return settle_ticket(ticket, 10, finish, r or rules(), dividends or {},
                             selected_at=AT, race_start=START, currency="HKD")

    def test_final_settlement_is_exact_ex_post(self):
        t = Ticket("r1", "WIN", ("1",))
        q = quote(t, status="final", convention="dividend", value="63.5", unit_stake=10,
                  quoted_at=START+timedelta(minutes=2), available_at=START+timedelta(minutes=3))
        s = self.settle(t, Finish((("1",), ("2",))), {t: q})
        self.assertEqual(s.gross_return, Decimal("63.5"))
        self.assertEqual(s.net_profit, Decimal("53.5"))
        self.assertEqual(s.status, "won_ex_post")

    def test_deadheat_qualification_and_per_combination_dividends(self):
        f = Finish((("1",), ("2", "3"), ("4",)))
        for pair in [("1", "2"), ("1", "3")]:
            self.assertTrue(ticket_wins(Ticket("r1", "QIN", pair), f, rules()))
        self.assertFalse(ticket_wins(Ticket("r1", "QIN", ("2", "3")), f, rules()))
        self.assertTrue(ticket_wins(Ticket("r1", "FORECAST", ("1", "3")), f, rules()))
        self.assertFalse(ticket_wins(Ticket("r1", "FORECAST", ("3", "1")), f, rules()))
        self.assertTrue(ticket_wins(Ticket("r1", "PLACE", ("3",)), f, rules()))
        for runner, dividend in [("1", "30"), ("2", "45")]:
            t = Ticket("r1", "WIN", (runner,))
            q = quote(t, status="final", convention="dividend", value=dividend,
                      quoted_at=START, available_at=START)
            s = self.settle(t, Finish((("1", "2"), ("3",))), {t:q})
            self.assertEqual(s.gross_return, Decimal(dividend))

    def test_scratch_refund_unknown_policy_and_abandonment(self):
        t = Ticket("r1", "QIN", ("1", "2"))
        f = Finish((("2",), ("3",)), scratches=("1",))
        s = self.settle(t, f)
        self.assertEqual(s.gross_return, Decimal(10))
        self.assertEqual(s.net_profit, 0)
        self.assertEqual(self.settle(t, f, r=rules(scratch_policy="unverified")).status, "unsupported_refund")
        self.assertEqual(self.settle(t, Finish((), abandoned=True)).status, "refund")

    def test_missing_dividend_and_loser_do_not_fabricate(self):
        t = Ticket("r1", "WIN", ("1",))
        self.assertIsNone(self.settle(t, Finish((("1",),))).gross_return)
        self.assertEqual(self.settle(t, Finish((("2",),))).net_profit, -10)
        with self.assertRaises(ValueError):
            self.settle(t, Finish((("1",),)), {t: quote(t)})
        with self.assertRaises(ValueError):
            Finish((("1",),), scratches=("1",))

    def test_replay_cli_and_no_lookahead(self):
        from scripts.evaluate_betting_ev import replay_payload
        record = payload()
        record["tickets"][0]["stake"] = "10"
        record["finish"] = {"groups": [["1"], ["2"], ["3"], ["4"]]}
        record["settled_at"] = (START+timedelta(minutes=5)).isoformat()
        final = dict(record["tickets"][0]["quote"], status="final",
                     quoted_at=(START+timedelta(minutes=1)).isoformat(),
                     available_at=(START+timedelta(minutes=2)).isoformat())
        record["dividends"] = [{"pool":"WIN", "runners":["1"], "quote":final}]
        data = {"records":[record]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"replay.json"
            path.write_text(json.dumps(data))
            command = [sys.executable, "-m", "scripts.evaluate_betting_ev", "--input", str(path),
                       "--replay", "--bankroll", "1000"]
            output = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
            self.assertEqual(output["final_bankroll"], "1050")
        record["decision_at"] = START.isoformat()
        with self.assertRaises(ValueError):
            replay_payload(data, bankroll=1000, currency="HKD")

    def test_replay_missing_and_not_yet_available_dividend(self):
        from scripts.evaluate_betting_ev import replay_payload
        record = payload()
        record["tickets"][0]["stake"] = "10"
        record["finish"] = {"groups":[["1"], ["2"], ["3"], ["4"]]}
        record["settled_at"] = (START+timedelta(minutes=1)).isoformat()
        with self.assertRaises(ValueError):
            replay_payload({"records":[record]}, bankroll=1000, currency="HKD")

    def test_zero_stake_requires_no_dividend(self):
        t = Ticket("r1", "WIN", ("1",))
        s = settle_ticket(t, 0, Finish((("1",),)), rules(), {},
                          selected_at=AT, race_start=START, currency="HKD")
        self.assertEqual(s.status, "no_bet")
        self.assertEqual(s.net_profit, 0)

    def test_frozen_chronological_controls_and_final_price_independence(self):
        from scripts.evaluate_betting_ev import replay_payload
        record = payload()
        record["tickets"][0]["stake"] = "10"
        record["finish"] = {"groups": [["1"], ["2"], ["3"], ["4"]]}
        record["settled_at"] = (START+timedelta(minutes=5)).isoformat()
        final = dict(record["tickets"][0]["quote"], status="final", value="1000",
                     quoted_at=(START+timedelta(minutes=1)).isoformat(),
                     available_at=(START+timedelta(minutes=2)).isoformat())
        record["dividends"] = [{"pool":"WIN", "runners":["1"], "quote":final}]
        record["stake_policy"] = {"max_race_fraction": .8, "max_ticket_fraction": .8,
                                  "minimum_cash_fraction": .1}
        data = {"records":[record]}
        sized = {}
        for strategy in ("fixed", "fractional_kelly", "full_kelly"):
            result = replay_payload(data, bankroll=1000, currency="HKD", strategy=strategy)
            sized[strategy] = result["total_stake"]
        self.assertEqual(sized["fixed"], 10)
        self.assertLessEqual(sized["fractional_kelly"], 10)
        self.assertLessEqual(sized["full_kelly"], 40)
        self.assertGreater(sized["full_kelly"], sized["fractional_kelly"])
        final["value"] = "20"
        again = replay_payload(data, bankroll=1000, currency="HKD", strategy="full_kelly")
        self.assertEqual(again["total_stake"], sized["full_kelly"])

    def test_replay_overlap_and_daily_budget(self):
        from scripts.evaluate_betting_ev import replay_payload
        record = payload()
        record["tickets"][0]["stake"] = "10"
        record["finish"] = {"groups":[["2"], ["1"], ["3"], ["4"]]}
        record["settled_at"] = (START+timedelta(minutes=5)).isoformat()
        record["stake_policy"] = {"remaining_day_budget": "20"}
        second = json.loads(json.dumps(record))
        with self.assertRaises(ValueError):
            replay_payload({"records":[record,second]}, bankroll=1000, currency="HKD")
        second["decision_at"] = (AT+timedelta(minutes=10)).isoformat()
        second["race_start"] = (START+timedelta(minutes=10)).isoformat()
        second["settled_at"] = (START+timedelta(minutes=15)).isoformat()
        second["tickets"][0]["quote"]["quoted_at"] = (AT+timedelta(minutes=9, seconds=55)).isoformat()
        second["tickets"][0]["quote"]["available_at"] = (AT+timedelta(minutes=9, seconds=56)).isoformat()
        result = replay_payload({"records":[record,second]}, bankroll=1000, currency="HKD", strategy="full_kelly")
        self.assertLessEqual(result["total_stake"], 20)

    def test_fixed_replay_enforces_supplied_risk_caps(self):
        from scripts.evaluate_betting_ev import replay_payload
        record = payload()
        record["tickets"][0]["stake"] = "800"
        record["finish"] = {"groups": [["2"], ["1"], ["3"], ["4"]]}
        record["settled_at"] = (START+timedelta(minutes=5)).isoformat()
        record["stake_policy"] = {"remaining_day_budget": "1", "max_race_fraction": .01,
                                  "minimum_cash_fraction": .99, "maximum_open_exposure": "1"}
        with self.assertRaises(ValueError):
            replay_payload({"records":[record]}, bankroll=1000, currency="HKD")
