from datetime import datetime, timedelta, timezone
from decimal import Decimal
import unittest

from ima.betting_contracts import Quote, RuleInputs, Ticket

AT = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
START = AT + timedelta(minutes=2)


def rules(**overrides):
    values = dict(version="synthetic-v1", source="synthetic test rules, not HKJC validation",
                  currency="HKD", effective_from=AT - timedelta(days=1),
                  effective_to=AT + timedelta(days=1), paid_places=2, qpl_places=3,
                  minimum_stake=Decimal(1), stake_increment=Decimal(1),
                  verified=True, scratch_policy="refund_selected")
    values.update(overrides)
    return RuleInputs(**values)


def quote(ticket=None, **overrides):
    values = dict(ticket=ticket or Ticket("r1", "WIN", ("1",)), value="6",
                  convention="decimal_return", unit_stake="10", currency="HKD",
                  quoted_at=AT - timedelta(seconds=5), available_at=AT - timedelta(seconds=4),
                  source_hash="synthetic-quote", rule_version="synthetic-v1", status="pre_race")
    values.update(overrides)
    return Quote(**values)


class BettingContractsTests(unittest.TestCase):
    def test_pool_alias_and_physical_combinations(self):
        self.assertEqual(Ticket("r1", "trio", ("3", "1", "2")).runners, ("1", "2", "3"))
        self.assertEqual(Ticket("r1", "FCT", ("2", "1")).pool, "FORECAST")
        self.assertEqual(Ticket("r1", "FORECAST", ("2", "1")).runners, ("2", "1"))
        for runners in [("1", "1"), ("1",), ("", "2")]:
            with self.assertRaises(ValueError):
                Ticket("r1", "QIN", runners)

    def test_all_conventions_convert_to_same_total_return(self):
        self.assertEqual(quote(value="60", convention="dividend").decimal_return, 6)
        self.assertEqual(quote(value="5", convention="net_odds").decimal_return, 6)
        self.assertEqual(quote(value="6").decimal_return, 6)
        self.assertEqual(quote(value="60", convention="dividend", unit_stake="100").decimal_return, .6)

    def test_invalid_quotes_fail_closed(self):
        for changes in [dict(unit_stake=0), dict(value="NaN"), dict(value=-1),
                        dict(quoted_at=AT.replace(tzinfo=None)), dict(available_at=AT-timedelta(hours=1)),
                        dict(convention="odds"), dict(source_hash="")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                quote(**changes)

    def test_availability_and_close_gates(self):
        self.assertTrue(quote().executable_at(AT, START, timedelta(minutes=1)))
        for changes in [dict(status="final"), dict(status="scenario"), dict(close_status="closed"),
                        dict(close_status="unknown"), dict(available_at=AT+timedelta(seconds=1)),
                        dict(quoted_at=AT-timedelta(hours=1), available_at=AT-timedelta(hours=1))]:
            self.assertFalse(quote(**changes).executable_at(AT, START, timedelta(minutes=1)))
        self.assertFalse(quote().executable_at(START, START, timedelta(minutes=5)))

    def test_rules_effective_units_currency_and_availability(self):
        r = rules(minimum_stake=10, stake_increment=5)
        self.assertEqual(r.validate_stake("15"), Decimal(15))
        for amount in [5, 11, -1, "Infinity"]:
            with self.assertRaises(ValueError):
                r.validate_stake(amount)
        with self.assertRaises(ValueError):
            r.validate_ticket(Ticket("r1", "WIN", ("1",)), AT, "USD")
        with self.assertRaises(ValueError):
            r.validate_ticket(Ticket("r1", "WIN", ("1",)), r.effective_to, "HKD")
        with self.assertRaises(ValueError):
            rules(minimum_stake=10, stake_increment=3)
        with self.assertRaises(ValueError):
            rules(available_pools=("WIN",)).validate_ticket(Ticket("r1", "QIN", ("1", "2")), AT, "HKD")
