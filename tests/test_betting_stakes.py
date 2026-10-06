from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ima.betting_contracts import Ticket
from ima.betting_ev import (conditional_choice_scenarios, evaluate_ticket, joint_return_matrix,
                            scenarios_from_orders)
from ima.betting_stakes import (StakePolicy, binary_log_growth, full_kelly_fraction,
                               physical_stake, size_portfolio, size_single)
from tests.test_betting_contracts import AT, START, quote, rules


def report(p=.2, d=6, ticket=None):
    ticket = ticket or Ticket("r1", "WIN", ("1",))
    return evaluate_ticket(ticket, p, 10, "HKD", rules(), at=AT, race_start=START,
                           quote=quote(ticket, value=d))


def policy(**changes):
    values = dict(bankroll=1000, currency="HKD", kelly_fraction=1, max_race_fraction=.8,
                  max_ticket_fraction=.8, minimum_cash_fraction=.1)
    values.update(changes)
    return StakePolicy(**values)


class BettingStakesTests(unittest.TestCase):
    def test_benter_arithmetic_and_fractional_kelly(self):
        self.assertAlmostEqual(full_kelly_fraction(.2, 6), .04)
        result = size_single(report(), policy(kelly_fraction=.25), rules(minimum_stake=10, stake_increment=10))
        self.assertEqual(result.stakes, (Decimal(10),))
        self.assertAlmostEqual(result.expected_profit, 2)
        self.assertAlmostEqual(result.expected_log_growth, binary_log_growth(.2, 6, .01))

    def test_extreme_probabilities_and_bad_returns(self):
        self.assertEqual(full_kelly_fraction(0, 6), 0)
        self.assertEqual(full_kelly_fraction(1, 6), 1)
        self.assertEqual(size_single(report(p=1), policy(), rules()).stakes, (Decimal(800),))
        self.assertEqual(size_single(report(p=.1), policy(), rules()).stakes, (Decimal(0),))
        for d in [0, 1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                full_kelly_fraction(.2, d)

    def test_rounding_never_upsizes_subminimum(self):
        r = rules(minimum_stake=10, stake_increment=5)
        self.assertEqual(physical_stake(.0099, policy(), r), 0)
        self.assertEqual(physical_stake(.0199, policy(), r), 15)
        self.assertEqual(size_single(report(), policy(bankroll=100), r).stakes, (Decimal(0),))

    def test_exposure_cash_day_and_open_caps(self):
        result = size_single(report(p=1), policy(remaining_day_budget=17,
                            maximum_open_exposure=30, current_open_exposure=19), rules())
        self.assertLessEqual(result.stakes[0], Decimal(11))
        result = size_single(report(p=1), policy(minimum_cash_fraction=.99), rules())
        self.assertLessEqual(result.stakes[0], Decimal(10))
        self.assertEqual(size_single(report(), policy(remaining_day_budget=0), rules()).stakes, (Decimal(0),))

    def test_missing_final_unverified_prices_and_duplicate_tickets(self):
        for r in [replace(report(), executable_evidence=False), replace(report(), mode="ex_post"),
                  replace(report(), decimal_return=None)]:
            with self.assertRaises(ValueError):
                size_single(r, policy(), rules())
        s = scenarios_from_orders([("1",), ("2",)], [.2, .8], assumptions="fixture", source_id="fixture")
        with self.assertRaises(ValueError):
            size_portfolio([report(), report()], s, np.array([[6,6],[0,0]]), policy(), rules())

    def test_single_portfolio_reduces_to_analytic_kelly(self):
        s = scenarios_from_orders([("1",), ("2",)], [.2, .8], assumptions="fixture", source_id="fixture")
        result = size_portfolio([report()], s, np.array([[6],[0]]), policy(), rules())
        self.assertTrue(result.success, result.solver_message)
        self.assertAlmostEqual(result.full_fractions[0], .04, places=5)
        self.assertLessEqual(result.stakes[0], 40)
        self.assertAlmostEqual(result.all_lose_probability, .8)
        self.assertLessEqual(result.optimality_gap, 1e-7)

    def test_jointly_winning_tickets_not_independent_allocations(self):
        s = scenarios_from_orders([("1","2","3"), ("2","3","1")], [.2, .8], assumptions="fixture", source_id="fixture")
        reports = [report(), report(ticket=Ticket("r1", "PLACE", ("1",)))]
        result = size_portfolio(reports, s, np.array([[6,6],[0,0]]), policy(), rules())
        self.assertTrue(result.success, result.solver_message)
        self.assertAlmostEqual(sum(result.full_fractions), .04, places=5)
        self.assertLessEqual(sum(result.stakes), 40)

    def test_mutually_exclusive_grid_control(self):
        s = scenarios_from_orders([("1",), ("2",), ("3",)], [.3,.3,.4], assumptions="fixture", source_id="fixture")
        reports = [report(.3, 4), report(.3, 4, Ticket("r1", "WIN", ("2",)))]
        gross = np.array([[4,0],[0,4],[0,0]])
        result = size_portfolio(reports, s, gross, policy(), rules())
        self.assertTrue(result.success, result.solver_message)
        f = np.asarray(result.full_fractions)
        growth = np.array(s.probabilities) @ np.log(1 + (gross-1) @ f)
        grid_best = max(np.array(s.probabilities) @ np.log(1 + (gross-1) @ np.array([a,b]))
                        for a in np.linspace(0,.3,61) for b in np.linspace(0,.3,61))
        self.assertGreaterEqual(growth+1e-8, grid_best)
        self.assertLessEqual(sum(result.stakes), Decimal(800))

    def test_losing_tails_preserved_for_overlap_and_refunds(self):
        s = conditional_choice_scenarios(["1","2","3","4"], [1]*4, depth=3)
        tickets = [Ticket("r1", "WIN", ("1",)), Ticket("r1", "QIN", ("1","2"))]
        gross = joint_return_matrix(tickets, s, rules(), [6,10])
        result = size_portfolio([report(.25,6,tickets[0]), report(1/6,10,tickets[1])],
                                s, gross, policy(max_race_fraction=.1), rules())
        self.assertTrue(result.success, result.solver_message)
        self.assertGreater(result.all_lose_probability, 0)
        self.assertLessEqual(sum(result.stakes), 100)

    def test_solver_failure_returns_cash_and_diagnostic(self):
        s = scenarios_from_orders([("1",), ("2",)], [.2,.8], assumptions="fixture", source_id="fixture")
        with patch("ima.betting_stakes.minimize", return_value=SimpleNamespace(
                x=np.array([.1]), success=False, message="forced failure")):
            result = size_portfolio([report()], s, np.array([[6],[0]]), policy(), rules())
        self.assertFalse(result.success)
        self.assertEqual(result.stakes, (Decimal(0),))
        self.assertEqual(result.status, "solver_failed")

    def test_false_solver_success_fails_independent_optimality_check(self):
        s = scenarios_from_orders([("1",), ("2",)], [.2,.8], assumptions="fixture", source_id="fixture")
        with patch("ima.betting_stakes.minimize", return_value=SimpleNamespace(
                x=np.array([0.]), success=True, message="false success")):
            result = size_portfolio([report()], s, np.array([[6],[0]]), policy(), rules())
        self.assertFalse(result.success)
        self.assertEqual(result.stakes, (Decimal(0),))

    def test_sampled_tail_admission_and_pool_impact_limits(self):
        s = scenarios_from_orders([("1",), ("2",)], [.2,.8], assumptions="empirical", source_id="fixture")
        for metadata in [{"draws":100}, {"draws":100,"converged":False}]:
            with self.assertRaises(ValueError):
                size_portfolio([report()], replace(s, metadata=metadata), np.array([[6],[0]]), policy(), rules())
        with self.assertRaises(ValueError):
            policy(own_impact_policy="unknown turnover")

    def test_malformed_return_matrix_and_probability_mismatch(self):
        s = scenarios_from_orders([("1",), ("2",)], [.2,.8], assumptions="fixture", source_id="fixture")
        with self.assertRaises(ValueError):
            size_portfolio([report()], s, np.array([[6],[6]]), policy(), rules())
        with self.assertRaises(ValueError):
            size_portfolio([report(.4)], s, np.array([[6],[0]]), policy(), rules())
        with self.assertRaises(ValueError):
            size_portfolio([report()], s, np.array([[600],[0]]), policy(), rules())

    def test_sampled_single_does_not_bypass_convergence_gate(self):
        r = replace(report(), probability_metadata={"draws": 100, "converged": False})
        with self.assertRaises(ValueError):
            size_single(r, policy(), rules())

    def test_mixed_refund_only_deep_ticket_and_shallow_win(self):
        from ima.betting_ev import JointScenarios
        from ima.betting_settlement import Finish
        shallow = Ticket("r1", "WIN", ("1",))
        deep = Ticket("r1", "QIN", ("1", "4"))
        s = JointScenarios((Finish((("1",),), scratches=("4",)),
                            Finish((("2",),), scratches=("4",))), (.2, .8), "fixture", "fixture")
        gross = joint_return_matrix([shallow,deep], s, rules(), [6,6])
        np.testing.assert_array_equal(gross, [[6,1],[0,1]])
        result = size_portfolio([report(.2,6,shallow), report(0,6,deep)], s, gross, policy(), rules())
        self.assertTrue(result.success, result.solver_message)

    def test_deadheat_hypothetical_portfolio_and_refund_payout_stress(self):
        from ima.betting_ev import JointScenarios
        from ima.betting_settlement import Finish
        t = Ticket("r1", "WIN", ("1",))
        s = JointScenarios((Finish((("1","2"), ("3",))),
                            Finish((("2",), ("3",)), scratches=("1",)),
                            Finish((("2",), ("1",), ("3",)))), (.4,.1,.5), "hypothetical deadheat", "fixture")
        dividends = [{t:quote(t,status="scenario",value=3)},{},{}]
        gross = joint_return_matrix([t], s, rules(), [6], scenario_dividends=dividends)
        result = size_portfolio([report(.4,6,t)], s, gross, policy(payout_multiplier=.9), rules(),
                                scenario_dividends=dividends)
        self.assertTrue(result.success, result.solver_message)
        f = result.rounded_fractions[0]
        expected = np.array(s.probabilities) @ np.log(1+np.array([1.7,0,-1])*f)
        self.assertAlmostEqual(result.expected_log_growth, expected)
