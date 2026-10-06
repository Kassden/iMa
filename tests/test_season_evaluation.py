import itertools
import unittest

import numpy as np
import pandas as pd

from ima.pools import SUPPORTED_POOLS, OrderExponents, rank_combinations
from ima.season_evaluation import evaluate_season, pool_result_status


def runners(race_id="r1", size=4, results=None, date="2026-09-01"):
    strengths = np.arange(size, 0, -1, dtype=float)
    return pd.DataFrame({
        "race_id": [race_id] * size, "date": [date] * size,
        "horse_no": list(range(1, size + 1)),
        "result": results if results is not None else list(range(1, size + 1)),
        "model_probability": strengths / strengths.sum(),
    })


def settlements(frame, dividends=None):
    dividends = dividends or dict(zip(SUPPORTED_POOLS, [30, 20, 100, 40, 200, 300, 400, 500]))
    records = {}
    for race_id, race in frame.groupby("race_id"):
        finish = race.sort_values("result")["horse_no"].astype(str).tolist()
        places = 3 if len(race) >= 7 else 2
        combinations = {
            "WIN": [(finish[0],)], "PLACE": [(x,) for x in finish[:places]],
            "QIN": [tuple(finish[:2])],
            "QPL": list(itertools.combinations(finish[:places], 2)),
            "TRI": [tuple(finish[:3])], "TIERCE": [tuple(finish[:3])],
            "FIRST4": [tuple(finish[:4])], "QUARTET": [tuple(finish[:4])],
        }
        records[race_id] = [
            {"pool": pool, "winning_combination": combination,
             "dividend_hkd_per_10": dividends[pool]}
            for pool, combos in combinations.items() for combination in combos
        ]
    return records


class SeasonEvaluationTests(unittest.TestCase):
    def test_eight_top1_tickets_gross_includes_stake(self):
        frame = runners()
        report = evaluate_season(frame, settlements(frame))
        self.assertEqual(8, len(report.ledger))
        self.assertEqual(80, report.summary["stake_hkd"])
        self.assertEqual(1590, report.summary["gross_hkd"])
        self.assertEqual(1510, report.summary["profit_hkd"])
        self.assertEqual(1510 / 80, report.summary["roi"])
        self.assertEqual(1, report.summary["hit_rate"])
        self.assertEqual(2510, report.summary["ending_bankroll_hkd"])
        self.assertEqual(80, report.summary["min_bankroll_capital_required_hkd"])
        for pool in SUPPORTED_POOLS:
            self.assertEqual(10, report.by_pool[pool]["stake_hkd"])
            self.assertEqual(1, report.by_pool[pool]["nsettled"])
        self.assertTrue(report.ledger["estimated_ev_hkd"].isna().all())

    def test_top_n_costs_ten_per_straight_ticket(self):
        frame = runners()
        report = evaluate_season(frame, settlements(frame), pools=["WIN"], top_n=3)
        self.assertEqual(["1", "2", "3"], report.ledger.combination.tolist())
        self.assertEqual([30, 0, 0], report.ledger.realized_gross_hkd.tolist())
        self.assertEqual([20, -10, -10], report.ledger.realized_net_hkd.tolist())
        self.assertEqual(30, report.summary["stake_hkd"])
        self.assertEqual(0, report.summary["profit_hkd"])
        self.assertEqual(1 / 3, report.summary["hit_rate"])

    def test_ordered_and_unordered_combinations(self):
        frame = runners(results=[2, 1, 4, 3])
        report = evaluate_season(frame, settlements(frame))
        hits = report.ledger.set_index("pool")["hit"].to_dict()
        self.assertTrue(hits["QIN"])
        self.assertTrue(hits["FIRST4"])
        self.assertFalse(hits["TRI"])
        self.assertFalse(hits["TIERCE"])
        self.assertFalse(hits["QUARTET"])

    def test_place_and_qpl_paid_positions_at_six_seven_boundary(self):
        for size, hit in [(6, False), (7, True)]:
            with self.subTest(size=size):
                frame = runners(size=size, results=[3, 1, 2] + list(range(4, size + 1)))
                report = evaluate_season(frame, settlements(frame), pools=["PLACE", "QPL"])
                self.assertEqual([hit, hit], report.ledger.hit.tolist())
                for row in report.ledger.itertuples():
                    expected = rank_combinations(
                        frame.horse_no.astype(str).tolist(), frame.model_probability.to_numpy(), row.pool,
                    )[0]
                    self.assertAlmostEqual(expected.probability, row.probability)

    def test_probabilities_and_break_even_are_normalized(self):
        frame = runners()
        frame.model_probability *= 100
        original = frame.copy(deep=True)
        report = evaluate_season(frame, settlements(frame), pools=["WIN"])
        self.assertAlmostEqual(0.4, report.ledger.probability.iloc[0])
        self.assertAlmostEqual(25, report.ledger.break_even_dividend_hkd_per_10.iloc[0])
        pd.testing.assert_frame_equal(original, frame)

    def test_selection_independent_of_results_quotes_and_dividends(self):
        first = runners()
        second = runners(results=[4, 3, 2, 1])
        poison = {"r1": [{"pool": "WIN", "combination": "4", "payout_hkd_per_10": 1e12,
                          "quote_scope": "full_ticket", "is_pre_race": True}]}
        a = evaluate_season(first, settlements(first))
        b = evaluate_season(second, settlements(second), quotes=poison)
        columns = ["pool", "rank", "combination", "probability"]
        pd.testing.assert_frame_equal(a.ledger[columns], b.ledger[columns])

    def test_frozen_order_exponents_pass_through_without_outcome_selection(self):
        frame = runners()
        exponents = OrderExponents(second=0.5, third=1.7)
        calibrated = evaluate_season(frame, settlements(frame), exponents=exponents)
        default = evaluate_season(frame, settlements(frame))
        for row in calibrated.ledger.itertuples():
            direct = rank_combinations(frame.horse_no.astype(str).tolist(),
                                       frame.model_probability.to_numpy(), row.pool, exponents)[0]
            self.assertEqual("/".join(direct.runners), row.combination)
            self.assertAlmostEqual(direct.probability, row.probability)
        self.assertFalse(np.allclose(calibrated.ledger.probability, default.ledger.probability))
        changed = runners(results=[4, 3, 2, 1])
        replay = evaluate_season(changed, settlements(changed), exponents=exponents)
        columns = ["pool", "combination", "probability"]
        pd.testing.assert_frame_equal(calibrated.ledger[columns], replay.ledger[columns])
        frame.model_probability = 1
        flat = evaluate_season(frame, settlements(frame), exponents=exponents)
        flat_replay = evaluate_season(frame.iloc[::-1], {}, exponents=exponents)
        pd.testing.assert_frame_equal(flat.ledger[columns], flat_replay.ledger[columns])
        self.assertAlmostEqual(1 / 24, flat.ledger.set_index("pool").loc["QUARTET", "probability"])

    def test_missing_settlement_is_never_a_loss(self):
        report = evaluate_season(runners(), {}, pools=["WIN"])
        self.assertEqual(0, report.summary["stake_hkd"])
        self.assertEqual(1, report.summary["nmissing"])
        self.assertIsNone(report.summary["roi"])
        self.assertIsNone(report.summary["hit_rate"])
        self.assertTrue(report.ledger.realized_gross_hkd.isna().all())
        self.assertTrue(report.ledger.hit.isna().all())
        self.assertEqual(1000, report.summary["ending_bankroll_hkd"])

    def test_missing_winner_dividend_excludes_selected_loser(self):
        frame = runners(results=[2, 1, 3, 4])
        official = {"r1": [{"pool": "WIN", "winning_combination": "2"}]}
        report = evaluate_season(frame, official, pools=["WIN"])
        self.assertEqual("missing_dividend", report.ledger.status.iloc[0])
        self.assertEqual(0, report.summary["nsettled"])
        self.assertTrue(report.ledger.realized_net_hkd.isna().all())

    def test_partial_place_and_qpl_records_exclude_whole_pool_race(self):
        frame = runners(size=7)
        official = settlements(frame)
        official["r1"] = [record for record in official["r1"]
                          if record["winning_combination"] not in {("3",), ("2", "3")}]
        report = evaluate_season(frame, official, pools=["PLACE", "QPL"])
        self.assertEqual(2, report.summary["nmissing"])
        self.assertEqual(0, report.summary["stake_hkd"])

    def test_quote_requires_exact_ticket_and_pre_race_full_ticket_provenance(self):
        frame = runners(results=[2, 1, 3, 4])
        quote = {"pool": "WIN", "combination": "1", "payout_hkd_per_10": 40,
                 "quote_scope": "full_ticket", "is_pre_race": True}
        report = evaluate_season(frame, settlements(frame), pools=["WIN"],
                                 quotes={"r1": [quote]})
        self.assertAlmostEqual(6, report.ledger.estimated_ev_hkd.iloc[0])
        self.assertEqual(0, report.ledger.realized_gross_hkd.iloc[0])
        for update in [{"quote_scope": "winner_only"}, {"is_pre_race": False},
                       {"payout_hkd_per_10": np.nan}, {"payout_hkd_per_10": np.inf},
                       {"payout_hkd_per_10": -20}, {"payout_hkd_per_10": True},
                       {"status": "final"}, {"winning_combination": "1"}]:
            invalid = evaluate_season(frame, settlements(frame), pools=["WIN"],
                                     quotes={"r1": [{**quote, **update}]})
            self.assertTrue(invalid.ledger.estimated_ev_hkd.isna().all())

    def test_final_dividends_never_supply_quotes_even_with_poison_fields(self):
        frame = runners()
        official = settlements(frame)
        for record in official["r1"]:
            record.update(payout_hkd_per_10=1e12, is_pre_race=True, quote_scope="full_ticket")
        report = evaluate_season(frame, official)
        self.assertTrue(report.ledger.payout_quote_hkd_per_10.isna().all())
        self.assertTrue(report.ledger.estimated_ev_hkd.isna().all())

    def test_refund_dead_heat_exclude_every_top_n_ticket_of_pool_race(self):
        frame = runners()
        for marker in [{"status": "refund"}, {"status": "dead_heat"},
                       {"refund": True}, {"dead_heat": True}]:
            official = settlements(frame)
            official["r1"].append({"pool": "WIN", **marker})
            report = evaluate_season(frame, official, pools=["WIN", "QIN"], top_n=2)
            self.assertEqual(2, report.by_pool["WIN"]["nexcluded"])
            self.assertEqual(0, report.by_pool["WIN"]["stake_hkd"])
            self.assertEqual(2, report.by_pool["QIN"]["nsettled"])

    def test_incomplete_and_tied_results_do_not_drop_starters(self):
        for results, status in [([1, 2, np.nan, 4], "excluded_incomplete_race"),
                                ([1, 2, 3, 5], "excluded_incomplete_race")]:
            report = evaluate_season(runners(results=results), {}, pools=["WIN"])
            self.assertEqual(status, report.ledger.status.iloc[0])
            self.assertEqual(1, report.summary["nexcluded"])
        frame = runners()
        frame["field_size"] = 5
        report = evaluate_season(frame, settlements(frame), pools=["WIN"])
        self.assertEqual("excluded_incomplete_race", report.ledger.status.iloc[0])

    def test_lower_sixth_place_dead_heat_leaves_all_eight_pools_eligible(self):
        frame = runners(size=12, results=[1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11, 12])
        frame["finishing_status"] = "FINISHED"
        report = evaluate_season(frame, settlements(frame))
        self.assertEqual(8, report.summary["nsettled"])
        self.assertEqual(80, report.summary["stake_hkd"])
        self.assertEqual(1590, report.summary["gross_hkd"])
        for pool in SUPPORTED_POOLS:
            self.assertEqual("eligible", pool_result_status(frame, pool))

    def test_paid_rank_tie_excludes_only_affected_pools(self):
        frame = runners(size=7, results=[1, 2, 3, 4, 4, 6, 7])
        for pool in SUPPORTED_POOLS:
            expected = "excluded_dead_heat" if pool in {"FIRST4", "QUARTET"} else "eligible"
            self.assertEqual(expected, pool_result_status(frame, pool))
        frame = runners(size=7, results=[1, 2, 2, 4, 5, 6, 7])
        self.assertEqual("eligible", pool_result_status(frame, "WIN"))
        self.assertEqual("excluded_dead_heat", pool_result_status(frame, "TRI"))

    def test_known_nonfinishers_keep_full_field_and_lose_without_invented_rank(self):
        for code in ["PU", "UR", "FE", "DNF", "DISQ", "TNP"]:
            frame = runners(size=7, results=[1, 2, 3, 4, 5, 6, np.nan])
            frame["finishing_status"] = ["FINISHED"] * 6 + [code]
            frame.loc[6, "model_probability"] = 0.99
            original = frame.copy(deep=True)
            report = evaluate_season(frame, settlements(frame))
            self.assertEqual(8, report.summary["nsettled"], code)
            self.assertEqual("7", report.ledger.set_index("pool").loc["WIN", "combination"])
            self.assertEqual(0, report.summary["gross_hkd"], code)
            self.assertEqual(-80, report.summary["profit_hkd"], code)
            pd.testing.assert_frame_equal(frame, original)

    def test_unknown_missing_position_quarantines_every_pool(self):
        frame = runners(size=7, results=[1, 2, 3, 4, 5, 6, np.nan])
        frame["finishing_status"] = ["FINISHED"] * 6 + ["UNKNOWN"]
        report = evaluate_season(frame, settlements(frame))
        self.assertEqual(8, report.summary["nexcluded"])
        self.assertEqual(0, report.summary["nsettled"])

    def test_fewer_known_finishers_excludes_only_pools_without_required_paid_ranks(self):
        frame = runners(size=7, results=[1, 2, 3, np.nan, np.nan, np.nan, np.nan])
        frame["finishing_status"] = ["FINISHED"] * 3 + ["PU", "UR", "FE", "DNF"]
        official = settlements(frame)
        for pool in SUPPORTED_POOLS:
            expected = "excluded_incomplete_race" if pool in {"FIRST4", "QUARTET"} else "eligible"
            self.assertEqual(expected, pool_result_status(frame, pool))
        report = evaluate_season(frame, official)
        self.assertEqual(6, report.summary["nsettled"])
        self.assertEqual(2, report.summary["nexcluded"])

    def test_normalized_runner_numbers_aliases_and_duplicate_records(self):
        frame = runners()
        frame.horse_no = ["01", "2.0", 3, 4.0]
        record = {"pool": "quinella", "winning_combination": "02/01", "dividend_hkd_per_10": 100}
        report = evaluate_season(frame, {"r1": [record, dict(record)]}, pools=["QIN"])
        self.assertEqual(100, report.summary["gross_hkd"])
        self.assertEqual(1, report.summary["nsettled"])
        conflict = evaluate_season(frame, {"r1": [record, {**record, "dividend_hkd_per_10": 101}]}, pools=["QIN"])
        self.assertEqual(1, conflict.summary["nexcluded"])

    def test_conflicting_duplicate_quotes_do_not_supply_ev(self):
        quote = {"pool": "WIN", "combination": "1", "payout_hkd_per_10": 40,
                 "quote_scope": "full_ticket", "is_pre_race": True}
        report = evaluate_season(runners(), {}, pools=["WIN"],
                                 quotes={"r1": [quote, {**quote, "payout_hkd_per_10": 50}]})
        self.assertEqual("conflicting", report.ledger.quote_status.iloc[0])
        self.assertTrue(report.ledger.estimated_ev_hkd.isna().all())

    def test_invalid_settlement_combinations_and_values(self):
        for combination, dividend in [("1/1", 20), ("1/9", 20), ("1", 20),
                                      ("1/2", np.nan), ("1/2", np.inf), ("1/2", 0)]:
            record = {"pool": "QIN", "winning_combination": combination,
                      "dividend_hkd_per_10": dividend}
            report = evaluate_season(runners(), {"r1": [record]}, pools=["QIN"])
            self.assertEqual(0, report.summary["nsettled"])
            self.assertTrue(report.ledger.realized_gross_hkd.isna().all())

    def test_unknown_settlement_status_is_not_assumed_payable(self):
        for status in ["pending_review", "nonpayable", "unparsed", "unavailable"]:
            record = {"pool": "WIN", "winning_combination": [1],
                      "dividend_hkd_per_10": 30, "status": status}
            report = evaluate_season(runners(), [dict(record, race_id="r1")], pools=["WIN"])
            self.assertEqual("excluded_unsupported_settlement", report.ledger.status.iloc[0])
            self.assertEqual(0, report.summary["nsettled"])

    def test_flat_settlement_and_quote_lists_match_mapping_adapter(self):
        frame = runners()
        official = settlements(frame)
        flat = [{"race_id": "r1", **record} for record in official["r1"]]
        a = evaluate_season(frame, official)
        b = evaluate_season(frame, flat)
        pd.testing.assert_frame_equal(a.ledger, b.ledger)
        quote = {"race_id": "r1", "pool": "WIN", "combination": "1",
                 "payout_hkd_per_10": 25, "quote_scope": "full_ticket", "is_pre_race": True}
        report = evaluate_season(frame, flat, pools=["WIN"], quotes=[quote])
        self.assertAlmostEqual(0, report.ledger.estimated_ev_hkd.iloc[0])

    def test_chronological_curve_and_simultaneous_portfolio_funding(self):
        frames = [runners(f"r{i:02d}", results=[2, 1, 3, 4], date=f"2026-09-{i + 1:02d}")
                  for i in range(14)]
        frame = pd.concat(frames[::-1], ignore_index=True)
        # Evaluate one losing WIN ticket per race.
        official = settlements(frame)
        report = evaluate_season(frame, official, pools=["WIN"], top_n=1)
        self.assertEqual([f"r{i:02d}" for i in range(14)], report.bankroll_curve.race_id.tolist())
        self.assertEqual(140, report.summary["maximum_drawdown_hkd"])
        self.assertEqual(140, report.summary["min_bankroll_capital_required_hkd"])
        self.assertTrue(report.summary["can_finance_with_1000"])
        # 110 straight WIN tickets cost 1100. One gross return of 1 occurs late,
        # and cannot finance the upfront stakes of that race.
        frame = runners(size=110)
        official = {"r1": [{"pool": "WIN", "winning_combination": "1",
                             "dividend_hkd_per_10": 1}]}
        report = evaluate_season(frame, official, pools=["WIN"], top_n=110)
        self.assertEqual(1100, report.summary["min_bankroll_capital_required_hkd"])
        self.assertEqual(1099, report.summary["maximum_drawdown_hkd"])
        self.assertFalse(report.summary["can_finance_with_1000"])
        self.assertEqual(100, report.summary["additional_capital_required_hkd"])
        self.assertFalse(report.bankroll_curve.financed_by_starting_1000.iloc[0])

    def test_fixed_eight_ticket_schedule_requires_external_funding_after_bankruptcy(self):
        frames = [runners(f"R{i + 1}", size=7, results=[7, 6, 5, 4, 3, 2, 1]) for i in range(14)]
        frame = pd.concat(frames[::-1], ignore_index=True)
        report = evaluate_season(frame, settlements(frame))
        self.assertEqual([f"R{i + 1}" for i in range(14)], report.bankroll_curve.race_id.tolist())
        self.assertEqual([80] * 14, report.bankroll_curve.stake_hkd.tolist())
        self.assertEqual(1120, report.summary["stake_hkd"])
        self.assertEqual(0, report.summary["gross_hkd"])
        self.assertEqual(-1, report.summary["roi"])
        self.assertEqual(-120, report.summary["ending_bankroll_hkd"])
        self.assertEqual(1120, report.summary["min_bankroll_capital_required_hkd"])
        self.assertEqual(1120, report.summary["maximum_drawdown_hkd"])
        self.assertFalse(report.summary["can_finance_with_1000"])
        self.assertEqual(120, report.summary["additional_capital_required_hkd"])
        self.assertEqual([True] * 12 + [False] * 2,
                         report.bankroll_curve.financed_by_starting_1000.tolist())

    def test_missing_races_are_excluded_from_bankroll_and_settled_totals(self):
        frame = pd.concat([runners("r1"), runners("r2")], ignore_index=True)
        report = evaluate_season(frame, settlements(runners("r1")))
        self.assertEqual(2, report.summary["nraces"])
        self.assertEqual(8, report.summary["nsettled"])
        self.assertEqual(8, report.summary["nmissing"])
        self.assertEqual(160, report.summary["planned_stake_hkd"])
        self.assertEqual(80, report.summary["stake_hkd"])
        self.assertEqual(["r1"], report.bankroll_curve.race_id.tolist())

    def test_bad_inputs_raise_and_deterministic_ties(self):
        for values in [[0, 0, 0, 0], [-1, 1, 1, 1], [np.nan, 1, 1, 1],
                       [np.inf, 1, 1, 1], ["bad", 1, 1, 1]]:
            frame = runners()
            frame.model_probability = values
            with self.assertRaises(ValueError):
                evaluate_season(frame, {})
        frame = runners()
        frame.horse_no = ["1", "01", "3", "4"]
        with self.assertRaises(ValueError):
            evaluate_season(frame, {})
        for top_n in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                evaluate_season(runners(), {}, top_n=top_n)
        for second in [0, -1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                evaluate_season(runners(), {}, exponents=OrderExponents(second=second))
        with self.assertRaises(ValueError):
            evaluate_season(runners(), {}, pools=["QIN", "quinella"])
        frame = runners()
        frame.model_probability = 1
        report = evaluate_season(frame.iloc[::-1], {}, pools=["WIN"], top_n=2)
        self.assertEqual(["1", "2"], report.ledger.combination.tolist())

    def test_empty_input_has_defined_zero_totals_and_undefined_rates(self):
        report = evaluate_season(runners().iloc[:0], {})
        self.assertTrue(report.ledger.empty)
        self.assertTrue(report.bankroll_curve.empty)
        self.assertEqual(0, report.summary["nraces"])
        self.assertEqual(0, report.summary["min_bankroll_capital_required_hkd"])
        self.assertEqual(1000, report.summary["ending_bankroll_hkd"])
        self.assertIsNone(report.summary["roi"])


if __name__ == "__main__":
    unittest.main()
