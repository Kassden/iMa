import unittest
from pathlib import Path

import pandas as pd

from ima.research_targets import (
    TargetContractError,
    apply_target_contract,
    target_contract,
    validate_no_forbidden_label_features,
)


FIXTURE = Path(__file__).parent / "fixtures" / "research_races.csv"


class ResearchTargetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.read_csv(FIXTURE, parse_dates=["date"])

    def test_win_probability_contract_validates_one_winner_and_totals(self):
        contract = target_contract("win_probability")
        labelled = apply_target_contract(self.frame, contract)
        self.assertIn("target_probability", labelled)
        broken = self.frame.copy()
        broken.loc[broken["race_id"].eq("R1"), "target_win"] = 0
        with self.assertRaisesRegex(TargetContractError, "exactly one winner"):
            apply_target_contract(broken, contract)

    def test_ranking_and_top_k_targets_are_materialized(self):
        ranking = apply_target_contract(self.frame, target_contract("ranking_strength"))
        self.assertIn("target_rank_score", ranking)
        self.assertGreater(ranking["target_rank_score"].max(), ranking["target_rank_score"].min())
        placing = apply_target_contract(
            self.frame, target_contract("placing_top_k", {"top_k": 2}),
        )
        self.assertEqual(12, int(placing["target_top_2"].sum()))

    def test_finish_time_target_requires_coverage_and_positive_distance(self):
        speed = apply_target_contract(
            self.frame,
            target_contract("adjusted_finish_time_or_speed", {"min_coverage": 1.0}),
        )
        self.assertIn("target_adjusted_speed", speed)
        broken = self.frame.copy()
        broken.loc[:4, "finish_time"] = None
        with self.assertRaisesRegex(TargetContractError, "coverage"):
            apply_target_contract(
                broken,
                target_contract("adjusted_finish_time_or_speed", {"min_coverage": 0.9}),
            )

    def test_market_odds_forecast_target_uses_market_probability(self):
        market = apply_target_contract(self.frame, target_contract("market_odds_forecast"))
        self.assertEqual(
            market["market_probability"].tolist(),
            market["target_market_probability"].tolist(),
        )

    def test_label_columns_cannot_be_features(self):
        contract = target_contract("win_probability")
        with self.assertRaisesRegex(TargetContractError, "forbids post-race"):
            validate_no_forbidden_label_features(contract, ["horse_age", "result"])

    def test_unknown_target_is_rejected(self):
        with self.assertRaisesRegex(TargetContractError, "Unknown target"):
            target_contract("magic")


if __name__ == "__main__":
    unittest.main()
