import unittest

import numpy as np
import pandas as pd

from ima.simulator import extract_market_prices, simulate_race
from ima.strategy import RiskBudget


class SimulatorTests(unittest.TestCase):
    @staticmethod
    def live_frame():
        return pd.DataFrame({
            "race_id": ["R1"] * 3,
            "horse_no": ["1", "2", "3"],
            "win_odds": [3.0, 5.0, 12.0],
            "place_odds": [1.8, 2.2, 3.5],
            "market_probability": [0.5, 0.3, 0.2],
            "place_market_probability": [0.5, 0.3, 0.2],
            "ratable": [True, True, True],
        })

    @staticmethod
    def raw_payload():
        return {"provider": {"pool_pages": {
            "WPQ": {"normalized_combinations": [
                {"pool": "QIN", "combination": ["2", "1"], "decimal_odds": 6.0},
                {"pool": "QPL", "combination": ["1", "2"], "decimal_odds": 2.5},
            ]},
            "TRI": {"top_combinations": [
                {"combination": ["3", "1", "2"], "decimal_odds": 9.0},
            ]},
        }}}

    @staticmethod
    def artifact():
        class Model:
            def predict_proba(self, frame):
                return np.array([0.6, 0.3, 0.1])

        class Calibrator:
            def transform(self, values, race_ids):
                return values

        class Blend:
            def transform(self, fundamental, market, race_ids):
                return fundamental

        return {"model": Model(), "calibrator": Calibrator(), "blend": Blend()}

    def test_extracts_win_place_and_exotic_market_prices(self):
        prices = extract_market_prices(self.raw_payload(), self.live_frame())
        keys = {(item.pool, item.combination) for item in prices}
        self.assertIn(("WIN", ("1",)), keys)
        self.assertIn(("PLACE", ("1",)), keys)
        self.assertIn(("QIN", ("1", "2")), keys)
        self.assertIn(("TRI", ("1", "2", "3")), keys)

    def test_simulator_uses_ten_dollar_units_and_exact_return_math(self):
        report = simulate_race(
            self.live_frame(), self.artifact(), self.raw_payload(), 1000.0, "test-model",
            RiskBudget(
                max_race_fraction=0.5, max_combination_fraction=0.1,
                uncertainty_z=0.0, minimum_edge=0.0,
                minimum_stake=10.0, stake_increment=10.0,
            ),
        )
        bets = report["recommendations"]
        self.assertGreater(len(bets), 0)
        self.assertTrue(all(item["stake"] % 10 == 0 for item in bets))
        cost = sum(item["stake"] for item in bets)
        gross = sum(
            item["stake"] * item["probability"] * item["current_decimal_odds"]
            for item in bets
        )
        self.assertAlmostEqual(cost, report["summary"]["total_cost"])
        self.assertAlmostEqual(gross, report["summary"]["expected_gross_return"])
        self.assertAlmostEqual(gross - cost, report["summary"]["expected_net_return"])


if __name__ == "__main__":
    unittest.main()
