import unittest

from scrapper.browser_pools import (
    parse_dom_odds, parse_leg_runners, parse_rectangular_matrix, parse_top_combinations,
)


class BrowserPoolTests(unittest.TestCase):
    def test_top_combination_parser(self):
        text = """Top 20\nBanker Top 10\nTrio Top 20\n4-5-6\n8.0\n5-6-9\n13\n"""
        rows = parse_top_combinations(text, "TRI")
        self.assertEqual(["4", "5", "6"], rows[0]["combination"])
        self.assertEqual(8.0, rows[0]["decimal_odds"])

    def test_rectangular_matrix_parser(self):
        text = """Forecast\n1st\u00a0Horse\n\t1\t2\t3\n2nd\u00a0Horse\n1\t\t12\t20\n2\t14\t\t30\nForecast Method :\n"""
        rows = parse_rectangular_matrix(text, "1st\u00a0Horse", "Forecast Method")
        self.assertIn({"combination": ["2", "1"], "decimal_odds": 12.0}, rows)
        self.assertIn({"combination": ["1", "2"], "decimal_odds": 14.0}, rows)

    def test_dom_odds_ids_normalize_pool_and_combination(self):
        rows = parse_dom_odds([
            {"id": "qb_QIN_1_2", "text": "8.5"},
            {"id": "qb_DBL_4_7", "text": "42"},
        ])
        self.assertEqual(
            {"pool": "QIN", "combination": ["1", "2"], "decimal_odds": 8.5}, rows[0]
        )

    def test_multi_race_legs_are_grouped(self):
        legs = parse_leg_runners([
            {"id": "runnerNo_6_1", "runner_no": "1", "horse_name": "A", "win_odds": "4.2"},
            {"id": "runnerNo_7_1", "runner_no": "1", "horse_name": "B", "win_odds": "5"},
        ])
        self.assertEqual("A", legs["6"][0]["horse_name"])
        self.assertEqual(5.0, legs["7"][0]["win_odds"])


if __name__ == "__main__":
    unittest.main()
