from pathlib import Path
import tempfile
import unittest

from ima.domain import PoolCandidate
from ima.execution import HKJCWebExecutor, PaperExecutor
from ima.strategy import RiskBudget, recommend_wagers


class StrategyTests(unittest.TestCase):
    def candidate(self, probability=0.35, odds=4.0):
        return PoolCandidate("race-1", "WIN", ("3",), probability, odds, model_version="m1")

    def test_strategy_abstains_without_edge(self):
        self.assertEqual([], recommend_wagers([self.candidate(0.20, 4.0)], 10000, RiskBudget()))

    def test_strategy_obeys_race_and_combination_caps(self):
        results = recommend_wagers([self.candidate()], 10000, RiskBudget())
        self.assertEqual(1, len(results))
        self.assertLessEqual(results[0].stake, 50)
        self.assertGreaterEqual(results[0].stake, 10)

    def test_paper_execution_is_idempotent_and_live_fails_closed(self):
        recommendation = recommend_wagers([self.candidate()], 10000, RiskBudget())[0]
        with tempfile.TemporaryDirectory() as directory:
            executor = PaperExecutor(Path(directory) / "ledger.jsonl")
            self.assertEqual("paper_accepted", executor.submit(recommendation).status)
            self.assertEqual("duplicate", executor.submit(recommendation).status)
        self.assertEqual("blocked", HKJCWebExecutor().submit(recommendation).status)


if __name__ == "__main__":
    unittest.main()
