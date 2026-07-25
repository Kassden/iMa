from pathlib import Path
import tempfile
import unittest

from ima.domain import PoolCandidate
from ima.ledger import PredictionLedger
from ima.registry import ModelRegistry, PromotionPolicy, eligible_for_promotion
from ima.strategy import RiskBudget, recommend_wagers


class LedgerRegistryTests(unittest.TestCase):
    def test_prediction_outcome_reconciliation(self):
        recommendation = recommend_wagers(
            [PoolCandidate("r1", "WIN", ("2",), 0.4, 4.0, model_version="m1")],
            10000,
            RiskBudget(),
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            ledger = PredictionLedger(Path(directory) / "ledger.sqlite")
            self.assertTrue(ledger.record(recommendation))
            self.assertFalse(ledger.record(recommendation))
            settlement = ledger.settle(recommendation.decision_id, ("2",), 4.2, "2026-01-01T00:00:00Z")
            self.assertTrue(settlement["won"])
            self.assertGreater(ledger.performance()["profit"], 0)

    def test_promotion_is_metric_gated_and_versioned(self):
        champion = {"race_log_loss": 2.1, "ece": 0.02}
        challenger = {"race_log_loss": 2.0, "ece": 0.019}
        allowed, reasons = eligible_for_promotion(champion, challenger, 800, PromotionPolicy())
        self.assertTrue(allowed, reasons)
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(Path(directory))
            registry.register("m2", {"metrics": challenger}, ["m2.joblib"])
            champion_path = registry.promote("m2", "operator", live_approved=False)
            self.assertTrue(champion_path.exists())


if __name__ == "__main__":
    unittest.main()
