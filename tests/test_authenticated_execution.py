from pathlib import Path
import tempfile
import unittest

from ima.authenticated_execution import GuardedHKJCExecutor, TransactionLimits
from ima.domain import WagerRecommendation


class FakeSession:
    def is_authenticated(self): return True
    def submit_and_confirm(self, recommendation): return "tx-1", "confirmed"


class AuthenticatedExecutionTests(unittest.TestCase):
    def recommendation(self, stake=10.0):
        return WagerRecommendation(
            f"d-{stake}", "r1", "WIN", ("3",), stake, 1000, 0.4, 2.5, 3.0, 0.2, 0.01, "m1"
        )

    def test_live_execution_requires_both_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = GuardedHKJCExecutor(FakeSession(), Path(directory) / "audit.jsonl")
            self.assertEqual("blocked", executor.submit(self.recommendation()).status)

    def test_limits_confirmation_and_idempotency(self):
        with tempfile.TemporaryDirectory() as directory:
            executor = GuardedHKJCExecutor(
                FakeSession(), Path(directory) / "audit.jsonl", TransactionLimits(),
                live_enabled=True, operator_approved=True,
            )
            self.assertEqual("accepted", executor.submit(self.recommendation()).status)
            self.assertEqual("duplicate", executor.submit(self.recommendation()).status)
            self.assertEqual("blocked", executor.submit(self.recommendation(20)).status)


if __name__ == "__main__":
    unittest.main()
