import tempfile
import unittest
from pathlib import Path

from ima.research_store import ResearchLedger


class ResearchStoreTests(unittest.TestCase):
    def test_attempt_reservation_is_idempotent_by_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory) / "ledger.sqlite")
            first = ledger.reserve_attempt("abc123", {"recipe": 1})
            second = ledger.reserve_attempt("abc123", {"recipe": 1})
        self.assertEqual(first.attempt_id, second.attempt_id)
        self.assertEqual("reserved", first.status)

    def test_completion_and_outbox_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory) / "ledger.sqlite")
            attempt = ledger.reserve_attempt("abc123", {"recipe": 1})
            ledger.complete_attempt(attempt.attempt_id, {"metric": 2.0})
            ledger.complete_attempt(attempt.attempt_id, {"metric": 2.0})
            outbox = ledger.pending_outbox()
            ledger.mark_uploaded(attempt.attempt_id, "mlflow-run")
            empty = ledger.pending_outbox()
        self.assertEqual(1, len(outbox))
        self.assertEqual([], empty)

    def test_completion_with_different_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = ResearchLedger(Path(directory) / "ledger.sqlite")
            attempt = ledger.reserve_attempt("abc123", {"recipe": 1})
            ledger.complete_attempt(attempt.attempt_id, {"metric": 2.0})
            with self.assertRaisesRegex(ValueError, "different result"):
                ledger.complete_attempt(attempt.attempt_id, {"metric": 3.0})


if __name__ == "__main__":
    unittest.main()
