from datetime import date
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import tools
from agent.objects import PurchaseItem


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 31)


class DeterministicRefundBenchmark(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.ticket_db = Path(self.temp_dir.name) / "benchmark_tickets.sqlite3"

        self.purchases = {
            "user123": {
                "book": {"price": "23.00", "date": "2026-10-01"},
                "pen": {"price": "7.00", "date": "2026-10-03"},
                "old_item": {"price": "10.00", "date": "2026-09-30"},
                "future_item": {"price": "5.00", "date": "2026-11-01"},
                "decimal_a": {"price": "0.10", "date": "2026-10-10"},
                "decimal_b": {"price": "0.20", "date": "2026-10-10"},
                "bad_date": {"price": "4.00", "date": "invalid"},
            }
        }

        self.data_patch = patch.object(tools, "users_data", self.purchases)
        self.date_patch = patch.object(tools, "date", FrozenDate)

        self.data_patch.start()
        self.date_patch.start()

        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.date_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

        original_create = tools.create_or_get_ticket

        self.create_patch = patch.object(
            tools,
            "create_or_get_ticket",
            side_effect=lambda **kwargs: original_create(
                **kwargs,
                db_path=self.ticket_db,
            ),
        )

        self.create_mock = self.create_patch.start()
        self.addCleanup(self.create_patch.stop)

    def items(self, *names):
        return [PurchaseItem(name=name) for name in names]

    def validate(self, user, *names):
        return tools.validate_refund(user, self.items(*names))

    def approved_quote(self, *names):
        result = self.validate("user123", *names)
        if not result["eligible"]:
            raise AssertionError(
                f"Cannot create an approval quote: {result}"
            )

        return {
            "user": result["user"],
            "items": sorted(
                result["items"],
                key=lambda item: item["item"],
            ),
            "refund_amount": result["refund_amount"],
        }

    def issue(self, request_id, *names, approved_quote=None):
        quote = (
            self.approved_quote(*names)
            if approved_quote is None
            else approved_quote
        )

        return tools.issue_approved_refund_ticket(
            request_id=request_id,
            user="user123",
            items=self.items(*names),
            approved_quote=quote,
        )

    # Eligibility and input validation

    def test_01_valid_single_item(self):
        result = self.validate("user123", "book")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["refund_amount"], "23.00")

    def test_02_valid_multiple_items(self):
        result = self.validate("user123", "book", "pen")
        self.assertTrue(result["eligible"])
        self.assertEqual(result["refund_amount"], "30.00")

    def test_03_exactly_30_days_is_eligible(self):
        result = self.validate("user123", "book")
        self.assertTrue(result["eligible"])

    def test_04_31_days_is_ineligible(self):
        result = self.validate("user123", "old_item")
        self.assertFalse(result["eligible"])

    def test_05_future_purchase_is_ineligible(self):
        result = self.validate("user123", "future_item")
        self.assertFalse(result["eligible"])

    def test_06_invalid_username_is_rejected(self):
        result = self.validate("unknown_user", "book")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "invalid_username")

    def test_07_unpurchased_item_is_rejected(self):
        result = self.validate("user123", "not_purchased")
        self.assertFalse(result["eligible"])

    def test_08_duplicate_items_are_rejected(self):
        result = self.validate("user123", "book", "book")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "duplicate_items")

    def test_09_empty_item_list_is_rejected(self):
        result = self.validate("user123")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "no_items_requested")

    def test_10_malformed_purchase_date_is_rejected(self):
        result = self.validate("user123", "bad_date")
        self.assertFalse(result["eligible"])

    def test_11_mixed_eligibility_rejects_entire_request(self):
        result = self.validate("user123", "book", "old_item")
        self.assertFalse(result["eligible"])

    # Ticket safety and idempotency

    def test_12_missing_request_id_fails_closed(self):
        result = self.issue("", "book")

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(result["reason"], "missing_request_id")
        self.create_mock.assert_not_called()

    def test_13_retry_returns_the_same_ticket(self):
        first = self.issue("benchmark-retry-001", "book", "pen")
        second = self.issue("benchmark-retry-001", "book", "pen")

        self.assertTrue(first["ticket_issued"])
        self.assertTrue(second["ticket_issued"])
        self.assertFalse(first["existing"])
        self.assertTrue(second["existing"])
        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertEqual(first["refund_amount"], "30.00")

        # Each request reaches the atomic SQLite store; the store
        # itself guarantees that retries do not create duplicate tickets.
        self.assertEqual(self.create_mock.call_count, 2)

    def test_14_reused_request_id_with_different_items_is_rejected(self):
        first = self.issue("benchmark-conflict-001", "book")
        second = self.issue("benchmark-conflict-001", "pen")

        self.assertTrue(first["ticket_issued"])
        self.assertFalse(second["ticket_issued"])
        self.assertEqual(
            second["reason"],
            "idempotency_key_payload_conflict",
        )
        # Each request reaches the atomic SQLite store; the store
        # itself guarantees that retries do not create duplicate tickets.
        self.assertEqual(self.create_mock.call_count, 2)



    def test_15_tampered_approved_quote_fails_closed(self):
        quote = self.approved_quote("book")
        quote["refund_amount"] = "999.00"

        result = self.issue(
            "benchmark-tampered-quote-001",
            "book",
            approved_quote=quote,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(result["reason"], "approved_quote_mismatch")
        self.create_mock.assert_not_called()

if __name__ == "__main__":
    main(verbosity=2)

