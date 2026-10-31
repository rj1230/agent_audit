import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import tools
from objects import PurchaseItem


class FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 31)


class RefundRulesTests(unittest.TestCase):

    def setUp(self):
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

    def item(self, name):
        return PurchaseItem(name=name)

    def test_rejects_invalid_username(self):
        result = tools.validate_refund("unknown", [self.item("book")])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "invalid_username")

    def test_rejects_empty_item_list(self):
        result = tools.validate_refund("user123", [])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "no_items_requested")

    def test_rejects_duplicate_items(self):
        result = tools.validate_refund(
            "user123",
            [self.item("book"), self.item("book")],
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "duplicate_items")

    def test_rejects_item_not_purchased(self):
        result = tools.validate_refund(
            "user123",
            [self.item("unknown_item")],
        )
        self.assertFalse(result["eligible"])

    def test_accepts_purchase_exactly_30_days_old(self):
        result = tools.validate_refund(
            "user123",
            [self.item("book")],
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["refund_amount"], "23.00")

    def test_rejects_purchase_31_days_old(self):
        result = tools.validate_refund(
            "user123",
            [self.item("old_item")],
        )
        self.assertFalse(result["eligible"])
        self.assertIn("refund window is 30 days", result["reason"])

    def test_rejects_future_purchase(self):
        result = tools.validate_refund(
            "user123",
            [self.item("future_item")],
        )
        self.assertFalse(result["eligible"])
        self.assertIn("future purchase date", result["reason"])

    def test_rejects_malformed_purchase_date(self):
        result = tools.validate_refund(
            "user123",
            [self.item("bad_date")],
        )
        self.assertFalse(result["eligible"])

    def test_uses_decimal_arithmetic_for_refund_total(self):
        result = tools.validate_refund(
            "user123",
            [self.item("decimal_a"), self.item("decimal_b")],
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["refund_amount"], "0.30")


if __name__ == "__main__":
    unittest.main()
