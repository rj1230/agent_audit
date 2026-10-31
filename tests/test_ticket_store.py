from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from ticket_store import create_or_get_ticket, get_existing_ticket


class TicketStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_tickets.sqlite3"

        self.items = [
            {
                "item": "book",
                "price": "23.00",
                "purchase_date": "2026-10-01",
            },
            {
                "item": "pen",
                "price": "7.00",
                "purchase_date": "2026-10-03",
            },
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_creates_new_ticket(self):
        result = create_or_get_ticket(
            "request-001",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        self.assertTrue(result["ticket_issued"])
        self.assertFalse(result["existing"])
        self.assertTrue(result["ticket_id"])
        self.assertEqual(result["status"], "ISSUED")
        self.assertEqual(result["refund_amount"], "30.00")

    def test_retry_returns_same_ticket(self):
        first = create_or_get_ticket(
            "request-002",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        second = create_or_get_ticket(
            "request-002",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertFalse(first["existing"])
        self.assertTrue(second["existing"])

    def test_same_items_in_different_order_are_same_payload(self):
        first = create_or_get_ticket(
            "request-003",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        second = create_or_get_ticket(
            "request-003",
            "user123",
            list(reversed(self.items)),
            "30.00",
            self.db_path,
        )

        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertTrue(second["existing"])

    def test_rejects_request_id_with_different_payload(self):
        create_or_get_ticket(
            "request-004",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        changed_items = [
            {
                "item": "book",
                "price": "23.00",
                "purchase_date": "2026-10-01",
            }
        ]

        result = create_or_get_ticket(
            "request-004",
            "user123",
            changed_items,
            "23.00",
            self.db_path,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(
            result["reason"],
            "idempotency_key_payload_conflict",
        )

    def test_rejects_request_id_reused_by_another_user(self):
        create_or_get_ticket(
            "request-005",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        result = create_or_get_ticket(
            "request-005",
            "another_user",
            self.items,
            "30.00",
            self.db_path,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(
            result["reason"],
            "idempotency_key_payload_conflict",
        )

    def test_ticket_persists_across_database_connections(self):
        first = create_or_get_ticket(
            "request-006",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        second = create_or_get_ticket(
            "request-006",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        self.assertEqual(first["ticket_id"], second["ticket_id"])
        self.assertTrue(second["existing"])

        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM refund_tickets"
            ).fetchone()[0]

        self.assertEqual(count, 1)

    def test_rejects_empty_request_id(self):
        with self.assertRaises(ValueError):
            create_or_get_ticket(
                "",
                "user123",
                self.items,
                "30.00",
                self.db_path,
            )

    def test_rejects_empty_user(self):
        with self.assertRaises(ValueError):
            create_or_get_ticket(
                "request-008",
                "",
                self.items,
                "30.00",
                self.db_path,
            )


    def test_existing_ticket_can_be_recovered_without_revalidation(self):
        created = create_or_get_ticket(
            "request-recovery",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        recovered = get_existing_ticket(
            "request-recovery",
            "user123",
            ["book", "pen"],
            self.db_path,
        )

        self.assertTrue(recovered["ticket_issued"])
        self.assertTrue(recovered["existing"])
        self.assertEqual(recovered["ticket_id"], created["ticket_id"])
        self.assertEqual(recovered["refund_amount"], "30.00")

    def test_existing_ticket_lookup_rejects_different_item(self):
        create_or_get_ticket(
            "request-recovery-conflict",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        result = get_existing_ticket(
            "request-recovery-conflict",
            "user123",
            ["book"],
            self.db_path,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(
            result["reason"],
            "idempotency_key_payload_conflict",
        )

    def test_existing_ticket_lookup_rejects_different_user(self):
        create_or_get_ticket(
            "request-recovery-user-conflict",
            "user123",
            self.items,
            "30.00",
            self.db_path,
        )

        result = get_existing_ticket(
            "request-recovery-user-conflict",
            "another_user",
            ["book", "pen"],
            self.db_path,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(
            result["reason"],
            "idempotency_key_payload_conflict",
        )


if __name__ == "__main__":
    unittest.main()
