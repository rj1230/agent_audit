from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from objects import PurchaseItem
from tools import issue_refund_ticket


class IssueRefundTicketTests(unittest.TestCase):

    def setUp(self):
        self.items = [PurchaseItem(name="Keyboard")]
        self.runtime = SimpleNamespace(
            context={"refund_request_id": "request-test-001"}
        )

    @patch("tools.create_or_get_ticket")
    @patch("tools.validate_refund")
    def test_eligible_refund_uses_trusted_request_id(
        self, mock_validate, mock_create
    ):
        validated = {
            "eligible": True,
            "user": "user-001",
            "items": [
                {
                    "item": "Keyboard",
                    "price": "49.99",
                    "purchase_date": "2026-10-20",
                }
            ],
            "refund_amount": "49.99",
        }
        expected = {
            "ticket_issued": True,
            "ticket_id": "ticket-test-001",
            "existing": False,
        }

        mock_validate.return_value = validated
        mock_create.return_value = expected

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=self.runtime,
        )

        self.assertEqual(result, expected)
        mock_create.assert_called_once_with(
            request_id="request-test-001",
            user="user-001",
            items=validated["items"],
            refund_amount="49.99",
        )

    @patch("tools.create_or_get_ticket")
    @patch("tools.validate_refund")
    def test_missing_request_id_fails_closed(
        self, mock_validate, mock_create
    ):
        mock_validate.return_value = {
            "eligible": True,
            "items": [{"item": "Keyboard", "price": "49.99"}],
            "refund_amount": "49.99",
        }

        runtime = SimpleNamespace(context={})

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=runtime,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(result["reason"], "missing_request_id")
        mock_create.assert_not_called()

    @patch("tools.create_or_get_ticket")
    @patch("tools.validate_refund")
    def test_ineligible_refund_never_creates_ticket(
        self, mock_validate, mock_create
    ):
        mock_validate.return_value = {
            "eligible": False,
            "reason": "outside_refund_window",
        }

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=self.runtime,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(result["reason"], "outside_refund_window")
        mock_create.assert_not_called()


    @patch("tools.validate_refund")
    @patch("tools.get_existing_ticket")
    def test_existing_ticket_is_returned_before_revalidation(
        self, mock_existing, mock_validate
    ):
        expected = {
            "ticket_issued": True,
            "ticket_id": "ticket-already-issued",
            "existing": True,
        }
        mock_existing.return_value = expected

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=self.runtime,
        )

        self.assertEqual(result, expected)
        mock_existing.assert_called_once_with(
            request_id="request-test-001",
            user="user-001",
            item_names=["Keyboard"],
        )
        mock_validate.assert_not_called()

    @patch("tools.validate_refund")
    @patch("tools.get_existing_ticket")
    def test_existing_ticket_payload_conflict_fails_before_revalidation(
        self, mock_existing, mock_validate
    ):
        conflict = {
            "ticket_issued": False,
            "reason": "idempotency_key_payload_conflict",
        }
        mock_existing.return_value = conflict

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=self.runtime,
        )

        self.assertEqual(result, conflict)
        mock_validate.assert_not_called()

    @patch("tools.create_or_get_ticket")
    @patch("tools.validate_refund")
    @patch("tools.get_existing_ticket", return_value=None)
    def test_new_ineligible_request_is_still_rejected(
        self, mock_existing, mock_validate, mock_create
    ):
        mock_validate.return_value = {
            "eligible": False,
            "reason": "outside_refund_window",
        }

        result = issue_refund_ticket.func(
            user="user-001",
            items=self.items,
            runtime=self.runtime,
        )

        self.assertFalse(result["ticket_issued"])
        self.assertEqual(result["reason"], "outside_refund_window")
        mock_validate.assert_called_once()
        mock_create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
