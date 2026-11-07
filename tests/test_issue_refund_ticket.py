from unittest.mock import patch

from agent.objects import PurchaseItem
from agent.tools import issue_approved_refund_ticket


REQUEST_ID = "request-test-001"
USER = "user-001"
ITEMS = [PurchaseItem(name="Keyboard")]

QUOTE = {
    "user": USER,
    "items": [
        {
            "item": "Keyboard",
            "price": "49.99",
            "purchase_date": "2026-10-20",
        }
    ],
    "refund_amount": "49.99",
}

VALIDATED = {
    "eligible": True,
    **QUOTE,
}


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_ticket_is_created_only_for_exact_approved_quote(
    validate_mock, create_mock
):
    expected = {
        "ticket_issued": True,
        "ticket_id": "ticket-test-001",
        "existing": False,
    }
    create_mock.return_value = expected

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == expected
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_called_once_with(
        request_id=REQUEST_ID,
        user=USER,
        items=QUOTE["items"],
        refund_amount="49.99",
    )


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund")
def test_missing_request_id_fails_closed(validate_mock, create_mock):
    result = issue_approved_refund_ticket(
        request_id=" ",
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == {
        "ticket_issued": False,
        "reason": "missing_request_id",
    }
    validate_mock.assert_not_called()
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch(
    "agent.tools.validate_refund",
    return_value={"eligible": False, "reason": "outside_refund_window"},
)
def test_ineligible_request_never_creates_ticket(
    validate_mock, create_mock
):
    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result["ticket_issued"] is False
    assert result["reason"] == "outside_refund_window"
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_changed_amount_does_not_create_ticket(validate_mock, create_mock):
    altered_quote = {
        **QUOTE,
        "refund_amount": "99.99",
    }

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=altered_quote,
    )

    assert result == {
        "ticket_issued": False,
        "reason": "approved_quote_mismatch",
    }
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_changed_purchase_details_do_not_create_ticket(
    validate_mock, create_mock
):
    altered_quote = {
        **QUOTE,
        "items": [
            {
                "item": "Keyboard",
                "price": "49.99",
                "purchase_date": "2026-10-21",
            }
        ],
    }

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=altered_quote,
    )

    assert result["ticket_issued"] is False
    assert result["reason"] == "approved_quote_mismatch"
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_changed_customer_does_not_create_ticket(
    validate_mock, create_mock
):
    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote={**QUOTE, "user": "user-999"},
    )

    assert result["ticket_issued"] is False
    assert result["reason"] == "approved_quote_mismatch"
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_existing_ticket_result_is_returned_for_same_approved_payload(
    validate_mock, create_mock
):
    expected = {
        "ticket_issued": True,
        "ticket_id": "ticket-existing-001",
        "existing": True,
    }
    create_mock.return_value = expected

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == expected
    create_mock.assert_called_once()
