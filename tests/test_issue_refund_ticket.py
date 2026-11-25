from unittest.mock import Mock, patch

import pytest

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


@pytest.fixture
def ticket_lookup(monkeypatch):
    """Keep tests isolated from the real SQLite ticket database."""
    lookup_mock = Mock(return_value=None)
    monkeypatch.setattr(
        "agent.tools.get_existing_ticket",
        lookup_mock,
    )
    return lookup_mock


def existing_ticket():
    """A stored ticket matching the approved quote."""
    return {
        "ticket_issued": True,
        "ticket_id": "ticket-existing-001",
        "request_id": REQUEST_ID,
        "user": USER,
        "items": [
            {
                "item": "Keyboard",
                "price": "49.99",
                "purchase_date": "2026-10-20",
            }
        ],
        "refund_amount": "49.99",
        "status": "issued",
        "created_at": "2026-10-20T10:00:00",
        "existing": True,
    }


# ---------------------------------------------------------
# New ticket creation and approval validation
# ---------------------------------------------------------


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_ticket_is_created_only_for_exact_approved_quote(
    validate_mock,
    create_mock,
    ticket_lookup,
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
    ticket_lookup.assert_called_once_with(
        REQUEST_ID,
        USER,
        ["Keyboard"],
    )
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_called_once_with(
        request_id=REQUEST_ID,
        user=USER,
        items=QUOTE["items"],
        refund_amount="49.99",
    )


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund")
def test_missing_request_id_fails_closed(
    validate_mock,
    create_mock,
    ticket_lookup,
):
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

    ticket_lookup.assert_not_called()
    validate_mock.assert_not_called()
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch(
    "agent.tools.validate_refund",
    return_value={
        "eligible": False,
        "reason": "outside_refund_window",
    },
)
def test_ineligible_request_never_creates_ticket(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result["ticket_issued"] is False
    assert result["reason"] == "outside_refund_window"

    ticket_lookup.assert_called_once_with(
        REQUEST_ID,
        USER,
        ["Keyboard"],
    )
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_changed_amount_does_not_create_ticket(
    validate_mock,
    create_mock,
    ticket_lookup,
):
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
    validate_mock,
    create_mock,
    ticket_lookup,
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

    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_changed_customer_does_not_create_ticket(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote={
            **QUOTE,
            "user": "user-999",
        },
    )

    assert result["ticket_issued"] is False
    assert result["reason"] == "approved_quote_mismatch"

    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_not_called()


# ---------------------------------------------------------
# Atomic creation fallback
# ---------------------------------------------------------


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund", return_value=VALIDATED)
def test_atomic_store_can_return_existing_ticket_after_lookup(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    """A concurrent request may create the ticket after the initial lookup."""
    expected = existing_ticket()
    create_mock.return_value = expected

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == expected
    ticket_lookup.assert_called_once_with(
        REQUEST_ID,
        USER,
        ["Keyboard"],
    )
    validate_mock.assert_called_once_with(USER, ITEMS)
    create_mock.assert_called_once()


# ---------------------------------------------------------
# Existing-ticket recovery regressions
# ---------------------------------------------------------


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund")
def test_matching_existing_ticket_is_recovered_without_revalidation(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    """An old ticket remains recoverable after the refund window expires."""
    expected = existing_ticket()
    ticket_lookup.return_value = expected

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == expected
    ticket_lookup.assert_called_once_with(
        REQUEST_ID,
        USER,
        ["Keyboard"],
    )
    validate_mock.assert_not_called()
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund")
def test_existing_ticket_with_different_amount_is_rejected(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    stored = existing_ticket()
    stored["refund_amount"] = "59.99"
    ticket_lookup.return_value = stored

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == {
        "ticket_issued": False,
        "reason": "approved_quote_mismatch",
    }

    validate_mock.assert_not_called()
    create_mock.assert_not_called()


@patch("agent.tools.create_or_get_ticket")
@patch("agent.tools.validate_refund")
def test_ticket_store_idempotency_conflict_is_returned_unchanged(
    validate_mock,
    create_mock,
    ticket_lookup,
):
    conflict = {
        "ticket_issued": False,
        "reason": "idempotency_key_payload_conflict",
    }
    ticket_lookup.return_value = conflict

    result = issue_approved_refund_ticket(
        request_id=REQUEST_ID,
        user=USER,
        items=ITEMS,
        approved_quote=QUOTE,
    )

    assert result == conflict

    ticket_lookup.assert_called_once_with(
        REQUEST_ID,
        USER,
        ["Keyboard"],
    )
    validate_mock.assert_not_called()
    create_mock.assert_not_called()
