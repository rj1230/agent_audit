import logging
from types import SimpleNamespace

import pytest

from agent.approval import (
    confirm_pending_approval,
    decline_pending_approval,
)


def make_session(approval=True):
    return SimpleNamespace(
        pending_approval=(
            {
                "request_id": "approval-test-001",
                "user": "user-001",
                "items": [{"name": "Keyboard"}],
                "quote": {"refund_amount": "49.99"},
            }
            if approval
            else None
        ),
        messages=[
            {"role": "user", "content": "Please refund my keyboard."}
        ],
        conversation_closed=False,
    )


def test_confirm_success_issues_ticket_and_closes_conversation():
    session = make_session()
    calls = []

    def issuer(**kwargs):
        calls.append(kwargs)
        return {
            "ticket_issued": True,
            "ticket_id": "TICKET-001",
            "existing": False,
        }

    confirm_pending_approval(
        session,
        ticket_issuer=issuer,
        logger=logging.getLogger("test"),
    )

    assert len(calls) == 1
    assert calls[0]["request_id"] == "approval-test-001"
    assert calls[0]["user"] == "user-001"
    assert [item.name for item in calls[0]["items"]] == ["Keyboard"]
    assert calls[0]["approved_quote"] == {"refund_amount": "49.99"}
    assert session.pending_approval is None
    assert session.conversation_closed is True
    assert session.messages[-1] == {
        "role": "assistant",
        "content": (
            "Your simulated refund ticket has been created. "
            "Ticket reference: TICKET-001. No money has been transferred."
        ),
    }


def test_confirm_existing_ticket_is_idempotent():
    session = make_session()

    confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: {
            "ticket_issued": True,
            "ticket_id": "TICKET-001",
            "existing": True,
        },
        logger=logging.getLogger("test"),
    )

    assert session.pending_approval is None
    assert session.conversation_closed is True
    assert "already recorded" in session.messages[-1]["content"]
    assert "TICKET-001" in session.messages[-1]["content"]


def test_confirm_exception_retains_approval_for_retry():
    session = make_session()
    original_approval = session.pending_approval

    def issuer(**kwargs):
        raise RuntimeError("database unavailable")

    confirm_pending_approval(
        session,
        ticket_issuer=issuer,
        logger=logging.getLogger("test"),
    )

    assert session.pending_approval is original_approval
    assert session.conversation_closed is False
    assert "approval is retained" in session.messages[-1]["content"]


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        [],
        {"ticket_issued": "yes"},
        {"ticket_issued": True},
        {
            "ticket_issued": True,
            "ticket_id": "",
            "existing": False,
        },
        {
            "ticket_issued": True,
            "ticket_id": "TICKET-001",
            "existing": "yes",
        },
    ],
)
def test_confirm_invalid_issuer_result_retains_approval(invalid_result):
    session = make_session()
    original_approval = session.pending_approval

    confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: invalid_result,
        logger=logging.getLogger("test"),
    )

    assert session.pending_approval is original_approval
    assert session.conversation_closed is False
    assert "approval is retained" in session.messages[-1]["content"]


def test_confirm_rejected_quote_clears_approval_without_ticket():
    session = make_session()

    confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: {
            "ticket_issued": False,
            "reason": "quote_validation_failed",
        },
        logger=logging.getLogger("test"),
    )

    assert session.pending_approval is None
    assert session.conversation_closed is False
    assert "No ticket was created." in session.messages[-1]["content"]


def test_confirm_without_pending_approval_is_harmless():
    session = make_session(approval=False)
    original_messages = list(session.messages)

    confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: pytest.fail(
            "Ticket issuer must not be called."
        ),
        logger=logging.getLogger("test"),
    )

    assert session.pending_approval is None
    assert session.messages == original_messages
    assert session.conversation_closed is False


def test_decline_clears_pending_quote_without_creating_ticket():
    session = make_session()

    decline_pending_approval(session)

    assert session.pending_approval is None
    assert session.messages[-1] == {
        "role": "assistant",
        "content": (
            "The refund request was declined. "
            "No ticket has been created."
        ),
    }


def test_decline_without_pending_quote_is_harmless():
    session = make_session(approval=False)
    original_messages = list(session.messages)

    decline_pending_approval(session)

    assert session.pending_approval is None
    assert session.messages == original_messages
