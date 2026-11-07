import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import approval


def make_session():
    return SimpleNamespace(
        pending_approval={
            "request_id": "private-request-123",
            "user": "private-user-456",
            "items": [{"name": "Private Product"}],
            "quote": {"refund_amount": "999.99"},
        },
        messages=[],
        conversation_closed=False,
    )


@pytest.mark.parametrize(
    ("issuer_result", "expected_outcome"),
    [
        (
            {"ticket_issued": True, "ticket_id": "PRIVATE-TICKET", "existing": False},
            "issued",
        ),
        (
            {"ticket_issued": True, "ticket_id": "PRIVATE-TICKET", "existing": True},
            "recovered_existing",
        ),
        (
            {"ticket_issued": False, "reason": "quote_validation_failed"},
            "failed",
        ),
    ],
)
def test_confirmation_records_metadata_only(
    tmp_path, monkeypatch, issuer_result, expected_outcome
):
    audit_file = tmp_path / "approval_audit.jsonl"
    monkeypatch.setattr(approval, "APPROVAL_AUDIT_FILE", audit_file)
    session = make_session()

    approval.confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: issuer_result,
        logger=logging.getLogger("approval-test"),
    )

    event = json.loads(audit_file.read_text(encoding="utf-8").strip())

    assert event["schema_version"] == 1
    assert event["outcome"] == expected_outcome
    assert len(event["request_id_hash"]) == 64
    assert event["request_id_hash"] != "private-request-123"
    assert event["duration_ms"] >= 0

    persisted = audit_file.read_text(encoding="utf-8")
    for private_value in (
        "private-request-123",
        "private-user-456",
        "Private Product",
        "999.99",
        "PRIVATE-TICKET",
    ):
        assert private_value not in persisted


def test_confirmation_exception_records_failure_and_retains_approval(
    tmp_path, monkeypatch
):
    audit_file = tmp_path / "approval_audit.jsonl"
    monkeypatch.setattr(approval, "APPROVAL_AUDIT_FILE", audit_file)
    session = make_session()
    original_approval = session.pending_approval

    def failing_issuer(**kwargs):
        raise RuntimeError("private database detail")

    approval.confirm_pending_approval(
        session,
        ticket_issuer=failing_issuer,
        logger=logging.getLogger("approval-test"),
    )

    event = json.loads(audit_file.read_text(encoding="utf-8").strip())
    assert event["outcome"] == "failed"
    assert session.pending_approval is original_approval
    assert session.conversation_closed is False


def test_decline_records_event_without_issuing_ticket(tmp_path, monkeypatch):
    audit_file = tmp_path / "approval_audit.jsonl"
    monkeypatch.setattr(approval, "APPROVAL_AUDIT_FILE", audit_file)
    session = make_session()

    approval.decline_pending_approval(session)

    event = json.loads(audit_file.read_text(encoding="utf-8").strip())
    assert event["outcome"] == "declined"
    assert session.pending_approval is None
    assert "No ticket has been created." in session.messages[-1]["content"]


def test_audit_write_failure_does_not_break_successful_confirmation(monkeypatch):
    def failing_open(*args, **kwargs):
        raise OSError("simulated audit storage failure")

    monkeypatch.setattr(Path, "open", failing_open)
    session = make_session()

    approval.confirm_pending_approval(
        session,
        ticket_issuer=lambda **kwargs: {
            "ticket_issued": True,
            "ticket_id": "PRIVATE-TICKET",
            "existing": False,
        },
        logger=logging.getLogger("approval-test"),
    )

    assert session.pending_approval is None
    assert session.conversation_closed is True
    assert "simulated refund ticket has been created" in session.messages[-1]["content"]

