"""Approval state transitions and privacy-safe audit events for AgentAudit."""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from agent.objects import PurchaseItem


APPROVAL_AUDIT_FILE = Path(__file__).with_name("traces") / "approval_audit.jsonl"


def _record_approval_event(
    *,
    request_id: str,
    outcome: str,
    started_at: float,
) -> None:
    """Append a metadata-only event. Audit failure must never change refund state."""
    try:
        request_hash = hashlib.sha256(
            request_id.encode("utf-8")
        ).hexdigest()

        event = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id_hash": request_hash,
            "outcome": outcome,
            "duration_ms": round(
                max(0.0, time.monotonic() - started_at) * 1000,
                2,
            ),
        }

        APPROVAL_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with APPROVAL_AUDIT_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception:
        # Audit logging is deliberately fail-safe and must not affect ticket state.
        pass


def confirm_pending_approval(
    session_state: Any,
    *,
    ticket_issuer: Callable[..., dict],
    logger: logging.Logger,
) -> None:
    """Issue a pending approved quote and safely update session state.

    Exceptions and malformed responses retain the approval for retry.
    The ticket issuer remains responsible for idempotency and quote validation.
    """
    approval = session_state.pending_approval

    if not isinstance(approval, dict):
        return

    started_at = time.monotonic()
    request_id = approval.get("request_id")
    if not isinstance(request_id, str):
        request_id = ""

    try:
        items = [
            PurchaseItem(name=item["name"])
            for item in approval["items"]
        ]

        result = ticket_issuer(
            request_id=approval["request_id"],
            user=approval["user"],
            items=items,
            approved_quote=approval["quote"],
        )

        if not isinstance(result, dict):
            raise TypeError("Ticket issuer returned a non-dictionary result.")

        if not isinstance(result.get("ticket_issued"), bool):
            raise TypeError("Ticket issuer returned an invalid ticket status.")

        if result["ticket_issued"]:
            ticket_id = result.get("ticket_id")

            if not isinstance(ticket_id, str) or not ticket_id.strip():
                raise TypeError("Ticket issuer returned no valid ticket ID.")

            if not isinstance(result.get("existing", False), bool):
                raise TypeError("Ticket issuer returned an invalid idempotency status.")

    except Exception:
        _record_approval_event(
            request_id=request_id,
            outcome="failed",
            started_at=started_at,
        )
        logger.exception("Approved ticket issuance failed.")
        session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't confirm the ticket outcome because of a "
                    "technical issue. Your approval is retained; retry the "
                    "same confirmation button."
                ),
            }
        )
        return

    if result["ticket_issued"]:
        existing = result.get("existing", False)
        _record_approval_event(
            request_id=request_id,
            outcome="recovered_existing" if existing else "issued",
            started_at=started_at,
        )

        ticket_id = result["ticket_id"]

        if existing:
            message = (
                "The simulated refund ticket was already recorded for "
                "this request. Ticket reference: {ticket_id}."
            ).format(ticket_id=ticket_id)
        else:
            message = (
                "Your simulated refund ticket has been created. "
                "Ticket reference: {ticket_id}."
            ).format(ticket_id=ticket_id)

        session_state.messages.append(
            {
                "role": "assistant",
                "content": message + " No money has been transferred.",
            }
        )
        session_state.pending_approval = None
        session_state.conversation_closed = True
        return

    _record_approval_event(
        request_id=request_id,
        outcome="failed",
        started_at=started_at,
    )
    reason = result.get("reason", "quote_validation_failed")
    session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"The quote could not be confirmed: {reason}. "
                "No ticket was created."
            ),
        }
    )
    session_state.pending_approval = None


def decline_pending_approval(session_state: Any) -> None:
    """Discard a pending refund quote without issuing a ticket."""
    approval = session_state.pending_approval

    if not isinstance(approval, dict):
        return

    started_at = time.monotonic()
    request_id = approval.get("request_id")
    if not isinstance(request_id, str):
        request_id = ""

    _record_approval_event(
        request_id=request_id,
        outcome="declined",
        started_at=started_at,
    )

    session_state.pending_approval = None
    session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "The refund request was declined. "
                "No ticket has been created."
            ),
        }
    )
