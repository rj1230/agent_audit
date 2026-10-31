import json
from contextlib import closing
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_DB_PATH = Path(__file__).with_name("refund_tickets.sqlite3")



def get_existing_ticket(
    request_id: str,
    user: str,
    item_names: list[str],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """Recover an existing ticket only for the same user and requested items."""

    if not request_id or not request_id.strip():
        raise ValueError("request_id must not be empty")

    if not user or not user.strip():
        raise ValueError("user must not be empty")

    database = Path(db_path)

    # A lookup must not create a database or table as a side effect.
    if not database.is_file():
        return None

    with closing(sqlite3.connect(database, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row

        table = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'refund_tickets'
            """
        ).fetchone()

        if table is None:
            return None

        existing = connection.execute(
            """
            SELECT ticket_id, request_id, user, items_json,
                   refund_amount, status, created_at
            FROM refund_tickets
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()

    if existing is None:
        return None

    stored_items = json.loads(existing["items_json"])
    stored_names = sorted(item["item"] for item in stored_items)

    if existing["user"] != user or stored_names != sorted(item_names):
        return {
            "ticket_issued": False,
            "reason": "idempotency_key_payload_conflict",
        }

    return {
        "ticket_issued": True,
        "ticket_id": existing["ticket_id"],
        "request_id": existing["request_id"],
        "user": existing["user"],
        "items": stored_items,
        "refund_amount": existing["refund_amount"],
        "status": existing["status"],
        "created_at": existing["created_at"],
        "existing": True,
    }


def create_or_get_ticket(
    request_id: str,
    user: str,
    items: list[dict],
    refund_amount: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Persist a simulated refund ticket idempotently."""

    if not request_id or not request_id.strip():
        raise ValueError("request_id must not be empty")

    if not user or not user.strip():
        raise ValueError("user must not be empty")

    normalized_items = sorted(
        items,
        key=lambda item: item["item"],
    )

    items_json = json.dumps(
        normalized_items,
        sort_keys=True,
        separators=(",", ":"),
    )

    payload = {
        "user": user,
        "items_json": items_json,
        "refund_amount": str(refund_amount),
    }

    database = Path(db_path)
    database.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(database, timeout=10)) as connection:
        with connection:
            connection.row_factory = sqlite3.Row

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS refund_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    user TEXT NOT NULL,
                    items_json TEXT NOT NULL,
                    refund_amount TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status = 'ISSUED'),
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT ticket_id, request_id, user, items_json,
                       refund_amount, status, created_at
                FROM refund_tickets
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

            if existing is not None:
                same_payload = (
                    existing["user"] == payload["user"]
                    and existing["items_json"] == payload["items_json"]
                    and existing["refund_amount"] == payload["refund_amount"]
                )

                if not same_payload:
                    return {
                        "ticket_issued": False,
                        "reason": "idempotency_key_payload_conflict",
                    }

                return {
                    "ticket_issued": True,
                    "ticket_id": existing["ticket_id"],
                    "request_id": existing["request_id"],
                    "user": existing["user"],
                    "items": json.loads(existing["items_json"]),
                    "refund_amount": existing["refund_amount"],
                    "status": existing["status"],
                    "created_at": existing["created_at"],
                    "existing": True,
                }

            ticket_id = str(uuid4())
            created_at = datetime.now(timezone.utc).isoformat()

            connection.execute(
                """
                INSERT INTO refund_tickets (
                    ticket_id,
                    request_id,
                    user,
                    items_json,
                    refund_amount,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, 'ISSUED', ?)
                """,
                (
                    ticket_id,
                    request_id,
                    payload["user"],
                    payload["items_json"],
                    payload["refund_amount"],
                    created_at,
                ),
            )

            return {
                "ticket_issued": True,
                "ticket_id": ticket_id,
                "request_id": request_id,
                "user": user,
                "items": normalized_items,
                "refund_amount": str(refund_amount),
                "status": "ISSUED",
                "created_at": created_at,
                "existing": False,
            }
