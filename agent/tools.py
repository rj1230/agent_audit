import json
import logging

from pathlib import Path
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from langchain.tools import tool
from dotenv import load_dotenv

from agent.ticket_store import create_or_get_ticket
from agent.objects import PurchaseItem

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Load the existing 1,000-user dataset without regenerating it.
DATA_FILE = Path(__file__).with_name("users_data_1000.json")

with DATA_FILE.open("r", encoding="utf-8") as file:
    users_data = json.load(file)


def parse_purchase_date(value: str) -> date:
    """Accept dates such as 2026-10-5 and 2026-10-05."""
    parts = value.split("-")

    if len(parts) != 3:
        raise ValueError("Invalid purchase date format")

    year, month, day = map(int, parts)
    return date(year, month, day)


def validate_refund(user: str, items: list[PurchaseItem]):
    """Apply deterministic refund eligibility rules."""

    if user not in users_data:
        return {
            "eligible": False,
            "reason": "invalid_username",
        }

    if not items:
        return {
            "eligible": False,
            "reason": "no_items_requested",
        }

    today = date.today()
    purchases = users_data[user]

    requested_names = [item.name for item in items]

    if len(requested_names) != len(set(requested_names)):
        return {
            "eligible": False,
            "reason": "duplicate_items",
        }

    total = Decimal("0.00")
    validated_items = []

    for item in items:
        purchase = purchases.get(item.name)

        if purchase is None:
            return {
                "eligible": False,
                "reason": f"{item.name} was not purchased by this customer",
            }

        try:
            purchase_date = parse_purchase_date(purchase["date"])
        except (ValueError, KeyError, TypeError):
            return {
                "eligible": False,
                "reason": f"Invalid purchase date for {item.name}",
            }

        age_days = (today - purchase_date).days

        if age_days < 0:
            return {
                "eligible": False,
                "reason": f"{item.name} has a future purchase date",
            }

        if age_days > 30:
            return {
                "eligible": False,
                "reason": f"{item.name} was purchased {age_days} days ago; refund window is 30 days",
            }

        price = Decimal(str(purchase["price"]))

        total += price

        validated_items.append(
            {
                "item": item.name,
                "price": str(price),
                "purchase_date": purchase_date.isoformat(),
            }
        )

    total = total.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )

    return {
        "eligible": True,
        "user": user,
        "items": validated_items,
        "refund_amount": str(total),
    }


@tool
def get_user_purchase_history(user: str) -> dict:
    """Verify a customer username and retrieve their purchase history.

    A result of {"status": "invalid_username"} means verification failed.
    This is a terminal result for the current request: stop refund processing
    and do not issue a refund ticket. Ask the customer to provide a valid
    username.
    """

    logging.info("Retrieving purchase history for %s", user)

    if user not in users_data:
        return {"status": "invalid_username"}

    return users_data[user]


def issue_approved_refund_ticket(
    *,
    request_id: str,
    user: str,
    items: list[PurchaseItem],
    approved_quote: dict,
) -> dict:
    """Issue a simulated ticket after explicit application-side approval."""

    if not isinstance(request_id, str) or not request_id.strip():
        return {
            "ticket_issued": False,
            "reason": "missing_request_id",
        }

    if not isinstance(approved_quote, dict):
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    if set(approved_quote) != {"user", "items", "refund_amount"}:
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    # Revalidate against the authoritative dataset and current policy.
    result = validate_refund(user, items)

    if not result["eligible"]:
        return {
            "ticket_issued": False,
            "reason": result["reason"],
        }

    current_items = sorted(
        result["items"],
        key=lambda item: item["item"],
    )

    try:
        approved_items = approved_quote["items"]
        if not isinstance(approved_items, list):
            raise TypeError("Approved items must be a list.")

        approved_items = sorted(
            approved_items,
            key=lambda item: item["item"],
        )
    except (TypeError, KeyError, AttributeError):
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    current_quote = {
        "user": result["user"],
        "items": current_items,
        "refund_amount": result["refund_amount"],
    }

    supplied_quote = {
        "user": approved_quote.get("user"),
        "items": approved_items,
        "refund_amount": approved_quote.get("refund_amount"),
    }

    if supplied_quote != current_quote:
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    # The ticket store performs the final atomic idempotency check.
    return create_or_get_ticket(
        request_id=request_id,
        user=result["user"],
        items=current_items,
        refund_amount=result["refund_amount"],
    )
