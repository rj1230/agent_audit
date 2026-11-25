import json
import logging

from pathlib import Path
from datetime import date
from decimal import Decimal, DecimalException, ROUND_HALF_UP

from langchain.tools import tool
from dotenv import load_dotenv

from agent.ticket_store import (
    create_or_get_ticket,
    get_existing_ticket,
)
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
                "reason": (f"{item.name} was not purchased by this customer"),
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
                "reason": (
                    f"{item.name} was purchased {age_days} days ago; "
                    "refund window is 30 days"
                ),
            }

        try:
            price = Decimal(str(purchase["price"]))
        except (DecimalException, ValueError, TypeError, KeyError):
            return {
                "eligible": False,
                "reason": f"Invalid purchase price for {item.name}",
            }

        if not price.is_finite() or price < 0:
            return {
                "eligible": False,
                "reason": f"Invalid purchase price for {item.name}",
            }

        try:
            total += price
        except DecimalException:
            return {
                "eligible": False,
                "reason": "Invalid refund amount",
            }

        validated_items.append(
            {
                "item": item.name,
                "price": str(price),
                "purchase_date": purchase_date.isoformat(),
            }
        )

    try:
        total = total.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    except DecimalException:
        return {
            "eligible": False,
            "reason": "Invalid refund amount",
        }

    return {
        "eligible": True,
        "user": user,
        "items": validated_items,
        "refund_amount": str(total),
    }


def _normalize_quote_items(items):
    """Validate and consistently sort the item records in an approved quote."""

    if not isinstance(items, list) or not items:
        raise ValueError("Approved items must be a non-empty list.")

    normalized = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each approved item must be a dictionary.")

        if set(item) != {"item", "price", "purchase_date"}:
            raise ValueError("Approved item fields do not match.")

        name = item["item"]
        price = item["price"]
        purchase_date = item["purchase_date"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Invalid approved item name.")

        if not isinstance(price, str) or not price.strip():
            raise ValueError("Invalid approved item price.")

        if not isinstance(purchase_date, str) or not purchase_date.strip():
            raise ValueError("Invalid approved purchase date.")

        normalized.append(
            {
                "item": name,
                "price": price,
                "purchase_date": purchase_date,
            }
        )

    return sorted(normalized, key=lambda item: item["item"])


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
    """Issue or recover a ticket after explicit application-side approval.

    Recovery is checked before current refund eligibility. This ensures that
    retrying an already completed request does not fail merely because the
    original purchase has since aged beyond the refund window.

    The stored ticket is still checked against the complete approved quote.
    New tickets require current eligibility and an exact quote match.
    """

    # ---------------------------------------------------------
    # 1. Validate the basic request structure.
    # ---------------------------------------------------------

    if not isinstance(request_id, str) or not request_id.strip():
        return {
            "ticket_issued": False,
            "reason": "missing_request_id",
        }

    if not isinstance(user, str) or not user.strip():
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    if not isinstance(items, list) or not items:
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
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

    requested_names = []

    for item in items:
        name = getattr(item, "name", None)

        if not isinstance(name, str) or not name.strip():
            return {
                "ticket_issued": False,
                "reason": "approved_quote_mismatch",
            }

        requested_names.append(name)

    if len(requested_names) != len(set(requested_names)):
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    # Validate the supplied quote before consulting the ticket store.
    try:
        approved_items = _normalize_quote_items(approved_quote["items"])
    except (TypeError, ValueError, KeyError, AttributeError):
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    approved_amount = approved_quote.get("refund_amount")

    if not isinstance(approved_amount, str) or not approved_amount.strip():
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    supplied_quote = {
        "user": approved_quote.get("user"),
        "items": approved_items,
        "refund_amount": approved_amount,
    }

    # ---------------------------------------------------------
    # 2. Recover an existing ticket BEFORE eligibility checks.
    # ---------------------------------------------------------

    existing = get_existing_ticket(
        request_id,
        user,
        requested_names,
    )

    if existing is not None:
        # Preserve the ticket store's idempotency conflict response.
        if not isinstance(existing, dict):
            return {
                "ticket_issued": False,
                "reason": "approved_quote_mismatch",
            }

        if existing.get("ticket_issued") is not True:
            return existing

        # A matching request ID and item names are not sufficient.
        # Verify the complete approved quote against the stored ticket.
        try:
            existing_items = _normalize_quote_items(existing["items"])

            existing_quote = {
                "user": existing["user"],
                "items": existing_items,
                "refund_amount": existing["refund_amount"],
            }

        except (TypeError, ValueError, KeyError, AttributeError):
            return {
                "ticket_issued": False,
                "reason": "approved_quote_mismatch",
            }

        if existing_quote != supplied_quote:
            return {
                "ticket_issued": False,
                "reason": "approved_quote_mismatch",
            }

        logging.info(
            "Recovered existing refund ticket for request_id=%s",
            request_id,
        )

        return existing

    # ---------------------------------------------------------
    # 3. No existing ticket: enforce current eligibility.
    # ---------------------------------------------------------

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

    current_quote = {
        "user": result["user"],
        "items": current_items,
        "refund_amount": result["refund_amount"],
    }

    # ---------------------------------------------------------
    # 4. Require an exact match with the approved quote.
    # ---------------------------------------------------------

    if supplied_quote != current_quote:
        return {
            "ticket_issued": False,
            "reason": "approved_quote_mismatch",
        }

    # ---------------------------------------------------------
    # 5. Create or recover atomically.
    #
    # The ticket store performs the final transactional
    # idempotency check to protect against concurrent requests.
    # ---------------------------------------------------------

    ticket = create_or_get_ticket(
        request_id=request_id,
        user=result["user"],
        items=current_items,
        refund_amount=result["refund_amount"],
    )

    return ticket
