import json
import logging

from pathlib import Path
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from langchain.tools import ToolRuntime, tool
from dotenv import load_dotenv

from agent_context import AgentContext
from ticket_store import create_or_get_ticket, get_existing_ticket

from objects import PurchaseItem

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
    """Retrieve purchase history for a customer username."""

    logging.info("Retrieving purchase history for %s", user)

    if user not in users_data:
        return {"status": "invalid_username"}

    return users_data[user]


@tool
def calculate_right_refund(
    user: str,
    items: list[PurchaseItem],
) -> dict:
    """Validate requested purchases and calculate the eligible refund amount."""

    logging.info("Calculating refund for %s", user)

    result = validate_refund(user, items)

    logging.info("Refund calculation result: %s", result)

    return result


@tool
def issue_refund_ticket(
    user: str,
    items: list[PurchaseItem],
    runtime: ToolRuntime[AgentContext],
) -> dict:
    """Revalidate eligibility and issue an idempotent simulated refund ticket."""

    logging.info("Processing refund ticket for %s", user)

    context = runtime.context
    request_id = (
        context.get("refund_request_id")
        if isinstance(context, dict)
        else None
    )

    if not isinstance(request_id, str) or not request_id.strip():
        logging.error("Refund ticket rejected: missing request context.")
        return {
            "ticket_issued": False,
            "reason": "missing_request_id",
        }

    # Recover an already-issued ticket before checking today's eligibility.
    # This allows a legitimate retry after the original refund window expires.
    existing = get_existing_ticket(
        request_id=request_id,
        user=user,
        item_names=[item.name for item in items],
    )

    if existing is not None:
        return existing

    # New tickets must always pass current deterministic eligibility checks.
    result = validate_refund(user, items)

    if not result["eligible"]:
        return {
            "ticket_issued": False,
            "reason": result["reason"],
        }

    # The store performs a second idempotency check inside its write
    # transaction, protecting against concurrent requests.
    return create_or_get_ticket(
        request_id=request_id,
        user=user,
        items=result["items"],
        refund_amount=result["refund_amount"],
    )


@tool
def get_today_date() -> str:
    """Return today's date in YYYY-MM-DD format."""

    return date.today().isoformat()
