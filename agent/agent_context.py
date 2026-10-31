from typing import TypedDict


class AgentContext(TypedDict):
    """Trusted application context for one refund operation."""

    refund_request_id: str
