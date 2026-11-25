from pydantic import BaseModel, ConfigDict, Field


class PurchaseItem(BaseModel):
    name: str = Field(description="Exact name of an item purchased by the customer")


class RefundProposal(BaseModel):
    username: str = Field(
        min_length=1, description="Customer username identified during the conversation"
    )
    items: list[PurchaseItem] = Field(
        min_length=1, description="Exact items the customer has requested to refund"
    )


class Answer(BaseModel):
    answer: str = Field(description="The response that should be shown to the customer")
    conversation: bool = Field(
        description=(
            "Whether the conversation should continue. True when more "
            "information or discussion is needed; False when the interaction "
            "is complete."
        ),
        default=True,
    )
    refund_proposal: RefundProposal | None = Field(
        default=None,
        description=(
            "Structured customer refund request when the username and exact "
            "requested items have been identified. This is a proposal only, "
            "not an eligibility decision, refund quote, or authorization."
        ),
    )


class StrictPurchaseItem(BaseModel):
    """Closed-schema item used only by strict structured output."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Exact name of an item purchased by the customer."
    )


class StrictRefundProposal(BaseModel):
    """Closed-schema proposal used only by strict structured output."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(
        description="Verified customer username identified during the conversation."
    )
    items: list[StrictPurchaseItem] = Field(
        description="Exact items the customer has requested to refund."
    )

class FinalAnswerOutput(BaseModel):
    """Strict provider-compatible output used by the tool-free finalizer."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="The response that should be shown to the customer."
    )
    conversation: bool = Field(
        description="Whether the conversation should continue."
    )
    refund_proposal: StrictRefundProposal | None = Field(
        description=(
            "A refund request proposal only when the username and exact "
            "requested items are established; otherwise null."
        )
    )
