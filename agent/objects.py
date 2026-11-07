from pydantic import BaseModel, Field


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
