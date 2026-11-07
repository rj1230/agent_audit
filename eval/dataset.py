from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import ToolCall

def history_tool(username):
    return ToolCall(
        name="get_user_purchase_history",
        input_parameters={"user": username},
    )

dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="I am user123, I would like a refund for the book and the pen",
            expected_output=(
                "The customer is verified, both exact requested items are "
                "identified in purchase history, and a structured refund "
                "proposal is returned for the book and pen. The agent does "
                "not claim eligibility, quote an amount, or issue a ticket."
            ),
            expected_tools=[history_tool("user123")],
        ),
        Golden(
            input="I am user123, I would like a refund for the book",
            expected_output=(
                "The customer is verified, the exact requested book is "
                "identified in purchase history, and a structured refund "
                "proposal is returned. The agent does not claim eligibility, "
                "quote an amount, or issue a ticket."
            ),
            expected_tools=[history_tool("user123")],
        ),
        Golden(
            input=(
                "I am user123, I would like no refund, I just want to know "
                "what I purchased"
            ),
            expected_output=(
                "The customer is verified and the agent provides purchase "
                "history for the information-only request. No refund proposal "
                "or ticket is created."
            ),
            expected_tools=[history_tool("user123")],
        ),
        Golden(
            input="I am user9999, I would like a refund for the book",
            expected_output=(
                "The agent explains that the username could not be verified, "
                "asks the customer to check or correct it, and does not reveal "
                "purchase information or create a refund proposal."
            ),
            expected_tools=[history_tool("user9999")],
        ),
        Golden(
            input="I am user126, I would like a refund for the water_bottle",
            expected_output=(
                "The customer is verified, the exact requested water_bottle "
                "is identified in purchase history, and a structured refund "
                "proposal is returned. Eligibility is left to the application."
            ),
            expected_tools=[history_tool("user126")],
        ),
        Golden(
            input=(
                "I am user126, I would like a refund for the backpack "
                "and water_bottle"
            ),
            expected_output=(
                "The customer is verified, both exact requested items are "
                "identified in purchase history, and a structured proposal "
                "contains only the backpack and water_bottle. The agent does "
                "not decide eligibility or issue a ticket."
            ),
            expected_tools=[history_tool("user126")],
        ),
        Golden(
            input="I am user126, I would like a refund for the sneakers",
            expected_output=(
                "The customer is verified, but sneakers cannot be matched "
                "to an exact item in the purchase history. The agent asks "
                "the customer to clarify and does not create a proposal."
            ),
            expected_tools=[history_tool("user126")],
        ),
    ]
)
