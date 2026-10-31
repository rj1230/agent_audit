import logging
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from models import model
from tools import (
    calculate_right_refund,
    get_today_date,
    get_user_purchase_history,
    issue_refund_ticket,
)
from prompts import agent_system_prompt
from objects import Answer
from tracing import ToolTraceCallback, write_agent_trace


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 8


agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    tools=[
        get_user_purchase_history,
        calculate_right_refund,
        issue_refund_ticket,
        get_today_date,
    ],
    response_format=Answer,
    checkpointer=InMemorySaver(),
)


def run_conversation(initial_message=None):
    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "thread_id": thread_id,
            "application": "refund-support-agent",
        },
    }

    print("\n========== REFUND SUPPORT AGENT ==========")
    print("Conversation ID:", thread_id)
    print("Type 'exit' to end the conversation.")
    print("===========================================\n")

    user_text = initial_message

    if not user_text:
        user_text = input("How can I help you today?\n> ").strip()

    for turn_number in range(1, MAX_CONVERSATION_TURNS + 1):
        if not user_text:
            print("Please enter a message.")
            user_text = input("> ").strip()
            continue

        if user_text.lower() in {"exit", "quit"}:
            print("Conversation ended.")
            return

        tool_callback = ToolTraceCallback()
        started = perf_counter()

        response = None
        error = None

        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={
                    **config,
                    "callbacks": [tool_callback],
                },
            )

            structured_response = response.get("structured_response")

            if not isinstance(structured_response, Answer):
                raise TypeError("Agent did not return the expected Answer structure.")

        except Exception as exc:
            error = exc

            logger.exception(
                "Agent execution failed. thread=%s turn=%s",
                thread_id,
                turn_number,
            )

        elapsed = perf_counter() - started

        write_agent_trace(
            thread_id=thread_id,
            turn_number=turn_number,
            elapsed_seconds=elapsed,
            tool_callback=tool_callback,
            response=response,
            error=error,
        )

        if error:
            print(
                "\nI'm unable to complete this request right now. "
                "No successful refund decision should be assumed."
            )
            return

        print("\nSupport Agent:")
        print(structured_response.answer)

        if not structured_response.conversation:
            print("\nConversation completed.")
            return

        if turn_number == MAX_CONVERSATION_TURNS:
            print(
                "\nMaximum conversation length reached. "
                "Please start a new support conversation if needed."
            )
            return

        user_text = input("\nYou:\n> ").strip()

    print("\nConversation ended.")


if __name__ == "__main__":
    run_conversation()
