import logging
from time import perf_counter
from uuid import uuid4
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from agent.two_stage import agent
from agent.approval import (
    confirm_pending_approval,
    decline_pending_approval,
)

from agent.objects import Answer
from agent.execution import _proposal_to_approval
from agent.tools import issue_approved_refund_ticket
from agent.tracing import ToolTraceCallback, write_agent_trace


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 8




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

        refund_request_id = str(uuid4())

        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={
                    **config,
                    "callbacks": [tool_callback],
                },
                context={
                    "refund_request_id": refund_request_id,
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

        proposal = structured_response.refund_proposal

        if proposal is None:
            print("\nSupport Agent:")
            print(structured_response.answer)

        cli_state = None

        if proposal is not None:
            try:
                approval, validation_error = _proposal_to_approval(
                    proposal,
                    refund_request_id,
                )

                if validation_error:
                    print(
                        "\nI couldn't prepare a valid refund quote: "
                        f"{validation_error}. No ticket was created."
                    )
                elif approval is not None:
                    print("\nSupport Agent:")
                    print(
                        "The application has prepared a refund quote "
                        "for your review. No ticket has been created."
                    )
                    quote = approval["quote"]

                    print("\n========== REFUND QUOTE ==========")
                    print(f"Customer: {quote['user']}")

                    for item in quote["items"]:
                        print(
                            f"- {item['item']}: "
                            f"{item['price']} "
                            f"(purchased {item['purchase_date']})"
                        )

                    print(f"Total refund: {quote['refund_amount']}")
                    print("Simulation only; no money will be transferred.")

                    confirmation = input(
                        "\nConfirm this exact refund quote? (y/N): "
                    ).strip().lower()

                    cli_state = SimpleNamespace(
                        pending_approval=approval,
                        messages=[],
                        conversation_closed=False,
                    )

                    if confirmation == "y":
                        while cli_state.pending_approval is not None:
                            previous_message_count = len(cli_state.messages)

                            confirm_pending_approval(
                                cli_state,
                                confirmed=True,
                                ticket_issuer=issue_approved_refund_ticket,
                                logger=logger,
                            )

                            if len(cli_state.messages) > previous_message_count:
                                message = cli_state.messages[-1]["content"].replace(
                                    "No money has been transferred.",
                                    "No money was transferred.",
                                )
                                print("\n" + message)

                            if cli_state.pending_approval is None:
                                break

                            retry = input(
                                "\\nThe ticket outcome is unresolved. "
                                "Retry the same request and quote? (y/N): "
                            ).strip().lower()

                            if retry != "y":
                                print(
                                    "\\nApproval remains unresolved. "
                                    "Ending this CLI session without "
                                    "starting another request."
                                )
                                return
                    else:
                        decline_pending_approval(cli_state)
                        print(
                            "\\nRefund quote declined. No ticket was created."
                        )

            except Exception:
                logger.exception(
                    "CLI approval processing failed. thread=%s turn=%s",
                    thread_id,
                    turn_number,
                )
                print(
                    "\nI couldn't complete the approval step. "
                    "No new ticket should be assumed."
                )

        if cli_state is not None and cli_state.conversation_closed:
            print("\nConversation completed.")
            return

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
