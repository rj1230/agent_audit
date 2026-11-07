from time import perf_counter
from uuid import uuid4

from agent.objects import PurchaseItem
from agent.tools import validate_refund


def _structured_field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _proposal_to_approval(proposal, request_id):
    """Validate an agent proposal and build an application-owned approval."""
    if proposal is None:
        return None, None

    username = _structured_field(proposal, "username")
    requested_items = _structured_field(proposal, "items")

    if not isinstance(username, str) or not username.strip():
        raise ValueError("Refund proposal has no valid username.")

    if not isinstance(requested_items, list) or not requested_items:
        raise ValueError("Refund proposal has no valid item list.")

    items = []
    for requested_item in requested_items:
        item_name = _structured_field(requested_item, "name")
        if not isinstance(item_name, str) or not item_name.strip():
            raise ValueError("Refund proposal contains an invalid item.")
        items.append(PurchaseItem(name=item_name))

    result = validate_refund(username, items)

    if not result.get("eligible"):
        reason = result.get("reason", "The request did not pass validation.")
        return None, str(reason)

    quote_items = sorted(
        result["items"],
        key=lambda item: item["item"],
    )
    quote = {
        "user": result["user"],
        "items": quote_items,
        "refund_amount": result["refund_amount"],
    }

    approval = {
        "request_id": request_id,
        "user": username,
        "items": [{"name": item.name} for item in items],
        "quote": quote,
    }
    return approval, None


def run_pending_operation(
    *,
    session_state,
    agent,
    callback_factory,
    trace_writer,
    input_builder,
    logger,
    max_conversation_turns,
):
    pending = session_state.pending_operation

    if not isinstance(pending, dict):
        return

    user_text = pending["user_text"]
    request_id = pending["refund_request_id"]
    thread_id = session_state.thread_id
    turn_number = pending["turn_number"]

    error_message = (
        "I encountered a technical issue while processing your request. "
        "The outcome could not be confirmed. You can retry below using "
        "the same request identifier."
    )

    if (
        pending.get("failed")
        and session_state.messages
        and session_state.messages[-1]
        == {"role": "assistant", "content": error_message}
    ):
        session_state.messages.pop()

    is_retry = bool(pending.get("failed"))
    pending["failed"] = False

    callback = callback_factory()
    started = perf_counter()
    response = None
    error = None

    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "turn_number": turn_number,
                "application": "agent-audit-streamlit",
            },
            "callbacks": [callback],
        }

        graph_input = input_builder(
            agent,
            config,
            is_retry=is_retry,
            user_text=user_text,
            request_id=request_id,
        )

        if isinstance(graph_input, dict) and "messages" in graph_input:
            response = agent.invoke(
                graph_input,
                config=config,
                context={"refund_request_id": request_id},
            )
        elif graph_input is None:
            response = agent.invoke(
                None,
                config=config,
                context={"refund_request_id": request_id},
            )
        else:
            response = {"structured_response": graph_input}

        structured = response.get("structured_response")
        if structured is None:
            raise ValueError("Agent did not return a structured response.")

        answer = _structured_field(structured, "answer")
        should_continue = _structured_field(structured, "conversation")
        proposal = _structured_field(structured, "refund_proposal")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Agent returned an invalid answer.")

        if not isinstance(should_continue, bool):
            raise ValueError("The conversation field must be a boolean.")

        approval, validation_error = _proposal_to_approval(
            proposal,
            request_id,
        )

        session_state.pending_approval = approval

        if proposal is None:
            # Ordinary conversation: preserve the agent's response.
            session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        elif validation_error:
            # The deterministic application rejection replaces the model
            # response so an invalid proposal cannot appear successful.
            session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "I couldn't prepare a refund quote: "
                        f"{validation_error}. No ticket was created."
                    ),
                }
            )
        elif approval is not None:
            # The application owns the status message; the model cannot
            # claim that a proposal was approved or a ticket was created.
            session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "The application has prepared a refund quote "
                        "for your review. Please review the quote below "
                        "and explicitly confirm or decline it. "
                        "No ticket has been created."
                    ),
                }
            )

        session_state.pending_operation = None

        if not should_continue and approval is None:
            session_state.conversation_closed = True

    except Exception as exc:
        error = exc
        logger.exception(
            "Agent failed: thread=%s turn=%s",
            thread_id,
            turn_number,
        )
        pending["failed"] = True
        session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )

    elapsed = perf_counter() - started

    try:
        trace_record = trace_writer(
            thread_id=thread_id,
            turn_number=turn_number,
            elapsed_seconds=elapsed,
            tool_callback=callback,
            response=response,
            error=error,
        )

        if not isinstance(trace_record, dict):
            raise TypeError("write_agent_trace must return a dictionary.")

        session_state.trace_records.append(trace_record)

    except Exception:
        logger.exception(
            "Failed to persist trace: thread=%s turn=%s",
            thread_id,
            turn_number,
        )
        session_state.trace_warning = True
        session_state.trace_records.append(
            {
                "trace_id": f"local-{uuid4()}",
                "turn_number": turn_number,
                "execution_time_seconds": elapsed,
                "status": "error" if error else "completed",
                "tool_call_count": 0,
                "successful_tool_count": 0,
                "failed_tool_count": 0,
                "tool_events": [],
                "trace_unavailable": True,
            }
        )

    if (
        session_state.pending_operation is None
        and session_state.pending_approval is None
        and turn_number >= max_conversation_turns
    ):
        session_state.conversation_closed = True
