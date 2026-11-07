"""Safe reconstruction of saved AgentAudit conversations."""

import json
from pathlib import Path
from uuid import UUID

from langchain.messages import HumanMessage

from agent.execution import _proposal_to_approval


RETRY_NOTICE = (
    "This request was interrupted before its outcome could be confirmed. "
    "You can retry the original request below. The original request ID is "
    "preserved so an already-issued simulated ticket will not be duplicated."
)


def _answer_fields(value):
    """Return a validated (answer, conversation) pair or None."""
    if isinstance(value, dict):
        answer = value.get("answer")
        conversation = value.get("conversation")
    else:
        answer = getattr(value, "answer", None)
        conversation = getattr(value, "conversation", None)

    if not isinstance(answer, str) or not answer.strip():
        return None
    if not isinstance(conversation, bool):
        return None

    return answer, conversation


def _read_thread_traces(trace_file, thread_id):
    path = Path(trace_file)
    if not path.is_file():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue

            if (
                isinstance(record, dict)
                and record.get("thread_id") == thread_id
            ):
                try:
                    int(record.get("turn_number"))
                except (TypeError, ValueError):
                    continue
                records.append(record)

    return sorted(
        records,
        key=lambda record: int(record["turn_number"]),
    )


def _metadata_turn_number(metadata, human_count):
    value = (metadata or {}).get("turn_number")
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = human_count

    return max(value, human_count, 0)


def list_saved_conversations(agent, checkpointer, limit=100):
    """Return one metadata-only entry per saved thread, newest first."""
    latest_by_thread = {}

    # Exhaust the SQLite checkpoint iterator before calling agent.get_state().
    # This prevents nested access while the checkpointer cursor is active.
    checkpoint_items = list(checkpointer.list(None, limit=limit))

    for item in checkpoint_items:
        configurable = (item.config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id")

        if not isinstance(thread_id, str) or not thread_id:
            continue
        if thread_id in latest_by_thread:
            continue

        config = {"configurable": {"thread_id": thread_id}}
        snapshot = agent.get_state(config)
        if snapshot is None:
            continue

        values = snapshot.values or {}
        human_count = sum(
            isinstance(message, HumanMessage)
            for message in values.get("messages", [])
        )

        metadata = item.metadata or {}
        checkpoint = item.checkpoint or {}

        latest_by_thread[thread_id] = {
            "thread_id": thread_id,
            "timestamp": checkpoint.get("ts", ""),
            "turn_number": _metadata_turn_number(metadata, human_count),
            "pending": bool(snapshot.next),
        }

    return sorted(
        latest_by_thread.values(),
        key=lambda item: str(item.get("timestamp") or ""),
        reverse=True,
    )


def load_saved_conversation(
    agent,
    *,
    thread_id,
    trace_file,
    welcome_message,
):
    """Rebuild UI state without invoking the graph or any agent tools."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid conversation ID is required.")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent.get_state(config)

    if snapshot is None:
        raise RuntimeError("No saved checkpoint exists for this conversation.")

    values = snapshot.values or {}
    raw_messages = values.get("messages", [])

    human_messages = [
        message
        for message in raw_messages
        if isinstance(message, HumanMessage)
        and isinstance(message.content, str)
        and message.content.strip()
    ]

    if not human_messages:
        raise RuntimeError("The saved conversation has no recoverable user messages.")

    pending_nodes = tuple(snapshot.next or ())
    is_pending = bool(pending_nodes)

    metadata = getattr(snapshot, "metadata", {}) or {}
    turn_number = _metadata_turn_number(metadata, len(human_messages))

    traces = _read_thread_traces(trace_file, thread_id)
    traces_by_turn = {}

    for record in traces:
        try:
            number = int(record.get("turn_number"))
        except (TypeError, ValueError):
            continue
        traces_by_turn[number] = record

    saved_structured = values.get("structured_response")
    saved_answer = _answer_fields(saved_structured)
    saved_proposal = (
        saved_structured.get("refund_proposal")
        if isinstance(saved_structured, dict)
        else getattr(saved_structured, "refund_proposal", None)
    )

    messages = [
        {"role": "assistant", "content": welcome_message}
    ]

    for index, human_message in enumerate(human_messages, start=1):
        messages.append(
            {"role": "user", "content": human_message.content}
        )

        record = traces_by_turn.get(index)
        trace_answer = None

        if isinstance(record, dict) and record.get("status") == "success":
            trace_answer = _answer_fields(record.get("final_response"))

        is_latest_saved_proposal = (
            index == len(human_messages)
            and not is_pending
            and saved_proposal is not None
        )

        if trace_answer is not None and not is_latest_saved_proposal:
            messages.append(
                {"role": "assistant", "content": trace_answer[0]}
            )
        elif (
            index == len(human_messages)
            and not is_pending
            and saved_proposal is None
        ):
            # The latest structured response is a safe fallback when its
            # corresponding trace record is absent or incomplete.
            saved_answer = _answer_fields(
                values.get("structured_response")
            )
            if saved_answer is not None:
                messages.append(
                    {"role": "assistant", "content": saved_answer[0]}
                )

    pending_operation = None
    pending_approval = None


    if is_pending:
        last_human = human_messages[-1]
        request_id = getattr(last_human, "id", None)

        try:
            UUID(str(request_id))
        except (TypeError, ValueError, AttributeError):
            raise RuntimeError(
                "The interrupted request has no valid request ID. "
                "Refusing to retry it automatically."
            ) from None

        if not last_human.content.strip():
            raise RuntimeError("The interrupted request has no recoverable text.")

        pending_turn = turn_number
        if pending_turn < 1:
            pending_turn = len(human_messages)

        pending_operation = {
            "user_text": last_human.content,
            "refund_request_id": str(request_id),
            "turn_number": pending_turn,
            "failed": True,
        }

        messages.append(
            {"role": "assistant", "content": RETRY_NOTICE}
        )

        conversation_closed = False
    else:
        if saved_answer is None:
            raise RuntimeError(
                "The completed checkpoint has no valid structured response."
            )

        conversation_closed = not saved_answer[1]

        if saved_proposal is not None:
            last_human = human_messages[-1]
            request_id = getattr(last_human, "id", None)

            try:
                request_id = str(UUID(str(request_id)))
            except (TypeError, ValueError, AttributeError):
                raise RuntimeError(
                    "The saved refund proposal has no valid request ID. "
                    "Refusing to reconstruct an approval."
                ) from None

            try:
                pending_approval, validation_error = _proposal_to_approval(
                    saved_proposal,
                    request_id,
                )
            except ValueError:
                pending_approval = None
                validation_error = (
                    "The saved refund proposal is invalid and cannot be "
                    "restored for approval."
                )

            if validation_error:
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I couldn't restore the refund quote: "
                            f"{validation_error} No ticket was created."
                        ),
                    }
                )
                pending_approval = None
            elif pending_approval is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "The application has revalidated the saved "
                            "refund quote for your review. Please review "
                            "the quote and explicitly confirm or decline it. "
                            "No ticket has been created."
                        ),
                    }
                )
                conversation_closed = False

    return {
        "thread_id": thread_id,
        "turn_number": turn_number,
        "messages": messages,
        "trace_records": traces,
        "conversation_closed": conversation_closed,
        "trace_warning": len(traces) < turn_number,
        "pending_operation": pending_operation,
        "pending_approval": pending_approval,
        "pending_prompt": None,
    }