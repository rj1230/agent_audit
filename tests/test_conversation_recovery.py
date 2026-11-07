import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from langchain.messages import HumanMessage

from agent.conversation_recovery import (
    list_saved_conversations,
    load_saved_conversation,
)
from agent.objects import Answer


class FakeAgent:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self.snapshots.get(thread_id)


class FakeCheckpointer:
    def __init__(self, items):
        self.items = items

    def list(self, config, limit=100):
        return iter(self.items[:limit])


def make_snapshot(messages, *, answer=None, next_nodes=(), turn_number=1):
    return SimpleNamespace(
        values={
            "messages": messages,
            "structured_response": answer,
        },
        next=next_nodes,
        metadata={"turn_number": turn_number},
    )


def write_trace(path, thread_id, turn_number, answer, conversation=True):
    record = {
        "thread_id": thread_id,
        "turn_number": turn_number,
        "status": "success",
        "final_response": {
            "answer": answer,
            "conversation": conversation,
        },
    }
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def test_completed_conversation_reconstructs_turns_from_traces(tmp_path):
    thread_id = str(uuid4())
    trace_file = tmp_path / "traces.jsonl"

    write_trace(trace_file, thread_id, 1, "First saved answer.")
    write_trace(trace_file, thread_id, 2, "Second saved answer.", False)

    snapshot = make_snapshot(
        [
            HumanMessage(content="First question", id=str(uuid4())),
            HumanMessage(content="Second question", id=str(uuid4())),
        ],
        answer=Answer(
            answer="Second saved answer.",
            conversation=False,
        ),
        turn_number=2,
    )
    agent = FakeAgent({thread_id: snapshot})

    restored = load_saved_conversation(
        agent,
        thread_id=thread_id,
        trace_file=trace_file,
        welcome_message="Welcome.",
    )

    assert restored["thread_id"] == thread_id
    assert restored["turn_number"] == 2
    assert restored["conversation_closed"] is True
    assert restored["pending_operation"] is None
    assert restored["trace_warning"] is False
    assert restored["messages"] == [
        {"role": "assistant", "content": "Welcome."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First saved answer."},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second saved answer."},
    ]


def test_malformed_trace_turn_number_is_ignored_during_recovery(tmp_path):
    thread_id = str(uuid4())
    trace_file = tmp_path / "traces.jsonl"

    write_trace(trace_file, thread_id, "not-a-number", "Malformed answer.")
    write_trace(trace_file, thread_id, 1, "Valid saved answer.")

    snapshot = make_snapshot(
        [HumanMessage(content="Question", id=str(uuid4()))],
        answer=Answer(answer="Valid saved answer.", conversation=True),
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    restored = load_saved_conversation(
        agent,
        thread_id=thread_id,
        trace_file=trace_file,
        welcome_message="Welcome.",
    )

    assert restored["trace_warning"] is False
    assert len(restored["trace_records"]) == 1
    assert restored["messages"] == [
        {"role": "assistant", "content": "Welcome."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Valid saved answer."},
    ]


def test_interrupted_conversation_preserves_original_request_id(tmp_path):
    thread_id = str(uuid4())
    request_id = str(uuid4())
    trace_file = tmp_path / "traces.jsonl"

    write_trace(trace_file, thread_id, 1, "Earlier completed answer.")

    snapshot = make_snapshot(
        [
            HumanMessage(content="Earlier question", id=str(uuid4())),
            HumanMessage(content="Retry this refund request", id=request_id),
        ],
        # This may be a stale response from an earlier turn. It must not
        # be presented as the result of the interrupted request.
        answer=Answer(answer="Old response.", conversation=True),
        next_nodes=("process",),
        turn_number=2,
    )
    agent = FakeAgent({thread_id: snapshot})

    restored = load_saved_conversation(
        agent,
        thread_id=thread_id,
        trace_file=trace_file,
        welcome_message="Welcome.",
    )

    assert restored["conversation_closed"] is False
    assert restored["pending_operation"] == {
        "user_text": "Retry this refund request",
        "refund_request_id": request_id,
        "turn_number": 2,
        "failed": True,
    }
    assert restored["messages"][-2]["role"] == "user"
    assert restored["messages"][-2]["content"] == "Retry this refund request"
    assert "interrupted" in restored["messages"][-1]["content"].lower()
    assert all(
        message["content"] != "Old response."
        for message in restored["messages"]
    )


def test_list_saved_conversations_returns_latest_entry_per_thread():
    thread_id = str(uuid4())
    snapshot = make_snapshot(
        [HumanMessage(content="Question", id=str(uuid4()))],
        answer=Answer(answer="Answer", conversation=True),
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    older = SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={"ts": "2026-10-30T10:00:00+00:00"},
        metadata={"turn_number": 1},
    )
    newer = SimpleNamespace(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={"ts": "2026-10-31T10:00:00+00:00"},
        metadata={"turn_number": 1},
    )

    results = list_saved_conversations(
        agent,
        FakeCheckpointer([newer, older]),
    )

    assert len(results) == 1
    assert results[0]["thread_id"] == thread_id
    assert results[0]["timestamp"] == "2026-10-31T10:00:00+00:00"
    assert results[0]["turn_number"] == 1


def test_interrupted_conversation_without_uuid_fails_closed(tmp_path):
    thread_id = str(uuid4())
    snapshot = make_snapshot(
        [HumanMessage(content="Uncorrelated request", id="not-a-uuid")],
        next_nodes=("process",),
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    import pytest

    with pytest.raises(RuntimeError, match="valid request ID"):
        load_saved_conversation(
            agent,
            thread_id=thread_id,
            trace_file=tmp_path / "missing.jsonl",
            welcome_message="Welcome.",
        )


def test_completed_refund_proposal_reconstructs_pending_approval(
    tmp_path,
    monkeypatch,
):
    from unittest.mock import patch

    from agent.objects import PurchaseItem, RefundProposal

    thread_id = str(uuid4())
    request_id = str(uuid4())
    trace_file = tmp_path / "traces.jsonl"

    proposal = RefundProposal(
        username="user-001",
        items=[PurchaseItem(name="Keyboard")],
    )
    answer = Answer(
        answer="The refund has been approved and issued.",
        conversation=False,
        refund_proposal=proposal,
    )

    snapshot = make_snapshot(
        [
            HumanMessage(
                content="Please prepare a refund for my Keyboard.",
                id=request_id,
            )
        ],
        answer=answer,
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    validated_quote = {
        "eligible": True,
        "user": "user-001",
        "items": [
            {
                "item": "Keyboard",
                "price": "49.99",
                "purchase_date": "2026-10-20",
            }
        ],
        "refund_amount": "49.99",
    }

    with patch(
        "agent.execution.validate_refund",
        return_value=validated_quote,
    ) as validate_mock:
        restored = load_saved_conversation(
            agent,
            thread_id=thread_id,
            trace_file=trace_file,
            welcome_message="Welcome.",
        )

    validate_mock.assert_called_once()
    approval = restored["pending_approval"]

    assert approval["request_id"] == request_id
    assert approval["user"] == "user-001"
    assert approval["items"] == [{"name": "Keyboard"}]
    assert approval["quote"]["refund_amount"] == "49.99"
    assert restored["pending_operation"] is None
    assert restored["conversation_closed"] is False
    assert "revalidated" in restored["messages"][-1]["content"].lower()
    assert "no ticket has been created" in restored["messages"][-1]["content"].lower()



def test_ineligible_saved_proposal_is_not_restored_as_approval(
    tmp_path,
):
    from unittest.mock import patch

    from agent.objects import PurchaseItem, RefundProposal

    thread_id = str(uuid4())
    request_id = str(uuid4())

    proposal = RefundProposal(
        username="user-001",
        items=[PurchaseItem(name="Keyboard")],
    )
    answer = Answer(
        answer="Your refund has been approved.",
        conversation=False,
        refund_proposal=proposal,
    )
    snapshot = make_snapshot(
        [
            HumanMessage(
                content="Prepare a refund for my Keyboard.",
                id=request_id,
            )
        ],
        answer=answer,
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    rejected_result = {
        "eligible": False,
        "reason": "The purchase is outside the 30-day refund window.",
    }

    with patch(
        "agent.execution.validate_refund",
        return_value=rejected_result,
    ) as validate_mock:
        restored = load_saved_conversation(
            agent,
            thread_id=thread_id,
            trace_file=tmp_path / "traces.jsonl",
            welcome_message="Welcome.",
        )

    validate_mock.assert_called_once()
    assert restored["pending_approval"] is None
    assert restored["pending_operation"] is None
    assert restored["conversation_closed"] is True

    final_message = restored["messages"][-1]["content"].lower()
    assert "outside the 30-day refund window" in final_message
    assert "no ticket was created" in final_message
    assert all(
        "approved" not in message["content"].lower()
        for message in restored["messages"]
        if message["role"] == "assistant"
    )


def test_completed_proposal_without_uuid_fails_closed(
    tmp_path,
):
    import pytest
    from unittest.mock import patch

    from agent.objects import PurchaseItem, RefundProposal

    thread_id = str(uuid4())
    proposal = RefundProposal(
        username="user-001",
        items=[PurchaseItem(name="Keyboard")],
    )
    answer = Answer(
        answer="A refund quote is ready.",
        conversation=True,
        refund_proposal=proposal,
    )
    snapshot = make_snapshot(
        [
            HumanMessage(
                content="Prepare a refund for my Keyboard.",
                id="not-a-uuid",
            )
        ],
        answer=answer,
        turn_number=1,
    )
    agent = FakeAgent({thread_id: snapshot})

    with patch("agent.execution.validate_refund") as validate_mock:
        with pytest.raises(RuntimeError, match="valid request ID"):
            load_saved_conversation(
                agent,
                thread_id=thread_id,
                trace_file=tmp_path / "traces.jsonl",
                welcome_message="Welcome.",
            )

    validate_mock.assert_not_called()
