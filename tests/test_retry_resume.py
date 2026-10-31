from typing import TypedDict

import pytest
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(AGENT_DIR))

from retry import build_agent_input


class State(TypedDict):
    messages: list
    attempts: int


def test_retry_resumes_checkpoint_without_duplicate_user_message():
    attempts = {"count": 0}

    def process(state: State) -> dict:
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise RuntimeError("Simulated transient failure")

        return {"attempts": attempts["count"]}

    builder = StateGraph(State)
    builder.add_node("process", process)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)

    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "retry-regression-test"}}
    request_id = "retry-request-1"
    original_message = HumanMessage(
        content="Check my refund",
        id=request_id,
    )

    with pytest.raises(RuntimeError, match="Simulated transient failure"):
        graph.invoke(
            {"messages": [original_message], "attempts": 0},
            config=config,
        )

    failed_snapshot = graph.get_state(config)

    assert failed_snapshot.next == ("process",)
    assert len(failed_snapshot.values["messages"]) == 1
    assert failed_snapshot.values["messages"][0].content == "Check my refund"

    # Exercise the same retry-input helper used by the Streamlit UI.
    graph_input = build_agent_input(
        graph,
        config,
        is_retry=True,
        user_text="Check my refund",
        request_id=request_id,
    )
    assert graph_input is None

    result = graph.invoke(graph_input, config=config)

    assert attempts["count"] == 2
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "Check my refund"
    assert result["attempts"] == 2



def test_retry_rejects_missing_checkpoint():
    class AgentWithoutCheckpoint:
        def get_state(self, config):
            return None

    with pytest.raises(RuntimeError, match="Refusing to replay"):
        build_agent_input(
            AgentWithoutCheckpoint(),
            {"configurable": {"thread_id": "missing-checkpoint"}},
            is_retry=True,
            user_text="Check my refund",
        )


def test_first_attempt_submits_original_message():
    class AgentWithoutCheckpoint:
        def get_state(self, config):
            raise AssertionError("First attempt must not inspect a checkpoint")

    graph_input = build_agent_input(
        AgentWithoutCheckpoint(),
        {},
        is_retry=False,
        user_text="Check my refund",
    )

    assert len(graph_input["messages"]) == 1
    assert graph_input["messages"][0].content == "Check my refund"


def test_retry_recovers_completed_checkpoint_dict():
    saved = {
        "answer": "Your simulated refund ticket has been created.",
        "conversation": True,
    }

    class AgentWithCompletedCheckpoint:
        def get_state(self, config):
            from types import SimpleNamespace

            return SimpleNamespace(
                next=(),
                values={"structured_response": saved},
            )

    result = build_agent_input(
        AgentWithCompletedCheckpoint(),
        {"configurable": {"thread_id": "completed-dict-test"}},
        is_retry=True,
        user_text="Check my refund",
    )

    assert result is saved
    assert result["answer"] == saved["answer"]
    assert result["conversation"] is True


def test_retry_recovers_completed_checkpoint_object():
    from types import SimpleNamespace
    from objects import Answer

    saved = Answer(
        answer="Your simulated refund ticket has been created.",
        conversation=False,
    )

    class AgentWithCompletedCheckpoint:
        def get_state(self, config):
            return SimpleNamespace(
                next=(),
                values={"structured_response": saved},
            )

    result = build_agent_input(
        AgentWithCompletedCheckpoint(),
        {"configurable": {"thread_id": "completed-object-test"}},
        is_retry=True,
        user_text="Check my refund",
    )

    assert result is saved
    assert result.answer == saved.answer
    assert result.conversation is False


def test_retry_rejects_completed_checkpoint_with_invalid_answer():
    from types import SimpleNamespace

    saved = {
        "answer": "   ",
        "conversation": True,
    }

    class AgentWithCompletedCheckpoint:
        def get_state(self, config):
            return SimpleNamespace(
                next=(),
                values={"structured_response": saved},
            )

    with pytest.raises(RuntimeError, match="valid structured answer"):
        build_agent_input(
            AgentWithCompletedCheckpoint(),
            {"configurable": {"thread_id": "invalid-answer-test"}},
            is_retry=True,
            user_text="Check my refund",
        )


def test_retry_rejects_completed_checkpoint_with_invalid_conversation():
    from types import SimpleNamespace

    saved = {
        "answer": "Your ticket status is available.",
        "conversation": "yes",
    }

    class AgentWithCompletedCheckpoint:
        def get_state(self, config):
            return SimpleNamespace(
                next=(),
                values={"structured_response": saved},
            )

    with pytest.raises(RuntimeError, match="invalid conversation field"):
        build_agent_input(
            AgentWithCompletedCheckpoint(),
            {"configurable": {"thread_id": "invalid-conversation-test"}},
            is_retry=True,
            user_text="Check my refund",
        )


def test_retry_rejects_completed_checkpoint_without_saved_response():
    from types import SimpleNamespace

    class AgentWithCompletedCheckpoint:
        def get_state(self, config):
            return SimpleNamespace(next=(), values={})

    with pytest.raises(RuntimeError, match="no recoverable structured response"):
        build_agent_input(
            AgentWithCompletedCheckpoint(),
            {"configurable": {"thread_id": "missing-response-test"}},
            is_retry=True,
            user_text="Check my refund",
        )
def test_retry_rejects_stale_completed_checkpoint():
    from types import SimpleNamespace

    old_message = HumanMessage(
        content="Check my refund",
        id="previous-request-id",
    )
    saved = {
        "answer": "This answer belongs to an earlier request.",
        "conversation": True,
    }

    class AgentWithStaleCheckpoint:
        def get_state(self, config):
            return SimpleNamespace(
                next=(),
                values={
                    "messages": [old_message],
                    "structured_response": saved,
                },
            )

    with pytest.raises(RuntimeError, match="does not match the pending request"):
        build_agent_input(
            AgentWithStaleCheckpoint(),
            {"configurable": {"thread_id": "stale-checkpoint-test"}},
            is_retry=True,
            user_text="Check my refund",
            request_id="new-request-id",
        )
