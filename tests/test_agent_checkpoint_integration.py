from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from tempfile import TemporaryDirectory

import sys
from pathlib import Path


from agent.objects import Answer


class StructuredAnswerFakeModel(BaseChatModel):
    """Test-only model that returns a deterministic structured tool call."""

    @property
    def _llm_type(self) -> str:
        return "structured-answer-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError(
            "The test must use structured tool binding, not plain generation."
        )

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if len(tools) != 1:
            raise AssertionError(
                "This test model must receive only the Answer schema."
            )

        schema = tools[0]
        schema_name = getattr(
            schema,
            "name",
            getattr(schema, "__name__", None),
        )

        if schema_name != "Answer":
            raise AssertionError(
                f"Unexpected structured-output schema: {schema_name}"
            )

        def return_structured_answer(_input):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Answer",
                        "args": {
                            "answer": "Checkpoint integration succeeded.",
                            "conversation": False,
                        },
                        "id": "answer-test-call-1",
                        "type": "tool_call",
                    }
                ],
            )

        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(return_structured_answer)


def test_create_agent_persists_answer_in_completed_checkpoint():
    checkpointer = InMemorySaver()

    graph = create_agent(
        model=StructuredAnswerFakeModel(),
        tools=[],
        response_format=Answer,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "agent-checkpoint-integration-test",
        }
    }

    response = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Run the checkpoint integration test.")
            ]
        },
        config=config,
    )

    returned_answer = response.get("structured_response")

    assert isinstance(returned_answer, Answer)
    assert returned_answer.answer == "Checkpoint integration succeeded."
    assert returned_answer.conversation is False

    snapshot = graph.get_state(config)

    assert snapshot.next == ()

    saved_answer = (snapshot.values or {}).get("structured_response")

    assert isinstance(saved_answer, Answer)
    assert saved_answer == returned_answer


def test_sqlite_checkpoint_preserves_structured_answer_after_reopen():
    def build_test_agent(checkpointer):
        return create_agent(
            model=StructuredAnswerFakeModel(),
            tools=[],
            response_format=Answer,
            checkpointer=checkpointer,
        )

    config = {
        "configurable": {
            "thread_id": "sqlite-agent-restart-integration-test",
        }
    }

    with TemporaryDirectory() as directory:
        database_path = Path(directory) / "agent_checkpoints.sqlite3"

        # First application lifetime: execute and save the structured answer.
        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            saver.setup()
            first_agent = build_test_agent(saver)

            response = first_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Run the persistent checkpoint test."
                        )
                    ]
                },
                config=config,
            )

            answer = response.get("structured_response")

            assert isinstance(answer, Answer)
            assert answer.answer == "Checkpoint integration succeeded."
            assert answer.conversation is False

            saved = first_agent.get_state(config)
            assert saved.next == ()
            assert saved.values["structured_response"] == answer

        # Second application lifetime: reopen the DB and create a new agent.
        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            saver.setup()
            second_agent = build_test_agent(saver)

            recovered = second_agent.get_state(config)

            assert recovered.next == ()
            recovered_answer = recovered.values["structured_response"]

            assert isinstance(recovered_answer, Answer)
            assert recovered_answer.answer == answer.answer
            assert recovered_answer.conversation == answer.conversation

def test_sqlite_pending_operation_resumes_after_reopen():
    import pytest
    from typing import TypedDict
    from uuid import uuid4

    from agent.conversation_recovery import load_saved_conversation
    from agent.retry import build_agent_input
    from langchain.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph

    class RecoveryState(TypedDict):
        messages: list
        attempts: int

    def build_graph(checkpointer, process_node):
        builder = StateGraph(RecoveryState)
        builder.add_node("process", process_node)
        builder.add_edge(START, "process")
        builder.add_edge("process", END)
        return builder.compile(checkpointer=checkpointer)

    thread_id = str(uuid4())
    request_id = str(uuid4())
    user_text = "Resume this interrupted operation."

    config = {"configurable": {"thread_id": thread_id}}

    with TemporaryDirectory() as directory:
        database_path = Path(directory) / "pending_checkpoints.sqlite3"
        trace_path = Path(directory) / "traces.jsonl"

        # First application lifetime: save the request, then fail.
        def failing_process(state: RecoveryState) -> dict:
            raise RuntimeError("Simulated application interruption")

        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            saver.setup()
            first_graph = build_graph(saver, failing_process)

            with pytest.raises(
                RuntimeError,
                match="Simulated application interruption",
            ):
                first_graph.invoke(
                    {
                        "messages": [
                            HumanMessage(content=user_text, id=request_id)
                        ],
                        "attempts": 0,
                    },
                    config=config,
                )

            failed_snapshot = first_graph.get_state(config)

            assert failed_snapshot.next == ("process",)
            assert len(failed_snapshot.values["messages"]) == 1
            assert failed_snapshot.values["messages"][0].id == request_id

        # Second application lifetime: reopen SQLite and rebuild the graph.
        def successful_process(state: RecoveryState) -> dict:
            return {"attempts": state["attempts"] + 1}

        with SqliteSaver.from_conn_string(str(database_path)) as saver:
            saver.setup()
            recovered_graph = build_graph(saver, successful_process)

            restored = load_saved_conversation(
                recovered_graph,
                thread_id=thread_id,
                trace_file=trace_path,
                welcome_message="Welcome.",
            )

            assert restored["thread_id"] == thread_id
            assert restored["conversation_closed"] is False
            assert restored["pending_operation"]["user_text"] == user_text
            assert restored["pending_operation"]["refund_request_id"] == request_id
            assert restored["pending_operation"]["failed"] is True

            # The retry helper must resume the pending node, not replay input.
            graph_input = build_agent_input(
                recovered_graph,
                config,
                is_retry=True,
                user_text=user_text,
                request_id=request_id,
            )

            assert graph_input is None

            result = recovered_graph.invoke(graph_input, config=config)

            assert result["attempts"] == 1
            assert len(result["messages"]) == 1
            assert result["messages"][0].id == request_id
            assert result["messages"][0].content == user_text

            completed_snapshot = recovered_graph.get_state(config)
            assert completed_snapshot.next == ()

def test_checkpoint_recovers_proposal_until_explicit_approval():
    import sqlite3
    import pytest
    from contextlib import closing
    from typing import TypedDict
    from unittest.mock import patch
    from uuid import uuid4

    from langchain_core.messages import HumanMessage
    from langgraph.graph import END, START, StateGraph

    from agent import ticket_store, tools
    from agent.objects import PurchaseItem

    class ProposalRecoveryState(TypedDict):
        messages: list
        attempts: int
        refund_proposal: dict

    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "checkpoints.sqlite3"
        ticket_path = Path(directory) / "tickets.sqlite3"

        thread_id = str(uuid4())
        request_id = str(uuid4())
        user_text = "Please prepare a refund proposal for Keyboard."
        config = {"configurable": {"thread_id": thread_id}}

        proposal = {
            "username": "user-001",
            "items": [{"name": "Keyboard"}],
        }

        first_execution = {"value": True}

        def process_node(state: ProposalRecoveryState) -> dict:
            if first_execution["value"]:
                first_execution["value"] = False

                # Simulate interruption after proposal preparation but before
                # LangGraph can save the node's successful output.
                raise RuntimeError("Simulated interruption after proposal")

            return {
                "attempts": state["attempts"] + 1,
                "refund_proposal": proposal,
            }

        def build_graph(checkpointer):
            builder = StateGraph(ProposalRecoveryState)
            builder.add_node("process", process_node)
            builder.add_edge(START, "process")
            builder.add_edge("process", END)
            return builder.compile(checkpointer=checkpointer)

        # First application lifetime: proposal processing is interrupted.
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            saver.setup()
            first_graph = build_graph(saver)

            with pytest.raises(
                RuntimeError,
                match="Simulated interruption after proposal",
            ):
                first_graph.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_text,
                                id=request_id,
                            )
                        ],
                        "attempts": 0,
                    },
                    config=config,
                )

            pending = first_graph.get_state(config)
            assert pending.next == ("process",)
            assert len(pending.values["messages"]) == 1

        # Second application lifetime: resume and recover the proposal.
        with SqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            saver.setup()
            recovered_graph = build_graph(saver)

            result = recovered_graph.invoke(None, config=config)

            assert result["attempts"] == 1
            assert result["refund_proposal"] == proposal

            completed = recovered_graph.get_state(config)
            assert completed.next == ()
            assert len(completed.values["messages"]) == 1

        # Graph execution must not even initialize the ticket database.
        assert not ticket_path.exists()

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

        def create_ticket(**kwargs):
            return ticket_store.create_or_get_ticket(
                **kwargs,
                db_path=ticket_path,
            )

        # Ticket issuance occurs only after the separate approval call.
        with (
            patch(
                "agent.tools.validate_refund",
                return_value=validated_quote,
            ) as validate_mock,
            patch(
                "agent.tools.create_or_get_ticket",
                side_effect=create_ticket,
            ) as create_mock,
        ):
            first_result = tools.issue_approved_refund_ticket(
                request_id=request_id,
                user="user-001",
                items=[PurchaseItem(name="Keyboard")],
                approved_quote={
                    "user": "user-001",
                    "items": validated_quote["items"],
                    "refund_amount": "49.99",
                },
            )

            assert first_result["ticket_issued"] is True
            assert first_result["existing"] is False

            retry_result = tools.issue_approved_refund_ticket(
                request_id=request_id,
                user="user-001",
                items=[PurchaseItem(name="Keyboard")],
                approved_quote={
                    "user": "user-001",
                    "items": validated_quote["items"],
                    "refund_amount": "49.99",
                },
            )

            assert retry_result["ticket_issued"] is True
            assert retry_result["existing"] is True
            assert retry_result["ticket_id"] == first_result["ticket_id"]

            assert validate_mock.call_count == 2
            assert create_mock.call_count == 2

        with closing(sqlite3.connect(ticket_path)) as connection:
            ticket_count_after_approval = connection.execute(
                "SELECT COUNT(*) FROM refund_tickets"
            ).fetchone()[0]

        assert ticket_count_after_approval == 1


from typing import TypedDict

import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from agent.objects import PurchaseItem, RefundProposal


def test_sqlite_checkpoint_preserves_nested_refund_proposal_after_reopen():
    class ProposalState(TypedDict):
        structured_response: Answer

    proposal = RefundProposal(
        username="user-001",
        items=[PurchaseItem(name="Keyboard")],
    )
    expected = Answer(
        answer="A refund proposal is ready for review.",
        conversation=True,
        refund_proposal=proposal,
    )

    def save_proposal(_state):
        return {"structured_response": expected}

    def build_graph(checkpointer):
        builder = StateGraph(ProposalState)
        builder.add_node("save", save_proposal)
        builder.add_edge(START, "save")
        builder.add_edge("save", END)
        return builder.compile(checkpointer=checkpointer)

    thread_id = "nested-refund-proposal-reopen-test"
    config = {"configurable": {"thread_id": thread_id}}

    with TemporaryDirectory() as directory:
        database_path = Path(directory) / "nested_proposal.sqlite3"

        # First application lifetime: persist the nested production model.
        connection = sqlite3.connect(str(database_path), check_same_thread=False)
        try:
            serializer = JsonPlusSerializer(
                allowed_msgpack_modules=[("agent.objects", "Answer")]
            )
            saver = SqliteSaver(connection, serde=serializer)
            saver.setup()

            graph = build_graph(saver)
            result = graph.invoke({}, config=config)

            assert result["structured_response"] == expected
            saved = graph.get_state(config)
            assert saved.next == ()
            assert saved.values["structured_response"] == expected
        finally:
            connection.close()

        # Second application lifetime: reopen using the same production
        # serializer allow-list and verify every nested field.
        connection = sqlite3.connect(str(database_path), check_same_thread=False)
        try:
            serializer = JsonPlusSerializer(
                allowed_msgpack_modules=[("agent.objects", "Answer")]
            )
            saver = SqliteSaver(connection, serde=serializer)
            saver.setup()

            graph = build_graph(saver)
            recovered = graph.get_state(config)

            assert recovered.next == ()
            restored = recovered.values["structured_response"]
            assert isinstance(restored, Answer)
            assert restored == expected
            assert isinstance(restored.refund_proposal, RefundProposal)
            assert isinstance(restored.refund_proposal.items[0], PurchaseItem)
            assert restored.refund_proposal.username == "user-001"
            assert restored.refund_proposal.items[0].name == "Keyboard"
        finally:
            connection.close()
