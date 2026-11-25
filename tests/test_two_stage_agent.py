from unittest.mock import Mock

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from agent.objects import FinalAnswerOutput, StrictRefundProposal, StrictPurchaseItem
from agent import two_stage


def make_tool_state(content, *, status="success", username="user123"):
    tool_call_id = "lookup-call-1"

    assistant_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_user_purchase_history",
                "args": {"user": username},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )

    tool_message = ToolMessage(
        content=content,
        name="get_user_purchase_history",
        tool_call_id=tool_call_id,
        status=status,
    )

    return {
        "messages": [assistant_message, tool_message],
    }


def test_record_tool_result_accepts_valid_purchase_history():
    history = {
        "book": {"date": "2026-10-05", "price": 23},
        "pen": {"date": "2026-10-06", "price": 5},
    }

    result = two_stage._record_tool_result(
        make_tool_state(history)
    )

    assert result["lookup_status"] == "verified"
    assert result["verified_username"] == "user123"
    assert result["purchase_history"] == history


def test_record_tool_result_rejects_invalid_username():
    result = two_stage._record_tool_result(
        make_tool_state({"status": "invalid_username"})
    )

    assert result["lookup_status"] == "invalid_username"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


def test_record_tool_result_rejects_malformed_json():
    result = two_stage._record_tool_result(
        make_tool_state("{not valid json")
    )

    assert result["lookup_status"] == "malformed"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


def test_record_tool_result_rejects_tool_error_even_if_content_looks_valid():
    history = {
        "book": {"date": "2026-10-05", "price": 23},
    }

    result = two_stage._record_tool_result(
        make_tool_state(history, status="error")
    )

    assert result["lookup_status"] != "verified"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


def test_record_tool_result_rejects_non_string_item_keys():
    result = two_stage._record_tool_result(
        make_tool_state({123: {"price": 10}})
    )

    assert result["lookup_status"] == "malformed"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


def test_finalizer_rejects_unrequested_item_and_replaces_claim(
    monkeypatch,
):
    parsed = FinalAnswerOutput(
        answer="Your refund has been submitted successfully.",
        conversation=True,
        refund_proposal=StrictRefundProposal(
            username="user123",
            items=[StrictPurchaseItem(name="book")],
        ),
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }

    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [
            HumanMessage(content="I have a question about my book."),
        ],
        "lookup_status": "verified",
        "verified_username": "user123",
        "purchase_history": {
            "book": {"date": "2026-10-05", "price": 23},
        },
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is None
    assert "submitted successfully" not in answer.answer.lower()
    assert "no ticket was created" in answer.answer.lower()


def test_finalizer_preserves_verified_items_when_replacing_success_claim(
    monkeypatch,
):
    parsed = FinalAnswerOutput(
        answer="Your refund has been approved and your ticket has been created.",
        conversation=False,
        refund_proposal=StrictRefundProposal(
            username="user123",
            items=[StrictPurchaseItem(name="book")],
        ),
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [
            HumanMessage(
                content="I'm user123 and I want a refund for my book."
            ),
        ],
        "lookup_status": "verified",
        "verified_username": "user123",
        "purchase_history": {
            "book": {"date": "2026-10-05", "price": 23},
        },
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is not None
    assert answer.refund_proposal.username == "user123"
    assert [item.name for item in answer.refund_proposal.items] == ["book"]
    assert "refund proposal for book" in answer.answer.lower()
    assert "not been submitted or approved" in answer.answer.lower()
    assert "explicit confirmation" in answer.answer.lower()
    assert "ticket has been created" not in answer.answer.lower()
    assert not two_stage._contains_unsupported_success_claim(answer.answer)

def test_finalizer_replaces_success_claim_without_proposal(monkeypatch):
    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="Your refund has been approved and your ticket has been created."
    )
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [HumanMessage(content="Where is my refund?")],
        "lookup_status": "not_attempted",
        "verified_username": None,
        "purchase_history": None,
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is None
    assert "can't confirm" in answer.answer.lower()
    assert "no refund action has been completed" in answer.answer.lower()


@pytest.mark.parametrize(
    "answer",
    [
        "No ticket has been created.",
        "Your refund has not been approved.",
        "I cannot confirm that your refund was processed.",
        "I have not created a ticket.",
        "How can I help you with your purchase?",
        "No refund request has been submitted.",
        "I can't confirm that your refund request was submitted.",
        "A refund proposal is ready for application review.",
        "Your refund request can be submitted after you confirm.",
        "Your request will not be forwarded for review.",
        "I cannot confirm that your request will be forwarded.",
        "Your request can be forwarded after you confirm.",
        "Your request will be forwarded after you confirm.",
        "Your refund request will be submitted once you approve.",
        "Your refund request has not been submitted.",
        "Your refund for the book has not been submitted.",
        "I have not forwarded your request.",
    ],
)
def test_success_claim_guard_preserves_negations_and_normal_answers(answer):
    assert not two_stage._contains_unsupported_success_claim(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Your refund has been approved.",
        "The refund was processed.",
        "Your refund is issued.",
        "Your ticket has been created.",
        "The ticket was submitted.",
        "Your refund has been submitted.",
        "The refund request was submitted.",
        "Your refund request has been submitted.",
        "Your refund request for water_bottle has been submitted for review.",
        "Your refund for the book and pen has been forwarded for review.",
        "Your refund for the book has been submitted.",
        "Your request will now be forwarded for review.",
        "Your request has been forwarded for review.",
        "I have forwarded your request for review.",
        "We will submit your refund request for review.",
        "The system has sent your refund request.",
    ],
)
def test_success_claim_guard_detects_affirmative_completion(answer):
    assert two_stage._contains_unsupported_success_claim(answer)

def test_finalizer_rejects_unverified_username(monkeypatch):
    parsed = FinalAnswerOutput(
        answer="A refund proposal is ready.",
        conversation=True,
        refund_proposal=StrictRefundProposal(
            username="different-user",
            items=[StrictPurchaseItem(name="book")],
        ),
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }

    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [
            HumanMessage(content="Please refund my book."),
        ],
        "lookup_status": "verified",
        "verified_username": "user123",
        "purchase_history": {
            "book": {"date": "2026-10-05", "price": 23},
        },
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is None
    assert answer.answer != parsed.answer
    assert "couldn't verify" in answer.answer.lower()
    assert "no ticket was created" in answer.answer.lower()
from langgraph.checkpoint.memory import InMemorySaver


def make_finalizer_result(answer="How can I help you?", proposal=None):
    parsed = FinalAnswerOutput(
        answer=answer,
        conversation=True,
        refund_proposal=proposal,
    )
    return {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }


def invoke_test_graph(graph, messages, thread_id):
    return graph.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
        context={"refund_request_id": f"test-{thread_id}"},
    )


def test_graph_normal_conversation_skips_tools(monkeypatch):
    conversation_mock = Mock()
    conversation_mock.invoke.return_value = AIMessage(
        content="Hello. How can I help with your purchase?"
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="Hello. How can I help with your purchase?"
    )

    monkeypatch.setattr(two_stage, "conversation_model", conversation_mock)
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    graph = two_stage.build_two_stage_graph(
        checkpointer_override=InMemorySaver()
    )

    result = invoke_test_graph(
        graph,
        [HumanMessage(content="Hello.")],
        "normal-conversation",
    )

    conversation_mock.invoke.assert_called_once()
    finalizer_mock.invoke.assert_called_once()

    answer = result["structured_response"]
    assert isinstance(answer, two_stage.Answer)
    assert answer.refund_proposal is None


def test_graph_executes_real_lookup_tool_and_resumes_conversation(
    monkeypatch,
):
    tool_call_id = "graph-lookup-1"

    lookup_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_user_purchase_history",
                "args": {"user": "user123"},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )

    conversation_done = AIMessage(
        content="I found your purchase history. Which item would you like help with?"
    )

    conversation_mock = Mock()
    conversation_mock.invoke.side_effect = [
        lookup_call,
        conversation_done,
    ]

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="I found your purchase history. Which item would you like help with?"
    )

    monkeypatch.setattr(two_stage, "conversation_model", conversation_mock)
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    graph = two_stage.build_two_stage_graph(
        checkpointer_override=InMemorySaver()
    )

    result = invoke_test_graph(
        graph,
        [HumanMessage(content="My username is user123.")],
        "successful-real-lookup",
    )

    assert conversation_mock.invoke.call_count == 2
    finalizer_mock.invoke.assert_called_once()

    answer = result["structured_response"]
    assert isinstance(answer, two_stage.Answer)
    assert answer.refund_proposal is None

    finalizer_messages = finalizer_mock.invoke.call_args.args[0]
    trusted_context_message = finalizer_messages[0]

    assert "verified" in trusted_context_message.content
    assert "user123" in trusted_context_message.content
    assert "book" in trusted_context_message.content


def test_graph_invalid_username_never_becomes_verified(monkeypatch):
    tool_call_id = "graph-invalid-lookup-1"

    lookup_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_user_purchase_history",
                "args": {"user": "definitely_not_a_real_user"},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )

    conversation_done = AIMessage(
        content="I could not find that username. Please check it."
    )

    conversation_mock = Mock()
    conversation_mock.invoke.side_effect = [
        lookup_call,
        conversation_done,
    ]

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="I could not verify that username. Please check it."
    )

    monkeypatch.setattr(two_stage, "conversation_model", conversation_mock)
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    graph = two_stage.build_two_stage_graph(
        checkpointer_override=InMemorySaver()
    )

    result = invoke_test_graph(
        graph,
        [HumanMessage(content="My username is definitely_not_a_real_user.")],
        "invalid-real-lookup",
    )

    assert conversation_mock.invoke.call_count == 2
    finalizer_mock.invoke.assert_called_once()

    finalizer_messages = finalizer_mock.invoke.call_args.args[0]
    trusted_context_message = finalizer_messages[0]

    assert "invalid_username" in trusted_context_message.content
    assert "verified_username" in trusted_context_message.content
    assert "definitely_not_a_real_user" not in (
        trusted_context_message.content
    )

    answer = result["structured_response"]
    assert answer.refund_proposal is None


def test_graph_exposes_only_purchase_lookup_tool():
    tool_names = {tool.name for tool in two_stage.TOOLS}

    assert tool_names == {"get_user_purchase_history"}
    assert "issue_refund_ticket" not in tool_names
    assert "issue_approved_refund_ticket" not in tool_names


def test_new_turn_clears_previous_verified_identity(monkeypatch):
    conversation_mock = Mock()
    conversation_mock.invoke.return_value = AIMessage(
        content="How can I help you?"
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="How can I help you?"
    )

    monkeypatch.setattr(two_stage, "conversation_model", conversation_mock)
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    graph = two_stage.build_two_stage_graph(
        checkpointer_override=InMemorySaver()
    )

    first_result = invoke_test_graph(
        graph,
        [HumanMessage(content="My username is user123.")],
        "stale-verification-thread",
    )

    # Simulate an existing checkpoint containing verified data.
    # The next invocation is a new user turn without a new lookup.
    second_result = invoke_test_graph(
        graph,
        [HumanMessage(content="Can you help me?")],
        "stale-verification-thread",
    )

    assert first_result["structured_response"] is not None
    assert second_result["lookup_status"] == "not_attempted"
    assert second_result["verified_username"] is None
    assert second_result["purchase_history"] is None

    finalizer_messages = finalizer_mock.invoke.call_args.args[0]
    trusted_context_message = finalizer_messages[0]

    assert '"lookup_status": "not_attempted"' in (
        trusted_context_message.content
    )
    assert '"verified_username": null' in trusted_context_message.content
    assert '"verified_purchase_records": []' in (
        trusted_context_message.content
    )


def test_record_tool_result_rejects_multiple_calls_in_latest_batch():
    first_call = {
        "name": "get_user_purchase_history",
        "args": {"user": "user123"},
        "id": "batch-call-1",
        "type": "tool_call",
    }
    second_call = {
        "name": "get_user_purchase_history",
        "args": {"user": "user124"},
        "id": "batch-call-2",
        "type": "tool_call",
    }

    state = make_tool_state({
        "book": {"date": "2026-10-05", "price": 23}
    })
    state["messages"][0] = AIMessage(
        content="",
        tool_calls=[first_call, second_call],
    )

    result = two_stage._record_tool_result(state)

    assert result["lookup_status"] == "malformed"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


def test_record_tool_result_does_not_reuse_older_lookup_result():
    old_state = make_tool_state({
        "book": {"date": "2026-10-05", "price": 23}
    })

    latest_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_user_purchase_history",
                "args": {"user": "user124"},
                "id": "new-lookup-call",
                "type": "tool_call",
            }
        ],
    )

    state = {
        "messages": [
            *old_state["messages"],
            latest_call,
        ]
    }

    result = two_stage._record_tool_result(state)

    assert result["lookup_status"] == "malformed"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None


@pytest.mark.parametrize(
    "history",
    [
        {"book": None},
        {"book": {"date": "2026-10-05"}},
        {"book": {"date": "not-a-date", "price": 23}},
        {"book": {"date": "2026-10-05", "price": 0}},
        {"book": {"date": "2026-10-05", "price": True}},
        {"": {"date": "2026-10-05", "price": 23}},
    ],
)
def test_record_tool_result_rejects_malformed_purchase_records(history):
    result = two_stage._record_tool_result(make_tool_state(history))

    assert result["lookup_status"] == "malformed"
    assert result["verified_username"] is None
    assert result["purchase_history"] is None



def test_two_stage_graph_persists_answer_after_sqlite_reopen(monkeypatch):
    import sqlite3
    from contextlib import closing
    from tempfile import TemporaryDirectory
    from pathlib import Path

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    conversation_mock = Mock()
    conversation_mock.invoke.return_value = AIMessage(
        content="Hello. How can I help with your purchase?"
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = make_finalizer_result(
        answer="Hello. How can I help with your purchase?"
    )

    monkeypatch.setattr(two_stage, "conversation_model", conversation_mock)
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    thread_id = "two-stage-sqlite-reopen-test"
    config = {"configurable": {"thread_id": thread_id}}

    with TemporaryDirectory() as directory:
        database_path = Path(directory) / "two_stage_checkpoints.sqlite3"

        # First application lifetime: execute the actual two-stage graph.
        with closing(sqlite3.connect(str(database_path), check_same_thread=False)) as connection:
            serializer = JsonPlusSerializer(
                allowed_msgpack_modules=[("agent.objects", "Answer")]
            )
            saver = SqliteSaver(connection, serde=serializer)
            saver.setup()

            graph = two_stage.build_two_stage_graph(
                checkpointer_override=saver
            )

            result = graph.invoke(
                {"messages": [HumanMessage(content="Hello.")]},
                config=config,
                context={"refund_request_id": "sqlite-reopen-test"},
            )

            answer = result["structured_response"]

            assert isinstance(answer, two_stage.Answer)
            assert answer.answer == "Hello. How can I help with your purchase?"
            assert answer.refund_proposal is None

            saved = graph.get_state(config)
            assert saved.next == ()
            assert saved.values["structured_response"] == answer

        # Second application lifetime: reopen SQLite and rebuild the graph.
        with closing(sqlite3.connect(str(database_path), check_same_thread=False)) as connection:
            serializer = JsonPlusSerializer(
                allowed_msgpack_modules=[("agent.objects", "Answer")]
            )
            saver = SqliteSaver(connection, serde=serializer)
            saver.setup()

            recovered_graph = two_stage.build_two_stage_graph(
                checkpointer_override=saver
            )

            recovered = recovered_graph.get_state(config)

            assert recovered.next == ()

            restored = recovered.values["structured_response"]

            assert isinstance(restored, two_stage.Answer)
            assert restored.answer == answer.answer
            assert restored.conversation is answer.conversation
            assert restored.refund_proposal is None

    # Reopening and reading the checkpoint must not invoke either model.
    conversation_mock.invoke.assert_called_once()
    finalizer_mock.invoke.assert_called_once()


def test_finalizer_normalizes_lunch_box_to_verified_purchase_key(monkeypatch):
    parsed = FinalAnswerOutput(
        answer="A refund proposal is ready for application review.",
        conversation=False,
        refund_proposal=StrictRefundProposal(
            username="user146",
            items=[StrictPurchaseItem(name="lunch box")],
        ),
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [
            HumanMessage(content="i m user146 and i want refund lunch box"),
        ],
        "lookup_status": "verified",
        "verified_username": "user146",
        "purchase_history": {
            "lunch_box": {"date": "2026-10-31", "price": 13.81},
        },
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is not None
    assert answer.refund_proposal.username == "user146"
    assert [item.name for item in answer.refund_proposal.items] == ["lunch_box"]
    assert answer.conversation is False


def test_finalizer_keeps_rejected_proposal_open_for_clarification(monkeypatch):
    parsed = FinalAnswerOutput(
        answer="A refund proposal is ready.",
        conversation=False,
        refund_proposal=StrictRefundProposal(
            username="user146",
            items=[StrictPurchaseItem(name="lunch box")],
        ),
    )

    finalizer_mock = Mock()
    finalizer_mock.invoke.return_value = {
        "raw": AIMessage(content="{}"),
        "parsed": parsed,
        "parsing_error": None,
    }
    monkeypatch.setattr(two_stage, "finalizer", finalizer_mock)

    state = {
        "messages": [
            HumanMessage(content="I want a refund for lunch box."),
        ],
        "lookup_status": "verified",
        "verified_username": "user146",
        "purchase_history": {
            "lunch_box": {"date": "2026-10-31", "price": 13.81},
            "lunch-box": {"date": "2026-10-30", "price": 12.00},
        },
    }

    result = two_stage._finalizer_node(state, config={})
    answer = result["structured_response"]

    assert answer.refund_proposal is None
    assert answer.conversation is True
    assert "please clarify" in answer.answer.lower()
    assert "no ticket was created" in answer.answer.lower()


@pytest.mark.parametrize("stage", ["conversation", "finalizer"])
def test_provider_diagnostics_preserves_stage_and_exception(stage, capsys):
    import json

    error = RuntimeError("synthetic provider failure")
    runnable = Mock()
    runnable.invoke.side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        two_stage._invoke_with_provider_diagnostics(
            stage,
            runnable,
            [],
            {},
        )

    assert exc_info.value is error

    captured = capsys.readouterr()
    first_line = captured.err.splitlines()[0]

    prefix = "[AGENTAUDIT_PROVIDER_FAILURE] "
    assert first_line.startswith(prefix)

    details = json.loads(first_line[len(prefix):])
    assert details["stage"] == stage
    assert details["exception_type"] == "RuntimeError"
    assert details["message"] == "synthetic provider failure"


def test_finalizer_uses_deterministic_eligibility_for_explicit_question(
    monkeypatch,
):
    parsed_result = make_finalizer_result(
        answer=(
            "Both purchases were made before the 30-day refund window "
            "ended on October 14, so they are ineligible."
        )
    )

    monkeypatch.setattr(
        two_stage,
        "_invoke_with_provider_diagnostics",
        lambda *args, **kwargs: parsed_result,
    )

    validator_calls = []

    def fake_validate_refund(username, items):
        validator_calls.append((username, items))
        return {
            "eligible": False,
            "reason": (
                "book was purchased 39 days ago; "
                "refund window is 30 days"
            ),
        }

    monkeypatch.setattr(two_stage, "validate_refund", fake_validate_refund)

    result = two_stage._finalizer_node(
        {
            "messages": [
                HumanMessage(
                    content="Is my book eligible for a refund?"
                )
            ],
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"date": "2026-10-05", "price": 23},
                "pen": {"date": "2026-10-03", "price": 7},
            },
        },
        {},
    )

    answer = result["structured_response"]

    assert len(validator_calls) == 1
    assert validator_calls[0][0] == "user123"
    assert validator_calls[0][1][0].name == "book"

    assert "39 days ago" in answer.answer
    assert "October 14" not in answer.answer
    assert "$" not in answer.answer
    assert "₹" not in answer.answer
    assert "no refund proposal" in answer.answer.lower()
    assert answer.refund_proposal is None


def test_finalizer_does_not_run_eligibility_check_for_purchase_history(
    monkeypatch,
):
    parsed_result = make_finalizer_result(
        answer="Your verified purchase history includes a book and a pen."
    )

    monkeypatch.setattr(
        two_stage,
        "_invoke_with_provider_diagnostics",
        lambda *args, **kwargs: parsed_result,
    )

    validator = Mock()
    monkeypatch.setattr(two_stage, "validate_refund", validator)

    result = two_stage._finalizer_node(
        {
            "messages": [
                HumanMessage(content="Show me my purchase history.")
            ],
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"date": "2026-10-05", "price": 23},
                "pen": {"date": "2026-10-03", "price": 7},
            },
        },
        {},
    )

    answer = result["structured_response"]

    validator.assert_not_called()
    assert answer.answer == parsed_result["parsed"].answer
    assert answer.refund_proposal is None


def test_finalizer_asks_for_exact_item_for_eligibility_question(
    monkeypatch,
):
    parsed_result = make_finalizer_result(
        answer="Both purchases are outside the refund window."
    )

    monkeypatch.setattr(
        two_stage,
        "_invoke_with_provider_diagnostics",
        lambda *args, **kwargs: parsed_result,
    )

    validator = Mock()
    monkeypatch.setattr(two_stage, "validate_refund", validator)

    result = two_stage._finalizer_node(
        {
            "messages": [
                HumanMessage(
                    content="Are any of my purchases eligible for a refund?"
                )
            ],
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"date": "2026-10-05", "price": 23},
                "pen": {"date": "2026-10-03", "price": 7},
            },
        },
        {},
    )

    answer = result["structured_response"]

    validator.assert_not_called()
    assert "which exact purchased item" in answer.answer.lower()
    assert answer.refund_proposal is None

def test_finalizer_provider_exception_returns_safe_fallback(monkeypatch):
    invocation = Mock(side_effect=RuntimeError("synthetic provider failure"))
    monkeypatch.setattr(
        two_stage, "_invoke_with_provider_diagnostics", invocation
    )

    result = two_stage._finalizer_node(
        {
            "messages": [HumanMessage(content="Please refund my book.")],
            "lookup_status": "not_attempted",
            "verified_username": None,
            "purchase_history": None,
        },
        {},
    )

    answer = result["structured_response"]
    assert isinstance(answer, two_stage.Answer)
    assert answer.refund_proposal is None
    assert answer.conversation is True
    assert "no refund proposal was prepared" in answer.answer.lower()
    assert "no refund action was taken" in answer.answer.lower()
    invocation.assert_called_once()


@pytest.mark.parametrize(
    "malformed_result",
    [
        {
            "raw": AIMessage(content="plain text instead of JSON"),
            "parsed": None,
            "parsing_error": ValueError("synthetic parse failure"),
        },
        {"raw": AIMessage(content="{}"), "parsed": None, "parsing_error": None},
        {"unexpected": "result shape"},
    ],
)
def test_finalizer_malformed_output_returns_safe_fallback(
    monkeypatch, malformed_result
):
    invocation = Mock(return_value=malformed_result)
    monkeypatch.setattr(
        two_stage, "_invoke_with_provider_diagnostics", invocation
    )

    result = two_stage._finalizer_node(
        {
            "messages": [HumanMessage(content="Please refund my book.")],
            "lookup_status": "not_attempted",
            "verified_username": None,
            "purchase_history": None,
        },
        {},
    )

    answer = result["structured_response"]
    assert isinstance(answer, two_stage.Answer)
    assert answer.refund_proposal is None
    assert answer.conversation is True
    assert "no refund proposal was prepared" in answer.answer.lower()
    assert "no refund action was taken" in answer.answer.lower()
    invocation.assert_called_once()



@pytest.mark.parametrize(
    ("message", "item_name", "expected"),
    [
        (
            "I don't want a refund for the book, but I do want a refund for the pen.",
            "book",
            False,
        ),
        (
            "I don't want a refund for the book, but I do want a refund for the pen.",
            "pen",
            True,
        ),
        (
            "Please refund the book, but do not refund the pen.",
            "book",
            True,
        ),
        (
            "Please refund the book, but do not refund the pen.",
            "pen",
            False,
        ),
        (
            "I want a refund for the book, but I changed my mind: "
            "I do not want a refund for the book.",
            "book",
            False,
        ),
        (
            "Please refund the book and pen.",
            "book",
            True,
        ),
        (
            "Please refund the book and pen.",
            "pen",
            True,
        ),
        (
            "Can I get a refund for the pen?",
            "pen",
            False,
        ),
    ],
)
def test_customer_requested_item_is_item_specific(
    message,
    item_name,
    expected,
):
    from langchain_core.messages import HumanMessage

    result = two_stage._customer_requested_item(
        [HumanMessage(content=message)],
        item_name,
    )

    assert result is expected
