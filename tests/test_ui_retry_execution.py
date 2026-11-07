import logging

from langchain.messages import HumanMessage

from agent.execution import run_pending_operation
from agent.retry import build_agent_input
from agent import tools as refund_tools


def test_agent_proposal_waits_for_approval_without_creating_ticket(
    monkeypatch,
):
    request_id = "streamlit-approval-request-test"
    thread_id = "streamlit-approval-thread-test"
    user_text = "Please refund my damaged keyboard."

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

    validation_calls = []

    def validate(user, items):
        validation_calls.append((user, [item.name for item in items]))
        return validated_quote

    def forbidden_ticket_creation(**kwargs):
        raise AssertionError(
            "The agent execution path must never issue a refund ticket."
        )

    monkeypatch.setattr("agent.execution.validate_refund", validate)
    monkeypatch.setattr(
        refund_tools,
        "create_or_get_ticket",
        forbidden_ticket_creation,
    )

    session_state = type("SessionState", (), {})()
    session_state.pending_operation = {
        "user_text": user_text,
        "refund_request_id": request_id,
        "turn_number": 1,
        "failed": False,
    }
    session_state.pending_approval = None
    session_state.thread_id = thread_id
    session_state.messages = [
        {"role": "user", "content": user_text},
    ]
    session_state.trace_records = []
    session_state.trace_warning = False
    session_state.conversation_closed = False

    class Callback:
        pass

    class ProposalAgent:
        def __init__(self):
            self.invocations = []

        def invoke(self, graph_input, *, config, context):
            message = graph_input["messages"][0]
            assert isinstance(message, HumanMessage)
            assert message.id == request_id
            assert config["configurable"]["thread_id"] == thread_id
            assert context["refund_request_id"] == request_id

            self.invocations.append(
                {
                    "thread_id": config["configurable"]["thread_id"],
                    "request_id": context["refund_request_id"],
                }
            )

            return {
                "structured_response": {
                    "answer": (
                        "Your refund proposal has been created and "
                        "will be processed shortly."
                    ),
                    "conversation": True,
                    "refund_proposal": {
                        "username": "user-001",
                        "items": [{"name": "Keyboard"}],
                    },
                }
            }

    fake_agent = ProposalAgent()

    def trace_writer(**kwargs):
        return {
            "status": "error" if kwargs["error"] else "completed",
            "turn_number": kwargs["turn_number"],
        }

    run_pending_operation(
        session_state=session_state,
        agent=fake_agent,
        callback_factory=Callback,
        trace_writer=trace_writer,
        input_builder=build_agent_input,
        logger=logging.getLogger("agent-audit-approval-test"),
        max_conversation_turns=8,
    )

    assert len(fake_agent.invocations) == 1
    assert fake_agent.invocations[0] == {
        "thread_id": thread_id,
        "request_id": request_id,
    }

    assert validation_calls == [("user-001", ["Keyboard"])]
    assert session_state.pending_operation is None
    assert session_state.pending_approval == {
        "request_id": request_id,
        "user": "user-001",
        "items": [{"name": "Keyboard"}],
        "quote": {
            "user": "user-001",
            "items": validated_quote["items"],
            "refund_amount": "49.99",
        },
    }

    assert session_state.messages[-1] == {
        "role": "assistant",
        "content": (
            "The application has prepared a refund quote "
            "for your review. Please review the quote below "
            "and explicitly confirm or decline it. "
            "No ticket has been created."
        ),
    }
    assert "processed shortly" not in str(session_state.messages)
    assert "proposal has been created" not in str(session_state.messages)
    assert session_state.conversation_closed is False
    assert len(session_state.trace_records) == 1
    assert session_state.trace_records[0]["status"] == "completed"
