import json
import logging
from types import SimpleNamespace
from uuid import uuid4

from agent.execution import run_pending_operation
from agent.retry import build_agent_input
from agent.tracing import ToolTraceCallback, write_agent_trace


def test_execution_wrapper_writes_privacy_safe_trace(
    tmp_path,
    monkeypatch,
):
    trace_path = tmp_path / "agent_traces.jsonl"
    monkeypatch.setattr("agent.tracing.TRACE_FILE", trace_path)

    thread_id = "synthetic-trace-thread"
    request_id = "synthetic-trace-request"
    private_customer_text = "PRIVATE_CUSTOMER_MARKER_73921"
    private_tool_result = "PRIVATE_TOOL_RESULT_MARKER_84216"
    private_model_answer = "PRIVATE_MODEL_ANSWER_MARKER_95317"

    session_state = SimpleNamespace(
        pending_operation={
            "user_text": private_customer_text,
            "refund_request_id": request_id,
            "turn_number": 1,
            "failed": False,
        },
        pending_approval=None,
        thread_id=thread_id,
        messages=[{"role": "user", "content": private_customer_text}],
        trace_records=[],
        trace_warning=False,
        conversation_closed=False,
    )

    class SyntheticAgent:
        def invoke(self, graph_input, *, config, context):
            assert config["configurable"]["thread_id"] == thread_id
            assert context["refund_request_id"] == request_id

            callback = config["callbacks"][0]
            run_id = uuid4()

            callback.on_tool_start(
                {"name": "synthetic_lookup"},
                private_customer_text,
                run_id=run_id,
                inputs={"customer_text": private_customer_text},
            )
            callback.on_tool_end(
                private_tool_result,
                run_id=run_id,
            )

            return {
                "structured_response": {
                    "answer": private_model_answer,
                    "conversation": True,
                    "refund_proposal": None,
                },
                "messages": [],
            }

    run_pending_operation(
        session_state=session_state,
        agent=SyntheticAgent(),
        callback_factory=ToolTraceCallback,
        trace_writer=write_agent_trace,
        input_builder=build_agent_input,
        logger=logging.getLogger("synthetic-trace-integration-test"),
        max_conversation_turns=8,
    )

    assert session_state.pending_operation is None
    assert session_state.pending_approval is None
    assert session_state.trace_warning is False
    assert len(session_state.trace_records) == 1

    record = session_state.trace_records[0]
    assert record["status"] == "success"
    assert record["tool_call_count"] == 1
    assert record["successful_tool_count"] == 1
    assert record["failed_tool_count"] == 0
    assert record["tool_events"][0]["tool_name"] == "synthetic_lookup"
    assert record["tool_events"][0]["status"] == "success"
    assert record["tool_events"][0]["arguments"] == "[OMITTED]"
    assert record["tool_events"][0]["result"] == "[OMITTED]"
    assert record["final_response"] == "[OMITTED]"

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    persisted = json.loads(lines[0])
    assert persisted["trace_id"] == record["trace_id"]
    assert persisted["thread_id"] == thread_id
    assert persisted["tool_call_count"] == 1

    serialized = trace_path.read_text(encoding="utf-8")
    assert private_customer_text not in serialized
    assert private_tool_result not in serialized
    assert private_model_answer not in serialized
