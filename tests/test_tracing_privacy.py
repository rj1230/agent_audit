import json
from types import SimpleNamespace
from uuid import uuid4

from agent import tracing


def test_tool_callback_never_retains_arguments_or_results():
    callback = tracing.ToolTraceCallback()
    run_id = uuid4()

    callback.on_tool_start(
        {"name": "get_user_purchase_history"},
        '{"user": "private-user-123"}',
        run_id=run_id,
        inputs={"user": "private-user-123"},
    )
    callback.on_tool_end(
        '{"purchase": "private-purchase-data"}',
        run_id=run_id,
    )

    event = callback.events[0]
    serialized = json.dumps(event)

    assert event["tool_name"] == "get_user_purchase_history"
    assert event["status"] == "success"
    assert event["arguments"] == tracing.OMITTED
    assert event["result"] == tracing.OMITTED
    assert "private-user-123" not in serialized
    assert "private-purchase-data" not in serialized
    assert event["duration_seconds"] >= 0


def test_tool_error_does_not_retain_exception_message():
    callback = tracing.ToolTraceCallback()
    run_id = uuid4()

    callback.on_tool_start(
        {"name": "get_user_purchase_history"},
        "private input",
        run_id=run_id,
    )
    callback.on_tool_error(
        RuntimeError("private-user-123 database details"),
        run_id=run_id,
    )

    event = callback.events[0]
    assert event["status"] == "error"
    assert event["error"]["type"] == "RuntimeError"
    assert event["error"]["message"] == tracing.OMITTED
    assert "private-user-123" not in json.dumps(event)


def test_model_tool_call_arguments_are_omitted():
    message = SimpleNamespace(
        tool_calls=[
            {
                "id": "call-001",
                "name": "get_user_purchase_history",
                "args": {"user": "private-user-123"},
            }
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 20},
    )
    response = {"messages": [message]}

    calls = tracing.extract_model_tool_calls(response)

    assert calls == [{
        "message_index": 0,
        "tool_call_id": "call-001",
        "tool_name": "get_user_purchase_history",
        "arguments": tracing.OMITTED,
    }]
    assert "private-user-123" not in json.dumps(calls)


def test_trace_writer_omits_customer_payloads_and_keeps_metrics(
    tmp_path, monkeypatch
):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(tracing, "TRACE_FILE", trace_file)

    callback = tracing.ToolTraceCallback()
    run_id = uuid4()
    callback.on_tool_start(
        {"name": "get_user_purchase_history"},
        "private-user-123",
        run_id=run_id,
        inputs={"user": "private-user-123"},
    )
    callback.on_tool_end(
        "private purchase history",
        run_id=run_id,
    )

    message = SimpleNamespace(
        tool_calls=[
            {
                "id": "call-001",
                "name": "get_user_purchase_history",
                "args": {"user": "private-user-123"},
            }
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 20},
    )
    response = {
        "messages": [message],
        "structured_response": {
            "answer": "Private customer response",
            "username": "private-user-123",
        },
    }

    record = tracing.write_agent_trace(
        thread_id="test-thread",
        turn_number=1,
        elapsed_seconds=1.25,
        tool_callback=callback,
        response=response,
    )

    persisted = trace_file.read_text(encoding="utf-8")
    assert record["schema_version"] == 2
    assert record["tool_call_count"] == 1
    assert record["successful_tool_count"] == 1
    assert record["failed_tool_count"] == 0
    assert record["tool_events"][0]["arguments"] == tracing.OMITTED
    assert record["tool_events"][0]["result"] == tracing.OMITTED
    assert record["model_tool_calls"][0]["arguments"] == tracing.OMITTED
    assert record["final_response"] == tracing.OMITTED
    assert record["token_usage"][0]["input_tokens"] == 100

    for private_value in (
        "private-user-123",
        "private purchase history",
        "Private customer response",
    ):
        assert private_value not in persisted


def test_trace_writer_omits_raw_top_level_exception_message(tmp_path, monkeypatch):
    trace_file = tmp_path / "traces.jsonl"
    monkeypatch.setattr(tracing, "TRACE_FILE", trace_file)

    record = tracing.write_agent_trace(
        thread_id="test-thread",
        turn_number=2,
        elapsed_seconds=0.5,
        tool_callback=tracing.ToolTraceCallback(),
        error=RuntimeError("private-user-123 secret database detail"),
    )

    assert record["status"] == "error"
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["message"] == tracing.OMITTED
    assert "private-user-123" not in trace_file.read_text(encoding="utf-8")
