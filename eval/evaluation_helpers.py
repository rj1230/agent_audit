from uuid import uuid4

from deepeval.integrations.langchain import CallbackHandler
from deepeval.test_case import ToolCall
from deepeval.tracing import trace_manager
from deepeval.tracing.utils import make_json_serializable


def invoke_agent_for_golden(agent, golden, *, capture_trace=False):
    """Invoke an agent and optionally capture its DeepEval trace.

    Default callers retain the historical two-value return.
    Trace capture returns (response, tools_called, trace_dict, trace_error).
    A trace-capture failure never discards a successful agent response.
    """
    request_id = str(uuid4())
    thread_id = str(uuid4())
    callback_handler = CallbackHandler()

    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [callback_handler],
        },
        context={"refund_request_id": request_id},
    )

    tools_called = extract_executed_tool_calls(response["messages"])

    if not capture_trace:
        return response, tools_called

    try:
        trace_uuid = callback_handler.trace_uuid
        if not trace_uuid:
            raise RuntimeError(
                "DeepEval CallbackHandler did not capture a trace UUID."
            )

        trace = next(
            (
                item for item in trace_manager.get_all_traces()
                if str(item.uuid) == str(trace_uuid)
            ),
            None,
        )
        if trace is None:
            raise RuntimeError(
                f"Completed DeepEval trace {trace_uuid} was not retained."
            )

        status = getattr(trace.status, "value", trace.status)
        if status != "SUCCESS":
            raise RuntimeError(
                f"DeepEval trace {trace_uuid} is not complete: {status}"
            )

        if not trace.root_spans:
            raise RuntimeError(
                f"DeepEval trace {trace_uuid} has no root span."
            )

        trace_dict = trace_manager.create_nested_spans_dict(
            trace.root_spans[0]
        )
        trace_dict = make_json_serializable(trace_dict)

        if not isinstance(trace_dict, dict) or not trace_dict:
            raise RuntimeError("DeepEval produced an empty trace artifact.")

        return response, tools_called, trace_dict, None

    except Exception as exc:
        return response, tools_called, None, (
            f"{type(exc).__name__}: {exc}"
        )


def extract_executed_tool_calls(messages):
    """Return tool calls only when a matching ToolMessage confirms execution."""
    calls_by_id = {}

    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            call_id = call.get("id")
            if call_id:
                calls_by_id[call_id] = call

    executed = []

    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue

        call_id = getattr(message, "tool_call_id", None)
        call = calls_by_id.get(call_id)

        if call is None:
            continue

        message_name = getattr(message, "name", None)
        if message_name != call.get("name"):
            continue

        executed.append(
            ToolCall(
                name=call["name"],
                input_parameters=call.get("args", {}),
            )
        )

    return executed
