from uuid import uuid4

from deepeval.integrations.langchain import CallbackHandler
from deepeval.integrations.langchain import utils as deepeval_trace_utils
from deepeval.tracing.types import TraceSpanStatus
from deepeval.test_case import ToolCall
from deepeval.tracing import trace_manager
from deepeval.tracing.utils import make_json_serializable


class SafeCallbackHandler(CallbackHandler):
    """Close failed LangChain chain spans and preserve trace error status."""

    def _finalize_if_quiescent(self, trace_uuid):
        """Finalize a trace when its last callback span closes after its root."""
        if not trace_uuid:
            return

        trace_uuid = str(trace_uuid)
        trace = trace_manager.active_traces.get(trace_uuid)
        if trace is None:
            return

        active_for_trace = [
            span
            for span in trace_manager.active_spans.values()
            if str(span.trace_uuid) == trace_uuid
        ]
        if active_for_trace:
            return

        if not trace.root_spans or any(
            root.end_time is None for root in trace.root_spans
        ):
            return

        # end_trace preserves ERRORED status; it only defaults IN_PROGRESS
        # traces to SUCCESS.
        trace_manager.end_trace(trace_uuid)

    def on_chain_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        span = trace_manager.get_span_by_uuid(str(run_id))
        trace_uuid = span.trace_uuid if span is not None else None
        result = super().on_chain_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._finalize_if_quiescent(trace_uuid)
        return result

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        span = trace_manager.get_span_by_uuid(str(run_id))
        trace_uuid = span.trace_uuid if span is not None else None
        result = super().on_tool_end(
            output,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._finalize_if_quiescent(trace_uuid)
        return result

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        span = trace_manager.get_span_by_uuid(str(run_id))
        trace_uuid = span.trace_uuid if span is not None else None
        result = super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._finalize_if_quiescent(trace_uuid)
        return result

    def on_chat_model_end(
        self, response, *, run_id, parent_run_id=None, **kwargs
    ):
        span = trace_manager.get_span_by_uuid(str(run_id))
        trace_uuid = span.trace_uuid if span is not None else None
        result = super().on_chat_model_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
        self._finalize_if_quiescent(trace_uuid)
        return result

    def _handle_callback_error(
        self, error, *, run_id, parent_run_id=None
    ):
        """Close a failed callback span without losing its error status."""
        span_uuid = str(run_id)
        span = trace_manager.get_span_by_uuid(span_uuid)

        if span is None or span.end_time is not None:
            return

        trace_uuid = str(span.trace_uuid)

        with self._ctx(run_id=run_id, parent_run_id=parent_run_id):
            deepeval_trace_utils.exit_current_context(
                uuid_str=span_uuid,
                exc_type=type(error),
                exc_val=error,
            )

        trace = trace_manager.active_traces.get(trace_uuid)
        if trace is None:
            trace = trace_manager.get_trace_by_uuid(trace_uuid)

        if trace is not None:
            trace.status = TraceSpanStatus.ERRORED

        self._restore_observe_parent(parent_run_id)
        self._finalize_if_quiescent(trace_uuid)

    def on_chain_error(
        self, error, *, run_id, parent_run_id=None, **kwargs
    ):
        return self._handle_callback_error(
            error, run_id=run_id, parent_run_id=parent_run_id
        )

    def on_tool_error(
        self, error, *, run_id, parent_run_id=None, **kwargs
    ):
        return self._handle_callback_error(
            error, run_id=run_id, parent_run_id=parent_run_id
        )

    def on_llm_error(
        self, error, *, run_id, parent_run_id=None, **kwargs
    ):
        return self._handle_callback_error(
            error, run_id=run_id, parent_run_id=parent_run_id
        )

    def on_chat_model_error(
        self, error, *, run_id, parent_run_id=None, **kwargs
    ):
        return self._handle_callback_error(
            error, run_id=run_id, parent_run_id=parent_run_id
        )



def invoke_agent_for_golden(
    agent, golden, *, capture_trace=False, defer_trace_capture=False
):
    """Invoke an agent and optionally expose its DeepEval trace UUID.

    Default callers retain the historical two-value return.
    Immediate capture returns (response, tools_called, trace_dict, error).
    Deferred capture returns (response, tools_called, trace_uuid, None).
    """
    request_id = str(uuid4())
    thread_id = str(uuid4())
    callback_handler = SafeCallbackHandler()

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

    trace_uuid = callback_handler.trace_uuid
    if defer_trace_capture:
        if not trace_uuid:
            return (
                response,
                tools_called,
                None,
                "RuntimeError: DeepEval CallbackHandler did not capture a trace UUID.",
            )
        return response, tools_called, str(trace_uuid), None

    trace_dict, trace_error = capture_trace_artifact(trace_uuid)
    return response, tools_called, trace_dict, trace_error


def capture_trace_artifact(trace_uuid):
    """Retrieve and serialize a completed DeepEval trace by UUID."""
    try:
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

        return trace_dict, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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
