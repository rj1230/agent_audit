from time import perf_counter
from uuid import uuid4

from deepeval.tracing.types import Trace, BaseSpan, TraceSpanStatus
from deepeval.tracing import trace_manager
import deepeval.integrations.langchain.utils as utils


def test_root_closing_before_child_leaves_trace_in_progress():
    manager = type(trace_manager)()

    trace_id = str(uuid4())
    root_id = str(uuid4())
    child_id = str(uuid4())

    root = BaseSpan(
        uuid=root_id,
        trace_uuid=trace_id,
        status=TraceSpanStatus.IN_PROGRESS,
        start_time=perf_counter(),
        name="root",
    )
    child = BaseSpan(
        uuid=child_id,
        trace_uuid=trace_id,
        parent_uuid=root_id,
        status=TraceSpanStatus.IN_PROGRESS,
        start_time=perf_counter(),
        name="child",
    )
    root.children.append(child)

    trace = Trace(
        uuid=trace_id,
        status=TraceSpanStatus.IN_PROGRESS,
        root_spans=[root],
        start_time=perf_counter(),
    )

    manager.active_traces[trace_id] = trace
    manager.traces.append(trace)
    manager.add_span(root)
    manager.add_span(child)

    original_manager = utils.trace_manager

    try:
        utils.trace_manager = manager

        utils.exit_current_context(root_id)
        assert trace.status == TraceSpanStatus.IN_PROGRESS
        assert child_id in manager.active_spans

        utils.exit_current_context(child_id)

        assert not manager.active_spans
        assert trace.status == TraceSpanStatus.IN_PROGRESS
        assert trace_id in manager.active_traces

    finally:
        utils.trace_manager = original_manager
        manager.active_spans.clear()
        manager.active_traces.clear()
        manager.traces.clear()

def test_safe_callback_finalizes_successful_trace_after_last_child(monkeypatch):
    from uuid import uuid4

    from eval.evaluation_helpers import SafeCallbackHandler

    manager = trace_manager
    monkeypatch.setattr(manager, "post_trace", lambda trace: None)

    handler = SafeCallbackHandler()
    root_id = uuid4()
    child_id = uuid4()

    handler.on_chain_start(
        {"name": "synthetic_agent"},
        {"input": "test"},
        run_id=root_id,
    )

    trace_id = str(handler.trace_uuid)
    trace = manager.active_traces[trace_id]

    try:
        handler.on_chain_start(
            {"name": "synthetic_child"},
            {"input": "child"},
            run_id=child_id,
            parent_run_id=root_id,
        )

        handler.on_chain_end(
            {"output": "root complete"},
            run_id=root_id,
        )

        # The child is still active, so the trace must remain open.
        assert trace_id in manager.active_traces
        assert trace.end_time is None

        handler.on_chain_end(
            {"output": "child complete"},
            run_id=child_id,
            parent_run_id=root_id,
        )

        assert trace.status == TraceSpanStatus.SUCCESS
        assert trace.end_time is not None
        assert trace_id not in manager.active_traces
        assert not any(
            str(span.trace_uuid) == trace_id
            for span in manager.active_spans.values()
        )

    finally:
        manager.active_traces.pop(trace_id, None)
        for span_id, span in list(manager.active_spans.items()):
            if str(span.trace_uuid) == trace_id:
                manager.active_spans.pop(span_id, None)
        if trace in manager.traces:
            manager.traces.remove(trace)

def test_safe_callback_finalizes_failed_child_after_root_closes(monkeypatch):
    from eval.evaluation_helpers import SafeCallbackHandler

    manager = trace_manager
    monkeypatch.setattr(manager, "post_trace", lambda trace: None)

    handler = SafeCallbackHandler()
    root_id = uuid4()
    child_id = uuid4()

    handler.on_chain_start(
        {"name": "synthetic_agent"},
        {"input": "test"},
        run_id=root_id,
    )

    trace_id = str(handler.trace_uuid)
    trace = manager.active_traces[trace_id]

    try:
        handler.on_chain_start(
            {"name": "synthetic_tool"},
            {"input": "child"},
            run_id=child_id,
            parent_run_id=root_id,
        )

        handler.on_chain_end(
            {"output": "root complete"},
            run_id=root_id,
        )

        assert trace_id in manager.active_traces
        assert trace.end_time is None

        handler.on_tool_error(
            RuntimeError("synthetic tool failure"),
            run_id=child_id,
            parent_run_id=root_id,
        )

        assert trace.status == TraceSpanStatus.ERRORED
        assert trace.end_time is not None
        assert trace_id not in manager.active_traces
        assert not any(
            str(span.trace_uuid) == trace_id
            for span in manager.active_spans.values()
        )
    finally:
        manager.active_traces.pop(trace_id, None)
        for span_id, span in list(manager.active_spans.items()):
            if str(span.trace_uuid) == trace_id:
                manager.active_spans.pop(span_id, None)
        if trace in manager.traces:
            manager.traces.remove(trace)


import pytest


@pytest.mark.parametrize(
    "error_callback",
    ["on_llm_error", "on_chat_model_error"],
)
def test_safe_callback_finalizes_failed_model_child_after_root_closes(
    monkeypatch, error_callback
):
    from eval.evaluation_helpers import SafeCallbackHandler

    manager = trace_manager
    monkeypatch.setattr(manager, "post_trace", lambda trace: None)

    handler = SafeCallbackHandler()
    root_id = uuid4()
    child_id = uuid4()

    handler.on_chain_start(
        {"name": "synthetic_agent"},
        {"input": "test"},
        run_id=root_id,
    )

    trace_id = str(handler.trace_uuid)
    trace = manager.active_traces[trace_id]

    try:
        handler.on_chain_start(
            {"name": "synthetic_model"},
            {"input": "child"},
            run_id=child_id,
            parent_run_id=root_id,
        )

        handler.on_chain_end(
            {"output": "root complete"},
            run_id=root_id,
        )

        assert trace_id in manager.active_traces
        assert trace.end_time is None

        callback = getattr(handler, error_callback)
        callback(
            RuntimeError("synthetic model failure"),
            run_id=child_id,
            parent_run_id=root_id,
        )

        assert trace.status == TraceSpanStatus.ERRORED
        assert trace.end_time is not None
        assert trace_id not in manager.active_traces
        assert not any(
            str(span.trace_uuid) == trace_id
            for span in manager.active_spans.values()
        )
    finally:
        manager.active_traces.pop(trace_id, None)
        for span_id, span in list(manager.active_spans.items()):
            if str(span.trace_uuid) == trace_id:
                manager.active_spans.pop(span_id, None)
        if trace in manager.traces:
            manager.traces.remove(trace)
