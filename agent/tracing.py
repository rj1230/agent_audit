"""Privacy-minimized execution tracing for AgentAudit."""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

TRACE_DIR = Path(__file__).with_name("traces")
TRACE_FILE = TRACE_DIR / "agent_traces.jsonl"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

_FILE_LOCK = threading.Lock()

OMITTED = "[OMITTED]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_value(value, max_chars: int = 4000):
    """Convert non-customer operational metadata to bounded JSON-safe data."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()

    if isinstance(value, dict):
        result = {}
        sensitive_keys = {
            "api_key", "access_token", "authorization",
            "password", "secret",
        }
        for key, item in value.items():
            key_string = str(key)
            if any(token in key_string.lower() for token in sensitive_keys):
                result[key_string] = "[REDACTED]"
            else:
                result[key_string] = safe_value(item, max_chars)
        return result

    if isinstance(value, (list, tuple)):
        return [safe_value(item, max_chars) for item in value]

    if isinstance(value, str):
        return value[:max_chars] + "...[TRUNCATED]" if len(value) > max_chars else value

    if value is None or isinstance(value, (bool, int, float)):
        return value

    return str(value)[:max_chars]


class ToolTraceCallback(BaseCallbackHandler):
    """Capture tool execution metadata without retaining tool payloads."""

    def __init__(self):
        self.events = []
        self._active_runs = {}
        self._lock = threading.Lock()

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        started = perf_counter()
        tool_name = (serialized or {}).get("name", "unknown_tool")

        event = {
            "sequence": 0,
            "tool_run_id": str(run_id),
            "tool_name": str(tool_name),
            "arguments": OMITTED,
            "status": "running",
            "started_at": utc_now(),
            "duration_seconds": None,
            "result": OMITTED,
            "error": None,
        }

        with self._lock:
            event["sequence"] = len(self.events) + 1
            self.events.append(event)
            self._active_runs[str(run_id)] = {
                "event": event,
                "started": started,
            }

    def on_tool_end(
        self,
        output,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        self._finish_tool(run_id=run_id, status="success")

    def on_tool_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        self._finish_tool(
            run_id=run_id,
            status="error",
            error_type=type(error).__name__,
        )

    def _finish_tool(self, run_id, status, error_type=None):
        with self._lock:
            active = self._active_runs.pop(str(run_id), None)
            if active is None:
                return

            event = active["event"]
            event["status"] = status
            event["duration_seconds"] = round(
                perf_counter() - active["started"], 4
            )
            event["finished_at"] = utc_now()

            if status == "error":
                event["error"] = {
                    "type": str(error_type or "UnknownError"),
                    "message": OMITTED,
                }


def extract_model_tool_calls(response):
    """Keep model-requested tool identity, never its arguments or content."""
    if not response:
        return []

    calls = []
    for message_index, message in enumerate(response.get("messages", [])):
        for tool_call in getattr(message, "tool_calls", []):
            calls.append({
                "message_index": message_index,
                "tool_call_id": tool_call.get("id"),
                "tool_name": tool_call.get("name"),
                "arguments": OMITTED,
            })
    return calls


def extract_usage(response):
    """Collect token usage metadata when the provider exposes it."""
    usage = []
    if not response:
        return usage

    for message in response.get("messages", []):
        metadata = getattr(message, "usage_metadata", None)
        if metadata:
            usage.append(safe_value(metadata))
    return usage


def write_agent_trace(
    *,
    thread_id,
    turn_number,
    elapsed_seconds,
    tool_callback,
    response=None,
    error=None,
):
    """Persist operational telemetry without customer or model payloads."""
    tool_events = list(tool_callback.events)

    trace_record = {
        "schema_version": 2,
        "trace_id": str(uuid4()),
        "timestamp": utc_now(),
        "thread_id": str(thread_id),
        "turn_number": turn_number,
        "status": "error" if error else "success",
        "execution_time_seconds": round(elapsed_seconds, 4),
        "tool_call_count": len(tool_events),
        "successful_tool_count": sum(
            event["status"] == "success" for event in tool_events
        ),
        "failed_tool_count": sum(
            event["status"] == "error" for event in tool_events
        ),
        "tool_events": tool_events,
        "model_tool_calls": extract_model_tool_calls(response),
        "token_usage": extract_usage(response),
        "final_response": OMITTED if response else None,
        "error": (
            {
                "type": type(error).__name__,
                "message": OMITTED,
            }
            if error else None
        ),
    }

    serialized = json.dumps(
        trace_record,
        ensure_ascii=False,
        default=str,
    )

    with _FILE_LOCK:
        with TRACE_FILE.open("a", encoding="utf-8") as file:
            file.write(serialized + "\n")
            file.flush()

    logger.info(
        "Agent trace saved: thread=%s turn=%s tools=%s status=%s",
        thread_id,
        turn_number,
        len(tool_events),
        trace_record["status"],
    )

    return trace_record
