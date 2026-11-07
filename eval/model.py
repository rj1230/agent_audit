import asyncio
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import os
import threading
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

from eval.quota_guard import (
    block_judge_requests,
    ensure_judge_available,
    is_daily_token_quota_error,
)


# Load the project-level .env before initializing the Groq client.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Keep the evaluator separate from the agent model where account access allows.
JUDGE_MODEL_ID = os.getenv(
    "AGENT_AUDIT_EVAL_MODEL",
    "openai/gpt-oss-120b",
).strip()

if not JUDGE_MODEL_ID:
    raise ValueError("AGENT_AUDIT_EVAL_MODEL cannot be empty.")

MAX_PROMPT_CHARS = 12_000
JUDGE_MIN_INTERVAL_SECONDS = float(
    os.getenv("AGENT_AUDIT_JUDGE_MIN_INTERVAL_SECONDS", "30")
)
if JUDGE_MIN_INTERVAL_SECONDS < 0:
    raise ValueError("AGENT_AUDIT_JUDGE_MIN_INTERVAL_SECONDS cannot be negative.")

_JUDGE_RATE_LOCK = threading.Lock()
_LAST_JUDGE_REQUEST_STARTED = 0.0


@contextmanager
def _judge_request_slot():
    """Serialize sync judge calls and space their start times."""
    global _LAST_JUDGE_REQUEST_STARTED

    with _JUDGE_RATE_LOCK:
        ensure_judge_available()
        delay = JUDGE_MIN_INTERVAL_SECONDS - (
            time.monotonic() - _LAST_JUDGE_REQUEST_STARTED
        )
        if delay > 0:
            time.sleep(delay)

        _LAST_JUDGE_REQUEST_STARTED = time.monotonic()
        yield


@asynccontextmanager
async def _async_judge_request_slot():
    """Share the same rate gate without blocking the asyncio event loop."""
    global _LAST_JUDGE_REQUEST_STARTED

    await asyncio.to_thread(_JUDGE_RATE_LOCK.acquire)
    try:
        ensure_judge_available()
        delay = JUDGE_MIN_INTERVAL_SECONDS - (
            time.monotonic() - _LAST_JUDGE_REQUEST_STARTED
        )
        if delay > 0:
            await asyncio.sleep(delay)

        _LAST_JUDGE_REQUEST_STARTED = time.monotonic()
        yield
    finally:
        _JUDGE_RATE_LOCK.release()



def _run_sync_judge_request(request):
    """Run one serialized judge request and latch daily quota exhaustion."""
    ensure_judge_available()
    try:
        with _judge_request_slot():
            return request()
    except Exception as exc:
        if is_daily_token_quota_error(exc):
            block_judge_requests(exc)
        raise


async def _run_async_judge_request(request):
    """Async counterpart of the guarded synchronous judge request."""
    ensure_judge_available()
    try:
        async with _async_judge_request_slot():
            return await request()
    except Exception as exc:
        if is_daily_token_quota_error(exc):
            block_judge_requests(exc)
        raise


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    if len(prompt) <= max_chars:
        return prompt

    marker = (
        "\n\n[TRUNCATED: middle portion of the agent trace removed "
        "to fit model input limits]\n\n"
    )

    if max_chars <= len(marker):
        return prompt[:max_chars]

    available = max_chars - len(marker)
    keep_start = available // 2
    keep_end = available - keep_start

    return (
        prompt[:keep_start]
        + marker
        + (prompt[-keep_end:] if keep_end else "")
    )


def _normalize_schema_result(result, schema):
    """Normalize occasional class-name-wrapped JSON and validate the schema."""
    if schema is None or isinstance(result, schema):
        return result

    if not isinstance(result, dict):
        return result

    payload = dict(result)
    wrapper_key = getattr(schema, "__name__", None)

    if wrapper_key and wrapper_key in payload:
        wrapped = payload.pop(wrapper_key)

        if isinstance(wrapped, dict):
            payload = {**wrapped, **payload}
        elif "score" in getattr(schema, "model_fields", {}):
            payload = {"score": wrapped, **payload}

    validator = getattr(schema, "model_validate", None)
    if validator is None:
        raise TypeError(
            f"Expected a Pydantic schema with model_validate: {schema!r}"
        )

    try:
        return validator(payload)
    except Exception as exc:
        raise ValueError(
            f"Groq judge returned data that does not match "
            f"{getattr(schema, '__name__', schema)!r}. "
            f"Returned keys: {sorted(payload.keys())}"
        ) from exc



class GroqJudge(DeepEvalBaseLLM):
    """DeepEval-compatible evaluator backed by a Groq chat model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self):
        return ChatGroq(
            model=self.model_name,
            temperature=0,
            max_tokens=1024,
            timeout=60,
            max_retries=0,
        )

    def generate(self, prompt: str, schema=None, audio=None):
        if audio is not None:
            raise NotImplementedError("GroqJudge does not support audio inputs.")

        prompt = _truncate_prompt(prompt)

        if schema is not None:
            prompt = prompt + "\n\nReturn valid json matching the requested schema. Output json only.\n"
            result = _run_sync_judge_request(
                lambda: self.model.with_structured_output(
                    schema,
                    method="json_mode",
                ).invoke(prompt)
            )
            return _normalize_schema_result(result, schema)

        result = _run_sync_judge_request(
            lambda: self.model.invoke(prompt)
        )
        content = result.content
        if not isinstance(content, str):
            content = str(content)

        return content

    async def a_generate(self, prompt: str, schema=None, audio=None):
        if audio is not None:
            raise NotImplementedError("GroqJudge does not support audio inputs.")

        prompt = _truncate_prompt(prompt)

        if schema is not None:
            prompt = prompt + "\n\nReturn valid json matching the requested schema. Output json only.\n"
            result = await _run_async_judge_request(
                lambda: self.model.with_structured_output(
                    schema,
                    method="json_mode",
                ).ainvoke(prompt)
            )
            return _normalize_schema_result(result, schema)

        result = await _run_async_judge_request(
            lambda: self.model.ainvoke(prompt)
        )
        content = result.content
        if not isinstance(content, str):
            content = str(content)

        return content

    def get_model_name(self) -> str:
        return self.model_name


eval_model = GroqJudge(model_name=JUDGE_MODEL_ID)
