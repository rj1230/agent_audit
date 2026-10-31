import logging
from html import escape
from time import perf_counter
from uuid import uuid4

import streamlit as st
from langchain.messages import HumanMessage

from agent import agent
from tracing import ToolTraceCallback, write_agent_trace
from retry import build_agent_input


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgentAudit | Refund support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 8

# Height of the scrolling area inside each panel (pixels).
# The trace panel has no chat input under it, so it gets a little extra
# height to make both panels end at the same line. Tweak TRACE_EXTRA_HEIGHT
# by a few pixels if your browser zoom makes them slightly uneven.
CHAT_BODY_HEIGHT = 400
TRACE_EXTRA_HEIGHT = 64
TRACE_BODY_HEIGHT = CHAT_BODY_HEIGHT + TRACE_EXTRA_HEIGHT

WELCOME_MESSAGE = (
    "Hello! I can help you review a purchase and request a refund.\n\n"
    "Describe the issue and include your order number if available."
)

SUGGESTED_PROMPTS = [
    "My order arrived damaged and I would like a refund.",
    "I was charged twice for the same order.",
    "Can you check if my order is eligible for a refund?",
]

ASSISTANT_AVATAR = ":material/support_agent:"
USER_AVATAR = ":material/person:"


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------


def initial_messages():
    return [{"role": "assistant", "content": WELCOME_MESSAGE}]


def initialize_session():
    defaults = {
        "thread_id": str(uuid4()),
        "turn_number": 0,
        "messages": initial_messages(),
        "trace_records": [],
        "conversation_closed": False,
        "show_tool_details": False,
        "trace_warning": False,
        "pending_prompt": None,
        "pending_operation": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_conversation():
    if st.session_state.pending_operation is not None:
        return

    st.session_state.thread_id = str(uuid4())
    st.session_state.turn_number = 0
    st.session_state.messages = initial_messages()
    st.session_state.trace_records = []
    st.session_state.conversation_closed = False
    st.session_state.trace_warning = False
    st.session_state.pending_prompt = None
    st.session_state.pending_operation = None


def queue_prompt(prompt):
    st.session_state.pending_prompt = prompt


initialize_session()


# ---------------------------------------------------------
# Agent execution (unchanged behaviour)
# ---------------------------------------------------------


def execute_pending_operation():
    pending = st.session_state.pending_operation

    if not isinstance(pending, dict):
        return

    user_text = pending["user_text"]
    request_id = pending["refund_request_id"]
    thread_id = st.session_state.thread_id
    turn_number = pending["turn_number"]

    error_message = (
        "I encountered a technical issue while processing your request. "
        "The outcome could not be confirmed. You can retry below using "
        "the same request identifier."
    )

    # Remove the previous retry error before attempting again.
    if (
        pending.get("failed")
        and st.session_state.messages
        and st.session_state.messages[-1]
        == {"role": "assistant", "content": error_message}
    ):
        st.session_state.messages.pop()

    is_retry = bool(pending.get("failed"))
    pending["failed"] = False

    callback = ToolTraceCallback()
    started = perf_counter()
    response = None
    error = None

    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "turn_number": turn_number,
                "application": "agent-audit-streamlit",
            },
            "callbacks": [callback],
        }

        graph_input = build_agent_input(
            agent,
            config,
            is_retry=is_retry,
            user_text=user_text,
            request_id=request_id,
        )

        if isinstance(graph_input, dict) and "messages" in graph_input:
            # First attempt: submit the original user message.
            response = agent.invoke(
                graph_input,
                config=config,
                context={"refund_request_id": request_id},
            )

        elif graph_input is None:
            # Interrupted graph: resume the existing checkpoint.
            response = agent.invoke(
                None,
                config=config,
                context={"refund_request_id": request_id},
            )

        else:
            # Completed graph: recover its saved response without rerunning.
            response = {"structured_response": graph_input}

        structured = response.get("structured_response")
        if structured is None:
            raise ValueError("Agent did not return a structured response.")

        if isinstance(structured, dict):
            answer = structured["answer"]
            should_continue = structured["conversation"]
        else:
            answer = structured.answer
            should_continue = structured.conversation

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Agent returned an invalid answer.")

        if not isinstance(should_continue, bool):
            raise ValueError("The conversation field must be a boolean.")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Clear the pending operation only after a valid response.
        st.session_state.pending_operation = None

        if not should_continue:
            st.session_state.conversation_closed = True

    except Exception as exc:
        error = exc
        logger.exception(
            "Agent failed: thread=%s turn=%s",
            thread_id,
            turn_number,
        )

        # Preserve the original message and request ID.
        pending["failed"] = True
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )

    elapsed = perf_counter() - started

    try:
        trace_record = write_agent_trace(
            thread_id=thread_id,
            turn_number=turn_number,
            elapsed_seconds=elapsed,
            tool_callback=callback,
            response=response,
            error=error,
        )

        if not isinstance(trace_record, dict):
            raise TypeError("write_agent_trace must return a dictionary.")

        st.session_state.trace_records.append(trace_record)

    except Exception:
        logger.exception(
            "Failed to persist trace: thread=%s turn=%s",
            thread_id,
            turn_number,
        )
        st.session_state.trace_warning = True
        st.session_state.trace_records.append(
            {
                "trace_id": f"local-{uuid4()}",
                "turn_number": turn_number,
                "execution_time_seconds": elapsed,
                "status": "error" if error else "completed",
                "tool_call_count": 0,
                "successful_tool_count": 0,
                "failed_tool_count": 0,
                "tool_events": [],
                "trace_unavailable": True,
            }
        )

    if (
        st.session_state.pending_operation is None
        and turn_number >= MAX_CONVERSATION_TURNS
    ):
        st.session_state.conversation_closed = True


def process_message(user_text):
    if (
        st.session_state.conversation_closed
        or st.session_state.turn_number >= MAX_CONVERSATION_TURNS
        or st.session_state.pending_operation is not None
    ):
        return

    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.messages.append(
        {"role": "user", "content": user_text}
    )
    st.session_state.turn_number += 1

    # One ID per operation, retained for every retry.
    st.session_state.pending_operation = {
        "user_text": user_text,
        "refund_request_id": str(uuid4()),
        "turn_number": st.session_state.turn_number,
        "failed": False,
    }

    execute_pending_operation()


# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --aa-bg: #0C1116;
    --aa-surface: #131A21;
    --aa-raised: #19222B;
    --aa-border: #25303B;
    --aa-border-strong: #33414F;
    --aa-text: #E4E9EE;
    --aa-muted: #8795A4;
    --aa-accent: #5EB5C4;
    --aa-accent-ink: #08242A;
    --aa-ok: #62C48F;
    --aa-warn: #E2B65A;
    --aa-err: #E57B72;
}

/* Base ------------------------------------------------- */
html, body, .stApp,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
button, input, textarea {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--aa-bg);
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Hide Streamlit chrome that doesn't belong in a product UI */
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
footer {
    display: none !important;
}

.block-container {
    max-width: 1360px;
    padding-top: 2.75rem;
    padding-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: var(--aa-surface);
    border-right: 1px solid var(--aa-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.25rem;
}

/* Header ------------------------------------------------ */
.aa-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
}

.aa-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.aa-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    flex-shrink: 0;
    border: 1px solid var(--aa-border-strong);
    border-radius: 9px;
    background: var(--aa-raised);
    color: var(--aa-accent);
}

.aa-title {
    color: var(--aa-text);
    font-size: 19px;
    font-weight: 600;
    letter-spacing: -0.2px;
    line-height: 1.25;
}

.aa-subtitle {
    margin-top: 2px;
    color: var(--aa-muted);
    font-size: 12.5px;
    line-height: 1.4;
}

.aa-header-pills {
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Pills ------------------------------------------------- */
.aa-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 3px 10px;
    border: 1px solid var(--aa-border-strong);
    border-radius: 999px;
    background: var(--aa-surface);
    color: var(--aa-muted);
    font-size: 12px;
    line-height: 1.5;
    white-space: nowrap;
}

.aa-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--aa-muted);
}

.aa-dot-ok { background: var(--aa-ok); }
.aa-dot-warn { background: var(--aa-warn); }
.aa-dot-err { background: var(--aa-err); }

/* Metrics strip ----------------------------------------- */
.aa-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    margin-bottom: 16px;
    border: 1px solid var(--aa-border);
    border-radius: 12px;
    background: var(--aa-surface);
}

.aa-metric {
    padding: 12px 18px;
    border-left: 1px solid var(--aa-border);
}

.aa-metric:first-child {
    border-left: none;
}

.aa-metric-label {
    color: var(--aa-muted);
    font-size: 12px;
    line-height: 1.3;
}

.aa-metric-value {
    margin-top: 3px;
    color: var(--aa-text);
    font-size: 21px;
    font-weight: 600;
    letter-spacing: -0.3px;
    line-height: 1.25;
    font-variant-numeric: tabular-nums;
}

.aa-metric-note {
    margin-top: 1px;
    color: var(--aa-muted);
    font-size: 11.5px;
    line-height: 1.3;
}

/* Panels ------------------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--aa-border);
    border-radius: 12px;
    background: var(--aa-surface);
}

.aa-panel-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--aa-border);
}

.aa-panel-title {
    color: var(--aa-text);
    font-size: 14px;
    font-weight: 600;
    line-height: 1.2;
}

/* Chat -------------------------------------------------- */
[data-testid="stChatMessage"] {
    margin-bottom: 8px;
    padding: 12px 14px;
    border: 1px solid var(--aa-border);
    border-radius: 10px;
    background: var(--aa-raised);
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-color: #26454D;
    background: #14232A;
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
    border: 1px solid var(--aa-border-strong);
    border-radius: 8px;
    background: var(--aa-surface) !important;
    color: var(--aa-accent) !important;
}

[data-testid="stChatMessage"] p {
    color: var(--aa-text);
    font-size: 14.5px;
    line-height: 1.6;
}

[data-testid="stChatInput"] {
    border: 1px solid var(--aa-border-strong);
    border-radius: 10px;
    background: var(--aa-raised);
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--aa-accent);
}

/* Buttons ----------------------------------------------- */
.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    min-height: 38px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}

.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    justify-content: flex-start;
    border: 1px solid var(--aa-border-strong);
    background: var(--aa-raised);
    color: var(--aa-text);
    text-align: left;
}

.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--aa-accent);
    color: var(--aa-accent);
}

.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    border: 1px solid var(--aa-accent);
    background: var(--aa-accent);
    color: var(--aa-accent-ink);
}

.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    border-color: #7CC8D5;
    background: #7CC8D5;
    color: var(--aa-accent-ink);
}

/* Expanders and trace ----------------------------------- */
[data-testid="stExpander"] {
    margin-bottom: 8px;
    border: 1px solid var(--aa-border);
    border-radius: 10px;
    background: var(--aa-raised);
}

[data-testid="stExpander"] summary {
    font-size: 13px;
    font-weight: 500;
}

.aa-empty {
    padding: 90px 16px 0;
    text-align: center;
}

.aa-empty-title {
    color: var(--aa-text);
    font-size: 14px;
    font-weight: 600;
}

.aa-empty-text {
    max-width: 260px;
    margin: 8px auto 0;
    color: var(--aa-muted);
    font-size: 12.5px;
    line-height: 1.65;
}

.aa-tool {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid var(--aa-border);
}

.aa-tool-main {
    display: flex;
    align-items: flex-start;
    gap: 10px;
}

.aa-tool-main .aa-dot {
    margin-top: 6px;
}

.aa-tool-name {
    color: var(--aa-text);
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12.5px;
    overflow-wrap: anywhere;
}

.aa-tool-status {
    margin-top: 2px;
    color: var(--aa-muted);
    font-size: 12px;
}

.aa-tool-time {
    color: var(--aa-muted);
    font-size: 12px;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
}

/* Sidebar ----------------------------------------------- */
.aa-side-label {
    margin: 18px 0 6px;
    color: var(--aa-text);
    font-size: 13px;
    font-weight: 600;
}

[data-testid="stCaptionContainer"] {
    color: var(--aa-muted);
}

@media (max-width: 900px) {
    .aa-metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .aa-metric:nth-child(3) {
        border-left: none;
    }

    .aa-metric:nth-child(n + 3) {
        border-top: 1px solid var(--aa-border);
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Display helpers
# ---------------------------------------------------------


def esc(value):
    return escape(str(value))


def number(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def duration(value):
    seconds = number(value)

    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"

    return f"{seconds:.2f} s"


def status_tone(status):
    value = str(status).lower()

    if value in {"success", "succeeded", "completed", "ok", "done"}:
        return "ok"

    if value in {"error", "failed", "failure", "exception"}:
        return "err"

    return ""


def pill(label, tone=""):
    dot_class = f"aa-dot-{tone}" if tone else ""
    return (
        f'<span class="aa-pill">'
        f'<span class="aa-dot {dot_class}"></span>{esc(label)}'
        f"</span>"
    )


def render_header(closed):
    state_pill = pill("Closed", "warn") if closed else pill("Ready", "ok")

    st.markdown(
        f"""
<div class="aa-header">
<div class="aa-brand">
<div class="aa-logo">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l7.5 3v5.5c0 4.6-3.1 8.2-7.5 9.5-4.4-1.3-7.5-4.9-7.5-9.5V6L12 3z"/><path d="M8.8 12.2l2.2 2.2 4.2-4.4"/></svg>
</div>
<div>
<div class="aa-title">Refund support workspace</div>
<div class="aa-subtitle">Policy-controlled conversations with an inspectable execution trace.</div>
</div>
</div>
<div class="aa-header-pills">{state_pill}{pill("Sandbox")}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(items):
    cells = "".join(
        f'<div class="aa-metric">'
        f'<div class="aa-metric-label">{esc(label)}</div>'
        f'<div class="aa-metric-value">{esc(value)}</div>'
        f'<div class="aa-metric-note">{esc(note)}</div>'
        f"</div>"
        for label, value, note in items
    )

    st.markdown(
        f'<div class="aa-metrics">{cells}</div>',
        unsafe_allow_html=True,
    )


def panel_heading(title, badge_html):
    st.markdown(
        f"""
<div class="aa-panel-heading">
<div class="aa-panel-title">{esc(title)}</div>
{badge_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_trace(record, expanded=False):
    turn = record.get("turn_number", "-")
    elapsed = duration(record.get("execution_time_seconds"))

    with st.expander(f"Turn {turn}, {elapsed}", expanded=expanded):
        if record.get("trace_unavailable"):
            st.warning(
                "Detailed tracing was unavailable for this turn. "
                "Processing time is retained locally."
            )

        status = record.get("status", "unknown")
        events = record.get("tool_events") or []
        count = record.get("tool_call_count", len(events))
        noun = "tool call" if count == 1 else "tool calls"

        st.caption(f"{str(status).capitalize()}, {count} {noun}")

        if not events and not record.get("trace_unavailable"):
            st.caption("No tool calls were made in this turn.")

        for event in events:
            name = event.get("tool_name", "Unknown tool")
            sequence = event.get("sequence", "-")
            event_status = event.get("status", "unknown")
            event_time = duration(event.get("duration_seconds"))
            tone = status_tone(event_status)
            dot_class = f"aa-dot-{tone}" if tone else ""

            st.markdown(
                f"""
<div class="aa-tool">
<div class="aa-tool-main">
<span class="aa-dot {dot_class}"></span>
<div>
<div class="aa-tool-name">{esc(sequence)}. {esc(name)}</div>
<div class="aa-tool-status">{esc(event_status)}</div>
</div>
</div>
<div class="aa-tool-time">{esc(event_time)}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.show_tool_details:
                st.caption("Arguments")
                st.json(event.get("arguments", {}), expanded=False)

                st.caption("Result")
                st.json(event.get("result", {}), expanded=False)

                if event.get("error"):
                    st.caption("Tool error")
                    st.json(event["error"], expanded=False)

        if record.get("error"):
            st.warning("This turn ended with an execution error.")

            if st.session_state.show_tool_details:
                st.json(record["error"], expanded=False)

        st.caption("Trace reference")
        st.code(
            str(record.get("trace_id", "Unavailable")),
            language=None,
        )


# ---------------------------------------------------------
# Session status
# ---------------------------------------------------------

pending_operation = st.session_state.pending_operation

turn_limit_reached = (
    st.session_state.turn_number >= MAX_CONVERSATION_TURNS
    and pending_operation is None
)

if turn_limit_reached:
    st.session_state.conversation_closed = True

closed = st.session_state.conversation_closed


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("### AgentAudit")
    st.caption("Observable refund support")

    st.markdown('<div class="aa-side-label">Workspace</div>', unsafe_allow_html=True)

    st.button(
        "New conversation",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pending_operation is not None,
        on_click=start_new_conversation,
    )

    st.markdown('<div class="aa-side-label">Session</div>', unsafe_allow_html=True)

    st.caption(f"{st.session_state.turn_number} of {MAX_CONVERSATION_TURNS} turns used")

    st.progress(min(st.session_state.turn_number / MAX_CONVERSATION_TURNS, 1.0))

    with st.expander("Session details"):
        st.caption("Conversation ID")
        st.code(st.session_state.thread_id, language=None)

    st.markdown('<div class="aa-side-label">Inspection</div>', unsafe_allow_html=True)

    st.toggle(
        "Inspect tool payloads",
        key="show_tool_details",
        help=(
            "Shows raw tool arguments, results and errors. "
            "This switch is not authentication or access control."
        ),
    )

    st.caption("Use synthetic customer data in public demos.")

    st.divider()

    st.caption(
        "Sandbox environment. Refund policy is enforced by deterministic tools. "
        "Ticket issuance is simulated and no bank transfer is performed."
    )


# ---------------------------------------------------------
# Header and metrics
# ---------------------------------------------------------

render_header(closed)

traces = st.session_state.trace_records

complete_traces = [record for record in traces if not record.get("trace_unavailable")]

total_tools = sum(
    int(number(record.get("tool_call_count"))) for record in complete_traces
)

successful_tools = sum(
    int(number(record.get("successful_tool_count"))) for record in complete_traces
)

failed_tools = sum(
    int(number(record.get("failed_tool_count"))) for record in complete_traces
)

total_time = sum(number(record.get("execution_time_seconds")) for record in traces)

has_missing_traces = any(record.get("trace_unavailable") for record in traces)

tool_count_display = str(total_tools)

success_display = f"{successful_tools / total_tools:.0%}" if total_tools else "-"

tool_detail = f"{successful_tools} succeeded, {failed_tools} failed"

if has_missing_traces:
    tool_count_display = f"{total_tools}+"
    tool_detail = "Partial tracing data"

render_metrics(
    [
        (
            "Conversation turns",
            f"{st.session_state.turn_number} / {MAX_CONVERSATION_TURNS}",
            "Current session",
        ),
        ("Tool executions", tool_count_display, tool_detail),
        ("Tool success rate", success_display, "Recorded executions only"),
        ("Processing time", duration(total_time), "Cumulative execution"),
    ]
)

if st.session_state.trace_warning:
    st.warning(
        "Some detailed traces could not be saved, so tool metrics may be "
        "incomplete. Check the application logs."
    )


# ---------------------------------------------------------
# Workspace
# ---------------------------------------------------------

conversation_column, trace_column = st.columns([1.6, 1], gap="medium")

user_input = None
retry_clicked = False

with conversation_column:
    with st.container(border=True):
        panel_heading(
            "Conversation", pill("Closed", "warn") if closed else pill("Ready", "ok")
        )

        with st.container(height=CHAT_BODY_HEIGHT, border=False):
            for message in st.session_state.messages:
                avatar = (
                    ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
                )

                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            if st.session_state.turn_number == 0 and not closed:
                st.caption("Try one of these to get started")

                for index, prompt in enumerate(SUGGESTED_PROMPTS):
                    st.button(
                        prompt,
                        key=f"suggested_prompt_{index}",
                        use_container_width=True,
                        on_click=queue_prompt,
                        args=(prompt,),
                    )

        if st.session_state.pending_operation is not None:
            st.warning(
                "A previous request has an unconfirmed outcome. "
                "Retry it before submitting another request."
            )
            retry_clicked = st.button(
                "Retry previous request",
                key="retry_pending_operation",
                type="primary",
                use_container_width=True,
            )

        user_input = st.chat_input(
            "Describe your order issue",
            key="refund_message",
            disabled=(
                closed
                or st.session_state.pending_operation is not None
            ),
            max_chars=1000,
        )

        if st.session_state.pending_operation is not None:
            st.caption(
                "Retry reuses the original request ID and conversation turn."
            )
        elif turn_limit_reached:
            st.caption("Turn limit reached. Start a new conversation to continue.")
        elif closed:
            st.caption("This conversation is closed. Start a new one to continue.")
        else:
            st.caption("Include your order number. Do not enter payment details.")


with trace_column:
    with st.container(border=True):
        panel_heading("Execution trace", pill(f"{len(traces)} recorded"))

        with st.container(height=TRACE_BODY_HEIGHT, border=False):
            if not traces:
                st.markdown(
                    """
<div class="aa-empty">
<div class="aa-empty-title">No activity yet</div>
<div class="aa-empty-text">Send a message to see each tool call with its order, timing and result.</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

            for index, record in enumerate(reversed(traces)):
                render_trace(record, expanded=index == 0)

        st.caption(
            "Turn on “Inspect tool payloads” in the sidebar for arguments and results. "
            "Success means the tool ran, not that the policy decision was correct."
        )


# ---------------------------------------------------------
# Request processing
# ---------------------------------------------------------

prompt = user_input or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if retry_clicked:
    with st.spinner("Retrying the original request..."):
        execute_pending_operation()

    st.rerun()

elif (
    prompt
    and prompt.strip()
    and not closed
    and st.session_state.pending_operation is None
):
    with st.spinner("Reviewing your request..."):
        process_message(prompt)

    st.rerun()
