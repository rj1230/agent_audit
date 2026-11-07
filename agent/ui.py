import logging
import sys
from html import escape
from pathlib import Path
from uuid import uuid4

# Ensure the repository root wins over Streamlit's script-directory path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from agent.agent import agent
from agent.tracing import ToolTraceCallback, write_agent_trace
from agent.retry import build_agent_input
from agent.execution import run_pending_operation
from agent.approval import (
    confirm_pending_approval as confirm_approval,
    decline_pending_approval as decline_approval,
)
from agent.objects import PurchaseItem
from agent.tools import issue_approved_refund_ticket


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgentAudit | Refund operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 8
PANEL_BODY_HEIGHT = 400  # decision inspector and execution trace bodies
CHAT_BODY_HEIGHT = 330  # leaves room for the chat input beneath the transcript

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
        "pending_approval": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_conversation():
    if (
        st.session_state.pending_operation is not None
        or st.session_state.pending_approval is not None
    ):
        return

    st.session_state.thread_id = str(uuid4())
    st.session_state.turn_number = 0
    st.session_state.messages = initial_messages()
    st.session_state.trace_records = []
    st.session_state.conversation_closed = False
    st.session_state.trace_warning = False
    st.session_state.pending_prompt = None
    st.session_state.pending_operation = None
    st.session_state.pending_approval = None


def queue_prompt(prompt):
    st.session_state.pending_prompt = prompt


initialize_session()


# ---------------------------------------------------------
# Agent execution — intentionally preserved
# ---------------------------------------------------------


def execute_pending_operation():
    return run_pending_operation(
        session_state=st.session_state,
        agent=agent,
        callback_factory=ToolTraceCallback,
        trace_writer=write_agent_trace,
        input_builder=build_agent_input,
        logger=logger,
        max_conversation_turns=MAX_CONVERSATION_TURNS,
    )


def process_message(user_text):
    if (
        st.session_state.conversation_closed
        or st.session_state.turn_number >= MAX_CONVERSATION_TURNS
        or st.session_state.pending_operation is not None
        or st.session_state.pending_approval is not None
    ):
        return

    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.turn_number += 1

    # One ID per operation, retained for every retry.
    st.session_state.pending_operation = {
        "user_text": user_text,
        "refund_request_id": str(uuid4()),
        "turn_number": st.session_state.turn_number,
        "failed": False,
    }

    execute_pending_operation()


def confirm_pending_approval():
    """Streamlit callback for explicitly confirming a refund quote."""
    confirm_approval(
        st.session_state,
        ticket_issuer=issue_approved_refund_ticket,
        logger=logger,
    )

def decline_pending_approval():
    """Discard the pending quote without issuing a ticket."""
    decline_approval(st.session_state)


# ---------------------------------------------------------
# Theme
# ---------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --aa-bg: #080E16;
    --aa-sidebar: #0C1520;
    --aa-surface: #101B28;
    --aa-surface-2: #142233;
    --aa-surface-3: #192B3E;
    --aa-border: #25384B;
    --aa-border-soft: #1D2D3D;
    --aa-text: #EAF1F7;
    --aa-muted: #91A4B8;
    --aa-subtle: #7A8FA5;
    --aa-teal: #55D6BE;
    --aa-teal-dark: #123B3B;
    --aa-green: #65D6A2;
    --aa-amber: #E9B968;
    --aa-red: #F08080;
    --aa-blue: #85B8F5;
}

html, body, .stApp, [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"], button, input, textarea {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}

* { scrollbar-width: thin; scrollbar-color: #2A4156 transparent; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--aa-bg);
    color: var(--aa-text);
}

/* Transparent header: it floats above the page, so the content below
   needs enough top padding to stay clear of it. */
[data-testid="stHeader"] { background: transparent; }

[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
footer {
    display: none !important;
}

.block-container {
    max-width: 1480px;
    padding-top: 3.6rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: var(--aa-sidebar);
    border-right: 1px solid var(--aa-border-soft);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.2rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 12px; line-height: 1.55;
}

p, li, label, [data-testid="stMarkdownContainer"] {
    color: var(--aa-text);
}
[data-testid="stCaptionContainer"] {
    color: var(--aa-muted) !important;
    font-size: 12px; line-height: 1.5;
}
hr {
    border-color: var(--aa-border-soft) !important;
}

/* Header: one compact row */
.aa-topbar {
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; flex-wrap:wrap; margin:0 0 16px 0;
}
.aa-brand { display:flex; align-items:center; gap:14px; }
.aa-mark {
    width:46px; height:46px; flex:none; display:flex; align-items:center;
    justify-content:center; border-radius:13px;
    color:var(--aa-teal); background:#123039;
    border:1px solid #28505A;
}
.aa-brand-sub {
    color:var(--aa-muted); font-size:10.5px; font-weight:600;
    letter-spacing:.1em; text-transform:uppercase;
}
.aa-brand-sub b { color:var(--aa-teal); font-weight:700; letter-spacing:.12em; }
.aa-page-title {
    color:var(--aa-text); font-size:21px; font-weight:600;
    letter-spacing:-.4px; margin-top:3px; line-height:1.25;
}
.aa-page-subtitle { color:var(--aa-muted); font-size:12.5px; margin-top:2px; }
.aa-header-right { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }

/* Status pills */
.aa-pill {
    display:inline-flex; align-items:center; gap:7px;
    padding:5px 10px; border-radius:999px;
    border:1px solid var(--aa-border); background:var(--aa-surface);
    color:var(--aa-muted); font-size:11px; font-weight:600;
    white-space:nowrap;
}
.aa-dot { width:7px; height:7px; border-radius:50%; display:inline-block; background:var(--aa-subtle); flex:none; }
.aa-dot-ok { background:var(--aa-green); }
.aa-dot-warn { background:var(--aa-amber); }
.aa-dot-err { background:var(--aa-red); }
.aa-dot-info { background:var(--aa-blue); }

/* Metric cards */
.aa-metrics {
    display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
    gap:10px; margin:0 0 14px 0;
}
.aa-metric {
    min-width:0; padding:10px 14px 11px;
    background:linear-gradient(145deg,#142233,#101B28);
    border:1px solid var(--aa-border); border-radius:12px;
}
.aa-metric-label {
    color:var(--aa-muted); font-size:10px; font-weight:600;
    text-transform:uppercase; letter-spacing:.08em;
}
.aa-metric-body {
    display:flex; align-items:baseline; flex-wrap:wrap;
    gap:2px 10px; margin-top:4px;
}
.aa-metric-value {
    color:var(--aa-text); font-size:22px; font-weight:600;
    letter-spacing:-.5px; line-height:1.2;
    font-variant-numeric:tabular-nums;
}
.aa-metric-value.ok { color:var(--aa-green); }
.aa-metric-value.warn { color:var(--aa-amber); }
.aa-metric-value.err { color:var(--aa-red); }
.aa-metric-note { color:var(--aa-subtle); font-size:11px; }

/* Panel surfaces: keyed containers (recent Streamlit) */
[class*="st-key-panel_"] {
    background:var(--aa-surface) !important;
    border:1px solid var(--aa-border) !important;
    border-radius:14px !important;
}
/* Make the three panels share one height per row (best effort). */
[data-testid="stColumn"]:has([class*="st-key-panel_"]) > [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has([class*="st-key-panel_"]) > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
[class*="st-key-panel_"] {
    height:100%;
}
a, a:visited { color:var(--aa-teal) !important; }
.aa-panel-heading {
    display:flex; align-items:center; justify-content:space-between;
    gap:10px; padding-bottom:12px; margin-bottom:8px;
    border-bottom:1px solid var(--aa-border-soft);
}
.aa-panel-title { color:var(--aa-text); font-size:14px; font-weight:600; white-space:nowrap; }
.aa-panel-kicker {
    color:var(--aa-subtle); font-size:10px; font-weight:600;
    letter-spacing:.09em; text-transform:uppercase; margin-bottom:4px;
    white-space:nowrap;
}
.aa-section-label {
    color:var(--aa-muted); font-size:10px; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase; margin:2px 0 6px;
}

/* Chat */
[data-testid="stChatMessage"] {
    background:var(--aa-surface-2);
    border:1px solid var(--aa-border-soft);
    border-radius:12px; padding:10px 12px;
}
[data-testid="stChatMessage"]:has(.aa-user-marker) {
    background:#122833; border-color:#28505A;
}
/* Marker used only to tell user messages apart; takes no space. */
[data-testid="stElementContainer"]:has(> .stMarkdown .aa-user-marker) {
    display:none;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color:var(--aa-text); font-size:13.5px; line-height:1.6;
}
[data-testid^="stChatMessageAvatar"] {
    background:#123039 !important;
    border:1px solid #28505A; border-radius:9px;
}
[data-testid^="stChatMessageAvatar"] * { color:var(--aa-teal) !important; }
[data-testid="stChatInput"] {
    border:1px solid var(--aa-border) !important;
    border-radius:12px !important; background:var(--aa-surface-2) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color:var(--aa-teal) !important;
    box-shadow:0 0 0 1px rgba(85,214,190,.12);
}
[data-testid="stChatInput"] textarea { color:var(--aa-text) !important; }

/* Buttons */
.stButton > button {
    min-height:38px; border-radius:9px;
    font-size:12.5px; font-weight:600;
    transition:all .15s ease;
}
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background:#172638 !important; color:var(--aa-text) !important;
    border:1px solid var(--aa-border) !important;
}
.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color:var(--aa-teal) !important; color:var(--aa-teal) !important;
}
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background:var(--aa-teal) !important; color:#06231F !important;
    border:1px solid var(--aa-teal) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background:#7CE4D1 !important; border-color:#7CE4D1 !important;
    color:#06231F !important;
}
.stButton > button:disabled { opacity:.42; }
.stButton > button:focus-visible {
    outline:2px solid var(--aa-teal); outline-offset:2px;
}

/* Quick-start prompts read as suggestion rows, not centred buttons */
[class*="st-key-suggested_prompt_"] button { justify-content:flex-start; }
[class*="st-key-suggested_prompt_"] button > div { width:100%; justify-content:flex-start; }
[class*="st-key-suggested_prompt_"] button p {
    text-align:left; font-weight:500; font-size:12.5px;
}

/* Approval card */
[class*="st-key-approval_card"] {
    background:var(--aa-surface-2) !important;
    border:1px solid #31515B !important;
    border-radius:12px !important;
}
.aa-quote-amount {
    color:#F1FAF8; font-size:26px; font-weight:600;
    letter-spacing:-.5px; margin:2px 0 2px;
    font-variant-numeric:tabular-nums;
}
.aa-quote-item { color:var(--aa-text); font-size:13px; font-weight:600; }
.aa-quote-meta { color:var(--aa-muted); font-size:11.5px; margin-bottom:6px; }

/* Alerts */
[data-testid="stAlertContainer"] {
    border-radius:10px !important;
    border:1px solid var(--aa-border-soft);
}
[data-testid="stAlertContainer"] p { font-size:12.5px; line-height:1.55; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    background:rgba(133,184,245,.08) !important; border-color:rgba(133,184,245,.28);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    background:rgba(233,185,104,.09) !important; border-color:rgba(233,185,104,.30);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    background:rgba(101,214,162,.09) !important; border-color:rgba(101,214,162,.30);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    background:rgba(240,128,128,.09) !important; border-color:rgba(240,128,128,.30);
}

/* Code blocks (trace and conversation IDs) */
[data-testid="stCode"] pre, [data-testid="stCode"] code { font-size:11.5px !important; }

/* Trace */
[data-testid="stExpander"] {
    background:var(--aa-surface-2);
    border:1px solid var(--aa-border-soft);
    border-radius:10px;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    background:transparent !important;
}
[data-testid="stExpander"] summary {
    color:var(--aa-text); font-size:12.5px; font-weight:600;
}
.aa-event {
    display:flex; justify-content:space-between; align-items:flex-start;
    gap:12px; padding:10px 0; border-bottom:1px solid var(--aa-border-soft);
}
.aa-event-left { display:flex; align-items:flex-start; gap:10px; min-width:0; }
.aa-event-name {
    color:var(--aa-text); font-family:'IBM Plex Mono',monospace;
    font-size:11.5px; overflow-wrap:anywhere;
}
.aa-event-status { color:var(--aa-muted); font-size:11px; margin-top:3px; }
.aa-event-time {
    color:var(--aa-muted); font-size:11px; white-space:nowrap;
    font-variant-numeric:tabular-nums;
}
.aa-empty {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; padding:0 16px;
}
.aa-empty-icon { color:var(--aa-teal); margin-bottom:12px; opacity:.9; }
.aa-empty-title { color:var(--aa-text); font-size:13px; font-weight:600; }
.aa-empty-copy {
    max-width:260px; margin:7px auto 0; color:var(--aa-muted);
    font-size:11.5px; line-height:1.6;
}

/* Decision inspector */
.aa-decision-card {
    padding:15px; border-radius:12px;
    border:1px solid #31515B;
    background:linear-gradient(145deg,#142D35,#12212E);
    margin:4px 0 12px;
}
.aa-decision-label {
    color:#9BB9BC; font-size:10px; font-weight:600;
    text-transform:uppercase; letter-spacing:.09em;
}
.aa-decision-amount {
    color:#F1FAF8; font-size:27px; font-weight:600;
    letter-spacing:-.5px; margin-top:5px;
}
.aa-decision-note { color:#9BB9BC; font-size:11px; margin-top:3px; }
.aa-check {
    display:flex; align-items:flex-start; gap:8px;
    color:#DCE9E9; font-size:12px; margin:9px 0;
}
.aa-check-mark { color:var(--aa-teal); font-weight:700; }
.aa-muted-box {
    border:1px solid var(--aa-border-soft); border-radius:10px;
    background:var(--aa-surface-2); padding:14px;
}
.aa-steps { margin-top:14px; }
.aa-step {
    display:flex; align-items:flex-start; gap:12px;
    position:relative; padding-bottom:14px;
}
.aa-step:last-child { padding-bottom:0; }
.aa-step:not(:last-child)::before {
    content:""; position:absolute; left:9px; top:22px; bottom:2px;
    width:1px; background:var(--aa-border);
}
.aa-step-marker {
    width:19px; height:19px; flex:none; border-radius:50%;
    border:1.5px solid var(--aa-border); background:var(--aa-surface-2);
    color:var(--aa-subtle); font-size:10px; font-weight:700;
    display:flex; align-items:center; justify-content:center;
}
.aa-step-title { color:var(--aa-text); font-size:12.5px; font-weight:600; line-height:19px; }
.aa-step-copy { color:var(--aa-muted); font-size:11.5px; line-height:1.5; }
.aa-foot { color:var(--aa-subtle); font-size:11px; line-height:1.6; margin-top:12px; }
.aa-mono {
    font-family:'IBM Plex Mono',monospace; overflow-wrap:anywhere;
}

/* Sidebar */
.aa-side-brand {
    display:flex; align-items:center; gap:10px;
    color:var(--aa-text); font-size:16px; font-weight:700;
}
.aa-side-kicker {
    color:var(--aa-muted); font-size:10px; letter-spacing:.1em;
    text-transform:uppercase; margin-top:3px;
}
.aa-side-label {
    color:var(--aa-muted); font-size:10px; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase;
    margin:20px 0 7px;
}
[data-testid="stProgressBarTrack"] {
    background:#1B2B3B !important; border-radius:999px; height:6px;
}
[data-testid="stProgressBarTrack"] > div {
    background:var(--aa-teal) !important; border-radius:999px;
}
[data-testid="stToggle"] label { color:var(--aa-text) !important; }

/* Responsive */
@media (max-width: 1050px) {
    .aa-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width: 700px) {
    .block-container { padding:3.4rem .8rem 2rem; }
    .aa-page-title { font-size:19px; }
    .aa-metrics { gap:7px; }
    .aa-metric { padding:9px 11px; }
    .aa-metric-value { font-size:20px; }
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
    if value in {"pending", "waiting", "approval_required"}:
        return "warn"
    return ""


def pill(label, tone=""):
    dot_class = f"aa-dot-{tone}" if tone else ""
    return (
        f'<span class="aa-pill">'
        f'<span class="aa-dot {dot_class}"></span>{esc(label)}'
        f"</span>"
    )


def current_session_status():
    if st.session_state.pending_operation is not None:
        return "Retry required", "warn"
    if st.session_state.pending_approval is not None:
        return "Approval required", "warn"
    if st.session_state.conversation_closed:
        return "Closed", "info"
    if st.session_state.turn_number >= MAX_CONVERSATION_TURNS:
        return "Turn limit reached", "warn"
    return "Ready", "ok"


def bordered(key, **kwargs):
    """Bordered container with a stable CSS hook (st-key-<key>).

    Falls back to an unkeyed container on Streamlit builds that predate
    container keys, so the app still runs (just without the panel styling).
    """
    try:
        return st.container(border=True, key=key, **kwargs)
    except TypeError:
        return st.container(border=True, **kwargs)


def section_label(text):
    st.markdown(
        f'<div class="aa-section-label">{esc(text)}</div>',
        unsafe_allow_html=True,
    )


def inspector_status():
    if st.session_state.pending_approval is not None:
        return "Pending", "warn"
    if st.session_state.pending_operation is not None:
        return "Unconfirmed", "warn"
    if st.session_state.conversation_closed:
        return "Closed", "info"
    return "Idle", ""


def render_header():
    label, tone = current_session_status()
    st.markdown(
        f"""
<div class="aa-topbar">
  <div class="aa-brand">
    <div class="aa-mark">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
           stroke-linejoin="round">
        <path d="M12 3l7.5 3v5.5c0 4.6-3.1 8.2-7.5 9.5-4.4-1.3-7.5-4.9-7.5-9.5V6L12 3z"/>
        <path d="M8.8 12.2l2.2 2.2 4.2-4.4"/>
      </svg>
    </div>
    <div>
      <div class="aa-brand-sub"><b>AgentAudit</b> · Refund operations · Observable AI</div>
      <div class="aa-page-title">Operations workspace</div>
      <div class="aa-page-subtitle">Review customer requests, policy decisions and agent execution.</div>
    </div>
  </div>
  <div class="aa-header-right">
    {pill("Sandbox", "info")}
    {pill(label, tone)}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(items):
    """items: (label, value, note) or (label, value, note, tone)."""
    cells = []
    for item in items:
        label, value, note = item[:3]
        tone = item[3] if len(item) > 3 else ""
        cells.append(
            f'<div class="aa-metric">'
            f'<div class="aa-metric-label">{esc(label)}</div>'
            f'<div class="aa-metric-body">'
            f'<span class="aa-metric-value {esc(tone)}">{esc(value)}</span>'
            f'<span class="aa-metric-note">{esc(note)}</span>'
            f"</div></div>"
        )
    st.markdown(
        f'<div class="aa-metrics">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def panel_heading(title, badge_html, kicker=None):
    kicker_html = f'<div class="aa-panel-kicker">{esc(kicker)}</div>' if kicker else ""
    st.markdown(
        f"""
<div class="aa-panel-heading">
  <div>{kicker_html}<div class="aa-panel-title">{esc(title)}</div></div>
  {badge_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_trace(record, expanded=False):
    turn = record.get("turn_number", "-")
    elapsed = duration(record.get("execution_time_seconds"))
    status = record.get("status", "unknown")
    events = record.get("tool_events") or []
    count = record.get("tool_call_count", len(events))

    with st.expander(
        f"Turn {turn}  ·  {str(status).upper()}  ·  {elapsed}", expanded=expanded
    ):
        if record.get("trace_unavailable"):
            st.warning(
                "Detailed tracing was unavailable for this turn. "
                "Processing time is retained locally."
            )

        noun = "tool call" if count == 1 else "tool calls"
        st.caption(f"{count} {noun} recorded for this turn.")

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
<div class="aa-event">
  <div class="aa-event-left">
    <span class="aa-dot {dot_class}" style="margin-top:5px"></span>
    <div>
      <div class="aa-event-name">{esc(sequence)}. {esc(name)}</div>
      <div class="aa-event-status">{esc(event_status)}</div>
    </div>
  </div>
  <div class="aa-event-time">{esc(event_time)}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

            if st.session_state.show_tool_details:
                with st.expander(f"Payload · {name}", expanded=False):
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
        st.code(str(record.get("trace_id", "Unavailable")), language=None)


POLICY_STEPS = [
    ("Customer verification", "Confirm who is making the request."),
    ("Purchase matching", "Find the order and items in question."),
    ("Eligibility check", "Apply the refund policy to each item."),
    ("Refund quote", "Propose an amount for your explicit approval."),
]


def render_decision_inspector():
    approval = st.session_state.pending_approval
    operation = st.session_state.pending_operation

    if isinstance(approval, dict):
        quote = approval.get("quote") or {}
        amount = quote.get("refund_amount", "Unavailable")
        st.markdown(
            f"""
<div class="aa-decision-card">
  <div class="aa-decision-label">Proposed refund amount</div>
  <div class="aa-decision-amount">{esc(amount)}</div>
  <div class="aa-decision-note">Quote awaiting your explicit confirmation</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Customer · {approval.get('user', 'Unavailable')}")
        section_label("Quoted items")
        for item in quote.get("items", []):
            st.markdown(
                f'<div class="aa-check"><span class="aa-check-mark">•</span>'
                f"<span>{esc(item.get('item', 'Unknown item'))} · "
                f"{esc(item.get('price', 'Price unavailable'))} · "
                f"Purchased {esc(item.get('purchase_date', 'date unavailable'))}</span></div>",
                unsafe_allow_html=True,
            )
        st.info(
            "No ticket has been issued yet. Confirming authorizes simulated "
            "ticket creation; no money is transferred."
        )
        return

    if operation is not None:
        section_label("Request recovery")
        st.warning(
            "The previous operation has an unconfirmed outcome. Retry the "
            "same request before submitting another message."
        )
        st.caption("Original request ID")
        st.code(str(operation.get("refund_request_id", "Unavailable")), language=None)
        st.caption(
            "Retry retains this request ID to avoid creating a new logical request."
        )
        return

    if st.session_state.conversation_closed:
        section_label("Session outcome")
        st.success("This conversation is closed.")
        st.caption("Start a new conversation to submit another request.")
        return

    steps = "".join(
        f'<div class="aa-step">'
        f'<div class="aa-step-marker">{index}</div>'
        f'<div><div class="aa-step-title">{esc(title)}</div>'
        f'<div class="aa-step-copy">{esc(copy)}</div></div>'
        f"</div>"
        for index, (title, copy) in enumerate(POLICY_STEPS, start=1)
    )
    st.markdown(
        f"""
<div class="aa-muted-box">
  <div class="aa-decision-label">Waiting for a request</div>
  <div class="aa-steps">{steps}</div>
  <div class="aa-foot">Evidence for each check appears here as the workflow produces it.</div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Session status
# ---------------------------------------------------------

pending_operation = st.session_state.pending_operation
pending_approval = st.session_state.pending_approval

turn_limit_reached = (
    st.session_state.turn_number >= MAX_CONVERSATION_TURNS
    and pending_operation is None
    and pending_approval is None
)

if turn_limit_reached:
    st.session_state.conversation_closed = True

closed = st.session_state.conversation_closed


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
<div class="aa-side-brand">
  <span style="color:#55D6BE;font-size:20px">◈</span>
  <div>
    <div>AgentAudit</div>
    <div class="aa-side-kicker">Refund support console</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="aa-side-label">Workspace</div>', unsafe_allow_html=True)
    st.button(
        "＋  New conversation",
        type="primary",
        use_container_width=True,
        disabled=(
            st.session_state.pending_operation is not None
            or st.session_state.pending_approval is not None
        ),
        on_click=start_new_conversation,
    )

    st.caption(
        "Cross-session conversation recovery is disabled until "
        "authenticated ownership controls are implemented."
    )

    st.markdown(
        '<div class="aa-side-label">Session usage</div>', unsafe_allow_html=True
    )
    st.caption(f"{st.session_state.turn_number} of {MAX_CONVERSATION_TURNS} turns")
    st.progress(min(st.session_state.turn_number / MAX_CONVERSATION_TURNS, 1.0))

    with st.expander("Session details"):
        st.caption("Conversation ID")
        st.code(st.session_state.thread_id, language=None)

    st.markdown(
        '<div class="aa-side-label">Developer inspection</div>', unsafe_allow_html=True
    )
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

render_header()

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

tool_count_display = f"{total_tools}+" if has_missing_traces else str(total_tools)
success_display = f"{successful_tools / total_tools:.0%}" if total_tools else "—"
tool_detail = (
    "Partial tracing data"
    if has_missing_traces
    else f"{successful_tools} succeeded · {failed_tools} failed"
)

render_metrics(
    [
        (
            "Conversation turns",
            f"{st.session_state.turn_number}/{MAX_CONVERSATION_TURNS}",
            "Current session",
        ),
        ("Tool executions", tool_count_display, tool_detail),
        (
            "Tool success rate",
            success_display,
            "Recorded executions only",
            "" if not total_tools else ("warn" if failed_tools else "ok"),
        ),
        ("Processing time", duration(total_time), "Cumulative execution time"),
    ]
)

if st.session_state.trace_warning:
    st.warning(
        "Some detailed traces could not be saved, so tool metrics may be "
        "incomplete. Check the application logs."
    )


# ---------------------------------------------------------
# Main workspace
# ---------------------------------------------------------

conversation_column, inspector_column, trace_column = st.columns(
    [1.25, 0.92, 1.05],
    gap="medium",
)

user_input = None
retry_clicked = False

# Give the transcript less height while a retry or approval card needs room.
action_pending = (
    st.session_state.pending_operation is not None
    or st.session_state.pending_approval is not None
)
chat_height = max(CHAT_BODY_HEIGHT - 110, 200) if action_pending else CHAT_BODY_HEIGHT

with conversation_column:
    with bordered("panel_conversation"):
        panel_heading(
            "Conversation",
            pill("Closed", "info") if closed else pill("Live session", "ok"),
            kicker="Customer interaction",
        )

        with st.container(height=chat_height, border=False):
            for message in st.session_state.messages:
                avatar = (
                    ASSISTANT_AVATAR if message["role"] == "assistant" else USER_AVATAR
                )
                with st.chat_message(message["role"], avatar=avatar):
                    if message["role"] == "user":
                        # Constant, invisible marker so CSS can tint user turns.
                        st.markdown(
                            '<div class="aa-user-marker"></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(message["content"])

            if st.session_state.turn_number == 0 and not closed:
                section_label("Quick start")
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
                "The previous request has an unconfirmed outcome. "
                "Retry it before submitting another request."
            )
            retry_clicked = st.button(
                "Retry previous request",
                key="retry_pending_operation",
                type="primary",
                use_container_width=True,
            )

        approval = st.session_state.pending_approval
        if isinstance(approval, dict):
            approval_quote = approval.get("quote") or {}
            with bordered("approval_card"):
                section_label("Review refund quote")
                st.caption(f"Customer · {approval.get('user', 'Unavailable')}")
                for item in approval_quote.get("items", []):
                    st.markdown(
                        f'<div class="aa-quote-item">{esc(item.get("item", "Unknown item"))}'
                        f" · {esc(item.get('price', 'Price unavailable'))}</div>"
                        f'<div class="aa-quote-meta">Purchased '
                        f"{esc(item.get('purchase_date', 'date unavailable'))}</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    '<div class="aa-quote-amount">'
                    f"{esc(approval_quote.get('refund_amount', 'Amount unavailable'))}"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Simulated ticket only. No money is transferred. "
                    "Confirming authorizes ticket creation."
                )
                confirm_col, decline_col = st.columns(2)
                with confirm_col:
                    st.button(
                        "Confirm refund",
                        key="confirm_pending_refund",
                        type="primary",
                        use_container_width=True,
                        on_click=confirm_pending_approval,
                    )
                with decline_col:
                    st.button(
                        "Decline",
                        key="decline_pending_refund",
                        use_container_width=True,
                        on_click=decline_pending_approval,
                    )

        user_input = st.chat_input(
            "Describe your order issue",
            key="refund_message",
            disabled=(
                closed
                or st.session_state.pending_operation is not None
                or st.session_state.pending_approval is not None
            ),
            max_chars=1000,
        )

        if st.session_state.pending_operation is not None:
            st.caption("Retry reuses the original request ID and conversation turn.")
        elif st.session_state.pending_approval is not None:
            st.caption(
                "Use the explicit approval controls. Chat messages cannot approve a quote."
            )
        elif turn_limit_reached:
            st.caption("Turn limit reached. Start a new conversation to continue.")
        elif closed:
            st.caption("This conversation is closed. Start a new one to continue.")
        else:
            st.caption("Include your order number. Do not enter payment details.")


with inspector_column:
    with bordered("panel_inspector"):
        inspector_label, inspector_tone = inspector_status()
        panel_heading(
            "Decision inspector",
            pill(inspector_label, inspector_tone),
            kicker="Policy and action",
        )
        with st.container(height=PANEL_BODY_HEIGHT, border=False):
            render_decision_inspector()


with trace_column:
    with bordered("panel_trace"):
        panel_heading(
            "Execution trace",
            pill(f"{len(traces)} recorded", "info"),
            kicker="Agent observability",
        )

        with st.container(height=PANEL_BODY_HEIGHT, border=False):
            if not traces:
                st.markdown(
                    f"""
<div class="aa-empty" style="min-height:{PANEL_BODY_HEIGHT - 24}px">
  <div class="aa-empty-icon">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
         stroke-linejoin="round">
      <path d="M3 12h4l2.5-6 5 12 2.5-6H21"/>
    </svg>
  </div>
  <div class="aa-empty-title">No execution activity yet</div>
  <div class="aa-empty-copy">
    Send a message to inspect tool calls, execution order, timing and results.
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

            for index, record in enumerate(reversed(traces)):
                render_trace(record, expanded=index == 0)

        st.caption(
            "Tool success means the tool executed successfully; it does not "
            "establish that the policy decision was correct."
        )


# ---------------------------------------------------------
# Request processing — preserved
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
    and st.session_state.pending_approval is None
):
    with st.spinner("Reviewing your request..."):
        process_message(prompt)
    st.rerun()
