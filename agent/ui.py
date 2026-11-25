import logging
import sys
from html import escape
import json
import sqlite3
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
    page_title="AgentAudit | Refund assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONVERSATION_TURNS = 8
CHAT_BODY_HEIGHT = 420  # transcript height; the input sits beneath it
PANEL_BODY_HEIGHT = 480  # case-details tab bodies (decision, activity log)

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
        confirmed=True,
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --aa-bg: #0E141B;
    --aa-sidebar: #0B1118;
    --aa-surface: #141C26;
    --aa-raised: #1A2431;
    --aa-border: #263444;
    --aa-border-soft: #1E2A38;
    --aa-text: #E8EDF2;
    --aa-muted: #9AA8B7;
    --aa-subtle: #7C8B9B;
    --aa-accent: #4FD1B5;
    --aa-accent-ink: #04231E;
    --aa-green: #5CC98F;
    --aa-amber: #E6B35A;
    --aa-red: #EF7B7B;
    --aa-blue: #7DB2F0;
}

html, body, .stApp, [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"], button, input, textarea {
    font-family: 'Public Sans', system-ui, sans-serif;
}

* { scrollbar-width: thin; scrollbar-color: #2C4054 transparent; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--aa-bg);
    color: var(--aa-text);
}

/* Transparent header floats above the page; padding keeps content clear. */
[data-testid="stHeader"] { background: transparent; }

[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
footer {
    display: none !important;
}

.block-container {
    max-width: 1320px;
    padding-top: 3.2rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: var(--aa-sidebar);
    border-right: 1px solid var(--aa-border-soft);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: 12.5px; line-height: 1.55; }

p, li, label, [data-testid="stMarkdownContainer"] { color: var(--aa-text); }
[data-testid="stCaptionContainer"] {
    color: var(--aa-muted) !important;
    font-size: 12.5px; line-height: 1.55;
}
hr { border-color: var(--aa-border-soft) !important; }
a, a:visited { color: var(--aa-accent) !important; }

/* Header */
.aa-topbar {
    display:flex; align-items:center; justify-content:space-between;
    gap:16px; flex-wrap:wrap; margin:0 0 18px 0;
}
.aa-brand { display:flex; align-items:center; gap:14px; }
.aa-mark {
    width:44px; height:44px; flex:none; display:flex; align-items:center;
    justify-content:center; border-radius:12px;
    color:var(--aa-accent); background:rgba(79,209,181,.10);
    border:1px solid rgba(79,209,181,.28);
}
.aa-page-title {
    color:var(--aa-text); font-size:22px; font-weight:700;
    letter-spacing:-.3px; line-height:1.2;
}
.aa-page-subtitle { color:var(--aa-muted); font-size:13.5px; margin-top:3px; }
.aa-header-right { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }

/* Status pills */
.aa-pill {
    display:inline-flex; align-items:center; gap:7px;
    padding:5px 11px; border-radius:999px;
    border:1px solid var(--aa-border); background:var(--aa-surface);
    color:var(--aa-muted); font-size:12px; font-weight:600; white-space:nowrap;
}
.aa-dot { width:7px; height:7px; border-radius:50%; display:inline-block; background:var(--aa-subtle); flex:none; }
.aa-dot-ok { background:var(--aa-green); }
.aa-dot-warn { background:var(--aa-amber); }
.aa-dot-err { background:var(--aa-red); }
.aa-dot-info { background:var(--aa-blue); }

/* Summary strip: one quiet row instead of four big cards */
.aa-stats {
    display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
    margin:0 0 16px 0; overflow:hidden;
    background:var(--aa-surface); border:1px solid var(--aa-border);
    border-radius:12px;
}
.aa-stat { min-width:0; padding:10px 16px; border-left:1px solid var(--aa-border-soft); }
.aa-stat:first-child { border-left:0; }
.aa-stat-label { color:var(--aa-muted); font-size:12px; font-weight:500; }
.aa-stat-body { display:flex; align-items:baseline; flex-wrap:wrap; gap:0 8px; margin-top:2px; }
.aa-stat-value {
    color:var(--aa-text); font-size:18px; font-weight:600;
    letter-spacing:-.2px; font-variant-numeric:tabular-nums;
}
.aa-stat-value.ok { color:var(--aa-green); }
.aa-stat-value.warn { color:var(--aa-amber); }
.aa-stat-value.err { color:var(--aa-red); }
.aa-stat-note { color:var(--aa-subtle); font-size:12px; }

/* Panels */
[class*="st-key-panel_"] {
    background:var(--aa-surface) !important;
    border:1px solid var(--aa-border) !important;
    border-radius:14px !important;
}
[data-testid="stColumn"]:has([class*="st-key-panel_"]) > [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has([class*="st-key-panel_"]) > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"],
[class*="st-key-panel_"] { height:100%; }

.aa-panel-heading {
    display:flex; align-items:center; justify-content:space-between;
    gap:10px; padding-bottom:12px; margin-bottom:6px;
    border-bottom:1px solid var(--aa-border-soft);
}
.aa-panel-title { color:var(--aa-text); font-size:15px; font-weight:600; }
.aa-panel-sub { color:var(--aa-muted); font-size:12.5px; margin-top:2px; }
.aa-section-label {
    color:var(--aa-muted); font-size:12.5px; font-weight:600; margin:4px 0 8px;
}

/* Tabs */
[data-baseweb="tab-list"] { gap:6px; }
button[data-baseweb="tab"] {
    background:transparent; color:var(--aa-muted);
    font-size:13.5px; font-weight:600; padding:8px 12px;
}
button[data-baseweb="tab"][aria-selected="true"] { color:var(--aa-text); }
button[data-baseweb="tab"]:focus-visible { outline:2px solid var(--aa-accent); outline-offset:-2px; }
[data-baseweb="tab-highlight"] { background:var(--aa-accent) !important; height:2px !important; }
[data-baseweb="tab-border"] { background:var(--aa-border-soft) !important; }

/* Chat */
[data-testid="stChatMessage"] {
    background:transparent; border:0; border-radius:12px; padding:8px 10px;
}
[data-testid="stChatMessage"]:has(.aa-user-marker) {
    background:rgba(79,209,181,.09); border:1px solid rgba(79,209,181,.20);
}
/* Marker used only to tell user messages apart; takes no space. */
[data-testid="stElementContainer"]:has(> .stMarkdown .aa-user-marker) { display:none; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color:var(--aa-text); font-size:14.5px; line-height:1.6;
}
[data-testid^="stChatMessageAvatar"] {
    background:var(--aa-raised) !important;
    border:1px solid var(--aa-border); border-radius:9px;
}
[data-testid^="stChatMessageAvatar"] * { color:var(--aa-accent) !important; }
[data-testid="stChatInput"] {
    border:1px solid var(--aa-border) !important;
    border-radius:12px !important; background:var(--aa-raised) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color:var(--aa-accent) !important;
    box-shadow:0 0 0 1px rgba(79,209,181,.18);
}
[data-testid="stChatInput"] textarea { color:var(--aa-text) !important; }

/* Buttons */
.stButton > button {
    min-height:40px; border-radius:10px;
    font-size:13.5px; font-weight:600; transition:border-color .15s ease, background .15s ease;
}
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background:var(--aa-raised) !important; color:var(--aa-text) !important;
    border:1px solid var(--aa-border) !important;
}
.stButton > button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color:var(--aa-accent) !important; color:var(--aa-accent) !important;
}
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background:var(--aa-accent) !important; color:var(--aa-accent-ink) !important;
    border:1px solid var(--aa-accent) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background:#79E0CB !important; border-color:#79E0CB !important;
    color:var(--aa-accent-ink) !important;
}
.stButton > button:disabled { opacity:.42; }
.stButton > button:focus-visible { outline:2px solid var(--aa-accent); outline-offset:2px; }

/* Suggested prompts read as rows, not centred buttons */
[class*="st-key-suggested_prompt_"] button { justify-content:flex-start; }
[class*="st-key-suggested_prompt_"] button > div { width:100%; justify-content:flex-start; }
[class*="st-key-suggested_prompt_"] button p { text-align:left; font-weight:500; font-size:13.5px; }

/* Approval card: the one place that asks for a decision */
[class*="st-key-approval_card"] {
    background:var(--aa-raised) !important;
    border:1px solid rgba(79,209,181,.38) !important;
    border-radius:12px !important;
}
.aa-quote-amount {
    color:#F2FAF8; font-size:28px; font-weight:700;
    letter-spacing:-.5px; margin:2px 0 4px;
    font-variant-numeric:tabular-nums;
}
.aa-quote-item { color:var(--aa-text); font-size:14px; font-weight:600; }
.aa-quote-meta { color:var(--aa-muted); font-size:12.5px; margin-bottom:6px; }

/* Alerts */
[data-testid="stAlertContainer"] {
    border-radius:10px !important; border:1px solid var(--aa-border-soft);
}
[data-testid="stAlertContainer"] p { font-size:13.5px; line-height:1.55; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
    background:rgba(125,178,240,.08) !important; border-color:rgba(125,178,240,.28);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    background:rgba(230,179,90,.09) !important; border-color:rgba(230,179,90,.30);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    background:rgba(92,201,143,.09) !important; border-color:rgba(92,201,143,.30);
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    background:rgba(239,123,123,.09) !important; border-color:rgba(239,123,123,.30);
}

/* Identifiers (request IDs, trace IDs, tool names) are the only monospace text */
[data-testid="stCode"] pre, [data-testid="stCode"] code { font-size:12px !important; }
.aa-mono { font-family:'IBM Plex Mono',monospace; overflow-wrap:anywhere; }

/* Expanders and trace */
[data-testid="stExpander"] {
    background:var(--aa-raised);
    border:1px solid var(--aa-border-soft); border-radius:10px;
}
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary { background:transparent !important; }
[data-testid="stExpander"] summary { color:var(--aa-text); font-size:13.5px; font-weight:600; }
.aa-event {
    display:flex; justify-content:space-between; align-items:flex-start;
    gap:12px; padding:10px 0; border-bottom:1px solid var(--aa-border-soft);
}
.aa-event-left { display:flex; align-items:flex-start; gap:10px; min-width:0; }
.aa-event-name {
    color:var(--aa-text); font-family:'IBM Plex Mono',monospace;
    font-size:12px; overflow-wrap:anywhere;
}
.aa-event-status { color:var(--aa-muted); font-size:12px; margin-top:3px; }
.aa-event-time { color:var(--aa-muted); font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }
.aa-empty {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; text-align:center; padding:0 16px;
}
.aa-empty-icon { color:var(--aa-accent); margin-bottom:12px; opacity:.9; }
.aa-empty-title { color:var(--aa-text); font-size:14px; font-weight:600; }
.aa-empty-copy {
    max-width:280px; margin:7px auto 0; color:var(--aa-muted);
    font-size:12.5px; line-height:1.6;
}

/* Refund decision */
.aa-decision-card {
    padding:16px; border-radius:12px;
    background:var(--aa-raised);
    border:1px solid var(--aa-border);
    border-left:3px solid var(--aa-accent);
    margin:6px 0 14px;
}
.aa-decision-label { color:var(--aa-muted); font-size:12.5px; font-weight:600; }
.aa-decision-amount {
    color:#F2FAF8; font-size:30px; font-weight:700;
    letter-spacing:-.5px; margin-top:4px; font-variant-numeric:tabular-nums;
}
.aa-decision-note { color:var(--aa-muted); font-size:12.5px; margin-top:3px; }
.aa-check {
    display:flex; align-items:flex-start; gap:8px;
    color:var(--aa-text); font-size:13.5px; margin:9px 0;
}
.aa-check-mark { color:var(--aa-accent); font-weight:700; }
.aa-intro { color:var(--aa-muted); font-size:13.5px; line-height:1.6; margin:6px 0 4px; }
.aa-steps { margin-top:14px; }
.aa-step {
    display:flex; align-items:flex-start; gap:12px;
    position:relative; padding-bottom:16px;
}
.aa-step:last-child { padding-bottom:0; }
.aa-step:not(:last-child)::before {
    content:""; position:absolute; left:10px; top:24px; bottom:2px;
    width:1px; background:var(--aa-border);
}
.aa-step-marker {
    width:21px; height:21px; flex:none; border-radius:50%;
    border:1.5px solid var(--aa-border); background:var(--aa-raised);
    color:var(--aa-muted); font-size:11px; font-weight:700;
    display:flex; align-items:center; justify-content:center;
}
.aa-step-title { color:var(--aa-text); font-size:14px; font-weight:600; line-height:21px; }
.aa-step-copy { color:var(--aa-muted); font-size:12.5px; line-height:1.5; }

/* Sidebar */
.aa-side-brand {
    display:flex; align-items:center; gap:10px;
    color:var(--aa-text); font-size:16px; font-weight:700;
}
.aa-side-sub { color:var(--aa-muted); font-size:12px; margin-top:2px; font-weight:400; }
.aa-side-label { color:var(--aa-text); font-size:13px; font-weight:600; margin:22px 0 8px; }
[data-testid="stProgressBarTrack"] { background:#1B2B3B !important; border-radius:999px; height:6px; }
[data-testid="stProgressBarTrack"] > div { background:var(--aa-accent) !important; border-radius:999px; }
[data-testid="stToggle"] label { color:var(--aa-text) !important; }

/* Responsive */
@media (max-width: 1050px) {
    .aa-stats { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .aa-stat:nth-child(3) { border-left:0; }
    .aa-stat:nth-child(n+3) { border-top:1px solid var(--aa-border-soft); }
}
@media (max-width: 700px) {
    .block-container { padding:3.2rem .8rem 2rem; }
    .aa-page-title { font-size:20px; }
    .aa-stat { padding:9px 12px; }
    .aa-stat-value { font-size:16px; }
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


EVALUATION_DB = PROJECT_ROOT / "eval" / "results" / "agent_audit.sqlite3"


def evaluation_db_connection():
    """Open the evaluation database in read-only mode."""
    if not EVALUATION_DB.exists():
        raise FileNotFoundError(f"Evaluation database not found: {EVALUATION_DB}")

    uri = EVALUATION_DB.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def evaluation_query(sql, params=()):
    """Run a read-only query and return dictionaries."""
    with evaluation_db_connection() as connection:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]


def is_quota_error(row):
    """Distinguish provider quota exhaustion from quality failures."""
    text = " ".join(str(row.get(key) or "") for key in ("error", "reason")).lower()

    return any(
        marker in text
        for marker in (
            "rate_limit_exceeded",
            "tokens per day",
            "quota",
            "tpd",
            "rate limit reached",
        )
    )


def metric_display_status(row):
    status = str(row.get("status") or "UNKNOWN").upper()

    if status == "EVALUATION_ERROR" and is_quota_error(row):
        return "QUOTA BLOCKED"

    return status


def render_evaluation_centre():
    st.markdown("### Evaluation runs")
    st.caption(
        "Saved evaluation runs, metric results and execution traces. "
        "This view is read-only and never calls the agent."
    )

    try:
        runs = evaluation_query(
            """
            SELECT run_id, status, started_at, finished_at,
                   dataset_fingerprint, config_fingerprint
            FROM runs
            ORDER BY started_at DESC
            """
        )
    except (sqlite3.Error, OSError, ValueError) as exc:
        st.error(
            f"Could not read the evaluation database: {exc}. "
            "Run `python -m eval.eval_all` to create it, then reload this page."
        )
        return

    if not runs:
        st.info(
            "No evaluation runs have been saved yet. "
            "Run `python -m eval.eval_all` to create one."
        )
        return

    run_ids = [row["run_id"] for row in runs]
    run_lookup = {row["run_id"]: row for row in runs}

    selected_run = st.selectbox(
        "Evaluation run",
        options=run_ids,
        index=0,
        format_func=lambda run_id: f"{run_id} · {run_lookup[run_id]['status']}",
        key="evaluation_selected_run",
    )

    run = run_lookup[selected_run]

    cases = evaluation_query(
        """
        SELECT case_id, case_number, input, expected_output,
               actual_output, tools_called_json,
               invocation_status, invocation_error, updated_at
        FROM cases
        WHERE run_id = ?
        ORDER BY case_number
        """,
        (selected_run,),
    )

    metrics = evaluation_query(
        """
        SELECT case_id, metric_name, status, score,
               threshold, reason, error, evaluated_at
        FROM metrics
        WHERE run_id = ?
        ORDER BY case_id, metric_name
        """,
        (selected_run,),
    )

    traces = evaluation_query(
        """
        SELECT case_id, schema_version, trace_json, created_at
        FROM trace_artifacts
        WHERE run_id = ?
        """,
        (selected_run,),
    )

    trace_lookup = {row["case_id"]: row for row in traces}

    statuses = [metric_display_status(row) for row in metrics]

    passed = statuses.count("PASS")
    failed = statuses.count("FAIL")
    skipped = statuses.count("SKIPPED")
    quota_blocked = statuses.count("QUOTA BLOCKED")
    other_errors = sum(status == "EVALUATION_ERROR" for status in statuses)

    completed_invocations = sum(
        str(case.get("invocation_status") or "").upper() == "COMPLETED"
        for case in cases
    )

    trace_count = len(traces)

    render_metrics(
        [
            (
                "Cases",
                str(len(cases)),
                f"{completed_invocations} completed",
            ),
            (
                "Saved traces",
                f"{trace_count}/{len(cases)}",
                "persisted",
                "ok" if trace_count == len(cases) and cases else "warn",
            ),
            (
                "Metric passes",
                str(passed),
                f"{failed} failed",
                "ok" if failed == 0 else "err",
            ),
            (
                "Quota blocked",
                str(quota_blocked),
                f"{skipped} skipped, {other_errors} other errors",
                "warn" if quota_blocked or other_errors else "",
            ),
        ]
    )

    st.caption(
        f"Run status: {run['status']} · "
        f"Started: {run['started_at'] or 'Unavailable'} · "
        f"Finished: {run['finished_at'] or 'In progress'}"
    )

    if quota_blocked:
        st.warning(
            f"{quota_blocked} metric evaluation(s) were blocked by provider "
            "quota limits. These are not quality failures."
        )

    if not cases:
        st.info("This run has no saved cases.")
        return

    case_lookup = {case["case_id"]: case for case in cases}
    case_ids = [case["case_id"] for case in cases]

    selected_case_id = st.selectbox(
        "Inspect case",
        options=case_ids,
        format_func=lambda case_id: (
            f"Case {case_lookup[case_id]['case_number']} · "
            f"{case_lookup[case_id]['invocation_status']}"
        ),
        key="evaluation_selected_case",
    )

    case = case_lookup[selected_case_id]

    case_metrics = [row for row in metrics if row["case_id"] == selected_case_id]

    trace = trace_lookup.get(selected_case_id)

    overview_tab, metrics_tab, trace_tab = st.tabs(
        ["Case overview", "Metric results", "Saved trace"]
    )

    with overview_tab:
        st.markdown("**Customer request**")
        st.write(case["input"])

        st.markdown("**Expected behaviour**")
        st.write(case["expected_output"])

        st.markdown("**Actual agent response**")
        st.info(case["actual_output"] or "No response recorded.")

        st.markdown("**Invocation status**")
        invocation_status = str(case["invocation_status"] or "UNKNOWN").upper()

        if invocation_status == "COMPLETED":
            st.success(invocation_status)
        else:
            st.warning(invocation_status)

        if case.get("invocation_error"):
            st.error(case["invocation_error"])

        st.markdown("**Tools called**")
        try:
            tools = json.loads(case["tools_called_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            tools = case["tools_called_json"]

        st.json(tools)

    with metrics_tab:
        if not case_metrics:
            st.info("No metric results have been recorded for this case.")
        else:
            metric_rows = []
            for row in case_metrics:
                metric_rows.append(
                    {
                        "Metric": row["metric_name"],
                        "Status": metric_display_status(row),
                        "Score": (
                            f"{row['score']:.3f}" if row["score"] is not None else "—"
                        ),
                        "Threshold": (
                            f"{row['threshold']:.3f}"
                            if row["threshold"] is not None
                            else "—"
                        ),
                    }
                )

            st.dataframe(
                metric_rows,
                use_container_width=True,
                hide_index=True,
            )

            for row in case_metrics:
                display_status = metric_display_status(row)

                with st.expander(f"{row['metric_name']} · {display_status}"):
                    if row.get("reason"):
                        st.markdown("**Reason**")
                        st.write(row["reason"])

                    if row.get("error"):
                        if display_status == "QUOTA BLOCKED":
                            st.warning(row["error"])
                        else:
                            st.error(row["error"])

    with trace_tab:
        if trace is None:
            st.warning("No saved trace exists for this case.")
        else:
            st.caption(
                f"Trace schema v{trace['schema_version']} · Saved {trace['created_at']}"
            )

            try:
                trace_data = json.loads(trace["trace_json"])
                st.json(trace_data, expanded=False)
            except (TypeError, json.JSONDecodeError):
                st.error("The saved trace could not be decoded as JSON.")

    with st.expander("Run metadata"):
        st.write(
            {
                "Run ID": run["run_id"],
                "Run status": run["status"],
                "Dataset fingerprint": run["dataset_fingerprint"],
                "Config fingerprint": run["config_fingerprint"],
            }
        )


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
        return "Retry needed", "warn"
    if st.session_state.pending_approval is not None:
        return "Waiting for approval", "warn"
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
        return "Needs your decision", "warn"
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
      <div class="aa-page-title">AgentAudit</div>
      <div class="aa-page-subtitle">Review refund requests and see exactly how the agent reached its decision.</div>
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
    """Summary strip. items: (label, value, note) or (label, value, note, tone)."""
    cells = []
    for item in items:
        label, value, note = item[:3]
        tone = item[3] if len(item) > 3 else ""
        cells.append(
            f'<div class="aa-stat">'
            f'<div class="aa-stat-label">{esc(label)}</div>'
            f'<div class="aa-stat-body">'
            f'<span class="aa-stat-value {esc(tone)}">{esc(value)}</span>'
            f'<span class="aa-stat-note">{esc(note)}</span>'
            f"</div></div>"
        )
    st.markdown(
        f'<div class="aa-stats">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def panel_heading(title, badge_html, subtitle=None):
    sub_html = f'<div class="aa-panel-sub">{esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="aa-panel-heading">
  <div><div class="aa-panel-title">{esc(title)}</div>{sub_html}</div>
  {badge_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def render_trace(record, expanded=False):
    turn = record.get("turn_number", "-")
    elapsed = duration(record.get("execution_time_seconds"))
    status = str(record.get("status", "unknown")).capitalize()
    events = record.get("tool_events") or []
    count = record.get("tool_call_count", len(events))

    with st.expander(f"Turn {turn} · {status} · {elapsed}", expanded=expanded):
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
                with st.expander(f"Inputs and outputs · {name}", expanded=False):
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

        st.caption("Trace ID")
        st.code(str(record.get("trace_id", "Unavailable")), language=None)


POLICY_STEPS = [
    ("Verify the customer", "Confirm who is making the request."),
    ("Match the purchase", "Find the order and items in question."),
    ("Check eligibility", "Apply the refund policy to each item."),
    ("Quote the refund", "Propose an amount for your approval."),
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
  <div class="aa-decision-label">Proposed refund</div>
  <div class="aa-decision-amount">{esc(amount)}</div>
  <div class="aa-decision-note">Waiting for your confirmation in the conversation panel</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Customer: {approval.get('user', 'Unavailable')}")
        section_label("Items in this quote")
        for item in quote.get("items", []):
            st.markdown(
                f'<div class="aa-check"><span class="aa-check-mark">•</span>'
                f"<span>{esc(item.get('item', 'Unknown item'))} · "
                f"{esc(item.get('price', 'Price unavailable'))} · "
                f"Purchased {esc(item.get('purchase_date', 'date unavailable'))}</span></div>",
                unsafe_allow_html=True,
            )
        st.info(
            "No ticket has been created yet. Confirming creates a simulated "
            "ticket; no money is transferred."
        )
        return

    if operation is not None:
        st.warning(
            "The last request didn't finish, so its outcome is unknown. "
            "Select Retry in the conversation panel before sending another message."
        )
        st.caption("Original request ID")
        st.code(str(operation.get("refund_request_id", "Unavailable")), language=None)
        st.caption("Retrying keeps this ID, so the request is never counted twice.")
        return

    if st.session_state.conversation_closed:
        st.success("This conversation is closed.")
        st.caption("Select New conversation to submit another request.")
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
<div class="aa-intro">
  Send a request and the agent works through these checks.
  The evidence for each one appears here.
</div>
<div class="aa-steps">{steps}</div>
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
action_pending = pending_operation is not None or pending_approval is not None


# ---------------------------------------------------------
# Sidebar: session info only. Primary actions live in the main view.
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
<div class="aa-side-brand">
  <span style="color:#4FD1B5;font-size:20px">◈</span>
  <div>
    <div>AgentAudit</div>
    <div class="aa-side-sub">Refund support console</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="aa-side-label">Session usage</div>', unsafe_allow_html=True
    )
    st.caption(f"{st.session_state.turn_number} of {MAX_CONVERSATION_TURNS} turns used")
    st.progress(min(st.session_state.turn_number / MAX_CONVERSATION_TURNS, 1.0))

    with st.expander("Session details"):
        st.caption("Conversation ID")
        st.code(st.session_state.thread_id, language=None)

    st.markdown('<div class="aa-side-label">Good to know</div>', unsafe_allow_html=True)
    st.caption(
        "Refund policy is enforced by deterministic tools. Ticket creation is "
        "simulated and no bank transfer is performed."
    )
    st.caption("Use synthetic customer data in public demos.")
    st.caption(
        "Conversations can't be resumed across sessions until "
        "authenticated ownership controls are in place."
    )


# ---------------------------------------------------------
# Header and main tabs
# ---------------------------------------------------------

render_header()

workspace_tab, evaluation_tab = st.tabs(["Workspace", "Evaluations"])

with workspace_tab:
    traces = st.session_state.trace_records
    complete_traces = [r for r in traces if not r.get("trace_unavailable")]

    total_tools = sum(int(number(r.get("tool_call_count"))) for r in complete_traces)
    successful_tools = sum(
        int(number(r.get("successful_tool_count"))) for r in complete_traces
    )
    failed_tools = sum(int(number(r.get("failed_tool_count"))) for r in complete_traces)
    total_time = sum(number(r.get("execution_time_seconds")) for r in traces)
    has_missing_traces = any(r.get("trace_unavailable") for r in traces)

    tool_count_display = f"{total_tools}+" if has_missing_traces else str(total_tools)
    success_display = f"{successful_tools / total_tools:.0%}" if total_tools else "—"
    tool_detail = (
        "partial data"
        if has_missing_traces
        else f"{successful_tools} ok, {failed_tools} failed"
    )

    render_metrics(
        [
            (
                "Turns used",
                f"{st.session_state.turn_number}/{MAX_CONVERSATION_TURNS}",
                "this session",
            ),
            ("Tool calls", tool_count_display, tool_detail),
            (
                "Tool success rate",
                success_display,
                "recorded calls",
                "" if not total_tools else ("warn" if failed_tools else "ok"),
            ),
            ("Processing time", duration(total_time), "total"),
        ]
    )

    if st.session_state.trace_warning:
        st.warning(
            "Some detailed traces could not be saved, so tool metrics may be "
            "incomplete. Check the application logs."
        )

    conversation_column, details_column = st.columns([1.5, 1], gap="medium")

    user_input = None
    retry_clicked = False

    # Give the transcript less height while a retry or approval card needs room.
    chat_height = (
        max(CHAT_BODY_HEIGHT - 120, 200) if action_pending else CHAT_BODY_HEIGHT
    )

    # ---------------- Conversation ----------------
    with conversation_column:
        with bordered("panel_conversation"):
            panel_heading(
                "Conversation",
                pill("Closed", "info") if closed else pill("Live", "ok"),
                subtitle="Chat with the refund assistant",
            )

            with st.container(height=chat_height, border=False):
                for message in st.session_state.messages:
                    avatar = (
                        ASSISTANT_AVATAR
                        if message["role"] == "assistant"
                        else USER_AVATAR
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
                    section_label("Try one of these")
                    for index, prompt_text in enumerate(SUGGESTED_PROMPTS):
                        st.button(
                            prompt_text,
                            key=f"suggested_prompt_{index}",
                            use_container_width=True,
                            on_click=queue_prompt,
                            args=(prompt_text,),
                        )

            if st.session_state.pending_operation is not None:
                st.warning(
                    "The last request didn't finish, so its outcome is unknown. "
                    "Retry it before sending another message."
                )
                retry_clicked = st.button(
                    "Retry last request",
                    key="retry_pending_operation",
                    type="primary",
                    use_container_width=True,
                )

            approval = st.session_state.pending_approval
            if isinstance(approval, dict):
                approval_quote = approval.get("quote") or {}
                with bordered("approval_card"):
                    section_label("Refund quote ready for your approval")
                    st.caption(f"Customer: {approval.get('user', 'Unavailable')}")
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
                        "Confirming creates the ticket."
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
                "Describe the problem and include your order number",
                key="refund_message",
                disabled=closed or action_pending,
                max_chars=1000,
            )

            hint_col, new_col = st.columns([2.6, 1.2], vertical_alignment="center")
            with hint_col:
                if st.session_state.pending_operation is not None:
                    st.caption("Retrying reuses the original request ID and turn.")
                elif st.session_state.pending_approval is not None:
                    st.caption(
                        "Use the buttons above to confirm or decline. "
                        "Chat messages can't approve a quote."
                    )
                elif turn_limit_reached:
                    st.caption(
                        "Turn limit reached. Start a new conversation to continue."
                    )
                elif closed:
                    st.caption(
                        "This conversation is closed. Start a new one to continue."
                    )
                else:
                    st.caption("Don't enter payment details.")
            with new_col:
                st.button(
                    "New conversation",
                    key="new_conversation_main",
                    icon=":material/add:",
                    use_container_width=True,
                    disabled=action_pending,
                    on_click=start_new_conversation,
                )

    # ---------------- Case details: decision + activity ----------------
    with details_column:
        with bordered("panel_details"):
            inspector_label, inspector_tone = inspector_status()
            panel_heading(
                "Case details",
                pill(inspector_label, inspector_tone),
                subtitle="What the agent decided and did",
            )

            decision_tab, activity_tab = st.tabs(["Refund decision", "Activity log"])

            with decision_tab:
                with st.container(height=PANEL_BODY_HEIGHT, border=False):
                    render_decision_inspector()

            with activity_tab:
                st.toggle(
                    "Show tool inputs and outputs",
                    key="show_tool_details",
                    help=(
                        "Developer view of raw tool arguments, results and errors. "
                        "This switch is not authentication or access control."
                    ),
                )
                with st.container(height=PANEL_BODY_HEIGHT - 48, border=False):
                    if not traces:
                        st.markdown(
                            f"""
<div class="aa-empty" style="min-height:{PANEL_BODY_HEIGHT - 72}px">
  <div class="aa-empty-icon">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.7" stroke-linecap="round"
         stroke-linejoin="round">
      <path d="M3 12h4l2.5-6 5 12 2.5-6H21"/>
    </svg>
  </div>
  <div class="aa-empty-title">No activity yet</div>
  <div class="aa-empty-copy">
    Send a message to see each tool call, its order, timing and result.
  </div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        noun = "turn" if len(traces) == 1 else "turns"
                        st.caption(f"{len(traces)} {noun} recorded. Newest first.")

                    for index, record in enumerate(reversed(traces)):
                        render_trace(record, expanded=index == 0)

                st.caption(
                    "A successful tool call means the tool ran without error. "
                    "It doesn't confirm the policy decision was correct."
                )

    # ---------------- Request processing — preserved ----------------
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


# ---------------------------------------------------------
# Evaluations — read-only
# ---------------------------------------------------------

with evaluation_tab:
    render_evaluation_centre()
