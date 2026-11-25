"""Two-stage, checkpointed refund-support graph.

Stage 1: tool-capable conversational model.
Stage 2: tool-free strict structured-output finalizer.

Only the existing Answer type is persisted in graph state.
"""

import ast
import json
import logging
import re
from typing import Annotated, Any, TypedDict

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.agent_context import AgentContext
from agent.checkpoint_store import checkpointer
from agent.models import model
from agent.objects import (
    Answer,
    FinalAnswerOutput,
    PurchaseItem,
    RefundProposal,
)
from agent.prompts import agent_system_prompt
from agent.tools import get_user_purchase_history, parse_purchase_date, validate_refund


logger = logging.getLogger(__name__)

TOOLS = [get_user_purchase_history]

conversation_model = model.bind_tools(TOOLS)

finalizer = model.bind(
    max_tokens=4096,
    reasoning_effort="low",
    include_reasoning=False,
).with_structured_output(
    FinalAnswerOutput,
    method="json_schema",
    strict=True,
    include_raw=True,
)


class RefundAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    structured_response: Answer | None

    # These fields are populated only from actual tool results.
    lookup_status: str
    verified_username: str | None
    purchase_history: dict[str, Any] | None


def _contains_unsupported_success_claim(answer: str) -> bool:
    """Detect affirmative refund, forwarding, and ticket-action claims.

    This graph stage never approves refunds, forwards requests, or creates
    tickets. Explicit negations, uncertainty, and conditional language
    should not be treated as completed or promised actions.
    """
    patterns = (
        # Refund approval, processing, issuance, completion, or submission.
        r"\b(?:your\s+)?refund(?:\s+request)?(?:\s+for\s+[^.!?;,:]+?)?\s+"
        r"(?:(?:has|was|is)\s+been|was|is)\s+"
        r"(?:approved|processed|issued|completed|sent|transferred|submitted|forwarded)\b",

        # Ticket creation or submission.
        r"\b(?:your\s+)?ticket\s+(?:(?:has|was|is)\s+been|was|is)\s+"
        r"(?:created|issued|submitted|opened)\b",

        # Passive request workflow claims:
        # "Your request will now be forwarded for review."
        # "Your request has been submitted."
        r"\b(?:your\s+)?(?:(?:refund|refund\s+proposal)\s+)?request"
        r"(?:\s+for\s+[^.!?;,:]+?)?\s+"
        r"(?:(?:will|shall)\s+(?:now\s+)?be|"
        r"(?:has|have|had)\s+been|"
        r"(?:is|are|was|were)\s+(?:now\s+)?being|"
        r"(?:is|are|was|were))\s+"
        r"(?:forwarded|submitted|sent|processed|approved|issued|created|opened)\b",

        # First-person or system claims that an action was performed:
        # "I have forwarded your request."
        r"\b(?:i|we|this agent|the system)\s+"
        r"(?:(?:have|has|had)\s+)?(?:now\s+)?"
        r"(?:forwarded|submitted|sent|processed|approved|issued|created|opened)\s+"
        r"(?:your\s+)?(?:(?:refund|refund\s+proposal)\s+)?request\b",

        # First-person or system commitments to perform an action:
        # "We will submit your refund request."
        r"\b(?:i|we|this agent|the system)\s+will\s+(?:now\s+)?"
        r"(?:forward|submit|send|process|approve|issue|create|open)\s+"
        r"(?:your\s+)?(?:(?:refund|refund\s+proposal)\s+)?request\b",
    )

    for pattern in patterns:
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            # Ignore explicit negation immediately before the subject.
            prefix = answer[max(0, match.start() - 12):match.start()]
            if re.search(r"\b(?:no|not|never)\s+$", prefix, re.IGNORECASE):
                continue

            # Respect sentence-level uncertainty.
            sentence_start = (
                max(
                    answer.rfind(".", 0, match.start()),
                    answer.rfind("!", 0, match.start()),
                    answer.rfind("?", 0, match.start()),
                    answer.rfind("\n", 0, match.start()),
                )
                + 1
            )
            sentence_prefix = answer[sentence_start:match.start()]

            if re.search(
                r"\b(?:cannot|can't|could not|couldn't|unable to|"
                r"do not|don't)\s+(?:confirm|verify|state|say|claim)\b",
                sentence_prefix,
                flags=re.IGNORECASE,
            ):
                continue

            # Ignore negation between the auxiliary and action verb.
            matched_text = match.group(0)
            if re.search(
                r"\b(?:has|have|had|was|were|is|are)\s+not\s+"
                r"(?:been\s+)?(?:forwarded|submitted|sent|processed|"
                r"approved|issued|created|opened|transferred|completed)\b",
                matched_text,
                flags=re.IGNORECASE,
            ):
                continue

            # A future action explicitly dependent on customer confirmation
            # is not an unconditional promise or a completed-action claim.
            if re.search(
                r"\b(?:if|after|once|when)\b[^.!?\n]{0,80}"
                r"\b(?:confirm|confirmation|approve|approval)\b",
                answer[match.end():],
                flags=re.IGNORECASE,
            ):
                continue

            return True

    return False

def _normalise_item_name(value: str) -> str:
    """Normalize harmless formatting differences in verified item names."""
    value = re.sub(r"[\s_-]+", " ", value.casefold().strip())
    value = re.sub(r"\s+", " ", value)

    for article in ("the ", "a ", "an "):
        if value.startswith(article):
            value = value[len(article):]
            break

    return value.strip()


def _resolve_verified_item_name(
    proposed_name: str,
    purchase_history: dict[str, Any],
) -> str | None:
    """Resolve a proposal to exactly one original purchase-history key."""
    if not isinstance(proposed_name, str) or not proposed_name.strip():
        return None

    # Preserve an exact verified key whenever one is supplied.
    if proposed_name in purchase_history:
        return proposed_name

    normalized = _normalise_item_name(proposed_name)
    if not normalized:
        return None

    matches = [
        key
        for key in purchase_history
        if isinstance(key, str) and _normalise_item_name(key) == normalized
    ]

    # Never guess when no key or more than one key matches.
    return matches[0] if len(matches) == 1 else None


def _customer_requested_item(
    messages: list,
    item_name: str,
) -> bool:
    """Require item-specific affirmative refund intent without conflicting negation."""
    expected = _normalise_item_name(item_name)

    if not expected:
        return False

    item_pattern = rf"(?<!\w){re.escape(expected)}(?!\w)"

    negative_intent_patterns = (
        r"\b(?:do not|don't|does not|doesn't|did not|didn't|"
        r"never|cannot|can't|will not|won't)\s+"
        r"(?:(?:want|wish|need|intend|plan)\s+(?:to\s+)?)?"
        r"(?:a\s+)?"
        r"(?:return|refund|send back)\b",
        r"\bnot\s+(?:returning|refunding|sending back)\b",
        r"\bno\s+refund\b",
    )

    positive_intent_pattern = r"\b(?:return|refund|send back)\b"
    clause_boundary_pattern = (
        r"\b(?:but|however|whereas|although|though|except)\b|[;,.!?]"
    )

    saw_positive_intent = False
    saw_negative_intent = False

    for message in messages:
        if not isinstance(message, HumanMessage):
            continue

        content = message.content
        if not isinstance(content, str):
            continue

        normalized = re.sub(
            r"[\s_-]+",
            " ",
            content.casefold().strip(),
        )
        normalized = re.sub(r"\s+", " ", normalized)

        # Questions about eligibility or advice are not refund instructions.
        if normalized.endswith("?") or re.match(
            r"^(?:can|could|should|would|may|might|do|does|did|is|are|"
            r"am|what|when|where|why|how)\b",
            normalized,
        ):
            continue

        # Evaluate intent only in clauses that mention this exact item.
        clauses = re.split(clause_boundary_pattern, normalized)

        for clause in clauses:
            if not re.search(item_pattern, clause):
                continue

            if any(
                re.search(pattern, clause)
                for pattern in negative_intent_patterns
            ):
                saw_negative_intent = True
            elif re.search(positive_intent_pattern, clause):
                saw_positive_intent = True

    # Conflicting intent for the same item is rejected conservatively.
    return saw_positive_intent and not saw_negative_intent

def _begin_turn(state: RefundAgentState) -> dict:
    """Clear verification data when a new user turn starts."""
    return {
        "lookup_status": "not_attempted",
        "verified_username": None,
        "purchase_history": None,
        "structured_response": None,
    }


def _invoke_with_provider_diagnostics(
    stage: str,
    runnable,
    messages: list,
    config: RunnableConfig,
):
    """Log provider failure details while preserving the original exception."""
    import json
    import sys
    import traceback

    try:
        return runnable.invoke(messages, config=config)
    except Exception as exc:
        details = {
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "status_code": getattr(exc, "status_code", None),
            "request_id": getattr(exc, "request_id", None),
            "body": getattr(exc, "body", None),
            "response_text": getattr(getattr(exc, "response", None), "text", None),
            "failed_generation": getattr(exc, "failed_generation", None),
        }
        print(
            "[AGENTAUDIT_PROVIDER_FAILURE] "
            + json.dumps(details, ensure_ascii=False, default=str),
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        raise


def _conversation_node(
    state: RefundAgentState,
    config: RunnableConfig,
) -> dict:
    """Generate the next conversational response or a tool call."""
    messages = [
        SystemMessage(content=agent_system_prompt),
        *state["messages"],
    ]

    response = _invoke_with_provider_diagnostics(
        "conversation", conversation_model, messages, config
    )

    if not isinstance(response, AIMessage):
        raise TypeError("Conversation model did not return an AIMessage.")

    return {"messages": [response]}


def _record_tool_result(
    state: RefundAgentState,
) -> dict:
    """Accept only one unambiguous lookup result from the latest tool batch."""
    messages = state["messages"]

    latest_tool_batch = None

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            latest_tool_batch = message.tool_calls
            break

    if not latest_tool_batch:
        return {
            "lookup_status": "unavailable",
            "verified_username": None,
            "purchase_history": None,
        }

    # This graph supports exactly one lookup call per assistant batch.
    # Reject multiple calls rather than guessing which identity to trust.
    if len(latest_tool_batch) != 1:
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    tool_call = latest_tool_batch[0]

    if tool_call.get("name") != "get_user_purchase_history":
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    tool_call_id = tool_call.get("id")
    args = tool_call.get("args")

    if not isinstance(tool_call_id, str) or not tool_call_id:
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    requested_username = args.get("user") if isinstance(args, dict) else None

    if not isinstance(requested_username, str) or not requested_username:
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    # Match only the result belonging to this exact call ID.
    matching_results = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id
    ]

    if len(matching_results) != 1:
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    tool_message = matching_results[0]

    if tool_message.name != "get_user_purchase_history":
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    if tool_message.status == "error":
        return {
            "lookup_status": "tool_error",
            "verified_username": None,
            "purchase_history": None,
        }

    content = tool_message.content

    if isinstance(content, dict):
        result = content
    elif isinstance(content, str):
        try:
            result = json.loads(content)
        except (TypeError, ValueError):
            try:
                result = ast.literal_eval(content)
            except (SyntaxError, ValueError):
                result = None
    else:
        result = None

    if not isinstance(result, dict):
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    if result.get("status") == "invalid_username":
        return {
            "lookup_status": "invalid_username",
            "verified_username": None,
            "purchase_history": None,
        }

    if "status" in result:
        return {
            "lookup_status": "malformed",
            "verified_username": None,
            "purchase_history": None,
        }

    # Validate the actual dataset shape before marking identity verified.
    for item_name, purchase in result.items():
        if not isinstance(item_name, str) or not item_name.strip():
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

        if not isinstance(purchase, dict):
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

        if set(purchase) != {"price", "date"}:
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

        price = purchase.get("price")
        purchase_date = purchase.get("date")

        if isinstance(price, bool) or not isinstance(price, (int, float)) or price <= 0:
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

        if not isinstance(purchase_date, str):
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

        try:
            parse_purchase_date(purchase_date)
        except (ValueError, TypeError):
            return {
                "lookup_status": "malformed",
                "verified_username": None,
                "purchase_history": None,
            }

    return {
        "lookup_status": "verified",
        "verified_username": requested_username,
        "purchase_history": result,
    }


def _eligibility_inquiry_answer(
    messages: list,
    lookup_status: str,
    verified_username: str | None,
    purchase_history: dict[str, Any] | None,
) -> str | None:
    """Answer explicit item-eligibility questions using deterministic policy."""
    if (
        lookup_status != "verified"
        or not verified_username
        or not isinstance(purchase_history, dict)
    ):
        return None

    latest_customer_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
        ),
        None,
    )

    if latest_customer_message is None or not isinstance(
        latest_customer_message.content, str
    ):
        return None

    customer_text = latest_customer_message.content
    normalized_text = re.sub(
        r"\s+",
        " ",
        re.sub(r"[^\w]+", " ", customer_text.casefold()),
    ).strip()

    eligibility_patterns = (
        r"\beligible\b",
        r"\beligibility\b",
        r"\brefundable\b",
        r"\brefund\s+window\b",
        r"\bqualif(?:y|ies|ied|ication)\b",
        r"\bwithin\s+(?:the\s+)?30\s+days?\b",
        r"\b(?:can|could|would|should|do)\s+i\b.{0,80}\brefund\b",
        r"\bpossible\s+to\s+get\s+(?:a\s+)?refund\b",
    )

    if not any(
        re.search(pattern, normalized_text)
        for pattern in eligibility_patterns
    ):
        return None

    # Match only exact verified item names. Prefer longer names when one
    # item name is contained within another.
    candidates = []
    for item_name in purchase_history:
        if not isinstance(item_name, str) or not item_name.strip():
            continue

        normalized_item = re.sub(
            r"\s+",
            " ",
            re.sub(r"[^\w]+", " ", item_name.casefold()),
        ).strip()

        if not normalized_item:
            continue

        match = re.search(
            rf"(?<!\w){re.escape(normalized_item)}(?!\w)",
            normalized_text,
        )
        if match:
            candidates.append(
                (match.start(), match.end(), item_name, len(normalized_item))
            )

    selected = []
    occupied_spans = []

    for start, end, item_name, _length in sorted(
        candidates,
        key=lambda candidate: (
            -candidate[3],
            candidate[0],
        ),
    ):
        overlaps = any(
            start < existing_end and end > existing_start
            for existing_start, existing_end in occupied_spans
        )
        if not overlaps:
            selected.append((start, item_name))
            occupied_spans.append((start, end))

    selected.sort(key=lambda item: item[0])
    requested_items = [item_name for _, item_name in selected]

    if not requested_items:
        return (
            "I can check refund eligibility using the 30-day policy. "
            "Which exact purchased item would you like me to check? "
            "This check does not create a refund proposal or ticket."
        )

    lines = []

    for item_name in requested_items:
        result = validate_refund(
            verified_username,
            [PurchaseItem(name=item_name)],
        )

        if not isinstance(result, dict):
            lines.append(
                f"- {item_name}: I couldn't verify the eligibility result."
            )
            continue

        if result.get("eligible") is True:
            validated_items = result.get("items") or []
            item_result = (
                validated_items[0]
                if validated_items and isinstance(validated_items[0], dict)
                else {}
            )
            purchase_date = item_result.get("purchase_date")
            purchase_price = item_result.get("price")

            details = []
            if purchase_date is not None:
                details.append(f"purchase date: {purchase_date}")
            if purchase_price is not None:
                details.append(f"recorded price: {purchase_price}")

            detail_text = (
                f" ({'; '.join(details)})" if details else ""
            )
            lines.append(
                f"- {item_name}: Eligible under the 30-day policy "
                f"for refund review{detail_text}."
            )
        else:
            reason = str(
                result.get("reason")
                or "The item did not pass the refund policy check."
            )
            lines.append(
                f"- {item_name}: Not eligible under the 30-day policy. "
                f"{reason}."
            )

    return (
        "Here is the deterministic eligibility check based on the "
        "verified purchase history:\n"
        + "\n".join(lines)
        + "\nThis is an informational eligibility check only. "
        "No refund proposal, approval, or ticket has been created."
    )


def _finalizer_node(
    state: RefundAgentState,
    config: RunnableConfig,
) -> dict:
    """Produce strict output without exposing any tools to this model call."""
    lookup_status = state.get("lookup_status", "not_attempted")
    verified_username = state.get("verified_username")
    identity_verified = (
        lookup_status == "verified"
        and isinstance(verified_username, str)
        and bool(verified_username.strip())
    )

    raw_purchase_history = state.get("purchase_history")
    trusted_purchase_history = (
        raw_purchase_history
        if identity_verified and isinstance(raw_purchase_history, dict)
        else {}
    )

    verified_purchase_records = [
        {
            "item": item_name,
            "purchase_date": purchase.get("date"),
            "purchase_price": purchase.get("price"),
        }
        for item_name, purchase in trusted_purchase_history.items()
        if isinstance(item_name, str) and isinstance(purchase, dict)
    ]

    trusted_context = {
        "lookup_status": lookup_status,
        "verified_username": verified_username if identity_verified else None,
        "verified_purchase_records": verified_purchase_records,
    }

    instructions = SystemMessage(
        content=(
            "You are the final structured-output stage of AgentAudit. "
            "Return only the requested strict structured output. "
            "Use the conversation and trusted verification context below. "
            "Never invent a verified username or purchased item. "
            "Never decide refund eligibility, calculate amounts, approve "
            "refunds, claim that a refund request or proposal was "
            "forwarded, submitted, sent for review, or processed, "
            "or claim a ticket was created. "
            "If identity or requested items are not established, set "
            "refund_proposal to null and ask one concise clarification. "
            "A proposal is only a request for application-side evaluation.\n\n"
            "TRUSTED VERIFICATION CONTEXT:\n"
            + json.dumps(trusted_context, ensure_ascii=False)
        )
    )

    # Groq Structured Outputs must not receive tool-protocol messages.
    # Keep customer messages and ordinary assistant replies only.
    finalizer_messages = [
        message
        for message in state["messages"]
        if isinstance(message, HumanMessage)
        or (
            isinstance(message, AIMessage)
            and not message.tool_calls
        )
    ]

    def safe_finalizer_fallback():
        return {
            "structured_response": Answer(
                answer=(
                    "I couldn't safely complete your refund-support response "
                    "because the final response could not be generated. "
                    "No refund proposal was prepared, and no refund action "
                    "was taken. Please try again."
                ),
                conversation=True,
                refund_proposal=None,
            )
        }

    try:
        result = _invoke_with_provider_diagnostics(
            "finalizer", finalizer, [instructions, *finalizer_messages], config
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Finalizer provider invocation failed; returning safe fallback."
        )
        return safe_finalizer_fallback()

    if (
        not isinstance(result, dict)
        or result.get("parsing_error") is not None
        or not isinstance(result.get("parsed"), FinalAnswerOutput)
    ):
        logging.getLogger(__name__).error(
            "Finalizer returned malformed structured output; "
            "returning safe fallback."
        )
        return safe_finalizer_fallback()

    parsed = result["parsed"]

    proposal = parsed.refund_proposal

    if proposal is not None:
        # No proposal is permitted without a successful lookup.
        if not identity_verified:
            proposal = None

        else:
            purchase_history = trusted_purchase_history

            if proposal.username != verified_username:
                proposal = None

            else:
                requested_items = []
                seen = set()
                valid_proposal = True

                for proposed_item in proposal.items:
                    item_name = proposed_item.name

                    if not isinstance(item_name, str) or not item_name.strip():
                        valid_proposal = False
                        break

                    # Resolve formatting variants to one verified dataset key.
                    canonical_item_name = _resolve_verified_item_name(
                        item_name,
                        purchase_history,
                    )
                    if canonical_item_name is None:
                        valid_proposal = False
                        break

                    normalized = _normalise_item_name(canonical_item_name)

                    if normalized in seen:
                        valid_proposal = False
                        break

                    if not _customer_requested_item(
                        state["messages"],
                        canonical_item_name,
                    ):
                        valid_proposal = False
                        break

                    seen.add(normalized)
                    requested_items.append(
                        PurchaseItem(name=canonical_item_name)
                    )

                if not valid_proposal or not requested_items:
                    proposal = None
                else:
                    proposal = RefundProposal(
                        username=verified_username,
                        items=requested_items,
                    )

    answer_text = parsed.answer
    # This graph stage cannot approve refunds or create tickets.
    # Never pass affirmative completion claims to either application UI.
    if _contains_unsupported_success_claim(answer_text):
        if proposal is not None:
            proposed_item_names = ", ".join(
                item.name for item in proposal.items
            )
            answer_text = (
                f"I've prepared a refund proposal for {proposed_item_names}. "
                "It has not been submitted or approved. The application "
                "must independently validate eligibility, and your explicit "
                "confirmation is required before any refund action."
            )
        else:
            answer_text = (
                "I can't confirm that a refund has been approved or a ticket "
                "created. No refund action has been completed by this agent. "
                "Any refund proposal must first pass application validation "
                "and your explicit confirmation."
            )

    # If the model proposed a refund but application validation rejected it,
    # discard its associated success language as well.
    proposal_rejected = parsed.refund_proposal is not None and proposal is None

    if proposal_rejected:
        answer_text = (
            "I couldn't verify the requested refund proposal against the "
            "customer's purchase history and request. Please clarify the "
            "customer or purchased item. No refund proposal was prepared, "
            "and no ticket was created."
        )

    # Do not let free-form model text decide eligibility. This path is
    # limited to explicit eligibility questions with exact verified items.
    if parsed.refund_proposal is None and proposal is None and not proposal_rejected:
        deterministic_eligibility_answer = _eligibility_inquiry_answer(
            messages=state["messages"],
            lookup_status=lookup_status,
            verified_username=verified_username if identity_verified else None,
            purchase_history=trusted_purchase_history,
        )
        if deterministic_eligibility_answer is not None:
            answer_text = deterministic_eligibility_answer

    answer = Answer(
        answer=answer_text,
        # A rejected proposal requires clarification, not conversation closure.
        conversation=True if proposal_rejected else parsed.conversation,
        refund_proposal=proposal,
    )

    return {"structured_response": answer}


def build_two_stage_graph(checkpointer_override=None):
    """Build the graph, optionally using an isolated test checkpointer."""
    builder = StateGraph(
        RefundAgentState,
        context_schema=AgentContext,
    )

    builder.add_node("begin_turn", _begin_turn)
    builder.add_node("conversation", _conversation_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("record_tool_result", _record_tool_result)
    builder.add_node("finalizer", _finalizer_node)

    builder.add_edge(START, "begin_turn")
    builder.add_edge("begin_turn", "conversation")

    builder.add_conditional_edges(
        "conversation",
        tools_condition,
        {
            "tools": "tools",
            END: "finalizer",
        },
    )

    builder.add_edge("tools", "record_tool_result")
    builder.add_edge("record_tool_result", "conversation")
    builder.add_edge("finalizer", END)

    selected_checkpointer = (
        checkpointer if checkpointer_override is None else checkpointer_override
    )
    return builder.compile(checkpointer=selected_checkpointer)


agent = build_two_stage_graph()
