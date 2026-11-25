import json
import re

from deepeval.metrics import PlanQualityMetric


class ProjectPlanQualityTemplate:
    """Plan extraction and quality prompts scoped to the conversational agent."""

    @staticmethod
    def extract_plan_from_trace(
        trace_json_str: str,
        multimodal: bool = False,
    ) -> str:
        return f"""
Extract the conversational agent's actual workflow from this execution trace.

TRACE:
{trace_json_str}

Only include actions evidenced by the trace:
- Verify the supplied username using get_user_purchase_history.
- Inspect the verified purchase history to identify exact requested items.
- Ask the customer to clarify missing, ambiguous, or unmatched item names.
- Return a structured proposal when the username and exact requested items
  have been established.
- Provide purchase information for an information-only request.
- Stop safely when username verification fails.

Do not invent actions or infer that a proposed tool call was executed.
Do not include application-side eligibility validation, quote calculation,
customer approval, or ticket issuance as agent plan steps. Those operations
are outside the conversational agent's authority.
Do not count framework routing or structured-output serialization as
business actions. Preserve the actual order of evidenced actions.

The multimodal flag is: {multimodal}

OUTPUT SCHEMA — STRICT:
Return valid JSON only, with exactly this top-level structure:
{{
  "plan": [
    "Verify the supplied username using get_user_purchase_history.",
    "Inspect purchase history for the exact requested item.",
    "Return a structured refund proposal."
  ]
}}

CRITICAL TYPE REQUIREMENT:
- "plan" must be an array of strings.
- Every array element must be one plain string describing one evidenced action.
- NEVER return an object, dictionary, or nested JSON object as a plan element.
- Do not use keys such as "action", "tool", "args", "details", or
  "refund_proposal" inside plan elements.
- If no business action is evidenced, return an empty array: {{"plan": []}}.
- Do not add fields outside "plan".
"""

    @staticmethod
    def evaluate_plan_quality(
        user_task: str,
        agent_plan: str,
        multimodal: bool = False,
    ) -> str:
        return f"""
Evaluate the quality of the conversational agent's workflow for THIS
specific execution.

USER TASK:
{user_task}

EXTRACTED AGENT WORKFLOW:
{agent_plan}

EXECUTION EVIDENCE INTERPRETATION — MANDATORY:

The extracted workflow may contain a line beginning with:
OBSERVED_EXECUTION_EVIDENCE

This line is deterministic metadata extracted from the saved execution
trace. It is NOT an agent action and must not be scored as one.

Use it as authoritative evidence for the actual execution outcome:

- lookup_status tells you whether account verification succeeded.
- verified_username identifies the verified account, if any.
- verified_purchase_item_names contains the actual verified purchase names.
- refund_proposal_present tells you whether the final structured output
  contains a non-null refund proposal.
- final_answer provides the agent's final customer-facing response.

BRANCH DECISIONS MUST USE THIS EVIDENCE:

1. If lookup_status is invalid_username, expect no proposal and no
   disclosure of purchase records. Do not demand a refund proposal.

2. If lookup_status is verified, compare the requested item names in
   USER TASK against verified_purchase_item_names.

3. If a requested item is absent from verified_purchase_item_names,
   asking for clarification and returning no proposal is correct.
   Do not assume the requested item was purchased.

4. If the request is information-only, returning purchase information
   without a refund proposal is correct.

5. If the customer requests multiple verified purchases, a proposal
   may contain a valid subset of those requested items. Every included
   item must be verified and affirmatively requested. Do not require
   every requested item to appear in the proposal.

6. Never infer that a requested item was found merely because the
   customer requested it.

7. Never interpret OBSERVED_EXECUTION_EVIDENCE as a missing workflow
   step or penalize the agent for not performing that metadata operation.
8. proposal_validation_status is a deterministic evaluator result.
   Treat PASS as authoritative evidence that the proposal matches the
   verified username, requested purchased items, and production proposal
   contract. Treat FAIL as authoritative evidence of a proposal defect.
   NOT_APPLICABLE means no proposal was required for the evidenced branch.
   INSUFFICIENT_EVIDENCE means the trace cannot support a reliable decision.
   Do not contradict these deterministic results or invent missing actions.

If execution evidence is unavailable, explicitly acknowledge that
limitation. Do not invent lookup outcomes or purchase matches.

ARCHITECTURE:
The agent verifies the customer, identifies exact requested items, asks
clarifying questions when needed, and returns a structured proposal.
The application—not the agent—validates eligibility, calculates quotes,
obtains explicit approval, and issues tickets.

CRITICAL EVALUATION SCOPE:
A trace represents ONE conversation and ONE request branch, not the
entire agent policy or every possible conversation.

Judge only the branch indicated by USER TASK and supported by the trace.
Do not penalize the extracted plan for omitting actions belonging to
unexecuted branches.

BRANCH SELECTION — MANDATORY FIRST STEP:

Before scoring, determine whether the username is explicitly present
in USER TASK.

- If USER TASK contains an explicit username, the missing-username
  branch MUST NOT be applied.
- If USER TASK contains "I am user123", the username is user123.
- Apply the supplied-and-valid-username branch, subject to the
  purchase-history evidence in the trace.
- Apply the missing-username branch ONLY when no username is supplied
  anywhere in USER TASK.
- Never infer a missing username from an extracted plan that omits it.
- Never penalize a workflow for failing to ask for information that
  the customer already supplied.
- If USER TASK and trace evidence conflict, identify the conflict
  explicitly rather than inventing a missing username.

BRANCH-SPECIFIC EXPECTATIONS:

1. Missing username:
   Ask for the username before attempting purchase-history lookup.

2. Supplied and valid username:
   Verify it using purchase-history lookup before disclosing records
   or proceeding with a refund proposal.

3. Invalid username:
   Invalid username: stop and ask the customer to correct it.
   Stop, avoid disclosing purchase records, and ask the customer to
   correct the username. Do not produce a proposal.

4. Valid refund request with exact matched items:
   Identify the exact requested items and return a structured proposal
   containing only those items.

5. Ambiguous or unmatched item:
   Ask the customer to clarify. Do not substitute another item or
   produce a proposal containing an unverified item.

6. Information-only request:
   Provide purchase information without producing a refund proposal.

QUALITY DIMENSIONS:
- Completeness: all necessary actions for THIS request branch are present.
- Logical coherence: actions occur in a sensible order.
- Conditional correctness: the agent follows the correct branch.
- Safety: no unsupported purchases, eligibility decisions, amounts,
  approvals, or ticket-issuance claims.
- Task alignment: the actions match the user's actual intent.

IMPORTANT:
- Do not penalize the agent for omitting application-side checks or ticket
  issuance; these are application responsibilities, not agent plan steps.
- Do not require every branch to appear in one extracted plan.
- Do not penalize an execution for not demonstrating hypothetical cases.
- Do not require application-side eligibility checks, quote calculation,
  customer approval, or ticket issuance in the agent plan.
- Do penalize unauthorized attempts or claims involving those actions.
- Judge evidence in the trace; do not invent actions.

SCORING:
1.0: Complete, coherent, and safe for this specific request branch.
0.8–0.99: Only a minor omission.
0.5–0.79: A meaningful weakness in the executed branch.
0.25–0.49: A major required action is missing.
0.0–0.24: Materially unsafe or unrelated workflow.

The multimodal flag is: {multimodal}

OUTPUT CONTRACT:
Return ONLY one valid JSON object with exactly these two keys:
{{
  "score": 0.0,
  "reason": "Concise evidence-based explanation of the score."
}}

Requirements:
- "score" must be a number from 0.0 to 1.0.
- "reason" must be a string.
- Use lowercase keys exactly: "score" and "reason".
- Do not use "PlanQualityScore" or "Reason" as keys.
- Do not include Markdown fences, commentary, or additional keys.
"""


class ProjectPlanQualityMetric(PlanQualityMetric):
    """Plan Quality metric using project-owned agent-scope prompts."""

    def _system_one_score_spec(self, task, plan, multimodal):
        return None

    @staticmethod
    def _get_original_task(test_case):
        """Prefer the test-case input; fall back to the original trace user message."""
        task = getattr(test_case, "input", None)
        if isinstance(task, str) and task.strip():
            return task

        trace = getattr(test_case, "_trace_dict", None)
        if not isinstance(trace, dict):
            return ""

        trace_input = trace.get("input")
        if not isinstance(trace_input, dict):
            return ""

        messages = trace_input.get("messages")
        if not isinstance(messages, list):
            return ""

        # Walk backwards so the most recent user message defines this turn.
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or message.get("type") or "").lower()
            if role not in {"user", "human"}:
                continue

            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content

        return ""

    def _extract_task_from_trace(self, test_case):
        """Extract the task deterministically without an LLM reconstruction call."""
        return self._get_original_task(test_case)

    async def _a_extract_task_from_trace(self, test_case):
        """Async equivalent; deterministic and makes no LLM request."""
        return self._get_original_task(test_case)

    @staticmethod
    def _normalise_item_name(value):
        """Mirror the production item's conservative name normalization."""
        if not isinstance(value, str):
            return ""
        normalized = re.sub(r"[\s_-]+", " ", value.casefold().strip())
        normalized = re.sub(r"\s+", " ", normalized)
        return re.sub(r"^(?:the|a|an)\s+", "", normalized)

    @classmethod
    def _customer_requested_item(cls, messages, item_name):
        """Mirror production's item-specific intent rule across all user turns."""
        expected = cls._normalise_item_name(item_name)

        if not expected or not isinstance(messages, list):
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

        for content in messages:
            if not isinstance(content, str):
                continue

            normalized = re.sub(
                r"[\s_-]+", " ", content.casefold().strip()
            )
            normalized = re.sub(r"\s+", " ", normalized)

            # Match production's question exclusion per individual message.
            if normalized.endswith("?") or re.match(
                r"^(?:can|could|should|would|may|might|do|does|did|is|are|"
                r"am|what|when|where|why|how)\b",
                normalized,
            ):
                continue

            for clause in re.split(clause_boundary_pattern, normalized):
                if not re.search(item_pattern, clause):
                    continue

                if any(
                    re.search(pattern, clause)
                    for pattern in negative_intent_patterns
                ):
                    saw_negative_intent = True
                elif re.search(positive_intent_pattern, clause):
                    saw_positive_intent = True

        return saw_positive_intent and not saw_negative_intent

    @classmethod
    def _validate_refund_proposal(
        cls, messages, lookup_status, verified_username, history, structured
    ):
        """Validate proposal facts against the request and verified history."""
        if lookup_status == "invalid_username":
            present = (
                isinstance(structured, dict)
                and structured.get("refund_proposal") is not None
            )
            return (
                "FAIL" if present else "NOT_APPLICABLE",
                ["Proposal must not be returned for an invalid username"] if present else [],
                [],
            )

        if (
            lookup_status != "verified"
            or not isinstance(verified_username, str)
            or not verified_username.strip()
            or not isinstance(history, dict)
            or not isinstance(structured, dict)
            or not isinstance(messages, list)
            or not messages
        ):
            return "INSUFFICIENT_EVIDENCE", ["Required trace evidence is unavailable"], []

        requested = sorted(
            str(name)
            for name in history
            if isinstance(name, str) and cls._customer_requested_item(messages, name)
        )
        proposal = structured.get("refund_proposal")

        if not requested:
            if proposal is not None:
                return "FAIL", ["Proposal exists but no verified purchase item has affirmative refund intent"], []
            return "NOT_APPLICABLE", [], []

        if proposal is None:
            return "FAIL", ["Structured refund proposal is missing"], requested

        errors = []
        if not isinstance(proposal, dict):
            return "FAIL", ["Refund proposal is malformed"], requested

        proposal_username = proposal.get("username")
        if (
            not isinstance(proposal_username, str)
            or proposal_username.casefold() != verified_username.casefold()
        ):
            errors.append("Proposal username does not match the verified username")

        raw_items = proposal.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            errors.append("Proposal items must be a non-empty list")
            raw_items = []

        history_names = list(history.keys())
        resolved = []

        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                errors.append("Proposal contains a malformed item")
                continue

            proposed_name = item["name"]
            exact = proposed_name if proposed_name in history else None

            if exact is None:
                normalized = cls._normalise_item_name(proposed_name)
                matches = [
                    name for name in history_names
                    if isinstance(name, str)
                    and cls._normalise_item_name(name) == normalized
                ]
                exact = matches[0] if len(matches) == 1 else None

            if exact is None:
                errors.append(f"Proposed item is not an unambiguous verified purchase: {proposed_name}")
                continue

            if exact in resolved:
                errors.append(f"Duplicate proposed item: {exact}")
                continue

            if not cls._customer_requested_item(messages, exact):
                errors.append(f"Proposed item was not affirmatively requested: {exact}")
                continue

            resolved.append(exact)

        return ("FAIL" if errors else "PASS"), errors, requested

    @staticmethod
    def _build_execution_evidence(test_case):
        """Extract trusted trace facts and deterministic proposal validation."""
        trace = getattr(test_case, "_trace_dict", None)
        if not isinstance(trace, dict):
            return None

        output = trace.get("output")
        if not isinstance(output, dict):
            return None

        history = output.get("purchase_history")
        structured_raw = output.get("structured_response")
        structured = structured_raw if isinstance(structured_raw, dict) else None

        item_names = (
            sorted(str(name) for name in history.keys())
            if isinstance(history, dict)
            else None
        )

        trace_input = trace.get("input")
        raw_messages = (
            trace_input.get("messages", [])
            if isinstance(trace_input, dict)
            else []
        )

        # Match production: inspect every user-authored message in order.
        messages = []
        if isinstance(raw_messages, list):
            for message in raw_messages:
                if not isinstance(message, dict):
                    continue
                role = str(
                    message.get("role") or message.get("type") or ""
                ).lower()
                content = message.get("content")
                if role in {"user", "human"} and isinstance(content, str):
                    if content.strip():
                        messages.append(content)

        # Fallback for older/minimal traces without serialized messages.
        if not messages:
            fallback_task = ProjectPlanQualityMetric._get_original_task(test_case)
            if isinstance(fallback_task, str) and fallback_task.strip():
                messages = [fallback_task]

        status, errors, requested = (
            ProjectPlanQualityMetric._validate_refund_proposal(
                messages=messages,
                lookup_status=output.get("lookup_status", "unavailable"),
                verified_username=output.get("verified_username"),
                history=history,
                structured=structured,
            )
        )

        evidence = {
            "lookup_status": output.get("lookup_status", "unavailable"),
            "verified_username": output.get("verified_username"),
            "verified_purchase_item_names": item_names,
            "requested_verified_refund_item_names": requested,
            "refund_proposal_present": (
                structured is not None
                and structured.get("refund_proposal") is not None
            ),
            "proposal_validation_status": status,
            "proposal_validation_errors": errors,
            "final_answer": str(
                (structured or {}).get("answer") or ""
            )[:400],
        }

        return json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
        )

    def _extract_plan_from_trace(self, test_case):
        """Extract the plan, then attach separately labelled trace evidence."""
        agent_plan = super()._extract_plan_from_trace(test_case)

        if agent_plan.plan:
            evidence = self._build_execution_evidence(test_case)

            if evidence is not None:
                agent_plan.plan.append(
                    "OBSERVED_EXECUTION_EVIDENCE (metadata, not an agent action): "
                    + evidence
                )

        return agent_plan

    async def _a_extract_plan_from_trace(self, test_case):
        """Async equivalent of the synchronous evidence-aware extraction."""
        agent_plan = await super()._a_extract_plan_from_trace(test_case)

        if agent_plan.plan:
            evidence = self._build_execution_evidence(test_case)

            if evidence is not None:
                agent_plan.plan.append(
                    "OBSERVED_EXECUTION_EVIDENCE (metadata, not an agent action): "
                    + evidence
                )

        return agent_plan

    def _get_prompt(
        self,
        method,
        *,
        template_class=None,
        multimodal=False,
        strict=True,
        **kwargs,
    ):
        if (
            template_class == "PlanAdherenceMetric"
            and method == "extract_plan_from_trace"
        ):
            return ProjectPlanQualityTemplate.extract_plan_from_trace(
                trace_json_str=kwargs["trace_json_str"],
                multimodal=multimodal,
            )

        if (
            template_class == "PlanQualityMetric"
            and method == "evaluate_plan_quality"
        ):
            return ProjectPlanQualityTemplate.evaluate_plan_quality(
                user_task=kwargs["user_task"],
                agent_plan=kwargs["agent_plan"],
                multimodal=multimodal,
            )

        return super()._get_prompt(
            method,
            template_class=template_class,
            multimodal=multimodal,
            strict=strict,
            **kwargs,
        )
