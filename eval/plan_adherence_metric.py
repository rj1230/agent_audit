from deepeval.metrics import PlanAdherenceMetric


class ProjectPlanAdherenceTemplate:
    """Project-specific prompts for extracting and evaluating committed plans."""

    @staticmethod
    def extract_plan_from_trace(
        trace_json_str: str,
        multimodal: bool = False,
    ) -> str:
        return f"""
You are identifying the agent's actual committed plan from an execution trace.

EXECUTION TRACE:
{trace_json_str}

Extract only actions the agent explicitly committed to doing as part of its
approach to the user's task. Preserve their intended order.

IMPORTANT DISTINCTIONS:
- Include an action when the agent clearly states or commits to doing it.
- Exclude hypothetical, conditional, speculative, or optional thoughts.
- Statements such as "could ask", "might check", "if the user wants",
  "one option would be", or "perhaps" are not committed plan steps unless
  the agent subsequently makes an explicit decision to perform that action.
- Do not turn a possible follow-up suggestion into a required plan step.
- A polite offer of further help in the final response is not a plan step.
- Do not infer an intention merely because an action would have been useful.
- Do not invent steps that are not supported by the trace.
- Do not treat framework routing, empty protocol messages, or structured
  output serialization as business-plan steps.
- Keep each genuine action as one concise, distinct step.
- If no committed plan is evidenced, return an empty plan.

The multimodal flag is: {multimodal}

Return exactly one JSON object matching this structure:
{{"plan": ["first committed action as a string", "second committed action as a string"]}}

STRICT SCHEMA CONTRACT:
- The top-level object must contain the "plan" field.
- "plan" must be an array of strings.
- Every array element must be one concise plain-text action description.
- Never return dictionaries or objects as plan elements.
- Never use nested "action" or "parameters" objects.
- Preserve tool names and relevant argument values inside the text string.
- Do not add fields other than "plan".
- If no committed plan is evidenced, return {{"plan": []}}.

VALID EXAMPLE:
{{"plan": ["Look up purchase history using get_user_purchase_history(user='user9999').", "Explain that the username could not be verified and ask the user to correct it."]}}

INVALID EXAMPLE:
{{"plan": [{{"action": "get_user_purchase_history", "parameters": {{"user": "user9999"}}}}]}}

The invalid example is prohibited because AgentPlan requires each plan
element to be a string, not an object.

Return valid JSON only. Do not include Markdown fences or commentary.
"""

    @staticmethod
    def evaluate_adherence(
        user_task: str,
        agent_plan: str,
        execution_trace_json: str,
        multimodal: bool = False,
    ) -> str:
        return f"""
You are evaluating whether an agent followed its committed plan.

USER TASK:
{user_task}

COMMITTED PLAN:
{agent_plan}

EXECUTION TRACE:
{execution_trace_json}

Evaluate the plan against the actual execution evidence.

SCORING RULES:
1. Check each committed plan step against the trace.
2. Treat an equivalent action as completion when it accomplishes the
   intended step; wording need not match exactly.
3. Do not mark a step complete merely because the agent proposed it.
   Require evidence that it was actually performed.
4. Do not count empty assistant protocol messages, framework routing,
   callback activity, or structured-output serialization as business actions.
5. In particular, a structured-output operation named Answer that only
   packages the final response is not an additional business action.
6. Do not penalize conditional thoughts, hypothetical options, optional
   follow-up suggestions, or polite offers that were not committed plan steps.
7. Do not require an optional follow-up question unless it was genuinely
   part of the committed plan and necessary for the user's task.
8. Penalize a genuinely omitted committed step.
9. Penalize additional meaningful business actions outside the plan when
   they materially exceed or contradict the user's intent.
10. Base every deduction on specific trace evidence. Do not invent actions
    or assume that a proposed tool call was executed.

A complete plan executed in order should receive a score near 1.0.
Use proportionate partial credit for incomplete or out-of-order execution.
Explain which committed steps were completed and identify any specific
deviation. If the trace does not establish a deviation, do not invent one.

The multimodal flag is: {multimodal}

Return a JSON object matching DeepEval's PlanAdherenceScore schema.
The object must contain exactly these top-level keys: "score" and "reason".
The "score" value must be a number between 0.0 and 1.0.
The "reason" value must be a concise, evidence-based string.
Use this exact JSON structure (the values shown are illustrative):
{{
  "score": 0.85,
  "reason": "Evidence-based explanation of plan adherence."
}}
Do not wrap these fields in a "PlanAdherenceScore" key or any other key.
Return JSON only; do not include Markdown fences or surrounding commentary.
"""


class ProjectPlanAdherenceMetric(PlanAdherenceMetric):
    """Plan Adherence metric using project-owned extraction and scoring prompts."""

    def _system_one_score_spec(self, task, plan, test_case):
        """Use the project prompts through DeepEval's regular LLM scoring path."""
        return None

    def _get_prompt(
        self,
        method,
        *,
        template_class=None,
        multimodal=False,
        strict=True,
        **kwargs,
    ):
        if template_class == "PlanAdherenceMetric":
            if method == "extract_plan_from_trace":
                return ProjectPlanAdherenceTemplate.extract_plan_from_trace(
                    trace_json_str=kwargs["trace_json_str"],
                    multimodal=multimodal,
                )
            if method == "evaluate_adherence":
                return ProjectPlanAdherenceTemplate.evaluate_adherence(
                    user_task=kwargs["user_task"],
                    agent_plan=kwargs["agent_plan"],
                    execution_trace_json=kwargs["execution_trace_json"],
                    multimodal=multimodal,
                )

        return super()._get_prompt(
            method,
            template_class=template_class,
            multimodal=multimodal,
            strict=strict,
            **kwargs,
        )
