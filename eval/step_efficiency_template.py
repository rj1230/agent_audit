class ProjectStepEfficiencyTemplate:
    """Efficiency rubric scoped to actual conversational-agent actions."""

    @staticmethod
    def get_execution_efficiency(
        task: str,
        trace_json_str: str,
        multimodal: bool = False,
    ) -> str:
        return f"""
Evaluate the execution efficiency of the AgentAudit conversational agent.

USER TASK:
{task}

EXECUTION TRACE:
{trace_json_str}

SCOPE:
The agent is responsible for username verification, purchase-history
retrieval, exact item identification, clarification, and structured
proposal creation. Application-side eligibility checks, quote calculation,
customer approval, and ticket issuance are not agent actions.

TRACE INTERPRETATION:
1. Judge agent execution, not evaluation-framework activity.
2. Do not count empty protocol messages or structured-output serialization
   as unnecessary business actions.
3. Count actual business-tool executions and meaningful agent decisions.
4. Do not assume a proposed tool call executed unless the trace confirms it.
5. Do not invent hidden actions, delays, costs, or retries.

TASK-RELATIVE EFFICIENCY:
- A request requiring purchase information normally needs one relevant
  purchase-history lookup and a clear answer.
- A refund proposal normally needs identity verification, inspection of
  purchase history, exact requested-item identification, and proposal
  creation when all required information is available.
- An invalid username should result in a safe stop after the failed lookup.
- An unmatched or ambiguous item should lead to clarification, not repeated
  unrelated lookups or substitution.
- Penalize redundant lookups, repeated equivalent calls, unrelated detours,
  avoidable retries, and actions outside the user's intent.
- Do not penalize the agent for not calculating eligibility or issuing
  tickets; those are application responsibilities.

SCORING:
Return a score from 0.0 to 1.0.
1.0 means meaningful agent actions were necessary and proportionate.
0.8 to 0.99 means execution was efficient with at most a minor avoidable step.
0.5 to 0.79 means a noticeable unnecessary action or inefficiency exists.
0.25 to 0.49 means multiple substantial inefficiencies exist.
0.0 to 0.24 means execution was highly wasteful or materially misdirected.

Explain the necessary actions and any specific unnecessary actions.
If the trace is ambiguous, identify the uncertainty rather than inventing
an inefficiency.

The multimodal flag is: {multimodal}

Return a concise evidence-based reason.
"""
