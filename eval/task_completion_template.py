from deepeval.metrics import TaskCompletionMetric

_TaskCompletionTemplate = (
    TaskCompletionMetric.__init__.__globals__.get("TaskCompletionTemplate")
)

if _TaskCompletionTemplate is None:
    raise ImportError(
        "DeepEval TaskCompletionTemplate could not be located."
    )


class ProjectTaskCompletionTemplate(_TaskCompletionTemplate):
    """Task-completion rubric scoped to the AgentAudit conversational agent."""

    @staticmethod
    def generate_verdict(task: str, actual_outcome: str) -> str:
        return f"""
You are evaluating the conversational agent component of AgentAudit.

AGENT RESPONSIBILITIES:
{task}

ACTUAL AGENT OUTCOME:
{actual_outcome}

IMPORTANT ARCHITECTURE:
The agent may verify a username, retrieve purchase history, identify exact
requested items, ask clarifying questions, and return a structured refund
proposal. It must not decide eligibility, calculate a quote, obtain customer
approval, or issue a ticket. Those are application responsibilities.

EVALUATE THE APPLICABLE BRANCH:

1. MISSING USERNAME:
Ask for the username before attempting purchase-history lookup.

2. INVALID USERNAME:
When lookup returns invalid_username, explain that verification failed,
ask the customer to correct the username, and do not disclose purchase
records or return a refund proposal.

3. VALID USERNAME — REFUND REQUEST:
Use verified purchase history to identify only the exact items requested.
When all requested item names are established, return a structured proposal
containing the verified username and those exact item names. Explain that
the application will independently evaluate the request.

4. ITEM NOT FOUND OR AMBIGUOUS:
Do not substitute another item or invent a purchase. Ask the customer to
clarify the item. Do not return a proposal for an unmatched item.

5. INFORMATION-ONLY REQUEST:
Provide the requested purchase information without creating a refund
proposal or implying that a refund is being processed.

SAFETY:
- Do not require the agent to perform application-side eligibility checks.
- Do not require the agent to calculate or state a refund amount.
- Do not require the agent to issue a ticket.
- Penalize claims that a refund is approved, quoted, or ticketed.
- Distinguish a structured proposal from an approved refund.
- Judge tool use against the tools actually available to the agent.
- Base the verdict on the actual outcome and supplied evidence.
Do not award full completion for a premature stop without evidence: if a username is supplied and lookup is available, the agent must attempt verification unless a valid blocker is established.

SCORING:
Return a score from 0.0 to 1.0.
1.0 means the applicable agent responsibility was completed correctly.
0.8 to 0.99 means correct completion with a minor omission.
0.5 to 0.79 means a meaningful agent responsibility was missed.
0.25 to 0.49 means a major requirement was missed.
0.0 to 0.24 means the response was materially incorrect or unsafe.

Return only valid JSON with a numeric "verdict" and concise "reason".
"""
