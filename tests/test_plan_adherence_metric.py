from eval.eval_all import build_metrics
from eval.plan_adherence_metric import (
    ProjectPlanAdherenceMetric,
    ProjectPlanAdherenceTemplate,
)


def test_plan_extraction_prompt_excludes_hypothetical_followups():
    trace = (
        "I will look up the purchase history. "
        "I could ask whether the user wants to discuss a specific item."
    )

    prompt = ProjectPlanAdherenceTemplate.extract_plan_from_trace(
        trace_json_str=trace,
        multimodal=False,
    )

    assert trace in prompt
    assert "hypothetical, conditional, speculative, or optional thoughts" in prompt
    assert "A polite offer of further help" in prompt


def test_adherence_prompt_checks_committed_steps_against_execution():
    task = "List user123's purchases without issuing a refund."
    plan = "1. Look up purchase history. 2. List the purchases."
    trace = '{"tool": "get_user_purchase_history", "executed": true}'

    prompt = ProjectPlanAdherenceTemplate.evaluate_adherence(
        user_task=task,
        agent_plan=plan,
        execution_trace_json=trace,
        multimodal=False,
    )

    assert task in prompt
    assert plan in prompt
    assert trace in prompt
    assert "Do not penalize conditional thoughts" in prompt
    assert "structured-output serialization" in prompt


def test_project_metric_routes_both_plan_prompts():
    _, _, _, metric = build_metrics()[0]

    assert isinstance(metric, ProjectPlanAdherenceMetric)
    assert metric.threshold == 0.8
    # The experimental system-one path requires a registered built-in prompt.
    # Returning None forces the regular path using our project-owned prompts.
    assert metric._system_one_score_spec("List purchases.", [], None) is None

    extraction = metric._get_prompt(
        "extract_plan_from_trace",
        template_class="PlanAdherenceMetric",
        trace_json_str='{"reasoning": "I will check purchase history."}',
        multimodal=False,
    )
    scoring = metric._get_prompt(
        "evaluate_adherence",
        template_class="PlanAdherenceMetric",
        user_task="List purchases.",
        agent_plan="1. Look up purchase history.",
        execution_trace_json='{"tool": "get_user_purchase_history"}',
        multimodal=False,
    )

    assert "committed plan" in extraction.lower()
    assert "COMMITTED PLAN" in scoring
    assert "List purchases." in scoring

def test_extraction_prompt_requires_string_plan_steps():
    prompt = ProjectPlanAdherenceTemplate.extract_plan_from_trace(
        trace_json_str='{"tool": "get_user_purchase_history"}',
        multimodal=False,
    )

    assert '"plan": ["first committed action as a string"' in prompt
    assert '"plan" must be an array of strings' in prompt
    assert "Never return dictionaries or objects as plan elements" in prompt
    assert '"action": "get_user_purchase_history"' in prompt
    assert "Return valid JSON only" in prompt

def test_adherence_prompt_requires_exact_deepeval_score_schema():
    prompt = ProjectPlanAdherenceTemplate.evaluate_adherence(
        user_task="List user123's purchases.",
        agent_plan="1. Look up purchase history.",
        execution_trace_json='{"tool": "get_user_purchase_history"}',
        multimodal=False,
    )

    assert '"score": 0.85' in prompt
    assert '"reason": "Evidence-based explanation' in prompt
    assert '"score" and "reason"' in prompt
    assert 'Do not wrap these fields in a "PlanAdherenceScore" key' in prompt


def test_adherence_prompt_preserves_scoring_evidence_requirements():
    prompt = ProjectPlanAdherenceTemplate.evaluate_adherence(
        user_task="List purchases.",
        agent_plan="1. Look up purchase history.",
        execution_trace_json='{"tool": "get_user_purchase_history"}',
        multimodal=False,
    )

    assert "between 0.0 and 1.0" in prompt
    assert "concise, evidence-based string" in prompt
    assert "Require evidence that it was actually performed" in prompt

