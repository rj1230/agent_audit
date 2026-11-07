from eval.eval_all import build_metrics
from eval.plan_quality_template import (
    ProjectPlanQualityMetric,
    ProjectPlanQualityTemplate,
)


def test_plan_extraction_prompt_captures_result_dependent_terminal_branch():
    trace = (
        '{"tool": "get_user_purchase_history", '
        '"output": {"status": "invalid_username"}, '
        '"final_response": "Please verify your username."}'
    )

    prompt = ProjectPlanQualityTemplate.extract_plan_from_trace(
        trace_json_str=trace,
        multimodal=False,
    )

    assert trace in prompt
    assert "invalid_username" in prompt
    assert "Stop safely when username verification fails." in prompt
    assert "Do not invent" in prompt


def test_plan_quality_prompt_accepts_safe_terminal_stop():
    task = "Refund the requested book for user9999."
    plan = (
        "1. Look up the supplied username. "
        "2. Stop after invalid_username. "
        "3. Ask the user to verify the username."
    )

    prompt = ProjectPlanQualityTemplate.evaluate_plan_quality(
        user_task=task,
        agent_plan=plan,
        multimodal=False,
    )

    assert task in prompt
    assert plan in prompt
    assert "Do not penalize the agent for omitting application-side checks or ticket" in prompt
    assert "Invalid username: stop and ask the customer to correct it." in prompt


def test_project_plan_quality_metric_is_registered():
    trace_metrics, _ = build_metrics()
    metric = trace_metrics[2]

    assert isinstance(metric, ProjectPlanQualityMetric)
    assert metric.threshold == 0.8
    assert metric._system_one_score_spec(
        "Refund a purchase.", [], False
    ) is None


def test_project_metric_routes_extraction_and_scoring_prompts():
    metric = None  # Prompt-routing tests do not require model initialization.

    extraction = ProjectPlanQualityMetric._get_prompt(None,
        "extract_plan_from_trace",
        template_class="PlanAdherenceMetric",
        trace_json_str='{"output": "invalid_username"}',
        multimodal=False,
    )
    scoring = ProjectPlanQualityMetric._get_prompt(None,
        "evaluate_plan_quality",
        template_class="PlanQualityMetric",
        user_task="Refund a purchase.",
        agent_plan="Stop after invalid username.",
        multimodal=False,
    )

    assert "Stop safely when username verification fails." in extraction
    assert "invalid_username" in extraction
    assert "Stop after invalid username." in scoring
    assert "conditional correctness" in scoring.lower()