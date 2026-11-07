from deepeval.metrics import (
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)

from eval.eval_all import build_metrics
from eval.step_efficiency_template import ProjectStepEfficiencyTemplate


def test_custom_efficiency_prompt_uses_task_and_trace():
    task = "List the purchases for user123 without issuing a refund."
    trace = '{"tool": "get_user_purchase_history", "executed": true}'

    prompt = ProjectStepEfficiencyTemplate.get_execution_efficiency(
        task=task,
        trace_json_str=trace,
        multimodal=False,
    )

    assert task in prompt
    assert trace in prompt
    assert "A refund proposal normally needs" in prompt
    assert "Do not penalize the agent for not calculating eligibility or issuing" in prompt


def test_build_metrics_uses_custom_efficiency_template_without_changing_other_metrics():
    trace_metrics, single_turn_metrics = build_metrics()

    assert len(trace_metrics) == 4
    assert len(single_turn_metrics) == 3

    task_metric, efficiency_metric, plan_quality, plan_adherence = trace_metrics

    assert isinstance(task_metric, TaskCompletionMetric)
    assert isinstance(efficiency_metric, StepEfficiencyMetric)
    assert isinstance(plan_quality, PlanQualityMetric)
    assert isinstance(plan_adherence, PlanAdherenceMetric)

    assert efficiency_metric.evaluation_template is ProjectStepEfficiencyTemplate
    assert efficiency_metric.threshold == 0.8
    assert efficiency_metric.model is not None

    assert task_metric.threshold == 0.8
    assert plan_quality.threshold == 0.8
    assert plan_adherence.threshold == 0.8
    assert single_turn_metrics[1].threshold == 0.9
