from deepeval.metrics import TaskCompletionMetric

from eval.eval_all import SUCCESSFUL_TASK_DESCRIPTION, build_metrics
from eval.task_completion_template import ProjectTaskCompletionTemplate


def test_invalid_username_terminal_branch_is_explicitly_complete():
    outcome = (
        "Purchase lookup returned invalid_username. "
        "The agent stopped, issued no ticket, and asked for correction."
    )
    prompt = ProjectTaskCompletionTemplate.generate_verdict(
        task=SUCCESSFUL_TASK_DESCRIPTION,
        actual_outcome=outcome,
    )

    assert outcome in prompt
    assert "INVALID USERNAME" in prompt
    assert "ask the customer to correct the username" in prompt
    assert "Do not penalize" in prompt


def test_premature_stop_is_not_a_successful_terminal_branch():
    prompt = ProjectTaskCompletionTemplate.generate_verdict(
        task=SUCCESSFUL_TASK_DESCRIPTION,
        actual_outcome=(
            "The agent stopped without looking up the supplied username "
            "or establishing a reason it could not proceed."
        ),
    )

    assert "Ask for the username before attempting purchase-history lookup." in prompt
    assert "premature stop without evidence" in prompt
    assert "the agent must attempt verification" in prompt


def test_task_completion_metric_registers_custom_template():
    trace_metrics, _ = build_metrics()
    metric = trace_metrics[0]

    assert isinstance(metric, TaskCompletionMetric)
    assert metric.threshold == 0.8
    assert metric.task == SUCCESSFUL_TASK_DESCRIPTION
    assert metric.evaluation_template is ProjectTaskCompletionTemplate


def test_registered_metric_renders_custom_verdict_prompt():
    trace_metrics, _ = build_metrics()
    metric = trace_metrics[0]

    outcome = (
        "Lookup returned invalid_username. "
        "The agent stopped and asked for username verification."
    )
    prompt = metric._get_prompt(
        "generate_verdict",
        task=metric.task,
        actual_outcome=outcome,
    )

    assert outcome in prompt
    assert "2. INVALID USERNAME:" in prompt
    assert "ask the customer to correct the username" in prompt
