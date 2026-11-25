import argparse
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import (
    GEval,
    PlanAdherenceMetric,

    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCallParams

if __package__:
    from .dataset import dataset
    from .model import eval_model
    from .evaluation_helpers import invoke_agent_for_golden
    from .step_efficiency_template import ProjectStepEfficiencyTemplate
    from .plan_adherence_metric import ProjectPlanAdherenceMetric
    from .plan_quality_template import ProjectPlanQualityMetric
    from .task_completion_template import ProjectTaskCompletionTemplate
else:
    from dataset import dataset
    from model import eval_model
    from evaluation_helpers import invoke_agent_for_golden
    from step_efficiency_template import ProjectStepEfficiencyTemplate
    from plan_adherence_metric import ProjectPlanAdherenceMetric
    from plan_quality_template import ProjectPlanQualityMetric
    from task_completion_template import ProjectTaskCompletionTemplate



def deterministic_argument_result(expected_tools, actual_tools, threshold=0.8):
    """Compare arguments using optimal one-to-one matching per tool name."""

    def read_field(call, field, default=None):
        if isinstance(call, dict):
            return call.get(field, default)
        return getattr(call, field, default)

    expected_by_name = {}
    actual_by_name = {}

    for call in expected_tools or []:
        name = read_field(call, "name")
        if name:
            expected_by_name.setdefault(name, []).append(call)

    for call in actual_tools or []:
        name = read_field(call, "name")
        if name:
            actual_by_name.setdefault(name, []).append(call)

    comparisons = []
    comparable_calls = 0

    for name, expected_calls in expected_by_name.items():
        actual_calls = actual_by_name.get(name, [])
        pair_count = min(len(expected_calls), len(actual_calls))

        if pair_count == 0:
            continue

        comparable_calls += pair_count
        pair_scores = []

        for expected_call in expected_calls:
            expected_args = read_field(
                expected_call, "input_parameters", {}
            ) or {}
            if not isinstance(expected_args, dict):
                raise TypeError(
                    f"Expected arguments for {name} must be a dictionary."
                )

            row = []
            for actual_call in actual_calls:
                actual_args = read_field(
                    actual_call, "input_parameters", {}
                ) or {}
                if not isinstance(actual_args, dict):
                    raise TypeError(
                        f"Actual arguments for {name} must be a dictionary."
                    )

                keys = set(expected_args) | set(actual_args)
                if not keys:
                    row.append((1, 1))
                else:
                    matches = sum(
                        key in expected_args
                        and key in actual_args
                        and expected_args[key] == actual_args[key]
                        for key in keys
                    )
                    row.append((matches, len(keys)))
            pair_scores.append(row)

        # DP explores one-to-one assignments and maximizes the aggregate
        # argument-match ratio, rather than making greedy local choices.
        states = {(0, 0, 0)}

        for row in pair_scores:
            next_states = set(states)
            for mask, matched_so_far, total_so_far in states:
                if mask.bit_count() >= pair_count:
                    continue

                for actual_index, (matches, total) in enumerate(row):
                    bit = 1 << actual_index
                    if mask & bit:
                        continue

                    next_states.add((
                        mask | bit,
                        matched_so_far + matches,
                        total_so_far + total,
                    ))
            states = next_states

        valid = [
            (matched, total)
            for mask, matched, total in states
            if mask.bit_count() == pair_count
        ]

        if valid:
            comparisons.append(
                max(valid, key=lambda item: item[0] / item[1])
            )

    if not comparisons:
        return {
            "status": "SKIPPED",
            "score": None,
            "reason": (
                "No actual tool call shared a name with an expected tool call. "
                "Tool selection is evaluated separately."
            ),
        }

    matched = sum(item[0] for item in comparisons)
    total = sum(item[1] for item in comparisons)
    score = matched / total

    return {
        "status": "PASS" if score >= threshold else "FAIL",
        "score": score,
        "reason": (
            f"Matched {matched} of {total} argument key/value comparisons "
            f"across {comparable_calls} comparable tool call(s). "
            f"Threshold: {threshold:.2f}."
        ),
    }


class DeterministicArgumentCorrectnessMetric:
    """Marker/configuration for the locally evaluated argument metric."""

    name = "Argument Correctness"
    threshold = 0.8
    deterministic = True


SUCCESSFUL_TASK_DESCRIPTION = """
Evaluate only the conversational agent's responsibilities.

The agent verifies the supplied username using purchase history, identifies
only exact requested items, asks for clarification when required information
is missing or an item cannot be matched, and returns a structured refund
proposal when the username and exact requested items are established.

For information-only requests, it provides purchase information without
creating a refund proposal.

A confirmed invalid username is a legitimate terminal condition: the agent
must stop, avoid disclosing purchase records, and ask the customer to correct
the username.

The agent must not decide refund eligibility, calculate a quote, obtain
customer approval, or issue a ticket. Those are application responsibilities.
Do not penalize the agent for omitting those downstream application actions.
Do penalize unsupported claims that a refund is approved, quoted, or ticketed.

Judge completion against the actual request branch and evidence in the trace.
"""


def build_metrics():
    """Keep the thresholds and evaluator model used by the individual scripts."""
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            model=eval_model,
            include_reason=True,
            evaluation_template=ProjectTaskCompletionTemplate,
        ),
        StepEfficiencyMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
            evaluation_template=ProjectStepEfficiencyTemplate,
        ),
        ProjectPlanQualityMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
        ProjectPlanAdherenceMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
    ]
    single_turn_metrics = [
        DeterministicArgumentCorrectnessMetric(),
        ToolCorrectnessMetric(
            threshold=0.9,
            model=eval_model,
            include_reason=True,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Treat expected_output as a behavioral specification, "
                "not as a sentence that the agent must repeat verbatim.",
                "Evaluate only the customer-facing actual_output. "
                "Do not require internal tool calls or structured proposal "
                "fields to be repeated in the natural-language answer.",
                "Check whether the answer appropriately acknowledges the "
                "customer's request and communicates the correct next step.",
                "For a valid refund proposal, it is acceptable to explain "
                "that the application will independently validate eligibility "
                "and may calculate a quote for the customer's review.",
                "Do not confuse explaining a downstream application step "
                "with the agent itself deciding eligibility, calculating "
                "or promising a refund amount, approving a refund, or "
                "claiming a ticket was issued.",
                "Penalize unsupported eligibility decisions, invented "
                "amounts, approval claims, ticket-issuance claims, "
                "incorrect item names, and failure to address the request.",
                "Judge the answer in the context of the user's actual "
                "request branch and the behavioral specification."
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


def main():
    parser = argparse.ArgumentParser(description="Run AgentAudit evaluations.")
    parser.add_argument(
        "--case-number",
        type=int,
        default=None,
        help="Run one 1-based golden case number; omit to run the full dataset.",
    )
    args = parser.parse_args()

    if args.case_number is not None and not 1 <= args.case_number <= len(dataset.goldens):
        parser.error(f"--case-number must be between 1 and {len(dataset.goldens)}.")

    selected_goldens = (
        dataset.goldens
        if args.case_number is None
        else [dataset.goldens[args.case_number - 1]]
    )
    run_dataset = EvaluationDataset(goldens=selected_goldens)

    load_dotenv()
    from agent.agent import agent

    trace_metrics, single_turn_metrics = build_metrics()
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    async_config = AsyncConfig(
        run_async=False,
        throttle_value=1,
        max_concurrent=1,
    )
    test_cases = []

    # The callback captures the trace required by the four agentic metrics.
    for golden in run_dataset.evals_iterator(
        metrics=trace_metrics,
        cache_config=cache_config,
        display_config=display_config,
        async_config=async_config,
    ):
        response, tools_called = invoke_agent_for_golden(agent, golden)

        # Reuse the same response for the three remaining metrics. Keep these
        # cases local so repeated runs do not accumulate dataset.test_cases.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=response["structured_response"].answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    deterministic_metrics = [
        metric for metric in single_turn_metrics
        if getattr(metric, "deterministic", False)
    ]
    judge_metrics = [
        metric for metric in single_turn_metrics
        if not getattr(metric, "deterministic", False)
    ]

    for metric in deterministic_metrics:
        print(f"\\n=== {metric.name} (DETERMINISTIC) ===")
        for index, (golden, test_case) in enumerate(
            zip(selected_goldens, test_cases), start=1
        ):
            result = deterministic_argument_result(
                expected_tools=golden.expected_tools or [],
                actual_tools=test_case.tools_called or [],
                threshold=metric.threshold,
            )
            score_text = (
                "n/a" if result["score"] is None
                else f"{result['score']:.2f}"
            )
            print(
                f"Case {index}: {result['status']} | "
                f"score={score_text} | {result['reason']}"
            )

    evaluation_result = evaluate(
        test_cases=test_cases,
        metrics=judge_metrics,
        async_config=async_config,
        cache_config=cache_config,
        display_config=display_config,
    )

    for test_result in evaluation_result.test_results:
        for metric_data in test_result.metrics_data or []:
            if metric_data.name == "Argument Correctness":
                print("\n=== ARGUMENT CORRECTNESS DIAGNOSTICS ===")
                print(metric_data.verbose_logs or metric_data.reason or "No diagnostic details returned.")

    print(f"\nCompleted all 7 metrics for {len(test_cases)} test cases.")


if __name__ == "__main__":
    main()
