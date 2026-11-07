import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase

from agent.agent import agent
from eval.dataset import dataset
from eval.evaluation_helpers import invoke_agent_for_golden
from eval.model import eval_model


def build_test_cases(case_number=None):
    """Build isolated cases using confirmed executed tool calls."""
    goldens = dataset.goldens

    if case_number is not None:
        if case_number < 1 or case_number > len(goldens):
            raise ValueError(
                f"Case number must be between 1 and {len(goldens)}."
            )
        selected_goldens = [goldens[case_number - 1]]
    else:
        selected_goldens = goldens

    test_cases = []

    for golden in selected_goldens:
        response, tools_called = invoke_agent_for_golden(agent, golden)
        structured_response = response["structured_response"]

        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=structured_response.answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=list(golden.expected_tools or []),
            )
        )

    return test_cases


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AgentAudit executed tool arguments."
    )
    parser.add_argument(
        "--case-number",
        type=int,
        help="Run one golden case (1-based). Default: all cases.",
    )
    args = parser.parse_args()

    test_cases = build_test_cases(args.case_number)
    print(f"Prepared {len(test_cases)} isolated test case(s).")

    metric = ArgumentCorrectnessMetric(
        threshold=0.8,
        model=eval_model,
        include_reason=True,
    )
    evaluate(test_cases=test_cases, metrics=[metric])


if __name__ == "__main__":
    main()
