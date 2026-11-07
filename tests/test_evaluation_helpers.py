from deepeval.test_case import ToolCall
from langchain.messages import AIMessage, ToolMessage

from eval.dataset import dataset
from eval.evaluation_helpers import extract_executed_tool_calls


def test_extracts_only_executed_tool_calls():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_user_purchase_history",
                    "args": {"user": "user123"},
                    "id": "call-1",
                    "type": "tool_call",
                },
                {
                    "name": "issue_refund_ticket",
                    "args": {
                        "user": "user123",
                        "items": [{"name": "book"}],
                    },
                    "id": "call-2",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(
            content='{"status": "ok"}',
            name="get_user_purchase_history",
            tool_call_id="call-1",
        ),
    ]

    result = extract_executed_tool_calls(messages)

    assert len(result) == 1
    assert result[0].name == "get_user_purchase_history"
    assert result[0].input_parameters == {"user": "user123"}


def test_ignores_tool_result_without_matching_call():
    messages = [
        ToolMessage(
            content="result",
            name="calculate_right_refund",
            tool_call_id="missing-call",
        )
    ]

    assert extract_executed_tool_calls(messages) == []


def test_ignores_mismatched_tool_result_name():
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "issue_refund_ticket",
                    "args": {"user": "user123", "items": [{"name": "book"}]},
                    "id": "call-3",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="result",
            name="calculate_right_refund",
            tool_call_id="call-3",
        ),
    ]

    assert extract_executed_tool_calls(messages) == []


def test_refund_proposal_goldens_use_only_history_lookup():
    for index in (0, 1, 4, 5):
        golden = dataset.goldens[index]

        assert [tool.name for tool in golden.expected_tools] == [
            "get_user_purchase_history"
        ]
        assert len(golden.expected_tools) == 1

        expected_user = golden.input.split("I am ", 1)[1].split(",", 1)[0]
        assert golden.expected_tools[0].input_parameters == {
            "user": expected_user
        }
        assert "proposal" in golden.expected_output.lower()


def test_information_only_golden_uses_history_without_proposal():
    golden = dataset.goldens[2]

    assert [tool.name for tool in golden.expected_tools] == [
        "get_user_purchase_history"
    ]
    assert "information-only" in golden.expected_output
    assert "No refund proposal" in golden.expected_output


def test_mixed_item_request_proposes_only_requested_items():
    golden = dataset.goldens[5]

    assert [tool.name for tool in golden.expected_tools] == [
        "get_user_purchase_history"
    ]
    assert "backpack and water_bottle" in golden.expected_output
    assert "does not decide eligibility" in golden.expected_output
    assert "ticket" in golden.expected_output.lower()


def test_invalid_user_and_unpurchased_item_stop_after_history():
    assert [
        tool.name for tool in dataset.goldens[3].expected_tools
    ] == ["get_user_purchase_history"]

    assert [
        tool.name for tool in dataset.goldens[6].expected_tools
    ] == ["get_user_purchase_history"]
