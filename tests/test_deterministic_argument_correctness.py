import pytest

from eval.eval_all import deterministic_argument_result


def tool(name, **arguments):
    return {
        "name": name,
        "input_parameters": arguments,
    }


def test_exact_argument_match_passes():
    result = deterministic_argument_result(
        [tool("get_user_purchase_history", user="user123")],
        [tool("get_user_purchase_history", user="user123")],
    )

    assert result["status"] == "PASS"
    assert result["score"] == 1.0


def test_wrong_argument_value_fails():
    result = deterministic_argument_result(
        [tool("get_user_purchase_history", user="user123")],
        [tool("get_user_purchase_history", user="user456")],
    )

    assert result["status"] == "FAIL"
    assert result["score"] == 0.0


def test_partial_argument_match_uses_key_value_ratio():
    result = deterministic_argument_result(
        [tool("refund", user="user123", item="book")],
        [tool("refund", user="user123", item="pen")],
    )

    assert result["status"] == "FAIL"
    assert result["score"] == 0.5


def test_extra_actual_argument_counts_as_mismatch():
    result = deterministic_argument_result(
        [tool("history", user="user123")],
        [tool("history", user="user123", limit=5)],
    )

    assert result["status"] == "FAIL"
    assert result["score"] == 0.5


def test_missing_actual_argument_counts_as_mismatch():
    result = deterministic_argument_result(
        [tool("refund", user="user123", item="book")],
        [tool("refund", user="user123")],
    )

    assert result["status"] == "FAIL"
    assert result["score"] == 0.5


def test_no_shared_tool_names_is_skipped():
    result = deterministic_argument_result(
        [tool("history", user="user123")],
        [tool("refund", user="user123")],
    )

    assert result["status"] == "SKIPPED"
    assert result["score"] is None


def test_no_expected_tools_is_skipped():
    result = deterministic_argument_result([], [tool("history", user="user123")])

    assert result["status"] == "SKIPPED"
    assert result["score"] is None


def test_empty_arguments_match():
    result = deterministic_argument_result(
        [tool("get_today_date")],
        [tool("get_today_date")],
    )

    assert result["status"] == "PASS"
    assert result["score"] == 1.0


def test_custom_threshold_is_respected():
    result = deterministic_argument_result(
        [tool("refund", user="user123", item="book")],
        [tool("refund", user="user123", item="pen")],
        threshold=0.5,
    )

    assert result["status"] == "PASS"
    assert result["score"] == 0.5

def test_repeated_tool_calls_use_global_best_matching():
    expected = [
        tool("lookup", a=1, b=1),
        tool("lookup", a=1),
    ]
    actual = [
        tool("lookup", a=1),
        tool("lookup", b=1),
    ]

    result = deterministic_argument_result(expected, actual)

    assert result["status"] == "FAIL"
    assert result["score"] == pytest.approx(2 / 3)

