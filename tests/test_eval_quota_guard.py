import pytest

from eval.quota_guard import (
    JudgeQuotaBlocked,
    block_judge_requests,
    ensure_judge_available,
    is_daily_token_quota_error,
    judge_block_reason,
    reset_judge_quota_breaker,
)


@pytest.fixture(autouse=True)
def reset_breaker():
    reset_judge_quota_breaker()
    yield
    reset_judge_quota_breaker()


@pytest.mark.parametrize(
    "message",
    [
        "429 RateLimitError: tokens per day (TPD) limit reached",
        "Daily token quota exhausted",
        "The daily token limit has been reached",
    ],
)
def test_detects_daily_token_quota(message):
    assert is_daily_token_quota_error(message)


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests: requests per minute limit",
        "Rate limit reached: tokens per minute",
        "APIConnectionError: Connection error",
    ],
)
def test_does_not_misclassify_other_errors(message):
    assert not is_daily_token_quota_error(message)


def test_breaker_blocks_requests_locally():
    block_judge_requests("429 tokens per day quota exhausted")

    with pytest.raises(JudgeQuotaBlocked, match="blocked locally"):
        ensure_judge_available()


def test_first_quota_reason_is_retained():
    block_judge_requests("first daily quota error")
    block_judge_requests("second daily quota error")

    assert judge_block_reason() == "first daily quota error"


def test_explicit_reset_reopens_breaker():
    block_judge_requests("daily token quota exhausted")
    reset_judge_quota_breaker()

    ensure_judge_available()
    assert judge_block_reason() is None
