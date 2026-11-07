"""Local circuit-breaker helpers for daily Groq token quotas."""

import threading


class JudgeQuotaBlocked(RuntimeError):
    """Raised locally after the judge's daily token quota is exhausted."""


_lock = threading.Lock()
_block_reason = None


def is_daily_token_quota_error(error):
    """Recognize daily token exhaustion without treating every 429 as daily."""
    message = str(error).lower()

    daily_markers = (
        "tokens per day",
        "tokens/day",
        "token per day",
        "token/day",
        "daily token limit",
        "daily token quota",
        "tpd",
    )

    return any(marker in message for marker in daily_markers)


def block_judge_requests(reason):
    """Latch the process-wide breaker. The first reason is retained."""
    global _block_reason

    with _lock:
        if _block_reason is None:
            _block_reason = str(reason)[:1000]


def judge_block_reason():
    with _lock:
        return _block_reason


def ensure_judge_available():
    """Fail locally when the daily quota breaker has been triggered."""
    reason = judge_block_reason()
    if reason is not None:
        raise JudgeQuotaBlocked(
            "Judge requests blocked locally because the daily token quota "
            f"was exhausted earlier. Original error: {reason}"
        )


def reset_judge_quota_breaker():
    """Reset explicitly, primarily for isolated tests or a deliberate retry."""
    global _block_reason

    with _lock:
        _block_reason = None
