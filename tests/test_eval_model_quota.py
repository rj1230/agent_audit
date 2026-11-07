import asyncio

import pytest

import eval.model as judge_module
from eval.quota_guard import (
    JudgeQuotaBlocked,
    judge_block_reason,
    reset_judge_quota_breaker,
)


class FakeSyncModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        raise RuntimeError(
            "Groq 429: tokens per day limit reached"
        )


class FakeAsyncModel:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        raise RuntimeError(
            "Groq 429: tokens per day limit reached"
        )


@pytest.fixture(autouse=True)
def isolate_quota_breaker(monkeypatch):
    reset_judge_quota_breaker()
    monkeypatch.setattr(
        judge_module,
        "JUDGE_MIN_INTERVAL_SECONDS",
        0,
    )
    yield
    reset_judge_quota_breaker()


def test_sync_judge_latches_daily_quota_and_blocks_next_call(monkeypatch):
    fake = FakeSyncModel()
    monkeypatch.setattr(judge_module.eval_model, "model", fake)

    with pytest.raises(RuntimeError, match="tokens per day"):
        judge_module.eval_model.generate("first evaluation prompt")

    assert fake.calls == 1
    assert judge_block_reason() is not None

    with pytest.raises(JudgeQuotaBlocked):
        judge_module.eval_model.generate("second evaluation prompt")

    assert fake.calls == 1


def test_async_judge_latches_daily_quota_and_blocks_next_call(monkeypatch):
    fake = FakeAsyncModel()
    monkeypatch.setattr(judge_module.eval_model, "model", fake)

    async def exercise():
        with pytest.raises(RuntimeError, match="tokens per day"):
            await judge_module.eval_model.a_generate(
                "first async evaluation prompt"
            )

        assert fake.calls == 1
        assert judge_block_reason() is not None

        with pytest.raises(JudgeQuotaBlocked):
            await judge_module.eval_model.a_generate(
                "second async evaluation prompt"
            )

        assert fake.calls == 1

    asyncio.run(exercise())
