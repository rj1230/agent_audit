from types import SimpleNamespace

import pytest

import eval.eval_persistent as runner
from eval.quota_guard import (
    block_judge_requests,
    reset_judge_quota_breaker,
)


class FakeStore:
    def __init__(self):
        self.case_saves = []
        self.metric_saves = []
        self.trace_artifacts = {}

    def save_case(self, **kwargs):
        self.case_saves.append(kwargs)

    def save_metric(self, *args, **kwargs):
        self.metric_saves.append((args, kwargs))

    def get_metric(self, run_id, case_id, metric_name):
        return None

    def save_trace_artifact(self, run_id, case_id, trace_dict):
        self.trace_artifacts[(run_id, case_id)] = {
            "schema_version": 1,
            "trace": trace_dict,
        }

    def get_trace_artifact(self, run_id, case_id):
        return self.trace_artifacts.get((run_id, case_id))


class FakeIterator:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(input="test")
        raise RuntimeError("simulated trace finalization failure")

    def close(self):
        self.closed = True


class FakeDataset:
    def __init__(self, goldens):
        self.goldens = goldens

    def evals_iterator(self, **kwargs):
        return fake_iterator


@pytest.fixture(autouse=True)
def reset_breaker():
    reset_judge_quota_breaker()
    yield
    reset_judge_quota_breaker()


def test_metric_status_marks_local_quota_block_as_skipped():
    data = SimpleNamespace(
        error="JudgeQuotaBlocked: blocked locally after daily quota exhaustion",
        score=None,
        success=None,
    )

    status, score, error = runner.metric_status(data)

    assert status == "SKIPPED"
    assert score is None
    assert runner.QUOTA_BLOCKED_MARKER in error


def test_metric_status_marks_provider_daily_quota_error():
    data = SimpleNamespace(
        error="Groq 429: tokens per day limit reached",
        score=None,
        success=None,
    )

    status, score, error = runner.metric_status(data)

    assert status == "EVALUATION_ERROR"
    assert runner.QUOTA_BLOCKED_MARKER in error


def test_metric_status_keeps_ordinary_evaluation_errors():
    data = SimpleNamespace(
        error="APIConnectionError: connection failed",
        score=None,
        success=None,
    )

    status, score, error = runner.metric_status(data)

    assert status == "EVALUATION_ERROR"
    assert error == "APIConnectionError: connection failed"
    assert runner.QUOTA_BLOCKED_MARKER not in error


def test_quota_marker_is_detected_in_persisted_metric():
    row = {
        "status": "EVALUATION_ERROR",
        "error": (
            "[JUDGE_DAILY_QUOTA_BLOCKED] "
            "Groq 429: tokens per day limit reached"
        ),
    }

    assert runner.is_quota_blocked_metric(row)
    assert not runner.is_quota_blocked_metric(
        {"status": "EVALUATION_ERROR", "error": "APIConnectionError"}
    )
    assert not runner.is_quota_blocked_metric(None)


def test_trace_finalization_failure_preserves_completed_agent_output(monkeypatch):
    global fake_iterator
    fake_iterator = FakeIterator()
    store = FakeStore()

    golden = SimpleNamespace(
        input="test input",
        expected_output="expected answer",
    )
    metric = SimpleNamespace(
        name="Trace Metric",
        threshold=0.5,
    )

    monkeypatch.setattr(
        runner,
        "EvaluationDataset",
        FakeDataset,
    )
    monkeypatch.setattr(
        runner,
        "invoke_agent_for_golden",
        lambda agent, yielded, *, capture_trace=False: (
            (
                {
                    "structured_response": SimpleNamespace(answer="saved answer")
                },
                ["lookup_tool"],
                None,
                "simulated trace capture unavailable",
            )
            if capture_trace
            else (
                {
                    "structured_response": SimpleNamespace(answer="saved answer")
                },
                ["lookup_tool"],
            )
        ),
    )

    answer, tools_called = runner.evaluate_trace_case(
        golden=golden,
        agent=object(),
        metrics=[metric],
        store=store,
        run_id="test-run",
        case_id="case-001",
        case_number=1,
    )

    assert answer == "saved answer"
    assert tools_called == ["lookup_tool"]
    assert len(store.case_saves) == 1
    assert store.case_saves[0]["invocation_status"] == "COMPLETED"
    assert store.case_saves[0]["actual_output"] == "saved answer"
    assert store.case_saves[0]["tools_called"] == ["lookup_tool"]
    assert fake_iterator.closed is True

    assert len(store.metric_saves) == 1
    assert store.metric_saves[0][1]["status"] == "EVALUATION_ERROR"


def test_trace_finalization_failure_marks_pending_metrics_skipped_on_quota():
    global fake_iterator
    fake_iterator = FakeIterator()
    store = FakeStore()

    golden = SimpleNamespace(
        input="test input",
        expected_output="expected answer",
    )
    metric = SimpleNamespace(
        name="Trace Metric",
        threshold=0.5,
    )

    block_judge_requests("Groq 429: tokens per day limit reached")

    class QuotaDataset:
        def __init__(self, goldens):
            pass

        def evals_iterator(self, **kwargs):
            return fake_iterator

    original_dataset = runner.EvaluationDataset
    original_invoke = runner.invoke_agent_for_golden

    try:
        runner.EvaluationDataset = QuotaDataset
        runner.invoke_agent_for_golden = lambda agent, yielded, *, capture_trace=False: (
            (
                {
                    "structured_response": SimpleNamespace(answer="saved answer")
                },
                ["lookup_tool"],
                None,
                "simulated trace capture unavailable",
            )
            if capture_trace
            else (
                {
                    "structured_response": SimpleNamespace(answer="saved answer")
                },
                ["lookup_tool"],
            )
        )

        answer, tools_called = runner.evaluate_trace_case(
            golden, object(), [metric], store, "test-run", "case-001", 1
        )
    finally:
        runner.EvaluationDataset = original_dataset
        runner.invoke_agent_for_golden = original_invoke

    assert answer == "saved answer"
    assert tools_called == ["lookup_tool"]
    assert len(store.case_saves) == 1
    assert store.case_saves[0]["invocation_status"] == "COMPLETED"
    assert store.metric_saves[0][1]["status"] == "SKIPPED"
    assert runner.QUOTA_BLOCKED_MARKER in store.metric_saves[0][1]["error"]



def test_saved_trace_metric_grades_without_agent_invocation(monkeypatch):
    store = FakeStore()
    golden = SimpleNamespace(
        input="test input",
        expected_output="expected answer",
        expected_tools=[runner.ToolCall(name="lookup_tool", input_parameters={"query": "test"})],
    )
    metric = SimpleNamespace(name="Trace Metric", threshold=0.5)
    saved_trace = {
        "root": {
            "name": "agent",
            "children": [{"name": "lookup_tool"}],
        }
    }

    captured = {}
    metric_data = SimpleNamespace(
        name="Trace Metric",
        score=1.0,
        success=True,
        threshold=0.5,
        reason="Trace behavior matches expectations.",
        error=None,
    )
    evaluation_result = SimpleNamespace(
        test_results=[
            SimpleNamespace(metrics_data=[metric_data])
        ]
    )

    def fake_evaluate(**kwargs):
        captured["test_case"] = kwargs["test_cases"][0]
        captured["metrics"] = kwargs["metrics"]
        return evaluation_result

    def unexpected_agent_invocation(*args, **kwargs):
        pytest.fail("Agent must not be invoked when grading a saved trace.")

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        runner,
        "invoke_agent_for_golden",
        unexpected_agent_invocation,
    )

    status = runner.evaluate_saved_trace_metric(
        golden=golden,
        answer="saved answer",
        tools_called=[
            {
                "name": "lookup_tool",
                "type": "FUNCTION",
                "input_parameters": {"query": "test"},
            }
        ],
        trace_dict=saved_trace,
        metric=metric,
        store=store,
        run_id="test-run",
        case_id="case-001",
    )

    assert status == "PASS"
    assert captured["test_case"]._trace_dict == saved_trace
    assert captured["test_case"].input == "test input"
    assert captured["test_case"].actual_output == "saved answer"
    assert captured["test_case"].expected_output == "expected answer"
    assert captured["metrics"] == [metric]
    restored = captured["test_case"].tools_called
    assert len(restored) == 1
    assert isinstance(restored[0], runner.ToolCall)
    assert restored[0].name == "lookup_tool"
    assert restored[0].input_parameters == {"query": "test"}
    assert captured["test_case"].expected_tools[0].name == "lookup_tool"
    assert len(store.metric_saves) == 1

    saved = store.metric_saves[0][1]
    assert saved["status"] == "PASS"
    assert saved["score"] == 1.0
    assert saved["reason"] == "Trace behavior matches expectations."

def _prepare_main_resume_test(
    monkeypatch,
    tmp_path,
    *,
    save_artifact=True,
    prior_metric=None,
    retry_quota=False,
):
    """Seed a compatible SQLite run and isolate main() from external services."""
    import sys
    from types import ModuleType

    golden = SimpleNamespace(
        input="test input",
        expected_output="expected answer",
        expected_tools=[],
        context=None,
        additional_metadata=None,
    )
    metric = SimpleNamespace(name="Trace Metric", threshold=0.5)
    run_id = "main-resume-test"
    db_path = tmp_path / "resume.sqlite3"

    store = runner.ResultStore(db_path)
    store.start_run(
        run_id=run_id,
        dataset_fingerprint=runner.fingerprint(
            [runner.golden_payload(golden)]
        ),
        config_fingerprint=runner.fingerprint(
            runner.metric_config([metric])
        ),
        metadata={"case_numbers": [1], "metrics": ["Trace Metric"]},
    )
    store.save_case(
        run_id,
        "case-001",
        1,
        golden.input,
        golden.expected_output,
        "saved answer",
        [],
        invocation_status="COMPLETED",
    )

    saved_trace = {
        "root": {
            "name": "agent",
            "children": [{"name": "lookup_tool"}],
        }
    }

    if save_artifact:
        store.save_trace_artifact(
            run_id, "case-001", saved_trace
        )

    if prior_metric is not None:
        store.save_metric(
            run_id,
            "case-001",
            "Trace Metric",
            status=prior_metric["status"],
            error=prior_metric.get("error"),
        )

    monkeypatch.setattr(
        runner,
        "dataset",
        SimpleNamespace(goldens=[golden]),
    )
    monkeypatch.setattr(
        runner,
        "build_metrics",
        lambda: ([metric], []),
    )
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner, "ensure_judge_available", lambda: None)

    agent_package = ModuleType("agent")
    agent_package.__path__ = []
    agent_module = ModuleType("agent.agent")
    agent_module.agent = object()

    monkeypatch.setitem(sys.modules, "agent", agent_package)
    monkeypatch.setitem(sys.modules, "agent.agent", agent_module)

    captured = {"graded": []}

    def fake_grade(
        golden_arg,
        answer,
        tools_called,
        trace_dict,
        metric_arg,
        store_arg,
        run_id_arg,
        case_id_arg,
    ):
        captured["graded"].append(metric_arg.name)
        assert answer == "saved answer"
        assert trace_dict == saved_trace
        store_arg.save_metric(
            run_id_arg,
            case_id_arg,
            runner.metric_storage_name(metric_arg),
            status="PASS",
            score=1.0,
            threshold=metric_arg.threshold,
            reason="Mocked saved-trace grading passed.",
        )
        return "PASS"

    monkeypatch.setattr(
        runner,
        "evaluate_saved_trace_metric",
        fake_grade,
    )

    def unexpected_agent_invocation(*args, **kwargs):
        pytest.fail(
            "main() must not reinvoke an already-completed agent."
        )

    monkeypatch.setattr(
        runner,
        "invoke_agent_for_golden",
        unexpected_agent_invocation,
    )

    argv = [
        "eval_persistent.py",
        "--resume",
        "--run-id",
        run_id,
        "--db",
        str(db_path),
    ]

    if retry_quota:
        argv.append("--retry-quota-blocked")

    monkeypatch.setattr(sys, "argv", argv)

    return store, run_id, captured


def test_main_resume_grades_saved_trace_without_agent_invocation(
    monkeypatch, tmp_path
):
    store, run_id, captured = _prepare_main_resume_test(
        monkeypatch, tmp_path
    )

    runner.main()

    assert captured["graded"] == ["Trace Metric"]
    assert store.get_metric(
        run_id, "case-001", "Trace Metric"
    )["status"] == "PASS"
    assert store.get_case(
        run_id, "case-001"
    )["invocation_status"] == "COMPLETED"


def test_main_resume_missing_trace_records_error_without_agent_invocation(
    monkeypatch, tmp_path
):
    store, run_id, captured = _prepare_main_resume_test(
        monkeypatch,
        tmp_path,
        save_artifact=False,
    )

    runner.main()

    assert captured["graded"] == []
    metric = store.get_metric(
        run_id, "case-001", "Trace Metric"
    )
    assert metric["status"] == "EVALUATION_ERROR"
    assert "no compatible saved trace artifact" in metric["error"]
    assert store.get_case(
        run_id, "case-001"
    )["invocation_status"] == "COMPLETED"


def test_main_resume_skips_quota_blocked_metric_by_default(
    monkeypatch, tmp_path
):
    prior = {
        "status": "EVALUATION_ERROR",
        "error": (
            runner.QUOTA_BLOCKED_MARKER
            + " Groq 429: tokens per day limit reached"
        ),
    }

    store, run_id, captured = _prepare_main_resume_test(
        monkeypatch,
        tmp_path,
        prior_metric=prior,
    )

    runner.main()

    assert captured["graded"] == []
    metric = store.get_metric(
        run_id, "case-001", "Trace Metric"
    )
    assert metric["status"] == "EVALUATION_ERROR"
    assert runner.is_quota_blocked_metric(metric)
    assert store.get_case(
        run_id, "case-001"
    )["invocation_status"] == "COMPLETED"


def test_main_explicit_quota_retry_uses_saved_trace_without_agent_invocation(
    monkeypatch, tmp_path
):
    prior = {
        "status": "EVALUATION_ERROR",
        "error": (
            runner.QUOTA_BLOCKED_MARKER
            + " Groq 429: tokens per day limit reached"
        ),
    }

    store, run_id, captured = _prepare_main_resume_test(
        monkeypatch,
        tmp_path,
        prior_metric=prior,
        retry_quota=True,
    )

    runner.main()

    assert captured["graded"] == ["Trace Metric"]
    assert store.get_metric(
        run_id, "case-001", "Trace Metric"
    )["status"] == "PASS"
    assert store.get_case(
        run_id, "case-001"
    )["invocation_status"] == "COMPLETED"
