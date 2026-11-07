import pytest

from eval.result_store import ResultStore, fingerprint


@pytest.fixture
def store(tmp_path):
    instance = ResultStore(tmp_path / "results.sqlite3")
    instance.start_run(
        run_id="test-run",
        dataset_fingerprint=fingerprint(["case-1"]),
        config_fingerprint=fingerprint({"model": "test"}),
    )
    return instance


def test_run_creation_and_resumption(store):
    store.start_run(
        run_id="test-run",
        dataset_fingerprint=fingerprint(["case-1"]),
        config_fingerprint=fingerprint({"model": "test"}),
    )

    assert store.get_run("test-run")["status"] == "RUNNING"


def test_run_rejects_changed_fingerprint(store):
    with pytest.raises(ValueError, match="fingerprints"):
        store.start_run(
            run_id="test-run",
            dataset_fingerprint=fingerprint(["changed"]),
            config_fingerprint=fingerprint({"model": "test"}),
        )


def test_case_round_trip(store):
    store.save_case(
        run_id="test-run",
        case_id="case-001",
        case_number=1,
        input_text="hello",
        expected_output="world",
        actual_output="world",
        tools_called=[{"name": "lookup", "input_parameters": {"id": 1}}],
        invocation_status="COMPLETED",
    )

    case = store.get_case("test-run", "case-001")
    assert case["actual_output"] == "world"
    assert case["invocation_status"] == "COMPLETED"
    assert case["tools_called"][0]["name"] == "lookup"


@pytest.mark.parametrize(
    ("status", "score", "error"),
    [
        ("PASS", 1.0, None),
        ("FAIL", 0.2, None),
        ("EVALUATION_ERROR", None, "judge quota exceeded"),
        ("SKIPPED", None, None),
    ],
)
def test_metric_round_trip(store, status, score, error):
    store.save_case(
        "test-run", "case-001", 1, "input", "expected",
        invocation_status="COMPLETED",
    )
    store.save_metric(
        "test-run", "case-001", "Answer Correctness",
        status=status, score=score, threshold=0.8, error=error,
    )

    metric = store.get_metric(
        "test-run", "case-001", "Answer Correctness"
    )
    assert metric["status"] == status
    assert metric["score"] == score
    assert metric["error"] == error


def test_metric_upsert_preserves_one_record(store):
    store.save_case(
        "test-run", "case-001", 1, "input", "expected",
        invocation_status="COMPLETED",
    )
    store.save_metric("test-run", "case-001", "Metric A", "FAIL", score=0.4)
    store.save_metric("test-run", "case-001", "Metric A", "PASS", score=1.0)

    metrics = store.get_metrics("test-run", "case-001")
    assert len(metrics) == 1
    assert metrics[0]["status"] == "PASS"


def test_summary_and_run_completion(store):
    store.save_case(
        "test-run", "case-001", 1, "input", "expected",
        invocation_status="COMPLETED",
    )
    store.save_metric("test-run", "case-001", "Metric A", "PASS", score=1.0)
    store.save_metric(
        "test-run", "case-001", "Metric B",
        "EVALUATION_ERROR", error="quota",
    )

    summary = store.get_run_summary("test-run")
    assert summary["cases"] == 1
    assert summary["metrics"] == 2
    assert summary["metric_statuses"] == {
        "PASS": 1,
        "EVALUATION_ERROR": 1,
    }

    store.complete_run("test-run", "PARTIAL")
    assert store.get_run("test-run")["status"] == "PARTIAL"

def test_sqlite_tool_calls_restore_as_deepeval_objects(store):
    from eval.eval_persistent import restore_tool_calls
    from deepeval.test_case import ToolCall

    store.save_case(
        run_id="test-run",
        case_id="case-toolcall",
        case_number=1,
        input_text="Find order details",
        expected_output="Order found",
        actual_output="Order found",
        tools_called=[
            {
                "name": "lookup_tool",
                "type": "FUNCTION",
                "input_parameters": {"order_id": "123"},
            }
        ],
        invocation_status="COMPLETED",
    )

    persisted_case = store.get_case("test-run", "case-toolcall")
    restored = restore_tool_calls(persisted_case["tools_called"])

    assert len(restored) == 1
    assert isinstance(restored[0], ToolCall)
    assert restored[0].name == "lookup_tool"
    assert restored[0].input_parameters == {"order_id": "123"}