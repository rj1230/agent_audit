import json

from eval.benchmark_reporting import (
    summarize_token_usage,
    write_benchmark_report,
)


def test_token_usage_is_aggregated():
    result = summarize_token_usage([
        {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
        {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
    ])

    assert result == {
        "available": True,
        "records_seen": 2,
        "input_tokens": 150,
        "output_tokens": 35,
        "total_tokens": 185,
    }


def test_missing_token_usage_is_unavailable():
    result = summarize_token_usage(None)

    assert result == {
        "available": False,
        "records_seen": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_report_separates_pass_fail_and_rate_limit(tmp_path):
    results = [
        {
            "name": "scenario_pass",
            "status": "PASS",
            "passed": True,
            "latency_seconds": 2.0,
            "token_usage": {
                "available": True,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
        },
        {
            "name": "scenario_fail",
            "status": "FAIL",
            "passed": False,
            "latency_seconds": 4.0,
            "token_usage": None,
        },
        {
            "name": "scenario_blocked",
            "status": "BLOCKED_RATE_LIMIT",
            "passed": False,
            "latency_seconds": 6.0,
            "token_usage": None,
        },
    ]

    report_path, summary = write_benchmark_report(
        results,
        output_dir=tmp_path,
    )

    assert report_path.exists()

    assert summary["scenario_count"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["blocked_rate_limit"] == 1
    assert summary["average_latency_seconds"] == 4.0
    assert summary["total_latency_seconds"] == 12.0
    assert summary["token_usage_scenarios"] == 1
    assert summary["token_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["summary"] == summary
    assert len(saved["scenarios"]) == 3


def test_empty_report_has_safe_empty_summary(tmp_path):
    report_path, summary = write_benchmark_report(
        [],
        output_dir=tmp_path,
    )

    assert report_path.exists()
    assert summary["scenario_count"] == 0
    assert summary["passed"] == 0
    assert summary["failed"] == 0
    assert summary["blocked_rate_limit"] == 0
    assert summary["average_latency_seconds"] is None
    assert summary["total_latency_seconds"] == 0
    assert summary["token_usage"] is None



def test_exception_summary_omits_raw_provider_details():
    from eval.benchmark_reporting import safe_exception_summary

    class FakeProviderError(Exception):
        pass

    exc = FakeProviderError(
        "organization org_private; API details and billing URL"
    )
    summary = safe_exception_summary(exc)

    assert summary.startswith("FakeProviderError:")
    assert "org_private" not in summary
    assert "billing URL" not in summary
    assert "details omitted" in summary



def test_benchmark_manifest_contains_reproducibility_metadata(tmp_path, monkeypatch):
    from eval import benchmark_reporting
    import platform

    monkeypatch.setattr(
        benchmark_reporting,
        "_git_value",
        lambda repo_root, *args: None,
    )

    manifest = benchmark_reporting.build_benchmark_manifest(
        model_id="qwen/qwen3.8-27b",
        evaluation_date="2026-10-31",
        scenario_names=["eligible_single_item", "invalid_username"],
        repo_root=tmp_path,
    )

    assert manifest["benchmark_version"] == "1.0.0"
    assert manifest["dataset_version"] == "refund-fixtures-v1"
    assert manifest["model_id"] == "qwen/qwen3.8-27b"
    assert manifest["evaluation_date"] == "2026-10-31"
    assert manifest["scenario_names"] == [
        "eligible_single_item",
        "invalid_username",
    ]
    assert manifest["python_version"] == platform.python_version()
    assert manifest["git_commit"] is None
    assert manifest["git_branch"] is None


def test_json_report_persists_manifest(tmp_path):
    from eval.benchmark_reporting import write_benchmark_report
    import json

    manifest = {
        "benchmark_version": "1.0.0",
        "model_id": "qwen/qwen3.8-27b",
        "git_commit": "abc123",
    }
    report_path, _ = write_benchmark_report(
        [{"name": "sample", "status": "PASS", "latency_seconds": 1.0}],
        output_dir=tmp_path,
        manifest=manifest,
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["manifest"] == manifest
