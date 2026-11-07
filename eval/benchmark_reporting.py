import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def summarize_token_usage(usage_records):
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    records_seen = 0

    for record in usage_records or []:
        if not isinstance(record, dict):
            continue

        fields = [
            key for key in totals
            if isinstance(record.get(key), (int, float))
        ]
        if not fields:
            continue

        records_seen += 1
        for key in fields:
            totals[key] += record[key]

    return {
        "available": records_seen > 0,
        "records_seen": records_seen,
        **totals,
    }





def _git_value(repo_root, *args):
    """Return a Git value, or None when Git metadata is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
        value = completed.stdout.strip()
        return value or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_benchmark_manifest(
    model_id,
    evaluation_date,
    scenario_names,
    repo_root,
    benchmark_version="1.0.0",
    dataset_version="refund-fixtures-v1",
):
    """Build reproducibility metadata without making model/API calls."""
    repo_root = Path(repo_root)

    return {
        "benchmark_version": benchmark_version,
        "dataset_version": dataset_version,
        "model_id": model_id,
        "evaluation_date": evaluation_date,
        "scenario_names": list(scenario_names),
        "python_version": platform.python_version(),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_value(repo_root, "branch", "--show-current"),
    }


def safe_exception_summary(exc):
    """Keep exception type while excluding provider-specific details."""
    return (
        f"{type(exc).__name__}: exception details omitted from report; "
        "inspect the local console output."
    )


def write_benchmark_report(results, output_dir=None, manifest=None):
    output_dir = Path(output_dir) if output_dir else (
        Path(__file__).resolve().parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    latencies = [
        item["latency_seconds"]
        for item in results
        if isinstance(item.get("latency_seconds"), (int, float))
    ]

    token_totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    usage_scenarios = 0

    for item in results:
        usage = item.get("token_usage")
        if not isinstance(usage, dict) or not usage.get("available"):
            continue
        usage_scenarios += 1
        for key in token_totals:
            token_totals[key] += usage.get(key, 0)

    summary = {
        "scenario_count": len(results),
        "passed": sum(item.get("status") == "PASS" for item in results),
        "failed": sum(item.get("status") == "FAIL" for item in results),
        "blocked_rate_limit": sum(
            item.get("status") == "BLOCKED_RATE_LIMIT" for item in results
        ),
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 3) if latencies else None
        ),
        "total_latency_seconds": round(sum(latencies), 3),
        "token_usage_scenarios": usage_scenarios,
        "token_usage": token_totals if usage_scenarios else None,
    }

    timestamp = datetime.now(timezone.utc)
    report = {
        "benchmark": "AgentAudit LLM integration benchmark",
        "generated_at_utc": timestamp.isoformat(),
        "manifest": manifest or {},
        "summary": summary,
        "scenarios": results,
    }

    filename = timestamp.strftime("agent_integration_%Y%m%d_%H%M%S_%f.json")
    report_path = output_dir / filename
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path, summary
