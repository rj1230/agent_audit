"""Resumable, SQLite-backed AgentAudit evaluation runner."""

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase, ToolCall

try:
    from .dataset import dataset
    from .eval_all import build_metrics
    from .evaluation_helpers import invoke_agent_for_golden
    from .result_store import ResultStore, fingerprint
except ImportError:
    from dataset import dataset
    from eval_all import build_metrics
    from evaluation_helpers import invoke_agent_for_golden
    from result_store import ResultStore, fingerprint

try:
    from .quota_guard import (
        JudgeQuotaBlocked,
        ensure_judge_available,
        is_daily_token_quota_error,
        judge_block_reason,
        reset_judge_quota_breaker,
    )
except ImportError:
    from quota_guard import (
        JudgeQuotaBlocked,
        ensure_judge_available,
        is_daily_token_quota_error,
        judge_block_reason,
        reset_judge_quota_breaker,
    )


QUOTA_BLOCKED_MARKER = "[JUDGE_DAILY_QUOTA_BLOCKED]"


def metric_lookup_names(metric):
    """Return DeepEval result name and compatibility aliases."""
    names = [
        getattr(metric, "__name__", None),
        getattr(metric, "name", None),
        metric.__class__.__name__,
    ]
    return list(dict.fromkeys(name for name in names if name))


def metric_storage_name(metric):
    """Use the name DeepEval uses when creating MetricData."""
    names = metric_lookup_names(metric)
    return names[0] if names else metric.__class__.__name__


def find_metric_data(metric_data_list, metric):
    """Find a metric result by its DeepEval name or a known alias."""
    by_name = {
        item.name: item
        for item in metric_data_list
        if getattr(item, "name", None)
    }
    for name in metric_lookup_names(metric):
        if name in by_name:
            return by_name[name]
    return None


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def golden_payload(golden):
    """Stable dataset representation for compatibility checks."""
    return {
        "input": golden.input,
        "expected_output": golden.expected_output,
        "expected_tools": [
            item.model_dump() if hasattr(item, "model_dump") else str(item)
            for item in (golden.expected_tools or [])
        ],
        "context": golden.context,
        "metadata": golden.additional_metadata,
    }


def metric_config(metrics):
    return [
        {
            "class": metric.__class__.__name__,
            "result_name": metric_storage_name(metric),
            "aliases": metric_lookup_names(metric),
            "threshold": getattr(metric, "threshold", None),
        }
        for metric in metrics
    ]


def metric_status(metric_data):
    """Classify quota blocks separately from ordinary evaluation errors."""
    error = getattr(metric_data, "error", None)
    score = getattr(metric_data, "score", None)
    success = getattr(metric_data, "success", None)

    if error:
        message = str(error)
        if isinstance(error, JudgeQuotaBlocked) or "blocked locally" in message.lower():
            return "SKIPPED", score, f"{QUOTA_BLOCKED_MARKER} {message}"
        if is_daily_token_quota_error(message):
            return "EVALUATION_ERROR", score, f"{QUOTA_BLOCKED_MARKER} {message}"
        return "EVALUATION_ERROR", score, message

    if score is None or success is None:
        return "NOT_EVALUATED", score, None
    return ("PASS" if success else "FAIL"), score, None


def is_quota_blocked_metric(row):
    """Whether a persisted metric was blocked by the judge's daily quota."""
    if not row:
        return False
    return QUOTA_BLOCKED_MARKER in str(row.get("error") or "")


def save_quota_skipped_metric(store, run_id, case_id, metric, reason=None):
    """Persist a local quota skip without making another judge request."""
    detail = reason or judge_block_reason() or "Judge daily quota breaker is active."
    message = f"{QUOTA_BLOCKED_MARKER} {detail}"
    store.save_metric(
        run_id=run_id,
        case_id=case_id,
        metric_name=metric_storage_name(metric),
        status="SKIPPED",
        threshold=getattr(metric, "threshold", None),
        error=message,
    )
    return "SKIPPED"


def save_metric_data(store, run_id, case_id, metric_data, metric):
    status, score, error = metric_status(metric_data)
    store.save_metric(
        run_id=run_id,
        case_id=case_id,
        metric_name=getattr(metric_data, "name", None) or metric_storage_name(metric),
        status=status,
        score=score,
        threshold=getattr(metric, "threshold", None),
        reason=getattr(metric_data, "reason", None),
        error=error,
    )
    return status


def evaluate_trace_case(golden, agent, metrics, store, run_id, case_id, case_number):
    """Invoke the agent once, persist its trace, and grade trace metrics."""
    iterator = EvaluationDataset(goldens=[golden]).evals_iterator(
        metrics=metrics,
        cache_config=CacheConfig(use_cache=False, write_cache=False),
        display_config=DisplayConfig(inspect_after_run=False),
        async_config=AsyncConfig(
            run_async=False, throttle_value=1, max_concurrent=1
        ),
        error_config=ErrorConfig(
            ignore_errors=True, skip_on_missing_params=False
        ),
    )

    answer = None
    tools_called = None
    invocation_completed = False
    trace_error = None

    try:
        yielded = next(iterator)
        response, tools_called, trace_dict, trace_error = (
            invoke_agent_for_golden(agent, yielded, capture_trace=True)
        )

        if trace_dict is None:
            print(f"  TRACE CAPTURE FAILED: {trace_error or 'Unknown capture failure'}")
        else:
            print(
                f"  TRACE CAPTURE SUCCEEDED: "
                f"{len(trace_dict.get('children', []))} root children"
            )

        answer = response["structured_response"].answer

        store.save_case(
            run_id=run_id,
            case_id=case_id,
            case_number=case_number,
            input_text=golden.input,
            expected_output=golden.expected_output,
            actual_output=answer,
            tools_called=tools_called,
            invocation_status="COMPLETED",
            invocation_error=trace_error,
        )
        invocation_completed = True

        if trace_dict is not None:
            store.save_trace_artifact(run_id, case_id, trace_dict)
            print("  TRACE ARTIFACT SAVED.")
        else:
            print("  WARNING: Agent response saved, but no trace artifact was saved.")

        try:
            next(iterator)
        except StopIteration as stop:
            result = stop.value
        else:
            raise RuntimeError("Trace iterator yielded unexpected extra cases.")

        if result is None:
            raise RuntimeError("DeepEval trace iterator returned no result.")

        test_results = getattr(result, "test_results", []) or []
        if not test_results:
            raise RuntimeError("DeepEval returned no trace test results.")

        metric_data_list = test_results[0].metrics_data or []

        for metric in metrics:
            data = find_metric_data(metric_data_list, metric)
            if data is not None:
                save_metric_data(store, run_id, case_id, data, metric)
            elif judge_block_reason() is not None:
                save_quota_skipped_metric(store, run_id, case_id, metric)
            else:
                store.save_metric(
                    run_id, case_id, metric_storage_name(metric),
                    status="EVALUATION_ERROR",
                    threshold=getattr(metric, "threshold", None),
                    error=trace_error or "DeepEval did not return this trace metric.",
                )

        return answer, tools_called

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if trace_error:
            message = f"Trace capture failed: {trace_error}; grading failed: {message}"

        if not invocation_completed:
            store.save_case(
                run_id=run_id,
                case_id=case_id,
                case_number=case_number,
                input_text=golden.input,
                expected_output=golden.expected_output,
                invocation_status="ERROR",
                invocation_error=message,
            )

        for metric in metrics:
            name = metric_storage_name(metric)
            prior = store.get_metric(run_id, case_id, name)
            if prior and prior.get("status") in {"PASS", "FAIL"}:
                continue

            if judge_block_reason() is not None or isinstance(exc, JudgeQuotaBlocked):
                save_quota_skipped_metric(
                    store, run_id, case_id, metric,
                    reason=judge_block_reason() or message,
                )
            else:
                store.save_metric(
                    run_id, case_id, name,
                    status="EVALUATION_ERROR",
                    threshold=getattr(metric, "threshold", None),
                    error=message,
                )

        print(f"  TRACE ERROR: {message}")
        traceback.print_exc()

        if invocation_completed:
            return answer, tools_called
        return None, None

    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


def restore_tool_calls(tool_calls):
    """Rebuild DeepEval ToolCall objects from SQLite JSON data."""
    restored = []

    for item in tool_calls or []:
        if isinstance(item, ToolCall):
            restored.append(item)
            continue

        if not isinstance(item, dict):
            raise TypeError(
                f"Unsupported persisted tool-call value: {type(item).__name__}"
            )

        allowed_fields = {
            "name",
            "type",
            "description",
            "reasoning",
            "output",
            "input_parameters",
        }
        payload = {
            key: value
            for key, value in item.items()
            if key in allowed_fields
        }

        if not payload.get("name"):
            raise ValueError("Persisted tool call is missing its name.")

        restored.append(ToolCall(**payload))

    return restored

def evaluate_saved_trace_metric(
    golden, answer, tools_called, trace_dict, metric,
    store, run_id, case_id,
):
    """Grade one trace metric from SQLite without invoking the agent."""
    name = metric_storage_name(metric)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        tools_called=restore_tool_calls(tools_called),
        expected_tools=golden.expected_tools or [],
    )
    test_case._trace_dict = trace_dict

    try:
        result = evaluate(
            test_cases=[test_case],
            metrics=[metric],
            async_config=AsyncConfig(
                run_async=False, throttle_value=1, max_concurrent=1
            ),
            cache_config=CacheConfig(use_cache=False, write_cache=False),
            display_config=DisplayConfig(inspect_after_run=False),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=False
            ),
        )

        test_results = getattr(result, "test_results", []) or []
        data = (
            find_metric_data(test_results[0].metrics_data or [], metric)
            if test_results else None
        )
        if data is None:
            store.save_metric(
                run_id, case_id, name,
                status="EVALUATION_ERROR",
                threshold=getattr(metric, "threshold", None),
                error="DeepEval returned no result for the restored trace metric.",
            )
            return "EVALUATION_ERROR"

        return save_metric_data(store, run_id, case_id, data, metric)

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, JudgeQuotaBlocked):
            return save_quota_skipped_metric(
                store, run_id, case_id, metric, reason=message
            )
        if is_daily_token_quota_error(message):
            message = f"{QUOTA_BLOCKED_MARKER} {message}"

        store.save_metric(
            run_id, case_id, name,
            status="EVALUATION_ERROR",
            threshold=getattr(metric, "threshold", None),
            error=message,
        )
        print(f"    {name}: EVALUATION_ERROR — {message}")
        return "EVALUATION_ERROR"

def evaluate_single_metric(golden, answer, tools_called, metric, store, run_id, case_id):
    """Evaluate one single-turn metric and persist it independently."""
    name = metric_storage_name(metric)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        tools_called=restore_tool_calls(tools_called),
        expected_tools=golden.expected_tools or [],
    )

    try:
        result = evaluate(
            test_cases=[test_case],
            metrics=[metric],
            async_config=AsyncConfig(
                run_async=False, throttle_value=1, max_concurrent=1
            ),
            cache_config=CacheConfig(use_cache=False, write_cache=False),
            display_config=DisplayConfig(inspect_after_run=False),
            error_config=ErrorConfig(
                ignore_errors=True, skip_on_missing_params=False
            ),
        )

        test_results = getattr(result, "test_results", []) or []
        data = None
        if test_results:
            data = find_metric_data(
                test_results[0].metrics_data or [], metric
            )

        if data is None:
            store.save_metric(
                run_id, case_id, name,
                status="EVALUATION_ERROR",
                threshold=getattr(metric, "threshold", None),
                error="DeepEval returned no metric result.",
            )
            return "EVALUATION_ERROR"

        status = save_metric_data(store, run_id, case_id, data, metric)

        if "Argument Correctness" in metric_lookup_names(metric):
            print(
                "    Argument diagnostics:",
                getattr(data, "verbose_logs", None)
                or getattr(data, "reason", None)
                or "(none)",
            )
        return status

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, JudgeQuotaBlocked):
            status = save_quota_skipped_metric(
                store, run_id, case_id, metric, reason=message
            )
        else:
            if is_daily_token_quota_error(message):
                message = f"{QUOTA_BLOCKED_MARKER} {message}"
            status = "EVALUATION_ERROR"
            store.save_metric(
                run_id, case_id, name,
                status=status,
                threshold=getattr(metric, "threshold", None),
                error=message,
            )
        print(f"    {name}: {status} — {message}")
        return status


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable AgentAudit evaluations."
    )
    parser.add_argument("--case-number", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-quota-blocked",
        action="store_true",
        help="Explicitly retry metrics previously blocked by the judge daily quota.",
    )
    parser.add_argument(
        "--recapture-missing-traces",
        action="store_true",
        help=(
            "Explicitly reinvoke completed cases whose trace artifacts are missing. "
            "Requires --resume and --run-id."
        ),
    )
    parser.add_argument(
        "--db",
        default="eval/results/agent_audit.sqlite3",
        help="SQLite result database path.",
    )
    args = parser.parse_args()

    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id.")
    if args.retry_quota_blocked and not args.resume:
        parser.error("--retry-quota-blocked requires --resume.")
    if args.recapture_missing_traces and not args.resume:
        parser.error("--recapture-missing-traces requires --resume.")
    if args.recapture_missing_traces and not args.run_id:
        parser.error("--recapture-missing-traces requires an explicit --run-id.")
    if args.retry_quota_blocked:
        reset_judge_quota_breaker()
    if args.case_number is not None and not 1 <= args.case_number <= len(dataset.goldens):
        parser.error(f"--case-number must be between 1 and {len(dataset.goldens)}.")

    indexed_goldens = list(enumerate(dataset.goldens, start=1))
    if args.case_number:
        indexed_goldens = [
            (number, golden)
            for number, golden in indexed_goldens
            if number == args.case_number
        ]

    trace_metrics, single_metrics = build_metrics()
    all_metrics = trace_metrics + single_metrics

    run_id = args.run_id or f"run-{utc_stamp()}"
    dataset_fp = fingerprint([golden_payload(g) for _, g in indexed_goldens])
    config_fp = fingerprint(metric_config(all_metrics))

    store = ResultStore(Path(args.db))
    if args.resume and store.get_run(run_id) is None:
        parser.error(f"Cannot resume unknown run ID: {run_id}")

    store.start_run(
        run_id=run_id,
        dataset_fingerprint=dataset_fp,
        config_fingerprint=config_fp,
        metadata={
            "case_numbers": [n for n, _ in indexed_goldens],
            "metrics": [metric_storage_name(m) for m in all_metrics],
        },
    )

    load_dotenv()
    from agent.agent import agent

    print(f"Run ID: {run_id}")
    print(f"Database: {Path(args.db).resolve()}")
    print(f"Cases selected: {len(indexed_goldens)}")
    print(f"Resume mode: {args.resume}")

    try:
        for case_number, golden in indexed_goldens:
            case_id = f"case-{case_number:03d}"
            print(f"\n[{case_id}] {golden.input}")

            existing_case = store.get_case(run_id, case_id)
            existing_metrics = store.get_metrics(run_id, case_id)
            metrics_by_name = {
                row["metric_name"]: row for row in existing_metrics
            }

            trace_pending = []
            for metric in trace_metrics:
                prior = metrics_by_name.get(metric_storage_name(metric))
                if prior and prior.get("status") in {"PASS", "FAIL"}:
                    continue
                if (
                    prior
                    and is_quota_blocked_metric(prior)
                    and not args.retry_quota_blocked
                ):
                    print(
                        f"  {metric_storage_name(metric)}: quota-blocked "
                        "(skip; use --retry-quota-blocked to retry)"
                    )
                    continue
                trace_pending.append(metric)

            answer = existing_case.get("actual_output") if existing_case else None
            tools_called = existing_case.get("tools_called") if existing_case else None
            invocation_completed = (
                existing_case is not None
                and existing_case.get("invocation_status") == "COMPLETED"
                and answer is not None
            )

            if trace_pending and invocation_completed:
                artifact = store.get_trace_artifact(run_id, case_id)
                if artifact and artifact.get("schema_version") == 1:
                    print("  Reusing saved trace; agent will not be invoked.")
                    for metric in trace_pending:
                        try:
                            ensure_judge_available()
                        except JudgeQuotaBlocked as exc:
                            save_quota_skipped_metric(
                                store, run_id, case_id, metric, reason=str(exc)
                            )
                            continue

                        print(f"  Grading saved trace: {metric_storage_name(metric)}...")
                        evaluate_saved_trace_metric(
                            golden, answer, tools_called, artifact["trace"],
                            metric, store, run_id, case_id,
                        )
                else:
                    if args.recapture_missing_traces:
                        print(
                            "  EXPLICIT TRACE RECOVERY: invoking agent again "
                            "because the saved trace artifact is missing."
                        )
                        answer, tools_called = evaluate_trace_case(
                            golden, agent, trace_pending, store,
                            run_id, case_id, case_number,
                        )
                    else:
                        message = (
                            "Completed agent invocation has no compatible saved trace artifact; "
                            "trace grading cannot resume without a new agent invocation. "
                            "Use --recapture-missing-traces to explicitly reinvoke this case."
                        )
                        for metric in trace_pending:
                            store.save_metric(
                                run_id, case_id, metric_storage_name(metric),
                                status="EVALUATION_ERROR",
                                threshold=getattr(metric, "threshold", None),
                                error=message,
                            )
                        print(f"  TRACE RECOVERY ERROR: {message}")

            elif trace_pending:
                print("  Running agent once and capturing its trace...")
                answer, tools_called = evaluate_trace_case(
                    golden, agent, trace_pending, store,
                    run_id, case_id, case_number,
                )

            elif answer is not None:
                print("  Trace metrics already complete; reusing saved agent output.")

            else:
                print("  WARNING: no saved agent output; invoking agent for single-turn metrics.")
                try:
                    response, tools_called = invoke_agent_for_golden(agent, golden)
                    answer = response["structured_response"].answer
                    store.save_case(
                        run_id, case_id, case_number, golden.input,
                        golden.expected_output, answer, tools_called,
                        invocation_status="COMPLETED",
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    store.save_case(
                        run_id, case_id, case_number, golden.input,
                        golden.expected_output,
                        invocation_status="ERROR", invocation_error=message,
                    )
                    for metric in single_metrics:
                        name = metric_storage_name(metric)
                        store.save_metric(
                            run_id, case_id, name,
                            status="EVALUATION_ERROR",
                            threshold=getattr(metric, "threshold", None),
                            error=message,
                        )
                    print(f"  INVOCATION ERROR: {message}")
                    continue

            if answer is None:
                print("  No answer available; single-turn metrics remain pending.")
                continue

            for metric in single_metrics:
                name = metric_storage_name(metric)
                prior = store.get_metric(run_id, case_id, name)

                if prior and prior["status"] in {"PASS", "FAIL"}:
                    print(f"  {name}: already {prior['status']} (skip)")
                    continue

                if (
                    prior
                    and is_quota_blocked_metric(prior)
                    and not args.retry_quota_blocked
                ):
                    print(
                        f"  {name}: quota-blocked "
                        "(skip; use --retry-quota-blocked to retry)"
                    )
                    continue

                try:
                    ensure_judge_available()
                except JudgeQuotaBlocked as exc:
                    status = save_quota_skipped_metric(
                        store, run_id, case_id, metric, reason=str(exc)
                    )
                    print(f"  {name}: {status} (local quota circuit breaker)")
                    continue

                print(f"  Evaluating {name}...")
                status = evaluate_single_metric(
                    golden, answer, tools_called, metric, store, run_id, case_id
                )
                print(f"    -> {status}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saved results are retained; resume with --run-id.")
        store.complete_run(run_id, status="PARTIAL")
        raise
    except Exception:
        store.complete_run(run_id, status="FAILED")
        raise

    summary = store.get_run_summary(run_id)
    total_expected = len(indexed_goldens) * len(all_metrics)
    persisted = summary["metrics"]
    terminal = (
        summary["metric_statuses"].get("PASS", 0)
        + summary["metric_statuses"].get("FAIL", 0)
    )

    if persisted < total_expected:
        final_status = "PARTIAL"
    elif terminal < total_expected:
        final_status = "COMPLETED_WITH_ERRORS"
    else:
        final_status = "COMPLETED"
    store.complete_run(run_id, status=final_status)

    print("\n=== PERSISTED EVALUATION SUMMARY ===")
    print(f"Run: {run_id}")
    print(f"Cases stored: {summary['cases']}")
    print(f"Metrics stored: {persisted}/{total_expected}")
    for status, count in sorted(summary["metric_statuses"].items()):
        print(f"  {status}: {count}")
    print(f"Run status: {final_status}")
    print(f"Database: {Path(args.db).resolve()}")


if __name__ == "__main__":
    main()
