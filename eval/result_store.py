"""Persistent evaluation results and resumable run state."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    """Stable SHA-256 fingerprint of JSON-compatible data."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_dump(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda obj: (
            obj.model_dump() if hasattr(obj, "model_dump")
            else str(obj)
        ),
    )


class ResultStore:
    """SQLite persistence for evaluation runs, cases, and metric results."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_fingerprint TEXT NOT NULL,
                    config_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS cases (
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    case_number INTEGER NOT NULL,
                    input TEXT NOT NULL,
                    expected_output TEXT,
                    actual_output TEXT,
                    tools_called_json TEXT,
                    invocation_status TEXT NOT NULL,
                    invocation_error TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, case_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    threshold REAL,
                    reason TEXT,
                    error TEXT,
                    evaluated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, case_id, metric_name),
                    FOREIGN KEY (run_id, case_id)
                        REFERENCES cases(run_id, case_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_status
                    ON metrics(run_id, status);

                CREATE TABLE IF NOT EXISTS trace_artifacts (
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, case_id),
                    FOREIGN KEY (run_id, case_id)
                        REFERENCES cases(run_id, case_id)
                        ON DELETE CASCADE
                );
            """)

    def start_run(
        self,
        run_id,
        dataset_fingerprint,
        config_fingerprint,
        metadata=None,
    ):
        """Create a run or reopen the same compatible run for resumption."""
        now = utc_now()
        metadata_json = json_dump(metadata or {})

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            if existing is not None:
                if (
                    existing["dataset_fingerprint"] != dataset_fingerprint
                    or existing["config_fingerprint"] != config_fingerprint
                ):
                    raise ValueError(
                        "Run ID exists with different dataset/config fingerprints."
                    )

                connection.execute(
                    """
                    UPDATE runs
                    SET status = 'RUNNING', finished_at = NULL
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )
                return

            connection.execute(
                """
                INSERT INTO runs (
                    run_id, dataset_fingerprint, config_fingerprint,
                    status, metadata_json, started_at
                ) VALUES (?, ?, ?, 'RUNNING', ?, ?)
                """,
                (
                    run_id,
                    dataset_fingerprint,
                    config_fingerprint,
                    metadata_json,
                    now,
                ),
            )

    def save_case(
        self,
        run_id,
        case_id,
        case_number,
        input_text,
        expected_output,
        actual_output=None,
        tools_called=None,
        invocation_status="PENDING",
        invocation_error=None,
    ):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cases (
                    run_id, case_id, case_number, input, expected_output,
                    actual_output, tools_called_json, invocation_status,
                    invocation_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, case_id) DO UPDATE SET
                    case_number = excluded.case_number,
                    input = excluded.input,
                    expected_output = excluded.expected_output,
                    actual_output = excluded.actual_output,
                    tools_called_json = excluded.tools_called_json,
                    invocation_status = excluded.invocation_status,
                    invocation_error = excluded.invocation_error,
                    updated_at = excluded.updated_at
                """,
                (
                    run_id,
                    case_id,
                    case_number,
                    input_text,
                    expected_output,
                    actual_output,
                    json_dump(tools_called) if tools_called is not None else None,
                    invocation_status,
                    invocation_error,
                    utc_now(),
                ),
            )

    def get_case(self, run_id, case_id):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM cases
                WHERE run_id = ? AND case_id = ?
                """,
                (run_id, case_id),
            ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["tools_called"] = (
            json.loads(result["tools_called_json"])
            if result["tools_called_json"] else None
        )
        return result

    def save_metric(
        self,
        run_id,
        case_id,
        metric_name,
        status,
        score=None,
        threshold=None,
        reason=None,
        error=None,
    ):
        allowed = {
            "PASS", "FAIL", "EVALUATION_ERROR", "SKIPPED",
            "NOT_EVALUATED",
        }
        if status not in allowed:
            raise ValueError(f"Unsupported metric status: {status}")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO metrics (
                    run_id, case_id, metric_name, status, score,
                    threshold, reason, error, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, case_id, metric_name) DO UPDATE SET
                    status = excluded.status,
                    score = excluded.score,
                    threshold = excluded.threshold,
                    reason = excluded.reason,
                    error = excluded.error,
                    evaluated_at = excluded.evaluated_at
                """,
                (
                    run_id,
                    case_id,
                    metric_name,
                    status,
                    score,
                    threshold,
                    reason,
                    error,
                    utc_now(),
                ),
            )

    def get_metric(self, run_id, case_id, metric_name):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM metrics
                WHERE run_id = ? AND case_id = ? AND metric_name = ?
                """,
                (run_id, case_id, metric_name),
            ).fetchone()
        return dict(row) if row else None

    def get_metrics(self, run_id, case_id):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM metrics
                WHERE run_id = ? AND case_id = ?
                ORDER BY metric_name
                """,
                (run_id, case_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_trace_artifact(
        self, run_id, case_id, trace_dict, schema_version=1
    ):
        """Persist a versioned DeepEval trace artifact for one completed case."""
        if not isinstance(trace_dict, dict) or not trace_dict:
            raise ValueError("trace_dict must be a non-empty dictionary.")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise ValueError("schema_version must be a positive integer.")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trace_artifacts (
                    run_id, case_id, schema_version, trace_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, case_id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    trace_json = excluded.trace_json,
                    created_at = excluded.created_at
                """,
                (
                    run_id,
                    case_id,
                    schema_version,
                    json_dump(trace_dict),
                    utc_now(),
                ),
            )

    def get_trace_artifact(self, run_id, case_id):
        """Return a decoded trace artifact, or None when none was saved."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT schema_version, trace_json, created_at
                FROM trace_artifacts
                WHERE run_id = ? AND case_id = ?
                """,
                (run_id, case_id),
            ).fetchone()

        if row is None:
            return None

        return {
            "schema_version": row["schema_version"],
            "trace": json.loads(row["trace_json"]),
            "created_at": row["created_at"],
        }

    def complete_run(self, run_id, status="COMPLETED"):
        allowed = {"COMPLETED", "COMPLETED_WITH_ERRORS", "PARTIAL", "FAILED"}
        if status not in allowed:
            raise ValueError(f"Unsupported run status: {status}")

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (status, utc_now(), run_id),
            )

    def get_run(self, run_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_run_summary(self, run_id):
        with self._connect() as connection:
            return {
                "cases": connection.execute(
                    "SELECT COUNT(*) FROM cases WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0],
                "metrics": connection.execute(
                    "SELECT COUNT(*) FROM metrics WHERE run_id = ?",
                    (run_id,),
                ).fetchone()[0],
                "metric_statuses": {
                    row["status"]: row["count"]
                    for row in connection.execute(
                        """
                        SELECT status, COUNT(*) AS count
                        FROM metrics WHERE run_id = ?
                        GROUP BY status
                        """,
                        (run_id,),
                    ).fetchall()
                },
            }
