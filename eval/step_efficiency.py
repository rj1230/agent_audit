"""
Retired standalone evaluation entry point.

The current AgentAudit evaluation architecture is maintained in:
    python -m eval.eval_all

For deterministic application-side checks, use:
    python -m eval.deterministic_benchmark

This file is retained for historical reference. Its previous implementation
is preserved in a timestamped .bak file.
"""

if __name__ == "__main__":
    raise SystemExit(
        "This standalone runner is retired. "
        "Use `python -m eval.eval_all` for the current seven-metric evaluation. "
        "Use `python -m eval.deterministic_benchmark` for deterministic checks."
    )
