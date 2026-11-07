# AgentAudit — Evaluation Report

**Date:** 2026-11-01
**Project:** Simulated Refund Support Agent
**LLM:** qwen/qwen3.8-27b via Groq

## 1. Deterministic Regression

| Metric | Result |
|---|---:|
| Tests passed | 58/58 |
| Tests failed | 0 |
| Runtime | 1.53 seconds |
| Status | PASS |

Command: `python -m pytest tests eval\deterministic_benchmark.py -q`

The benchmark uses a frozen date, synthetic fixtures, temporary SQLite databases, and no LLM API calls.

## 2. LLM Integration Benchmark

| Scenario | Result |
|---|---|
| Eligible single item | PASS |
| Invalid username | PASS |
| Eligible multiple items | PASS |
| Information only | BLOCKED — quota |
| Expired purchase | BLOCKED — quota |
| Mixed eligibility | BLOCKED — quota |
| Unpurchased item | BLOCKED — quota |

Three of seven scenarios passed in targeted runs. Four remain unverified because Groq returned HTTP 429 (daily token quota). These are blocked results, not confirmed agent failures.

The three passing scenarios verified expected tool execution, ticket behavior, refund amounts, and simulated-refund disclosure where applicable.

## 3. Benchmark Exit Status

| Exit code | Meaning |
|---:|---|
| 0 | All requested scenarios passed |
| 1 | Genuine evaluation failure |
| 2 | One or more scenarios blocked by rate limits, with no genuine failures |

## 4. Remaining Work

1. Rerun the four blocked scenarios when Groq quota is available.
2. Investigate any genuine failures.
3. Complete the seven-scenario integration benchmark.
4. Add repeat-run reliability, latency, token usage, and cost measurements.

## 5. Current Assessment

**Deterministic:** 58/58 passed.

**LLM integration:** 3/7 passed; 4/7 blocked by provider quota.

**Overall:** Partial evaluation baseline. Complete LLM validation is still required.

All refunds and tickets in this project are simulated. No actual payments or bank transfers are initiated.

## Latest verification update — 2026-11-01

### Deterministic and regression verification

| Check | Result |
|---|---:|
| Full regression suite (`tests` + deterministic benchmark) | **65 passed** |
| Regression runtime | 1.17 seconds |
| Benchmark reporting tests | **7 passed** |
| Deterministic refund benchmark cases | 14 |
| Python compilation | Passed |
| `git diff --check` | No whitespace errors |

The deterministic refund benchmark uses a frozen evaluation date,
temporary SQLite databases, and fixture purchase data. It does not call
the Groq API or modify the production ticket/checkpoint databases.

### LLM integration benchmark status

Three of seven scenarios have previously completed and passed:

- `eligible_single_item`
- `eligible_multiple_items`
- `invalid_username`

The following four scenarios were blocked by the Groq daily token limit
during the latest multi-scenario run:

- `information_only`
- `expired_purchase`
- `mixed_eligibility_all_or_nothing`
- `un_purchased_item`

These four are **unverified due to provider quota**, not benchmark
assertion failures. A fresh complete seven-scenario LLM run remains
outstanding. Do not interpret the three passing scenarios as a
seven-scenario pass.

### Reproducibility manifest

JSON benchmark reports now support a manifest containing:

| Field | Value in the verified local manifest |
|---|---|
| `benchmark_version` | `1.0.0` |
| `dataset_version` | `refund-fixtures-v1` |
| `model_id` | `qwen/qwen3.8-27b` |
| `evaluation_date` | `2026-10-31` |
| `scenario_names` | Names selected for that run |
| `python_version` | `3.13.15` |
| `git_commit` | `d3ab855bf6dc4309c9cc7bc120cba3f6d1ceaae3` |
| `git_branch` | `feature/idempotent-refund-tickets` |

The scenario list is generated from the selected cases, so a run using
`--case` records only those scenarios rather than incorrectly claiming
to be a full benchmark run. Manifest construction and JSON persistence
are covered by automated tests.

### Remaining verification

1. Wait for Groq quota availability before rerunning the four blocked
   scenarios (or the complete seven-scenario benchmark).
2. Generate a fresh successful LLM JSON report and verify its manifest,
   scenario counts, latency, and token-usage fields.
3. Keep deterministic test results and LLM integration results reported
   separately.

**Current conclusion:** deterministic and regression verification is
green. LLM integration evidence is partial; full seven-scenario
verification is still pending.

## Groq quota probe update — 2026-11-01

A single-scenario quota probe was attempted for `information_only`.

| Item | Result |
|---|---|
| Scenario | `information_only` |
| Benchmark outcome | `BLOCKED_RATE_LIMIT` |
| Passed / failed / blocked | 0 / 0 / 1 |
| Provider response | HTTP 429 — quota limit |
| Scenario completion | Blocked before completion |
| Token usage recorded | Unavailable |

The provider returned HTTP 429 because the available quota was insufficient for the request. The scenario did not complete, so this is not an assertion failure and must not be counted as a scenario failure. Exact account quota and token-consumption figures are intentionally omitted from this public report.

Generated report:
`eval/results/agent_integration_20261101_064606_609824.json`

### Updated LLM verification status

Three scenarios remain verified as passing:
- `eligible_single_item`
- `eligible_multiple_items`
- `invalid_username`

Four scenarios remain unverified:
- `information_only` — quota probe blocked
- `expired_purchase` — previously quota blocked
- `mixed_eligibility_all_or_nothing` — previously quota blocked
- `un_purchased_item` — previously quota blocked

Do not retry the LLM benchmark until sufficient Groq daily quota is available. The deterministic and regression test results are unchanged by this probe.

## DeepEval contract correction — 2026-11-01

### Offline validation

- Corrected successful-ticket golden arguments to match the current `issue_refund_ticket` schema: `user` and `items`.
- Corrected the mixed-eligibility golden to enforce the implemented all-or-nothing refund policy. No ticket is expected when any requested item is ineligible.
- Corrected the expired-purchase expected sequence to include `calculate_right_refund`.
- Added `eval/evaluation_helpers.py` to extract executed tool calls only when a matching `ToolMessage` confirms execution.
- Updated `eval/eval_all.py` and `eval/tool_correctness.py` to supply trusted `refund_request_id` context and use executed-tool extraction.
- Corrected standalone tool-correctness case collection to avoid resetting the local list inside the scenario loop and avoid accumulating cases on the shared dataset.
- Added `tests/test_evaluation_helpers.py` with six automated tests covering executed-call extraction and golden dataset contracts.

### Verified results

| Check | Result |
|---|---:|
| Offline golden and extractor validation | PASS |
| New evaluation-helper tests | 6 passed |
| Full local regression suite | 71 passed |
| Full regression duration | 1.51 seconds |
| Python compilation | PASS |
| Groq / LLM calls during these checks | None |

The LLM-based DeepEval run remains unverified. It was not executed because the Groq daily token quota was constrained. The offline checks do not establish live model metric scores.

### Working-tree safety

No commit or push was performed. Existing unrelated modifications and backup files remain untouched.

## Latest local verification — 2026-11-02

| Check | Result |
|---|---:|
| Full offline regression suite | **75 passed** |
| Regression runtime | 1.30 seconds |
| Python compilation | Passed in the preceding verification |
| Live Groq requests during this verification | None |

Command: `python -m pytest tests eval\deterministic_benchmark.py --ignore-glob="*.before-*.py" -q`

The offline suite passed. The live LLM integration benchmark remains partial: three scenarios have passed and four remain unverified because of provider quota limits. No full seven-scenario live pass is claimed.

