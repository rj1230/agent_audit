from eval.eval_all import build_metrics
from eval.plan_quality_template import (
    ProjectPlanQualityMetric,
    ProjectPlanQualityTemplate,
)


def test_plan_extraction_prompt_captures_result_dependent_terminal_branch():
    trace = (
        '{"tool": "get_user_purchase_history", '
        '"output": {"status": "invalid_username"}, '
        '"final_response": "Please verify your username."}'
    )

    prompt = ProjectPlanQualityTemplate.extract_plan_from_trace(
        trace_json_str=trace,
        multimodal=False,
    )

    assert trace in prompt
    assert "invalid_username" in prompt
    assert "Stop safely when username verification fails." in prompt
    assert "Do not invent" in prompt


def test_plan_quality_prompt_accepts_safe_terminal_stop():
    task = "Refund the requested book for user9999."
    plan = (
        "1. Look up the supplied username. "
        "2. Stop after invalid_username. "
        "3. Ask the user to verify the username."
    )

    prompt = ProjectPlanQualityTemplate.evaluate_plan_quality(
        user_task=task,
        agent_plan=plan,
        multimodal=False,
    )

    assert task in prompt
    assert plan in prompt
    assert "Do not penalize the agent for omitting application-side checks or ticket" in prompt
    assert "Invalid username: stop and ask the customer to correct it." in prompt


def test_project_plan_quality_metric_is_registered():
    trace_metrics, _ = build_metrics()
    metric = trace_metrics[2]

    assert isinstance(metric, ProjectPlanQualityMetric)
    assert metric.threshold == 0.8
    assert metric._system_one_score_spec(
        "Refund a purchase.", [], False
    ) is None


def test_project_metric_routes_extraction_and_scoring_prompts():
    metric = None  # Prompt-routing tests do not require model initialization.

    extraction = ProjectPlanQualityMetric._get_prompt(None,
        "extract_plan_from_trace",
        template_class="PlanAdherenceMetric",
        trace_json_str='{"output": "invalid_username"}',
        multimodal=False,
    )
    scoring = ProjectPlanQualityMetric._get_prompt(None,
        "evaluate_plan_quality",
        template_class="PlanQualityMetric",
        user_task="Refund a purchase.",
        agent_plan="Stop after invalid username.",
        multimodal=False,
    )

    assert "Stop safely when username verification fails." in extraction
    assert "invalid_username" in extraction
    assert "Stop after invalid username." in scoring
    assert "conditional correctness" in scoring.lower()

def _make_plan_quality_case(trace, task=""):
    from types import SimpleNamespace

    return SimpleNamespace(
        input=task,
        _trace_dict=trace,
    )


def test_execution_evidence_extracts_verified_purchase_and_proposal():
    import json

    case = _make_plan_quality_case({
        "input": {
            "messages": [
                {"role": "user", "content": "Refund the book for user123."}
            ]
        },
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"date": "2026-10-05", "price": 23},
                "water_bottle": {"date": "2026-10-06", "price": 15},
            },
            "structured_response": {
                "answer": "A refund proposal is ready.",
                "refund_proposal": {
                    "username": "user123",
                    "items": [{"name": "book"}],
                },
            },
        },
    })

    evidence = ProjectPlanQualityMetric._build_execution_evidence(case)
    parsed = json.loads(evidence)

    assert parsed["lookup_status"] == "verified"
    assert parsed["verified_username"] == "user123"
    assert parsed["verified_purchase_item_names"] == [
        "book",
        "water_bottle",
    ]
    assert parsed["refund_proposal_present"] is True
    assert parsed["final_answer"] == "A refund proposal is ready."


def test_execution_evidence_handles_invalid_username_without_proposal():
    import json

    case = _make_plan_quality_case({
        "output": {
            "lookup_status": "invalid_username",
            "verified_username": None,
            "purchase_history": {},
            "structured_response": {
                "answer": "Please verify your username.",
                "refund_proposal": None,
            },
        },
    })

    evidence = ProjectPlanQualityMetric._build_execution_evidence(case)
    parsed = json.loads(evidence)

    assert parsed["lookup_status"] == "invalid_username"
    assert parsed["verified_username"] is None
    assert parsed["verified_purchase_item_names"] == []
    assert parsed["refund_proposal_present"] is False
    assert parsed["final_answer"] == "Please verify your username."


def test_execution_evidence_handles_missing_structured_response():
    import json

    case = _make_plan_quality_case({
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
        },
    })

    evidence = ProjectPlanQualityMetric._build_execution_evidence(case)
    parsed = json.loads(evidence)

    assert parsed["lookup_status"] == "verified"
    assert parsed["verified_purchase_item_names"] == ["book"]
    assert parsed["refund_proposal_present"] is False
    assert parsed["final_answer"] == ""


def test_execution_evidence_returns_none_for_missing_or_malformed_output():
    assert ProjectPlanQualityMetric._build_execution_evidence(
        _make_plan_quality_case(None)
    ) is None

    assert ProjectPlanQualityMetric._build_execution_evidence(
        _make_plan_quality_case({"output": None})
    ) is None

    assert ProjectPlanQualityMetric._build_execution_evidence(
        _make_plan_quality_case({"output": "not-a-dictionary"})
    ) is None


def test_original_task_prefers_nonempty_test_case_input():
    case = _make_plan_quality_case(
        {
            "input": {
                "messages": [
                    {"role": "user", "content": "Trace task"}
                ]
            }
        },
        task="Canonical test-case task",
    )

    assert ProjectPlanQualityMetric._get_original_task(case) == (
        "Canonical test-case task"
    )


def test_original_task_falls_back_to_latest_user_message():
    case = _make_plan_quality_case({
        "input": {
            "messages": [
                {"role": "user", "content": "First user message"},
                {"role": "assistant", "content": "Assistant response"},
                {"role": "tool", "content": "Tool output"},
                {"role": "human", "content": "Latest user message"},
            ]
        },
    })

    assert ProjectPlanQualityMetric._get_original_task(case) == (
        "Latest user message"
    )


def test_original_task_ignores_non_user_messages_and_malformed_entries():
    case = _make_plan_quality_case({
        "input": {
            "messages": [
                None,
                {"role": "assistant", "content": "Assistant response"},
                {"role": "tool", "content": "Tool output"},
                {"role": "user", "content": "   "},
            ]
        },
    })

    assert ProjectPlanQualityMetric._get_original_task(case) == ""


def test_async_original_task_uses_same_deterministic_fallback():
    import asyncio

    case = _make_plan_quality_case({
        "input": {
            "messages": [
                {"role": "user", "content": "Original request"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Latest request"},
            ]
        },
    })

    metric = object.__new__(ProjectPlanQualityMetric)
    result = asyncio.run(
        metric._a_extract_task_from_trace(case)
    )

    assert result == "Latest request"


def test_execution_evidence_validates_the_saved_book_and_pen_proposal():
    import json

    case = _make_plan_quality_case({
        "input": {
            "messages": [{
                "role": "user",
                "content": "I am user123, I would like a refund for the book and the pen",
            }]
        },
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"date": "2026-10-05", "price": 23},
                "pen": {"date": "2026-10-03", "price": 7},
            },
            "structured_response": {
                "answer": "A proposal is ready.",
                "refund_proposal": {
                    "username": "user123",
                    "items": [{"name": "book"}, {"name": "pen"}],
                },
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "PASS"
    assert parsed["proposal_validation_errors"] == []
    assert parsed["requested_verified_refund_item_names"] == ["book", "pen"]


def test_execution_evidence_accepts_valid_subset_of_requested_items():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [{
            "role": "user",
            "content": "I want a refund for the book and pen",
        }]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"price": 23},
                "pen": {"price": 7},
            },
            "structured_response": {
                "refund_proposal": {
                    "username": "user123",
                    "items": [{"name": "book"}],
                },
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "PASS"
    assert parsed["proposal_validation_errors"] == []
    assert parsed["requested_verified_refund_item_names"] == ["book", "pen"]


def test_execution_evidence_fails_for_wrong_username_and_unrequested_item():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [{
            "role": "user",
            "content": "I want a refund for the book for user123",
        }]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {
                "book": {"price": 23},
                "pen": {"price": 7},
            },
            "structured_response": {
                "refund_proposal": {
                    "username": "user999",
                    "items": [{"name": "book"}, {"name": "pen"}],
                },
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "FAIL"
    assert any("username" in error for error in parsed["proposal_validation_errors"])
    assert any("not affirmatively requested" in error for error in parsed["proposal_validation_errors"])


def test_execution_evidence_marks_information_only_branch_not_applicable():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [{
            "role": "user",
            "content": "Show me my book purchase history for user123",
        }]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
            "structured_response": {
                "answer": "You purchased a book.",
                "refund_proposal": None,
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "NOT_APPLICABLE"
    assert parsed["proposal_validation_errors"] == []


def test_execution_evidence_rejects_duplicate_proposal_items():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [{
            "role": "user",
            "content": "Refund the book for user123",
        }]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
            "structured_response": {
                "refund_proposal": {
                    "username": "user123",
                    "items": [{"name": "book"}, {"name": "the book"}],
                },
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "FAIL"
    assert any("Duplicate" in error for error in parsed["proposal_validation_errors"])


def test_execution_evidence_uses_affirmative_intent_across_multiple_turns():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [
            {"role": "user", "content": "I am user123."},
            {"role": "assistant", "content": "How can I help?"},
            {"role": "user", "content": "I would like a refund for the book."},
        ]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
            "structured_response": {
                "refund_proposal": {
                    "username": "user123",
                    "items": [{"name": "book"}],
                },
            },
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "PASS"
    assert parsed["requested_verified_refund_item_names"] == ["book"]


def test_execution_evidence_rejects_conflicting_intent_across_multiple_turns():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [
            {"role": "user", "content": "I want a refund for the book."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Actually, do not refund the book."},
        ]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
            "structured_response": {"refund_proposal": None},
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "NOT_APPLICABLE"
    assert parsed["requested_verified_refund_item_names"] == []


def test_execution_evidence_ignores_questions_across_multiple_turns():
    import json

    case = _make_plan_quality_case({
        "input": {"messages": [
            {"role": "user", "content": "Can I get a refund for the book?"},
            {"role": "assistant", "content": "I can explain the policy."},
        ]},
        "output": {
            "lookup_status": "verified",
            "verified_username": "user123",
            "purchase_history": {"book": {"price": 23}},
            "structured_response": {"refund_proposal": None},
        },
    })

    parsed = json.loads(ProjectPlanQualityMetric._build_execution_evidence(case))
    assert parsed["proposal_validation_status"] == "NOT_APPLICABLE"
    assert parsed["requested_verified_refund_item_names"] == []
