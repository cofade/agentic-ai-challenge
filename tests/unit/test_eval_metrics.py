"""Issue #33: scoring math in :mod:`eval.metrics`.

Pure-Python unit tests over hand-built ``EvalTicket`` / ``Output`` pairs.
The runner integration is covered in ``tests/integration/test_eval_runner.py``;
this module focuses on metric correctness and edge cases (empty input,
all-error input, divide-by-zero ratios, single-class confusion).
"""

from __future__ import annotations

from typing import Any

import pytest
from eval.metrics import (
    AggregateMetrics,
    PerTicketResult,
    compute_metrics,
    score_error,
    score_one,
)

from wscad_triage.schemas import (
    Category,
    CoverageCase,
    EvalTicket,
    Output,
    Priority,
    ResolutionKind,
)


def _eval_ticket(
    ticket_id: str = "T-1",
    *,
    expected_category: Category = "Licensing",
    expected_priority: Priority = "High",
    should_clarify: bool = False,
    coverage_case: CoverageCase = "resolvable EN",
    notes: str = "",
) -> EvalTicket:
    return EvalTicket(
        ticket_id=ticket_id,
        text="dummy",
        expected_category=expected_category,
        expected_priority=expected_priority,
        should_clarify=should_clarify,
        coverage_case=coverage_case,
        notes=notes,
    )


def _output(
    ticket_id: str = "T-1",
    *,
    category: Category = "Licensing",
    priority: Priority = "High",
    resolution_kind: ResolutionKind = "solve",
    confidence: float = 0.7,
    cited_sources: list[str] | None = None,
) -> Output:
    body: dict[str, Any] = {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "resolution_kind": resolution_kind,
        "confidence": confidence,
        "cited_sources": cited_sources or [],
    }
    if resolution_kind == "solve":
        body["proposed_solution"] = "do the thing"
    else:
        body["preliminary_assessment"] = "needs more info"
    return Output(**body)


def test_score_one_all_correct() -> None:
    et = _eval_ticket()
    out = _output()
    row = score_one(et, out)
    assert row.category_correct is True
    assert row.priority_correct is True
    assert row.clarification_outcome == "TN"
    assert row.error is None
    assert row.actual_category == "Licensing"
    assert row.confidence == 0.7


def test_score_one_category_mismatch() -> None:
    et = _eval_ticket(expected_category="Licensing")
    out = _output(category="Installation")
    row = score_one(et, out)
    assert row.category_correct is False
    assert row.priority_correct is True


def test_score_one_priority_mismatch() -> None:
    et = _eval_ticket(expected_priority="High")
    out = _output(priority="Low")
    row = score_one(et, out)
    assert row.category_correct is True
    assert row.priority_correct is False


@pytest.mark.parametrize(
    ("should_clarify", "actual_kind", "expected_outcome"),
    [
        (True, "clarify", "TP"),
        (False, "clarify", "FP"),
        (True, "solve", "FN"),
        (False, "solve", "TN"),
    ],
)
def test_clarification_confusion_matrix(
    should_clarify: bool,
    actual_kind: ResolutionKind,
    expected_outcome: str,
) -> None:
    et = _eval_ticket(should_clarify=should_clarify)
    out = _output(resolution_kind=actual_kind)
    assert score_one(et, out).clarification_outcome == expected_outcome


def test_score_error_carries_exception_message() -> None:
    et = _eval_ticket(notes="vague crash; expect grounding fail")
    row = score_error(et, RuntimeError("boom"))
    assert row.error == "RuntimeError: boom"
    assert row.actual_category is None
    assert row.actual_priority is None
    assert row.actual_resolution_kind is None
    assert row.confidence is None
    assert row.category_correct is False
    assert row.priority_correct is False
    assert row.clarification_outcome is None
    assert row.notes == "vague crash; expect grounding fail"


def test_compute_metrics_empty_input() -> None:
    m = compute_metrics([])
    assert m.n_tickets == 0
    assert m.n_errors == 0
    assert m.category_accuracy is None
    assert m.priority_accuracy is None
    assert m.clarification_precision is None
    assert m.clarification_recall is None
    assert m.clarification_f1 is None
    assert m.mean_confidence_overall is None
    assert m.mean_confidence_solve is None
    assert m.mean_confidence_clarify is None
    assert m.category_confusion == {}
    assert m.per_coverage_case == {}


def test_compute_metrics_all_correct() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", expected_category="Licensing", should_clarify=False),
            _output("T-1", category="Licensing", resolution_kind="solve", confidence=0.8),
        ),
        score_one(
            _eval_ticket("T-2", expected_category="Errors", should_clarify=True),
            _output("T-2", category="Errors", resolution_kind="clarify", confidence=0.4),
        ),
    ]
    m = compute_metrics(rows)
    assert m.n_tickets == 2
    assert m.n_errors == 0
    assert m.category_accuracy == pytest.approx(1.0)
    assert m.priority_accuracy == pytest.approx(1.0)
    assert m.clarification_precision == pytest.approx(1.0)
    assert m.clarification_recall == pytest.approx(1.0)
    assert m.clarification_f1 == pytest.approx(1.0)
    assert m.mean_confidence_overall == pytest.approx(0.6)
    assert m.mean_confidence_solve == pytest.approx(0.8)
    assert m.mean_confidence_clarify == pytest.approx(0.4)


def test_compute_metrics_all_errors_returns_none_metrics() -> None:
    rows = [
        score_error(_eval_ticket("T-1"), RuntimeError("a")),
        score_error(_eval_ticket("T-2"), ValueError("b")),
    ]
    m = compute_metrics(rows)
    assert m.n_tickets == 2
    assert m.n_errors == 2
    assert m.category_accuracy is None
    assert m.priority_accuracy is None
    assert m.clarification_precision is None
    assert m.clarification_recall is None
    assert m.clarification_f1 is None
    assert m.mean_confidence_overall is None
    assert m.category_confusion == {}


def test_compute_metrics_no_positive_predictions_yields_none_precision() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", should_clarify=True),
            _output("T-1", resolution_kind="solve"),
        ),
        score_one(
            _eval_ticket("T-2", should_clarify=False),
            _output("T-2", resolution_kind="solve"),
        ),
    ]
    m = compute_metrics(rows)
    assert m.clarification_precision is None
    assert m.clarification_recall == pytest.approx(0.0)
    assert m.clarification_f1 is None


def test_compute_metrics_no_positive_labels_yields_none_recall() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", should_clarify=False),
            _output("T-1", resolution_kind="clarify"),
        ),
        score_one(
            _eval_ticket("T-2", should_clarify=False),
            _output("T-2", resolution_kind="solve"),
        ),
    ]
    m = compute_metrics(rows)
    assert m.clarification_precision == pytest.approx(0.0)
    assert m.clarification_recall is None
    assert m.clarification_f1 is None


def test_compute_metrics_only_solve_outputs_means_only_solve_bucket() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1"),
            _output("T-1", resolution_kind="solve", confidence=0.5),
        ),
        score_one(
            _eval_ticket("T-2"),
            _output("T-2", resolution_kind="solve", confidence=0.9),
        ),
    ]
    m = compute_metrics(rows)
    assert m.mean_confidence_solve == pytest.approx(0.7)
    assert m.mean_confidence_clarify is None


def test_compute_metrics_confusion_matrix_populated() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", expected_category="Licensing"),
            _output("T-1", category="Installation"),
        ),
        score_one(
            _eval_ticket("T-2", expected_category="Licensing"),
            _output("T-2", category="Licensing"),
        ),
        score_one(
            _eval_ticket("T-3", expected_category="Errors"),
            _output("T-3", category="Errors"),
        ),
    ]
    m = compute_metrics(rows)
    assert m.category_confusion == {
        "Licensing": {"Installation": 1, "Licensing": 1},
        "Errors": {"Errors": 1},
    }


def test_compute_metrics_excludes_errored_rows_from_confusion() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", expected_category="Licensing"),
            _output("T-1", category="Licensing"),
        ),
        score_error(_eval_ticket("T-2", expected_category="Errors"), RuntimeError("x")),
    ]
    m = compute_metrics(rows)
    assert m.n_tickets == 2
    assert m.n_errors == 1
    assert m.category_confusion == {"Licensing": {"Licensing": 1}}
    assert m.category_accuracy == pytest.approx(1.0)


def test_compute_metrics_per_coverage_case_breakdown() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1", coverage_case="resolvable EN", expected_category="Licensing"),
            _output("T-1", category="Licensing", confidence=0.6),
        ),
        score_one(
            _eval_ticket("T-2", coverage_case="resolvable EN", expected_category="Licensing"),
            _output("T-2", category="Errors", confidence=0.4),
        ),
        score_error(
            _eval_ticket("T-3", coverage_case="clarify vague crash"),
            RuntimeError("x"),
        ),
    ]
    m = compute_metrics(rows)
    en = m.per_coverage_case["resolvable EN"]
    assert en.n == 2
    assert en.n_errors == 0
    assert en.category_accuracy == pytest.approx(0.5)
    assert en.mean_confidence == pytest.approx(0.5)

    crash = m.per_coverage_case["clarify vague crash"]
    assert crash.n == 1
    assert crash.n_errors == 1
    assert crash.category_accuracy is None
    assert crash.mean_confidence is None


def test_aggregate_metrics_round_trips_through_json() -> None:
    rows = [
        score_one(
            _eval_ticket("T-1"),
            _output("T-1", confidence=0.5),
        ),
    ]
    m = compute_metrics(rows)
    payload = m.model_dump_json()
    restored = AggregateMetrics.model_validate_json(payload)
    assert restored == m


def test_per_ticket_result_round_trips_through_json() -> None:
    row = score_one(_eval_ticket("T-1"), _output("T-1", cited_sources=["a.md"]))
    restored = PerTicketResult.model_validate_json(row.model_dump_json())
    assert restored == row
