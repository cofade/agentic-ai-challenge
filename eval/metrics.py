"""Phase 5 evaluation metrics (issue #33, ADR-010).

Pure scoring layer for the eval harness: compares a pipeline ``Output``
against an ``EvalTicket``'s ground-truth labels and aggregates per-ticket
results into the metrics named in the issue's acceptance criteria —
category accuracy, priority accuracy, clarification precision/recall,
and mean grounded confidence.

The module is deliberately I/O free so it can be unit-tested without the
LLM, the KB, or the file system. ``runner.py`` is the I/O layer that
loads the eval set, runs the pipeline, and writes the JSON artefact.

Definitions (mirrors ADR-010):

- **Clarification confusion matrix.** Treat ``should_clarify=True`` as
  the positive label and ``actual_resolution_kind == "clarify"`` as the
  positive prediction. The four cells are TP/FP/TN/FN; precision and
  recall follow the standard formulas.
- **"Mean grounded confidence"** (acceptance-criteria phrasing) is the
  mean confidence over outputs that produced a ``proposed_solution``
  (``resolution_kind == "solve"``). The verifier's groundedness gate
  (ADR-008) is the only path to a ``solve`` outcome, so confidence on
  ``solve`` outputs is the grounded-confidence quantity.
- **Undefined ratios return ``None``.** When the denominator is zero
  (e.g. no positive predictions, no solve outputs) the corresponding
  metric is ``None`` rather than ``0.0`` or ``nan`` so downstream
  consumers cannot misread "no data" as "perfect zero".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wscad_triage.schemas import (
    Category,
    CoverageCase,
    EvalTicket,
    Output,
    Priority,
    ResolutionKind,
)

ClarificationOutcome = Literal["TP", "FP", "TN", "FN"]


class PerTicketResult(BaseModel):
    """One eval ticket scored against the pipeline's output.

    All ``actual_*`` fields and ``confidence`` are ``None`` when the
    pipeline raised — see :func:`score_error`. ``error`` carries a short
    ``"{ExceptionType}: {message}"`` string in that case so the run JSON
    is enough to triage failures without re-running.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    coverage_case: CoverageCase
    expected_category: Category
    expected_priority: Priority
    should_clarify: bool
    notes: str = ""
    actual_category: Category | None = None
    actual_priority: Priority | None = None
    actual_resolution_kind: ResolutionKind | None = None
    confidence: float | None = None
    cited_sources: list[str] = Field(default_factory=list)
    category_correct: bool = False
    priority_correct: bool = False
    clarification_outcome: ClarificationOutcome | None = None
    error: str | None = None


class CoverageCaseMetrics(BaseModel):
    """Per-coverage-case roll-up. ``n`` includes errored tickets; the
    accuracy/confidence values are computed over the error-free subset
    and are ``None`` when that subset is empty.
    """

    model_config = ConfigDict(extra="forbid")

    n: int
    n_errors: int
    category_accuracy: float | None
    priority_accuracy: float | None
    mean_confidence: float | None


class AggregateMetrics(BaseModel):
    """The four acceptance-criteria metrics plus diagnostics.

    ``category_confusion`` is a nested dict ``expected -> actual ->
    count`` populated from error-free results only. The keys are the
    Category literal values (str at the JSON layer); the model is
    serialised verbatim so reviewers can spot-check it.
    """

    model_config = ConfigDict(extra="forbid")

    n_tickets: int
    n_errors: int
    category_accuracy: float | None
    priority_accuracy: float | None
    clarification_precision: float | None
    clarification_recall: float | None
    clarification_f1: float | None
    mean_confidence_overall: float | None
    mean_confidence_solve: float | None
    mean_confidence_clarify: float | None
    category_confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    per_coverage_case: dict[str, CoverageCaseMetrics] = Field(default_factory=dict)


def _clarification_outcome(should_clarify: bool, actual: ResolutionKind) -> ClarificationOutcome:
    if should_clarify and actual == "clarify":
        return "TP"
    if not should_clarify and actual == "clarify":
        return "FP"
    if should_clarify and actual == "solve":
        return "FN"
    return "TN"


def score_one(eval_ticket: EvalTicket, output: Output) -> PerTicketResult:
    """Compare an :class:`Output` against the ticket's ground truth."""
    return PerTicketResult(
        ticket_id=eval_ticket.ticket_id,
        coverage_case=eval_ticket.coverage_case,
        expected_category=eval_ticket.expected_category,
        expected_priority=eval_ticket.expected_priority,
        should_clarify=eval_ticket.should_clarify,
        notes=eval_ticket.notes,
        actual_category=output.category,
        actual_priority=output.priority,
        actual_resolution_kind=output.resolution_kind,
        confidence=output.confidence,
        cited_sources=list(output.cited_sources),
        category_correct=output.category == eval_ticket.expected_category,
        priority_correct=output.priority == eval_ticket.expected_priority,
        clarification_outcome=_clarification_outcome(
            eval_ticket.should_clarify, output.resolution_kind
        ),
    )


def score_error(eval_ticket: EvalTicket, exc: BaseException) -> PerTicketResult:
    """Record a per-ticket pipeline failure without crashing the run.

    The ``actual_*`` fields stay ``None`` so downstream metrics can
    cleanly skip this row; ``error`` carries the exception type and
    message for triage.
    """
    return PerTicketResult(
        ticket_id=eval_ticket.ticket_id,
        coverage_case=eval_ticket.coverage_case,
        expected_category=eval_ticket.expected_category,
        expected_priority=eval_ticket.expected_priority,
        should_clarify=eval_ticket.should_clarify,
        notes=eval_ticket.notes,
        error=f"{type(exc).__name__}: {exc}",
    )


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _build_confusion(rows: list[PerTicketResult]) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.actual_category is None:
            continue
        confusion.setdefault(row.expected_category, {}).setdefault(row.actual_category, 0)
        confusion[row.expected_category][row.actual_category] += 1
    return confusion


def _per_case(rows: list[PerTicketResult]) -> dict[str, CoverageCaseMetrics]:
    by_case: dict[str, list[PerTicketResult]] = {}
    for row in rows:
        by_case.setdefault(row.coverage_case, []).append(row)

    out: dict[str, CoverageCaseMetrics] = {}
    for case, case_rows in by_case.items():
        successes = [r for r in case_rows if r.error is None]
        n_errors = len(case_rows) - len(successes)
        cat_correct = sum(1 for r in successes if r.category_correct)
        prio_correct = sum(1 for r in successes if r.priority_correct)
        confidences = [r.confidence for r in successes if r.confidence is not None]
        out[case] = CoverageCaseMetrics(
            n=len(case_rows),
            n_errors=n_errors,
            category_accuracy=_safe_ratio(cat_correct, len(successes)),
            priority_accuracy=_safe_ratio(prio_correct, len(successes)),
            mean_confidence=_safe_mean(confidences),
        )
    return out


def compute_metrics(results: list[PerTicketResult]) -> AggregateMetrics:
    """Aggregate scored results into the metrics dict for ``latest.json``.

    Errored rows are excluded from accuracy/confusion/confidence
    computations but counted in ``n_tickets`` and ``n_errors``.
    """
    successes = [r for r in results if r.error is None]
    n_errors = len(results) - len(successes)

    cat_correct = sum(1 for r in successes if r.category_correct)
    prio_correct = sum(1 for r in successes if r.priority_correct)

    tp = sum(1 for r in successes if r.clarification_outcome == "TP")
    fp = sum(1 for r in successes if r.clarification_outcome == "FP")
    fn = sum(1 for r in successes if r.clarification_outcome == "FN")

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)

    overall_conf = [r.confidence for r in successes if r.confidence is not None]
    solve_conf = [
        r.confidence
        for r in successes
        if r.actual_resolution_kind == "solve" and r.confidence is not None
    ]
    clarify_conf = [
        r.confidence
        for r in successes
        if r.actual_resolution_kind == "clarify" and r.confidence is not None
    ]

    return AggregateMetrics(
        n_tickets=len(results),
        n_errors=n_errors,
        category_accuracy=_safe_ratio(cat_correct, len(successes)),
        priority_accuracy=_safe_ratio(prio_correct, len(successes)),
        clarification_precision=precision,
        clarification_recall=recall,
        clarification_f1=_f1(precision, recall),
        mean_confidence_overall=_safe_mean(overall_conf),
        mean_confidence_solve=_safe_mean(solve_conf),
        mean_confidence_clarify=_safe_mean(clarify_conf),
        category_confusion=_build_confusion(successes),
        per_coverage_case=_per_case(results),
    )


__all__ = [
    "AggregateMetrics",
    "ClarificationOutcome",
    "CoverageCaseMetrics",
    "PerTicketResult",
    "compute_metrics",
    "score_error",
    "score_one",
]
