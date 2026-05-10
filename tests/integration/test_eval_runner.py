"""Issue #33: integration test for ``eval.runner.run_eval``.

The runner kernel is decoupled from the actual triage pipeline via the
``pipeline_run`` keyword-only injection point so this test can exercise
the loop, scoring, and JSON-write steps without scripting all five
agents. The agent layer's contract is covered by
``tests/integration/test_pipeline.py``.

Three scenarios:

1. Happy path with two scripted outputs — verifies ``latest.json`` is
   written, parses back as ``EvalResults``, and the per-ticket results
   carry the expected fields.
2. Per-ticket exception is recorded and does not abort the loop.
3. ``RunMetadata`` provenance fields are persisted verbatim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from eval.metrics import PerTicketResult
from eval.runner import EvalResults, RunMetadata, run_eval

from wscad_triage.kb import HybridRetriever
from wscad_triage.llm.client import LLMClient
from wscad_triage.schemas import (
    Category,
    CoverageCase,
    EvalTicket,
    Output,
    Priority,
    ResolutionKind,
    Ticket,
)

pytestmark = pytest.mark.integration


def _eval_ticket(
    ticket_id: str,
    *,
    expected_category: Category = "Licensing",
    expected_priority: Priority = "High",
    should_clarify: bool = False,
    coverage_case: CoverageCase = "resolvable EN",
) -> EvalTicket:
    return EvalTicket(
        ticket_id=ticket_id,
        text="dummy",
        expected_category=expected_category,
        expected_priority=expected_priority,
        should_clarify=should_clarify,
        coverage_case=coverage_case,
    )


def _output(
    ticket_id: str,
    *,
    category: Category = "Licensing",
    priority: Priority = "High",
    resolution_kind: ResolutionKind = "solve",
    confidence: float = 0.7,
) -> Output:
    if resolution_kind == "solve":
        return Output(
            ticket_id=ticket_id,
            category=category,
            priority=priority,
            resolution_kind="solve",
            proposed_solution="do the thing",
            confidence=confidence,
        )
    return Output(
        ticket_id=ticket_id,
        category=category,
        priority=priority,
        resolution_kind="clarify",
        preliminary_assessment="needs more info",
        confidence=confidence,
    )


def _metadata(eval_set_path: Path) -> RunMetadata:
    return RunMetadata(
        timestamp_utc="2026-05-10T12:00:00+00:00",
        provider="ollama",
        model="gpt-oss:20b",
        kb_chunk_count=42,
        eval_set_path=str(eval_set_path),
        eval_set_sha256="0" * 64,
        git_commit="deadbeef",
        wscad_triage_version="0.0.0+test",
    )


def _fake_llm() -> LLMClient:
    """The runner passes the LLM through to ``pipeline_run``; the fake
    callable in these tests does not use it. Cast the unused stand-in
    so mypy is satisfied without constructing a real backend.
    """
    return cast(LLMClient, object())


def _fake_retriever() -> HybridRetriever:
    return cast(HybridRetriever, object())


def test_run_eval_happy_path_writes_latest_json(tmp_path: Path) -> None:
    eval_tickets = [
        _eval_ticket("T-1", expected_category="Licensing"),
        _eval_ticket("T-2", expected_category="Errors", should_clarify=True),
    ]
    scripted = {
        "T-1": _output("T-1", category="Licensing", confidence=0.8),
        "T-2": _output("T-2", category="Errors", resolution_kind="clarify", confidence=0.4),
    }

    def fake_pipeline_run(t: Ticket, _llm: LLMClient, _r: HybridRetriever) -> Output:
        return scripted[t.ticket_id]

    metadata = _metadata(tmp_path / "eval_set.json")

    results = run_eval(
        eval_tickets,
        _fake_llm(),
        _fake_retriever(),
        metadata,
        tmp_path / "results",
        pipeline_run=fake_pipeline_run,
        progress=None,
    )

    latest = tmp_path / "results" / "latest.json"
    assert latest.is_file(), "run_eval must write latest.json"

    parsed = EvalResults.model_validate_json(latest.read_text(encoding="utf-8"))
    assert parsed == results
    assert parsed.aggregate.n_tickets == 2
    assert parsed.aggregate.n_errors == 0
    assert parsed.aggregate.category_accuracy == pytest.approx(1.0)
    ids = [r.ticket_id for r in parsed.per_ticket]
    assert ids == ["T-1", "T-2"]


def test_run_eval_writes_timestamped_sibling(tmp_path: Path) -> None:
    eval_tickets = [_eval_ticket("T-1")]

    def fake_pipeline_run(t: Ticket, _llm: LLMClient, _r: HybridRetriever) -> Output:
        return _output(t.ticket_id)

    metadata = _metadata(tmp_path / "eval_set.json")
    run_eval(
        eval_tickets,
        _fake_llm(),
        _fake_retriever(),
        metadata,
        tmp_path / "results",
        pipeline_run=fake_pipeline_run,
        progress=None,
    )

    timestamped = list((tmp_path / "results").glob("*.json"))
    # latest.json + one timestamped file = 2
    assert len(timestamped) == 2
    names = {p.name for p in timestamped}
    assert "latest.json" in names
    other = next(n for n in names if n != "latest.json")
    assert other.endswith(".json")
    assert "2026-05-10T12-00-00" in other


def test_run_eval_records_per_ticket_exception(tmp_path: Path) -> None:
    eval_tickets = [
        _eval_ticket("T-1"),
        _eval_ticket("T-2"),
        _eval_ticket("T-3"),
    ]

    def fake_pipeline_run(t: Ticket, _llm: LLMClient, _r: HybridRetriever) -> Output:
        if t.ticket_id == "T-2":
            raise RuntimeError("fabricated mid-run failure")
        return _output(t.ticket_id)

    metadata = _metadata(tmp_path / "eval_set.json")
    results = run_eval(
        eval_tickets,
        _fake_llm(),
        _fake_retriever(),
        metadata,
        tmp_path / "results",
        pipeline_run=fake_pipeline_run,
        progress=None,
    )

    assert results.aggregate.n_tickets == 3
    assert results.aggregate.n_errors == 1
    by_id: dict[str, PerTicketResult] = {r.ticket_id: r for r in results.per_ticket}
    assert by_id["T-2"].error == "RuntimeError: fabricated mid-run failure"
    assert by_id["T-2"].actual_category is None
    assert by_id["T-1"].error is None
    assert by_id["T-3"].error is None


def test_run_eval_persists_metadata(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "eval_set.json")

    def fake_pipeline_run(t: Ticket, _llm: LLMClient, _r: HybridRetriever) -> Output:
        return _output(t.ticket_id)

    run_eval(
        [_eval_ticket("T-1")],
        _fake_llm(),
        _fake_retriever(),
        metadata,
        tmp_path / "results",
        pipeline_run=fake_pipeline_run,
        progress=None,
    )

    payload = json.loads((tmp_path / "results" / "latest.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["provider"] == "ollama"
    assert payload["metadata"]["model"] == "gpt-oss:20b"
    assert payload["metadata"]["kb_chunk_count"] == 42
    assert payload["metadata"]["git_commit"] == "deadbeef"
    assert payload["metadata"]["wscad_triage_version"] == "0.0.0+test"
