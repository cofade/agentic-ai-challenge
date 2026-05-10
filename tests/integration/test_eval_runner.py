"""Issue #33: integration tests for ``eval.runner``.

Covers two entry points:

- ``run_eval`` — pure kernel exercised via ``pipeline_run`` injection.
- ``main(argv)`` — argparse entry point; exercised via monkeypatching
  ``make_client``, ``load_kb``, and ``run_eval`` so no real LLM or KB
  is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from eval.metrics import PerTicketResult, compute_metrics, score_error, score_one
from eval.runner import EvalResults, RunMetadata, main, run_eval

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
    assert other == "20260510T120000Z.json"


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


# ---------------------------------------------------------------------------
# main() CLI entry-point tests
# ---------------------------------------------------------------------------


def _minimal_eval_set(tmp_path: Path) -> Path:
    """Write a single-ticket eval set file that Pydantic can validate."""
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "ticket_id": "T-1",
                    "text": "dummy",
                    "expected_category": "Licensing",
                    "expected_priority": "High",
                    "should_clarify": False,
                    "coverage_case": "resolvable EN",
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_main_missing_eval_set_exits_2(tmp_path: Path) -> None:
    rc = main(
        [
            "--eval-set",
            str(tmp_path / "does_not_exist.json"),
            "--kb-dir",
            str(tmp_path),
            "--results-dir",
            str(tmp_path / "r"),
        ]
    )
    assert rc == 2


def test_main_missing_kb_dir_exits_2(tmp_path: Path) -> None:
    eval_set = _minimal_eval_set(tmp_path)
    rc = main(
        [
            "--eval-set",
            str(eval_set),
            "--kb-dir",
            str(tmp_path / "no_such_kb"),
            "--results-dir",
            str(tmp_path / "r"),
        ]
    )
    assert rc == 2


def test_main_exits_1_when_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_set = _minimal_eval_set(tmp_path)
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()

    et = _eval_ticket("T-1")
    err_result = score_error(et, RuntimeError("oops"))
    fake = EvalResults(
        metadata=_metadata(eval_set),
        aggregate=compute_metrics([err_result]),
        per_ticket=[err_result],
    )
    monkeypatch.setattr("eval.runner.make_client", lambda _s: cast(object, object()))
    monkeypatch.setattr("eval.runner.load_kb", lambda _d: [])
    monkeypatch.setattr("eval.runner.Embedding", lambda _: object())
    monkeypatch.setattr("eval.runner.run_eval", lambda *_a, **_kw: fake)

    rc = main(
        ["--eval-set", str(eval_set), "--kb-dir", str(kb_dir), "--results-dir", str(tmp_path / "r")]
    )
    assert rc == 1


def test_main_exits_0_on_clean_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_set = _minimal_eval_set(tmp_path)
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()

    et = _eval_ticket("T-1", expected_category="Licensing")
    good_result = score_one(et, _output("T-1", category="Licensing", confidence=0.8))
    fake = EvalResults(
        metadata=_metadata(eval_set),
        aggregate=compute_metrics([good_result]),
        per_ticket=[good_result],
    )
    monkeypatch.setattr("eval.runner.make_client", lambda _s: cast(object, object()))
    monkeypatch.setattr("eval.runner.load_kb", lambda _d: [])
    monkeypatch.setattr("eval.runner.Embedding", lambda _: object())
    monkeypatch.setattr("eval.runner.run_eval", lambda *_a, **_kw: fake)

    rc = main(
        ["--eval-set", str(eval_set), "--kb-dir", str(kb_dir), "--results-dir", str(tmp_path / "r")]
    )
    assert rc == 0
