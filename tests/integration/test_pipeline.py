"""End-to-end pipeline integration tests (issue #25, ADR-005).

Five scenarios cover the ROADMAP's #25 acceptance criterion verbatim:

- resolvable EN ticket -> high-confidence solution
- clarify-required ticket -> follow-up questions
- multilingual (DE) ticket -> resolvable path
- ungrounded-claim trap -> verifier downgrades to clarify
- missing-OS metadata -> short-circuit through clarify

All scripts run through ``MockLLMClient`` and a ``StubRetriever`` so the
suite is fully deterministic and offline. Triggers use each agent's full
system-prompt opening (``"You are the {name} agent."``) rather than the
shorter ``"{name} agent"`` form -- the clarify agent's prompt mentions
the triage agent by name, which would otherwise cross-trigger the
triage script. Unit tests can use the short form because they run one
agent at a time. The supervisor itself makes no LLM calls (ADR-005,
deterministic edges).
"""

from __future__ import annotations

from typing import Any

import pytest

from wscad_triage import pipeline
from wscad_triage.llm import LLMResponse, ToolCall, Usage
from wscad_triage.schemas import (
    KBChunk,
    Output,
    RetrievalResult,
    Ticket,
    TicketMetadata,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers


def _empty_usage() -> Usage:
    return Usage(input_tokens=0, output_tokens=0)


def _tool_response(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name=name, arguments=args)],
        stop_reason="tool_use",
        usage=_empty_usage(),
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=_empty_usage(),
    )


def _retrieval(chunk_id: str, source_file: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id=chunk_id, source_file=source_file, text=text),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


# ---------------------------------------------------------------------------
# Scenario 1: resolvable EN ticket -> finalize_solve


def test_resolvable_ticket_returns_solve_output(mock_llm: Any, stub_retriever_factory: Any) -> None:
    ticket = Ticket(
        ticket_id="T-001",
        text="Application fails to start. Error 504 appears.",
        metadata=TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11"),
    )
    retrieval = _retrieval(
        "Common_Errors.md#0",
        "Common_Errors.md",
        "Error 504 indicates licensing.",
    )
    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "Error 504 indicates licensing.",
                },
            ),
            "You are the retrieve agent.": _text_response("error 504 licensing offline"),
            "You are the reason agent.": _tool_response(
                "emit_draft",
                {
                    "solution": "Re-activate the offline license via License Manager.",
                    "claims": [
                        {
                            "claim": "Error 504 indicates licensing.",
                            "chunk_id": "Common_Errors.md#0",
                            "quote": "Error 504 indicates licensing.",
                        }
                    ],
                },
            ),
            "You are the verify agent.": _tool_response(
                "emit_verdict",
                {
                    "grounding_score": 0.85,
                    "per_claim": [
                        {
                            "claim": "Error 504 indicates licensing.",
                            "grounded": True,
                            "rationale": "Verbatim match.",
                        }
                    ],
                },
            ),
        }
    )
    retriever = stub_retriever_factory(retrieval)

    output = pipeline.run(ticket, llm, retriever)

    assert isinstance(output, Output)
    assert output.ticket_id == "T-001"
    assert output.resolution_kind == "solve"
    assert output.proposed_solution == "Re-activate the offline license via License Manager."
    assert output.preliminary_assessment is None
    assert output.confidence == 0.85
    assert output.confidence_breakdown == {"verifier": 0.85}
    assert output.cited_sources == ["Common_Errors.md"]
    # Trace ordering: triage -> retrieve -> reason -> verify -> finalize_solve.
    actors = [s.actor for s in output.reasoning_trace]
    assert actors == ["triage", "retrieve", "reason", "verify", "supervisor"]
    assert output.reasoning_trace[-1].action == "finalize_solve"


# ---------------------------------------------------------------------------
# Scenario 2: missing OS + version -> finalize_clarify (no retrieval)


def test_clarify_required_when_metadata_missing(mock_llm: Any, stub_retriever_factory: Any) -> None:
    ticket = Ticket(
        ticket_id="T-002",
        text="License stopped working on an offline machine.",
        metadata=TicketMetadata(product="WSCAD Suite"),  # version + os missing
    )
    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": ["version", "os"],
                    "rationale": "Need OS and version to proceed.",
                },
            ),
            "You are the clarify agent.": _tool_response(
                "emit_questions",
                {
                    "questions": [
                        "Which version of WSCAD Suite is installed?",
                        "Which operating system and OS version is on the affected machine?",
                    ]
                },
            ),
        }
    )
    retriever = stub_retriever_factory()  # never called

    output = pipeline.run(ticket, llm, retriever)

    assert output.resolution_kind == "clarify"
    assert output.proposed_solution is None
    assert output.preliminary_assessment is not None
    assert len(output.followup_questions) == 2
    assert output.confidence < 0.5  # placeholder is below threshold by construction
    # Retriever was never called -- short-circuit through clarify.
    assert retriever.calls == []
    actors = [s.actor for s in output.reasoning_trace]
    assert actors == ["triage", "clarify", "supervisor"]


# ---------------------------------------------------------------------------
# Scenario 3: multilingual (DE) ticket -> resolvable


def test_multilingual_ticket_solves(mock_llm: Any, stub_retriever_factory: Any) -> None:
    ticket = Ticket(
        ticket_id="T-003",
        text="Lizenz funktioniert nach dem Update nicht mehr.",
        metadata=TicketMetadata(product="ELECTRIX AI", version="7.3.2.4", os="Windows 11"),
    )
    de_chunk_id = "v7.3.2.4.de.md#0"
    retrieval = _retrieval(
        de_chunk_id,
        "v7.3.2.4.de.md",
        "Der License Manager prueft Hostname und Lizenzschluessel.",
    )
    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "Lizenzproblem nach Update.",
                },
            ),
            "You are the retrieve agent.": _text_response("lizenz update license manager"),
            "You are the reason agent.": _tool_response(
                "emit_draft",
                {
                    "solution": "License Manager neu starten und Lizenz erneut aktivieren.",
                    "claims": [
                        {
                            "claim": "License Manager prueft Hostname und Lizenzschluessel.",
                            "chunk_id": de_chunk_id,
                            "quote": "License Manager prueft Hostname und Lizenzschluessel.",
                        }
                    ],
                },
            ),
            "You are the verify agent.": _tool_response(
                "emit_verdict",
                {
                    "grounding_score": 0.78,
                    "per_claim": [
                        {
                            "claim": "License Manager prueft Hostname und Lizenzschluessel.",
                            "grounded": True,
                            "rationale": "Wortwoertlich im KB-Chunk.",
                        }
                    ],
                },
            ),
        }
    )
    retriever = stub_retriever_factory(retrieval)

    output = pipeline.run(ticket, llm, retriever)

    assert output.resolution_kind == "solve"
    assert output.confidence == 0.78
    # The DE chunk_id surfaces in the reason step's evidence_refs.
    reason_step = next(s for s in output.reasoning_trace if s.actor == "reason")
    assert de_chunk_id in reason_step.evidence_refs
    assert output.cited_sources == ["v7.3.2.4.de.md"]


# ---------------------------------------------------------------------------
# Scenario 4: ungrounded-claim trap -> finalize_clarify_downgrade


def test_ungrounded_claim_trap_downgrades_to_clarify(
    mock_llm: Any, stub_retriever_factory: Any
) -> None:
    """ADR-008 gate: aggregate grounding < 0.4 forces clarify even with full metadata."""
    ticket = Ticket(
        ticket_id="T-004",
        text="Application fails to start. Error 504 appears.",
        metadata=TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11"),
    )
    retrieval = _retrieval(
        "Common_Errors.md#0",
        "Common_Errors.md",
        "Error 504 indicates licensing.",
    )
    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "Error 504.",
                },
            ),
            "You are the retrieve agent.": _text_response("error 504 licensing"),
            "You are the reason agent.": _tool_response(
                "emit_draft",
                {
                    "solution": "Reinstall the application from a USB stick.",
                    "claims": [
                        {
                            "claim": "Error 504 is fixed by reinstalling from USB.",
                            "chunk_id": "Common_Errors.md#0",
                            # Substring of the chunk text -- passes layer-1 (b)
                            # but the verifier flags it as not grounded in
                            # support of the claim.
                            "quote": "Error 504",
                        }
                    ],
                },
            ),
            "You are the verify agent.": _tool_response(
                "emit_verdict",
                {
                    "grounding_score": 0.30,
                    "per_claim": [
                        {
                            "claim": "Error 504 is fixed by reinstalling from USB.",
                            "grounded": False,
                            "rationale": "Cited chunk says nothing about USB reinstall.",
                        }
                    ],
                },
            ),
        }
    )
    retriever = stub_retriever_factory(retrieval)

    output = pipeline.run(ticket, llm, retriever)

    assert output.resolution_kind == "clarify"
    assert output.proposed_solution is None
    assert output.preliminary_assessment is not None
    assert "Error 504 is fixed by reinstalling from USB." in output.preliminary_assessment
    assert output.followup_questions == []  # downgrade path emits no questions
    assert output.confidence == 0.3
    # The downgrade path runs reason+verify but skips the clarify worker.
    actors = [s.actor for s in output.reasoning_trace]
    assert actors == ["triage", "retrieve", "reason", "verify", "supervisor"]
    assert output.reasoning_trace[-1].action == "finalize_clarify_downgrade"


# ---------------------------------------------------------------------------
# Scenario 5: missing-OS metadata short-circuits before retrieval


def test_missing_os_short_circuits_before_retrieval(
    mock_llm: Any, stub_retriever_factory: Any
) -> None:
    """Triage's deterministic gap pass plus the supervisor's route_after_triage
    edge ensure the retriever is never invoked when critical metadata is
    missing -- saves a vector-DB round-trip on tickets we can't answer.
    """
    ticket = Ticket(
        ticket_id="T-005",
        text="Application crashes intermittently.",
        metadata=TicketMetadata(product="WSCAD Suite", version="2.3"),  # os missing
    )
    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Errors",
                    "priority": "Medium",
                    "missing_critical_fields": ["os"],
                    "rationale": "Need OS to reproduce.",
                },
            ),
            "You are the clarify agent.": _tool_response(
                "emit_questions",
                {
                    "questions": [
                        "Which operating system and version is the machine running?",
                        "Does the crash reproduce on a different operating system?",
                    ]
                },
            ),
        }
    )
    retriever = stub_retriever_factory()  # would raise if called (no results queued)

    output = pipeline.run(ticket, llm, retriever)

    assert output.resolution_kind == "clarify"
    assert retriever.calls == []
    # The supervisor's deterministic edge skipped retrieve -> reason -> verify.
    actors = [s.actor for s in output.reasoning_trace]
    assert "retrieve" not in actors
    assert "reason" not in actors
    assert "verify" not in actors
    assert actors == ["triage", "clarify", "supervisor"]
