"""Round-trip and validation tests for the typed contracts in ``schemas``.

The acceptance criterion for issue #8 is "round-trip JSON test for every
schema": every model is constructed, serialised, reparsed, and asserted
equal. The Output cross-field validator and the Ticket/TicketMetadata
``extra`` policies are exercised with targeted negative tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from wscad_triage import (
    ClaimEvidence,
    ClaimVerdict,
    Classification,
    DraftSolution,
    KBChunk,
    Output,
    ReasoningStep,
    RetrievalResult,
    Ticket,
    TicketMetadata,
    TicketState,
    VerifierVerdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def assert_round_trip(model: BaseModel) -> None:
    """Serialise a model to JSON and reparse it; assert equality."""
    parsed = type(model).model_validate_json(model.model_dump_json())
    assert parsed == model


def test_ticket_metadata_round_trip() -> None:
    meta = TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11")
    assert_round_trip(meta)


def test_ticket_metadata_allows_extra_fields() -> None:
    raw = {"product": "WSCAD Suite", "vendor_internal_id": "X-99"}
    meta = TicketMetadata.model_validate(raw)
    parsed = TicketMetadata.model_validate_json(meta.model_dump_json())
    assert parsed.__pydantic_extra__ == {"vendor_internal_id": "X-99"}


def test_ticket_round_trip() -> None:
    ticket = Ticket(
        ticket_id="T-001",
        text="Application fails to start after update. Error 504 appears.",
        metadata=TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11"),
    )
    assert_round_trip(ticket)


def test_ticket_forbids_extra_fields() -> None:
    raw = {"ticket_id": "T-X", "text": "hi", "metadata": {}, "stray": "nope"}
    with pytest.raises(ValidationError):
        Ticket.model_validate(raw)


def test_ticket_parses_real_tickets_file() -> None:
    """Schema must match the WSCAD-provided tickets.json byte-for-byte."""
    data = json.loads((REPO_ROOT / "tickets" / "tickets.json").read_text(encoding="utf-8"))
    tickets = [Ticket.model_validate(t) for t in data]
    assert [t.ticket_id for t in tickets] == ["T-001", "T-002"]
    assert tickets[1].metadata.version is None


def test_kbchunk_round_trip() -> None:
    chunk = KBChunk(
        chunk_id="Common_Errors.md#0",
        source_file="Common_Errors.md",
        text="Error 504 commonly indicates missing or expired licensing information.",
        language="en",
        metadata={"product_line": "ELECTRIX AI"},
    )
    assert_round_trip(chunk)


def test_retrieval_result_round_trip() -> None:
    chunk = KBChunk(
        chunk_id="Licensing_Offline_Activation.md#0",
        source_file="Licensing_Offline_Activation.md",
        text="Offline licenses must be activated using the License Manager.",
    )
    result = RetrievalResult(chunk=chunk, score=0.87, rank=1, retriever="rrf")
    assert_round_trip(result)


def test_classification_round_trip() -> None:
    classification = Classification(
        category="Licensing",
        priority="High",
        missing_critical_fields=["license_type"],
        rationale="Error 504 with no offline/online distinction stated.",
    )
    assert_round_trip(classification)


def test_reasoning_step_round_trip() -> None:
    step = ReasoningStep(
        actor="retrieve",
        action="hybrid_search",
        evidence_refs=["Common_Errors.md#0", "Licensing_Offline_Activation.md#0"],
        rationale="Top-2 RRF hits both reference licensing/error-504.",
    )
    assert_round_trip(step)


def test_output_solve_round_trip() -> None:
    output = Output(
        ticket_id="T-001",
        category="Licensing",
        priority="High",
        resolution_kind="solve",
        proposed_solution="Re-activate the offline license via License Manager.",
        followup_questions=["Was the license reactivated after the update?"],
        confidence=0.81,
        confidence_breakdown={"rubric": 0.85, "verifier": 0.81},
        cited_sources=["Common_Errors.md", "Licensing_Offline_Activation.md"],
    )
    assert_round_trip(output)


def test_output_clarify_round_trip() -> None:
    output = Output(
        ticket_id="T-003",
        category="Installation",
        priority="Medium",
        resolution_kind="clarify",
        preliminary_assessment="Multiple plausible root causes; OS and runtime unknown.",
        followup_questions=[
            "Which operating system and version?",
            "Which version of WSCAD Suite?",
        ],
        confidence=0.32,
    )
    assert_round_trip(output)


def test_output_validator_solve_requires_solution() -> None:
    with pytest.raises(ValidationError):
        Output(
            ticket_id="T-001",
            category="Licensing",
            priority="High",
            resolution_kind="solve",
            proposed_solution=None,
            confidence=0.5,
        )


def test_output_validator_solve_rejects_assessment() -> None:
    with pytest.raises(ValidationError):
        Output(
            ticket_id="T-001",
            category="Licensing",
            priority="High",
            resolution_kind="solve",
            proposed_solution="Re-activate.",
            preliminary_assessment="Should not be set.",
            confidence=0.5,
        )


def test_output_validator_clarify_requires_assessment() -> None:
    with pytest.raises(ValidationError):
        Output(
            ticket_id="T-003",
            category="Installation",
            priority="Medium",
            resolution_kind="clarify",
            preliminary_assessment=None,
            confidence=0.3,
        )


def test_output_validator_clarify_rejects_solution() -> None:
    with pytest.raises(ValidationError):
        Output(
            ticket_id="T-003",
            category="Installation",
            priority="Medium",
            resolution_kind="clarify",
            preliminary_assessment="Need more info.",
            proposed_solution="Should not be set.",
            confidence=0.3,
        )


def test_output_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        Output(
            ticket_id="T-001",
            category="Licensing",
            priority="High",
            resolution_kind="solve",
            proposed_solution="Re-activate.",
            confidence=1.5,
        )


def test_claim_evidence_round_trip() -> None:
    evidence = ClaimEvidence(
        claim="Error 504 indicates a licensing problem.",
        chunk_id="Common_Errors.md#0",
        quote="Error 504 indicates licensing.",
    )
    assert_round_trip(evidence)


def test_draft_solution_round_trip() -> None:
    draft = DraftSolution(
        solution="Re-activate the offline license via License Manager.",
        claims=[
            ClaimEvidence(
                claim="Error 504 is a licensing error.",
                chunk_id="Common_Errors.md#0",
                quote="Error 504 indicates licensing.",
            ),
        ],
    )
    assert_round_trip(draft)


def test_draft_solution_allows_empty_claims() -> None:
    draft = DraftSolution(solution="No retrieval; nothing to claim.")
    assert draft.claims == []
    assert_round_trip(draft)


def test_claim_verdict_round_trip() -> None:
    verdict = ClaimVerdict(
        claim="Error 504 is a licensing error.",
        grounded=True,
        rationale="Verbatim match against Common_Errors.md#0.",
    )
    assert_round_trip(verdict)


def test_verifier_verdict_round_trip() -> None:
    verdict = VerifierVerdict(
        grounding_score=0.85,
        per_claim=[
            ClaimVerdict(
                claim="Error 504 is a licensing error.",
                grounded=True,
                rationale="Cited chunk literally states this.",
            ),
        ],
    )
    assert_round_trip(verdict)


def test_verifier_verdict_score_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        VerifierVerdict(grounding_score=1.5)
    with pytest.raises(ValidationError):
        VerifierVerdict(grounding_score=-0.1)


def test_ticket_state_with_draft_and_verdict_round_trip() -> None:
    ticket = Ticket(ticket_id="T-001", text="hi")
    state = TicketState(
        ticket=ticket,
        draft_solution=DraftSolution(solution="ok", claims=[]),
        verifier_verdict=VerifierVerdict(grounding_score=0.7, per_claim=[]),
    )
    assert_round_trip(state)


def test_ticket_state_round_trip() -> None:
    ticket = Ticket(
        ticket_id="T-001",
        text="App fails to start.",
        metadata=TicketMetadata(product="WSCAD Suite"),
    )
    chunk = KBChunk(
        chunk_id="Common_Errors.md#0",
        source_file="Common_Errors.md",
        text="Error 504 indicates licensing.",
    )
    state = TicketState(
        ticket=ticket,
        classification=Classification(
            category="Licensing",
            priority="High",
            rationale="Error 504 hint.",
        ),
        retrievals=[RetrievalResult(chunk=chunk, score=0.9, rank=1, retriever="bm25")],
        reasoning_trace=[
            ReasoningStep(actor="triage", action="classified", rationale="Error code"),
        ],
        confidence_components={"retrieval_strength": 0.9},
        final_confidence=0.81,
        resolution_kind="solve",
        proposed_solution="Re-activate.",
    )
    assert_round_trip(state)
