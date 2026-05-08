"""Tests for the deterministic supervisor (issue #24, ADR-005).

Two routing functions and three finalize sinks. Pinned behaviour:

- ``route_after_triage`` returns ``"clarify"`` iff
  ``classification.missing_critical_fields`` is non-empty.
- ``route_after_verify`` returns ``"finalize_clarify_downgrade"`` iff
  ``verifier_verdict.grounding_score < GROUNDING_THRESHOLD`` (0.4 by ADR-008).
  Boundary-strict: 0.4 -> solve, 0.39999 -> downgrade.
- ``finalize_solve`` requires both ``draft_solution`` and
  ``verifier_verdict``; raises otherwise.
- ``finalize_clarify_downgrade`` lists every ungrounded claim in the
  synthesised ``preliminary_assessment``.
- ``state_to_output`` populates ``cited_sources`` from claim chunk_ids'
  source files (deduplicated, sorted).
"""

from __future__ import annotations

import pytest

from wscad_triage.agents import supervisor
from wscad_triage.pipeline import state_to_output
from wscad_triage.schemas import (
    ClaimEvidence,
    ClaimVerdict,
    Classification,
    DraftSolution,
    KBChunk,
    Output,
    RetrievalResult,
    Ticket,
    TicketState,
    VerifierVerdict,
)


def _retrieval(chunk_id: str, source_file: str, text: str = "...") -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id=chunk_id, source_file=source_file, text=text),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


def _ticket() -> Ticket:
    return Ticket(ticket_id="T-001", text="App fails to start.")


def _classification(*, gaps: list[str] | None = None) -> Classification:
    return Classification(
        category="Licensing",
        priority="High",
        rationale="Error 504 hint.",
        missing_critical_fields=gaps or [],
    )


def _draft() -> DraftSolution:
    return DraftSolution(
        solution="Re-activate the offline license.",
        claims=[
            ClaimEvidence(
                claim="Error 504 is a licensing problem.",
                chunk_id="Common_Errors.md#0",
                quote="Error 504",
            )
        ],
    )


def _verdict(score: float, *, grounded: bool = True) -> VerifierVerdict:
    return VerifierVerdict(
        grounding_score=score,
        per_claim=[
            ClaimVerdict(
                claim="Error 504 is a licensing problem.",
                grounded=grounded,
                rationale="checked",
            )
        ],
    )


# ---------------------------------------------------------------------------
# route_after_triage


def test_route_after_triage_no_gaps_routes_to_retrieve() -> None:
    state = TicketState(ticket=_ticket(), classification=_classification())
    assert supervisor.route_after_triage(state) == "retrieve"


def test_route_after_triage_with_gaps_routes_to_clarify() -> None:
    state = TicketState(ticket=_ticket(), classification=_classification(gaps=["os", "version"]))
    assert supervisor.route_after_triage(state) == "clarify"


def test_route_after_triage_no_classification_raises() -> None:
    """Routing assumes triage just ran. Missing classification is a graph bug."""
    state = TicketState(ticket=_ticket())
    with pytest.raises(ValueError, match=r"state.classification is None"):
        supervisor.route_after_triage(state)


# ---------------------------------------------------------------------------
# route_after_verify


def test_route_after_verify_high_score_routes_to_finalize_solve() -> None:
    state = TicketState(
        ticket=_ticket(),
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.85),
    )
    assert supervisor.route_after_verify(state) == "finalize_solve"


def test_route_after_verify_low_score_routes_to_downgrade() -> None:
    state = TicketState(
        ticket=_ticket(),
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.30, grounded=False),
    )
    assert supervisor.route_after_verify(state) == "finalize_clarify_downgrade"


def test_route_after_verify_threshold_boundary_solve() -> None:
    """At exactly 0.4 we solve (>= threshold)."""
    state = TicketState(
        ticket=_ticket(),
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.40),
    )
    assert supervisor.route_after_verify(state) == "finalize_solve"


def test_route_after_verify_threshold_boundary_downgrade() -> None:
    """Just below 0.4 we downgrade."""
    state = TicketState(
        ticket=_ticket(),
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.39999, grounded=False),
    )
    assert supervisor.route_after_verify(state) == "finalize_clarify_downgrade"


def test_route_after_verify_no_verdict_raises() -> None:
    state = TicketState(ticket=_ticket(), draft_solution=_draft())
    with pytest.raises(ValueError, match=r"state.verifier_verdict is None"):
        supervisor.route_after_verify(state)


# ---------------------------------------------------------------------------
# finalize_solve


def test_finalize_solve_promotes_draft_solution() -> None:
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.85),
    )
    new = supervisor.finalize_solve(state)
    assert new.resolution_kind == "solve"
    assert new.proposed_solution == "Re-activate the offline license."
    assert new.preliminary_assessment is None
    assert new.final_confidence == 0.85
    assert new.reasoning_trace[-1].actor == "supervisor"
    assert new.reasoning_trace[-1].action == "finalize_solve"


def test_finalize_solve_no_draft_raises() -> None:
    state = TicketState(
        ticket=_ticket(),
        verifier_verdict=_verdict(0.85),
    )
    with pytest.raises(ValueError, match=r"state.draft_solution is None"):
        supervisor.finalize_solve(state)


def test_finalize_solve_no_verdict_raises() -> None:
    state = TicketState(ticket=_ticket(), draft_solution=_draft())
    with pytest.raises(ValueError, match=r"state.verifier_verdict is None"):
        supervisor.finalize_solve(state)


# ---------------------------------------------------------------------------
# finalize_clarify


def test_finalize_clarify_synthesises_assessment_and_uses_placeholder_confidence() -> None:
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(gaps=["os", "version"]),
        followup_questions=["Which OS?", "Which version?"],
    )
    new = supervisor.finalize_clarify(state)
    assert new.resolution_kind == "clarify"
    assert new.preliminary_assessment is not None
    # Whole-word membership against the comma-separated gap list rather
    # than naked substrings (which would falsely match "os" inside
    # "lots", "post", etc.).
    gap_list = new.preliminary_assessment.split("Awaiting clarification on: ")[1]
    assert "os" in {g.strip().rstrip(".") for g in gap_list.split(",")}
    assert "version" in {g.strip().rstrip(".") for g in gap_list.split(",")}
    assert new.proposed_solution is None
    assert new.final_confidence == supervisor.CLARIFY_CONFIDENCE_PLACEHOLDER
    # Below 0.5 by construction so a clarify never masquerades as a solve.
    assert new.final_confidence < 0.5


def test_finalize_clarify_no_classification_raises() -> None:
    state = TicketState(ticket=_ticket())
    with pytest.raises(ValueError, match=r"state.classification is None"):
        supervisor.finalize_clarify(state)


# ---------------------------------------------------------------------------
# finalize_clarify_downgrade


def test_finalize_clarify_downgrade_lists_ungrounded_claims() -> None:
    verdict = VerifierVerdict(
        grounding_score=0.30,
        per_claim=[
            ClaimVerdict(claim="claim A", grounded=False, rationale="no"),
            ClaimVerdict(claim="claim B", grounded=True, rationale="yes"),
            ClaimVerdict(claim="claim C", grounded=False, rationale="no"),
        ],
    )
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        draft_solution=_draft(),
        verifier_verdict=verdict,
    )
    new = supervisor.finalize_clarify_downgrade(state)
    assert new.resolution_kind == "clarify"
    assert new.proposed_solution is None
    assert new.preliminary_assessment is not None
    # Both ungrounded claims surface; the grounded one does not.
    assert "claim A" in new.preliminary_assessment
    assert "claim C" in new.preliminary_assessment
    assert "claim B" not in new.preliminary_assessment
    assert new.final_confidence == supervisor.CLARIFY_CONFIDENCE_PLACEHOLDER
    assert new.followup_questions == []  # downgrade path emits no questions


def test_finalize_clarify_downgrade_aggregate_disagreement() -> None:
    """Edge case: aggregate score < 0.4 but every per-claim verdict is grounded.

    The supervisor surfaces the disagreement explicitly rather than
    crashing or silently emitting an empty assessment.
    """
    verdict = VerifierVerdict(
        grounding_score=0.30,
        per_claim=[ClaimVerdict(claim="x", grounded=True, rationale="ok")],
    )
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        draft_solution=_draft(),
        verifier_verdict=verdict,
    )
    new = supervisor.finalize_clarify_downgrade(state)
    assert new.preliminary_assessment is not None
    assert "0.30" in new.preliminary_assessment
    assert "threshold" in new.preliminary_assessment


def test_finalize_clarify_downgrade_no_verdict_raises() -> None:
    state = TicketState(ticket=_ticket(), draft_solution=_draft())
    with pytest.raises(ValueError, match=r"state.verifier_verdict is None"):
        supervisor.finalize_clarify_downgrade(state)


# ---------------------------------------------------------------------------
# state_to_output


def test_state_to_output_solve_path() -> None:
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        retrievals=[_retrieval("Common_Errors.md#0", "Common_Errors.md")],
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.85),
        confidence_components={"verifier": 0.85},
    )
    finalized = supervisor.finalize_solve(state)
    out = state_to_output(finalized)
    assert isinstance(out, Output)
    assert out.resolution_kind == "solve"
    assert out.proposed_solution == "Re-activate the offline license."
    assert out.preliminary_assessment is None
    assert out.confidence == 0.85
    assert out.confidence_breakdown == {"verifier": 0.85}
    assert out.cited_sources == ["Common_Errors.md"]


def test_state_to_output_clarify_path() -> None:
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(gaps=["os"]),
        followup_questions=["Which OS?"],
    )
    finalized = supervisor.finalize_clarify(state)
    out = state_to_output(finalized)
    assert out.resolution_kind == "clarify"
    assert out.proposed_solution is None
    assert out.preliminary_assessment is not None
    assert out.followup_questions == ["Which OS?"]
    assert out.cited_sources == []  # no draft; nothing to cite


def test_state_to_output_dedupes_and_sorts_cited_sources() -> None:
    """Multiple chunks from the same source file collapse to one entry."""
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        retrievals=[
            _retrieval("Licensing_Offline_Activation.md#0", "Licensing_Offline_Activation.md"),
            _retrieval("Licensing_Offline_Activation.md#3", "Licensing_Offline_Activation.md"),
            _retrieval("Common_Errors.md#0", "Common_Errors.md"),
        ],
        draft_solution=DraftSolution(
            solution="x",
            claims=[
                ClaimEvidence(
                    claim="a",
                    chunk_id="Licensing_Offline_Activation.md#0",
                    quote="q",
                ),
                ClaimEvidence(
                    claim="b",
                    chunk_id="Licensing_Offline_Activation.md#3",
                    quote="q",
                ),
                ClaimEvidence(
                    claim="c",
                    chunk_id="Common_Errors.md#0",
                    quote="q",
                ),
            ],
        ),
        verifier_verdict=_verdict(0.9),
    )
    finalized = supervisor.finalize_solve(state)
    out = state_to_output(finalized)
    assert out.cited_sources == ["Common_Errors.md", "Licensing_Offline_Activation.md"]


def test_state_to_output_downgrade_path_omits_cited_sources() -> None:
    """The verifier explicitly rejected these claims; surfacing their source
    files as ``cited_sources`` would misrepresent them as supporting an
    outcome with no proposed_solution. Pin the choice in a test so the
    contract can't drift silently.
    """
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        retrievals=[_retrieval("Common_Errors.md#0", "Common_Errors.md")],
        draft_solution=_draft(),
        verifier_verdict=_verdict(0.30, grounded=False),
    )
    finalized = supervisor.finalize_clarify_downgrade(state)
    out = state_to_output(finalized)
    assert out.resolution_kind == "clarify"
    assert out.cited_sources == []
    assert out.proposed_solution is None


def test_state_to_output_pre_finalize_state_raises() -> None:
    """A state that hasn't reached a finalize node has no resolution_kind."""
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
    )
    with pytest.raises(ValueError, match=r"state.resolution_kind is None"):
        state_to_output(state)


def test_state_to_output_no_classification_raises() -> None:
    state = TicketState(ticket=_ticket(), resolution_kind="clarify", final_confidence=0.3)
    with pytest.raises(ValueError, match=r"state.classification is None"):
        state_to_output(state)


def test_state_to_output_no_final_confidence_raises() -> None:
    """The defensive check exists in case a future finalize-node author
    forgets to set ``final_confidence``. Pin it so the check stays live.
    """
    state = TicketState(
        ticket=_ticket(),
        classification=_classification(),
        resolution_kind="clarify",
        preliminary_assessment="x",
        # final_confidence intentionally None
    )
    with pytest.raises(ValueError, match=r"state.final_confidence is None"):
        state_to_output(state)
