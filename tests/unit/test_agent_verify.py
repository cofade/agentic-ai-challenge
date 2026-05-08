"""Tests for the verify agent (issue #23, ADR-008).

Pinned behaviour:

- Returns a ``VerifierVerdict`` with a 0-1 ``grounding_score`` and one
  ``ClaimVerdict`` per claim.
- ``state.verifier_verdict`` is set; ``state.confidence_components["verifier"]``
  records the score.
- When ``state.draft_solution is None`` the agent **raises** (supervisor
  routing bug; better to fail loudly than degrade silently).
- When ``state.draft_solution.claims`` is empty the agent records a
  zero-score verdict deterministically and does NOT call the LLM.
- Out-of-range scores raise via the schema's ``Field(ge=0, le=1)``.
- Exactly one ReasoningStep with ``actor="verify"`` is appended.
- The agent advertises the ``emit_verdict`` ToolSpec to the LLM.
"""

from __future__ import annotations

from typing import Any

import pytest

from wscad_triage.agents import verify
from wscad_triage.llm import LLMResponse, MockLLMClient, ToolCall, Usage
from wscad_triage.schemas import (
    ClaimEvidence,
    DraftSolution,
    KBChunk,
    RetrievalResult,
    Ticket,
    TicketMetadata,
    TicketState,
)


def _verdict_response(args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name="emit_verdict", arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _retrieval(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id=chunk_id, source_file=chunk_id.split("#")[0], text=text),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


def _state_with_draft() -> TicketState:
    return TicketState(
        ticket=Ticket(
            ticket_id="T-001",
            text="App fails to start.",
            metadata=TicketMetadata(),
        ),
        retrievals=[
            _retrieval("Common_Errors.md#0", "Error 504 indicates a licensing problem."),
        ],
        draft_solution=DraftSolution(
            solution="Re-activate the offline license.",
            claims=[
                ClaimEvidence(
                    claim="Error 504 is a licensing problem.",
                    chunk_id="Common_Errors.md#0",
                    quote="Error 504 indicates a licensing problem.",
                )
            ],
        ),
    )


def test_grounding_score_in_unit_interval() -> None:
    """Acceptance gate for #23 part 1: schema enforces 0 ≤ grounding_score ≤ 1."""
    state = _state_with_draft()
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.85,
                    "per_claim": [
                        {
                            "claim": "Error 504 is a licensing problem.",
                            "grounded": True,
                            "rationale": "Verbatim match against the cited chunk.",
                        }
                    ],
                }
            )
        }
    )

    new_state = verify.run(state, mock)
    assert new_state.verifier_verdict is not None
    assert 0.0 <= new_state.verifier_verdict.grounding_score <= 1.0
    assert new_state.confidence_components["verifier"] == 0.85
    assert len(new_state.reasoning_trace) == 1
    assert new_state.reasoning_trace[0].actor == "verify"


def test_per_claim_verdicts_match_draft_claims() -> None:
    """Acceptance gate for #23 part 2: one verdict per claim."""
    state = _state_with_draft()
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.9,
                    "per_claim": [
                        {
                            "claim": "Error 504 is a licensing problem.",
                            "grounded": True,
                            "rationale": "Cited chunk states this.",
                        }
                    ],
                }
            )
        }
    )

    new_state = verify.run(state, mock)
    assert new_state.verifier_verdict is not None
    assert len(new_state.verifier_verdict.per_claim) == len(state.draft_solution.claims)


def test_ungrounded_claim_low_score() -> None:
    """A 0.30 score is the canonical ungrounded-trap signal for the supervisor."""
    state = _state_with_draft()
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.30,
                    "per_claim": [
                        {
                            "claim": "Error 504 is a licensing problem.",
                            "grounded": False,
                            "rationale": "Cited chunk does not actually say this.",
                        }
                    ],
                }
            )
        }
    )

    new_state = verify.run(state, mock)
    assert new_state.verifier_verdict is not None
    assert new_state.verifier_verdict.grounding_score == 0.30
    assert new_state.verifier_verdict.per_claim[0].grounded is False
    assert new_state.confidence_components["verifier"] == 0.30


def test_score_above_one_raises() -> None:
    from pydantic import ValidationError

    state = _state_with_draft()
    mock = MockLLMClient(
        script={"verify agent": _verdict_response({"grounding_score": 1.5, "per_claim": []})}
    )
    with pytest.raises(ValidationError):
        verify.run(state, mock)


def test_no_draft_solution_raises() -> None:
    """Hard-assert precondition: supervisor must not route here without a draft."""
    state = TicketState(ticket=Ticket(ticket_id="T", text="x"))
    mock = MockLLMClient(
        script={"verify agent": _verdict_response({"grounding_score": 0.0, "per_claim": []})}
    )

    with pytest.raises(ValueError, match=r"draft_solution is None"):
        verify.run(state, mock)


def test_empty_claims_score_zero_without_llm_call() -> None:
    """A draft with zero claims gets grounding_score=0.0 deterministically."""
    state = TicketState(
        ticket=Ticket(ticket_id="T", text="x"),
        draft_solution=DraftSolution(solution="empty-claims draft", claims=[]),
    )
    mock = MockLLMClient(
        script={"verify agent": _verdict_response({"grounding_score": 0.7, "per_claim": []})}
    )

    new_state = verify.run(state, mock)

    assert new_state.verifier_verdict is not None
    assert new_state.verifier_verdict.grounding_score == 0.0
    assert new_state.verifier_verdict.per_claim == []
    assert new_state.confidence_components["verifier"] == 0.0
    assert new_state.reasoning_trace[-1].actor == "verify"
    # No LLM call; the empty-claims branch is deterministic.
    assert mock.calls == []


def test_per_claim_length_must_match_draft_claims() -> None:
    """A misbehaving LLM returning a high score with empty per_claim must NOT
    sneak past the supervisor's threshold gate. The agent enforces 1:1.
    """
    state = _state_with_draft()  # draft has exactly one claim
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {"grounding_score": 1.0, "per_claim": []}  # length mismatch
            )
        }
    )
    with pytest.raises(ValueError, match=r"per_claim length 0 does not match"):
        verify.run(state, mock)


def test_per_claim_too_many_verdicts_also_raises() -> None:
    """Symmetry: extra verdicts for non-existent claims also fail the 1:1 check."""
    state = _state_with_draft()  # draft has exactly one claim
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.9,
                    "per_claim": [
                        {"claim": "real", "grounded": True, "rationale": "ok"},
                        {"claim": "ghost", "grounded": True, "rationale": "made up"},
                    ],
                }
            )
        }
    )
    with pytest.raises(ValueError, match=r"per_claim length 2 does not match"):
        verify.run(state, mock)


def test_advertises_emit_verdict_toolspec() -> None:
    """Refactor-resistance: a refactor that drops the structured-output
    tool call should fail this test."""
    from wscad_triage.schemas import VerifierVerdict

    state = _state_with_draft()
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.7,
                    "per_claim": [
                        {
                            "claim": "Error 504 is a licensing problem.",
                            "grounded": True,
                            "rationale": "matches",
                        }
                    ],
                }
            )
        }
    )

    verify.run(state, mock)

    _messages, tools, response_format = mock.calls[0]
    assert tools is not None and len(tools) == 1
    assert tools[0].name == "emit_verdict"
    assert tools[0].input_schema == VerifierVerdict.model_json_schema()
    assert response_format is VerifierVerdict


def test_no_tool_call_raises() -> None:
    state = _state_with_draft()
    mock = MockLLMClient(
        script={
            "verify agent": LLMResponse(
                content="text",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        }
    )
    with pytest.raises(ValueError, match="no tool calls"):
        verify.run(state, mock)


def test_existing_confidence_components_preserved() -> None:
    state = _state_with_draft().model_copy(
        update={"confidence_components": {"retrieval_strength": 0.9}}
    )
    mock = MockLLMClient(
        script={
            "verify agent": _verdict_response(
                {
                    "grounding_score": 0.7,
                    "per_claim": [
                        {
                            "claim": "Error 504 is a licensing problem.",
                            "grounded": True,
                            "rationale": "matches",
                        }
                    ],
                }
            )
        }
    )

    new_state = verify.run(state, mock)
    assert new_state.confidence_components == {
        "retrieval_strength": 0.9,
        "verifier": 0.7,
    }
