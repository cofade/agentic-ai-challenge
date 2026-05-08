"""Tests for the reason agent (issue #21).

Pinned behaviour:

- Drafts a ``DraftSolution`` via the ``emit_draft`` tool call.
- Every claim's ``chunk_id`` MUST be present in ``state.retrievals``;
  fabricated ids raise ``ValueError``. This is the trivial-hallucination
  guard (ADR-008 layer-1 (a)).
- Every claim's ``quote`` MUST be a substring of the cited chunk's text;
  otherwise ``ValueError``. Cheap deterministic check that protects the
  verifier from spending an LLM call on a fail-fast case (ADR-008
  layer-1 (b)).
- ``state.draft_solution`` is populated; ``state.proposed_solution`` is
  NOT mirrored from the draft — the renderer (PR3) decides whether to
  surface the solution based on the supervisor's resolution_kind.
- Exactly one ReasoningStep with ``actor="reason"`` is appended.
- The agent advertises the ``emit_draft`` ToolSpec to the LLM.
"""

from __future__ import annotations

from typing import Any

import pytest

from wscad_triage.agents import reason
from wscad_triage.llm import LLMResponse, MockLLMClient, ToolCall, Usage
from wscad_triage.schemas import (
    DraftSolution,
    KBChunk,
    RetrievalResult,
    Ticket,
    TicketMetadata,
    TicketState,
)


def _draft_response(args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name="emit_draft", arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _retrieval(chunk_id: str, source_file: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id=chunk_id, source_file=source_file, text=text),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


def _state_with_retrievals() -> TicketState:
    return TicketState(
        ticket=Ticket(
            ticket_id="T-001",
            text="Application fails to start. Error 504 appears.",
            metadata=TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11"),
        ),
        retrievals=[
            _retrieval(
                "Common_Errors.md#0",
                "Common_Errors.md",
                "Error 504 indicates a licensing problem.",
            ),
            _retrieval(
                "Licensing_Offline_Activation.md#0",
                "Licensing_Offline_Activation.md",
                "Re-activate the offline license via License Manager.",
            ),
        ],
    )


def test_drafts_solution_with_grounded_claims() -> None:
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": _draft_response(
                {
                    "solution": "Re-activate the offline license via License Manager.",
                    "claims": [
                        {
                            "claim": "Error 504 indicates a licensing problem.",
                            "chunk_id": "Common_Errors.md#0",
                            "quote": "Error 504 indicates a licensing problem.",
                        },
                        {
                            "claim": "Use License Manager to re-activate offline.",
                            "chunk_id": "Licensing_Offline_Activation.md#0",
                            "quote": "Re-activate the offline license via License Manager.",
                        },
                    ],
                }
            )
        }
    )

    new_state = reason.run(state, mock)

    assert new_state.draft_solution is not None
    assert len(new_state.draft_solution.claims) == 2
    # PR2 deliberately does NOT mirror draft.solution into proposed_solution —
    # PR3's renderer derives that from resolution_kind.
    assert new_state.proposed_solution is None
    assert len(new_state.reasoning_trace) == 1
    assert new_state.reasoning_trace[0].actor == "reason"
    assert "Common_Errors.md#0" in new_state.reasoning_trace[0].evidence_refs


def test_rejects_fabricated_chunk_id() -> None:
    """Acceptance gate for #21: every claim must map to a retrieved chunk_id."""
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": _draft_response(
                {
                    "solution": "x",
                    "claims": [
                        {
                            "claim": "Made up.",
                            "chunk_id": "does_not_exist.md#0",
                            "quote": "...",
                        },
                    ],
                }
            )
        }
    )
    with pytest.raises(ValueError, match=r"not in state\.retrievals"):
        reason.run(state, mock)


def test_empty_claims_allowed() -> None:
    """Reason can produce a solution with zero claims (verifier scores it 0)."""
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={"reason agent": _draft_response({"solution": "Generic guidance.", "claims": []})}
    )

    new_state = reason.run(state, mock)
    assert new_state.draft_solution is not None
    assert new_state.draft_solution.claims == []
    assert new_state.proposed_solution is None


def test_no_tool_call_raises() -> None:
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": LLMResponse(
                content="text only",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        }
    )
    with pytest.raises(ValueError, match="no tool calls"):
        reason.run(state, mock)


def test_user_message_contains_chunk_text() -> None:
    """Sanity: the LLM is given full chunk text, not just metadata."""
    state = _state_with_retrievals()
    mock = MockLLMClient(script={"reason agent": _draft_response({"solution": "x", "claims": []})})

    reason.run(state, mock)

    messages, _, _ = mock.calls[0]
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    assert "Error 504 indicates a licensing problem." in user_text
    assert "Common_Errors.md#0" in user_text


def test_quote_not_in_chunk_text_raises() -> None:
    """ADR-008 layer-1 (b): the cited quote must appear verbatim in the cited chunk."""
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": _draft_response(
                {
                    "solution": "x",
                    "claims": [
                        {
                            "claim": "Error 504 indicates the user should reboot.",
                            "chunk_id": "Common_Errors.md#0",
                            # The chunk text is "Error 504 indicates a licensing problem."
                            # No "reboot" anywhere; the quote is fabricated.
                            "quote": "reboot the machine to clear the error",
                        }
                    ],
                }
            )
        }
    )
    with pytest.raises(ValueError, match=r"not substrings of the cited chunks"):
        reason.run(state, mock)


def test_quote_substring_check_accepts_partial_quotes() -> None:
    """A quote shorter than the chunk is fine, as long as it's verbatim contained."""
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": _draft_response(
                {
                    "solution": "Re-activate the offline license.",
                    "claims": [
                        {
                            "claim": "Error 504 is licensing.",
                            "chunk_id": "Common_Errors.md#0",
                            "quote": "indicates a licensing problem",  # partial substring
                        }
                    ],
                }
            )
        }
    )
    new_state = reason.run(state, mock)
    assert new_state.draft_solution is not None


def test_advertises_emit_draft_toolspec() -> None:
    """A refactor that drops the structured-output tool call should fail this test."""
    state = _state_with_retrievals()
    mock = MockLLMClient(script={"reason agent": _draft_response({"solution": "x", "claims": []})})

    reason.run(state, mock)

    _messages, tools, response_format = mock.calls[0]
    assert tools is not None and len(tools) == 1
    assert tools[0].name == "emit_draft"
    assert tools[0].input_schema == DraftSolution.model_json_schema()
    assert response_format is DraftSolution


def test_one_fabricated_claim_among_valid_ones_still_raises() -> None:
    state = _state_with_retrievals()
    mock = MockLLMClient(
        script={
            "reason agent": _draft_response(
                {
                    "solution": "x",
                    "claims": [
                        {
                            "claim": "real",
                            "chunk_id": "Common_Errors.md#0",
                            "quote": "Error 504 indicates a licensing problem.",
                        },
                        {
                            "claim": "fake",
                            "chunk_id": "ghost.md#0",
                            "quote": "...",
                        },
                    ],
                }
            )
        }
    )
    with pytest.raises(ValueError, match=r"\['ghost\.md#0'\]"):
        reason.run(state, mock)
