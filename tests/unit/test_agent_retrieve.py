"""Tests for the retrieve agent (issue #20).

Pinned behaviour:

- The agent rewrites the ticket text via one LLM call (system prompt
  triggers on ``"retrieve agent"``).
- The rewritten query is passed to ``HybridRetriever.retrieve(query, k)``.
- An empty / whitespace rewrite falls back to ``state.ticket.text``.
- ``state.retrievals`` is populated and one ``ReasoningStep`` is appended
  with ``actor="retrieve"``.

The HybridRetriever is faked with a stub that records the call —
exercising the agent's wiring without requiring the embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wscad_triage.agents import retrieve
from wscad_triage.llm import LLMResponse, MockLLMClient, Usage
from wscad_triage.schemas import (
    KBChunk,
    RetrievalResult,
    Ticket,
    TicketMetadata,
    TicketState,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


@dataclass
class _StubRetriever:
    """Minimal HybridRetriever stand-in.

    Returns a fixed ``results`` list and records every ``retrieve`` call so
    tests can assert on the rewritten query and ``k``.
    """

    results: list[RetrievalResult]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return self.results[:k]


def _state(text: str = "Application fails to start. Error 504 appears.") -> TicketState:
    return TicketState(
        ticket=Ticket(
            ticket_id="T-001",
            text=text,
            metadata=TicketMetadata(product="WSCAD Suite"),
        )
    )


def _fixture_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=KBChunk(
                chunk_id="Common_Errors.md#0",
                source_file="Common_Errors.md",
                text="Error 504 indicates licensing.",
            ),
            score=0.9,
            rank=1,
            retriever="rrf",
        ),
        RetrievalResult(
            chunk=KBChunk(
                chunk_id="Licensing_Offline_Activation.md#0",
                source_file="Licensing_Offline_Activation.md",
                text="Offline license re-activation steps.",
            ),
            score=0.7,
            rank=2,
            retriever="rrf",
        ),
    ]


def test_retrieves_topk_with_provenance() -> None:
    """Acceptance gate for #20: returns top-k chunks with provenance + reasoning step."""
    state = _state()
    mock_llm = MockLLMClient(
        script={"retrieve agent": _text_response("error 504 licensing offline")}
    )
    retriever_stub = _StubRetriever(results=_fixture_results())

    new_state = retrieve.run(state, mock_llm, retriever=retriever_stub, k=5)

    assert len(new_state.retrievals) == 2
    assert new_state.retrievals[0].chunk.source_file == "Common_Errors.md"
    assert new_state.retrievals[0].chunk.chunk_id == "Common_Errors.md#0"
    assert retriever_stub.calls == [("error 504 licensing offline", 5)]

    assert len(new_state.reasoning_trace) == 1
    step = new_state.reasoning_trace[0]
    assert step.actor == "retrieve"
    assert step.action == "hybrid_search"
    assert "Common_Errors.md#0" in step.evidence_refs


def test_empty_rewrite_falls_back_to_ticket_text() -> None:
    state = _state("license stopped working on offline machine")
    mock_llm = MockLLMClient(script={"retrieve agent": _text_response("   ")})
    retriever_stub = _StubRetriever(results=_fixture_results())

    new_state = retrieve.run(state, mock_llm, retriever=retriever_stub)

    # Fallback used → query is the raw ticket text.
    assert retriever_stub.calls == [("license stopped working on offline machine", 5)]
    assert "fell back to ticket text" in new_state.reasoning_trace[0].rationale


def test_default_k_is_five() -> None:
    state = _state()
    mock_llm = MockLLMClient(script={"retrieve agent": _text_response("query")})
    retriever_stub = _StubRetriever(results=_fixture_results())

    retrieve.run(state, mock_llm, retriever=retriever_stub)
    assert retriever_stub.calls[0][1] == 5


def test_reasoning_step_lists_only_top3_files() -> None:
    """Many hits — rationale only enumerates top-3 to keep the trace scannable."""
    state = _state()
    extra: list[RetrievalResult] = []
    for i in range(5):
        extra.append(
            RetrievalResult(
                chunk=KBChunk(
                    chunk_id=f"file_{i}.md#0",
                    source_file=f"file_{i}.md",
                    text=f"text {i}",
                ),
                score=0.5,
                rank=i + 1,
                retriever="rrf",
            )
        )
    mock_llm = MockLLMClient(script={"retrieve agent": _text_response("query")})
    retriever_stub = _StubRetriever(results=extra)

    new_state = retrieve.run(state, mock_llm, retriever=retriever_stub, k=5)
    rationale = new_state.reasoning_trace[0].rationale

    assert "file_0.md" in rationale
    assert "file_1.md" in rationale
    assert "file_2.md" in rationale
    # Top-3 only — file 3+ should not appear in rationale.
    assert "file_3.md" not in rationale


def test_classification_hint_passed_to_rewriter() -> None:
    """When triage has run, the rewrite prompt sees the category."""
    from wscad_triage.schemas import Classification

    state = _state()
    state = state.model_copy(
        update={
            "classification": Classification(
                category="Errors", priority="High", rationale="Error 504"
            )
        }
    )
    mock_llm = MockLLMClient(script={"retrieve agent": _text_response("rewritten")})
    retriever_stub = _StubRetriever(results=_fixture_results())

    retrieve.run(state, mock_llm, retriever=retriever_stub)

    # The mock captures (messages, tools, response_format) — verify category landed in user message.
    messages, _, _ = mock_llm.calls[0]
    user_msg_content = "\n".join(m.content for m in messages if m.role == "user")
    assert "Category hint: Errors" in user_msg_content


def test_no_classification_means_no_category_hint() -> None:
    state = _state()
    mock_llm = MockLLMClient(script={"retrieve agent": _text_response("rewritten")})
    retriever_stub = _StubRetriever(results=_fixture_results())

    retrieve.run(state, mock_llm, retriever=retriever_stub)

    messages, _, _ = mock_llm.calls[0]
    user_msg_content = "\n".join(m.content for m in messages if m.role == "user")
    assert "Category hint" not in user_msg_content


# Sanity: type-check that the stub satisfies the structural contract the agent uses.
def test_stub_retriever_is_call_compatible() -> None:
    stub = _StubRetriever(results=[])
    out: list[Any] = stub.retrieve("q", k=3)
    assert out == []
