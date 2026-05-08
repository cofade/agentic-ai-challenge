"""Retrieve agent (Phase 3 — issue #20).

Two-step worker:

1. **Query rewrite (LLM).** One text response. The agent reformulates the
   raw ticket text — plus the classification hints from triage — into a
   focused KB search query (5-15 words, no PII). Empty/whitespace
   fallback to ``state.ticket.text``.
2. **Hybrid retrieval.** ``HybridRetriever.retrieve(query, k)`` returns
   the top-k chunks with provenance.

State update: ``state.retrievals`` is replaced (the retrieve agent is the
only writer); one ReasoningStep records the rewritten query, k, and the
top-3 source files.
"""

from __future__ import annotations

from wscad_triage.agents._state import append_step
from wscad_triage.kb import HybridRetriever
from wscad_triage.llm import LLMClient, Message
from wscad_triage.observability import get_logger
from wscad_triage.schemas import ReasoningStep, TicketState

_DEFAULT_K = 5
_SYSTEM_PROMPT = """You are the retrieve agent.

Rewrite the ticket as a focused KB search query, 5 to 15 words, with no \
personally identifying information. Output the query as plain text only — \
no preamble, no quotes, no JSON.
"""


def run(
    state: TicketState,
    llm: LLMClient,
    *,
    retriever: HybridRetriever,
    k: int = _DEFAULT_K,
) -> TicketState:
    """Rewrite the ticket into a KB query and run hybrid retrieval."""
    log = get_logger(state.ticket.ticket_id)
    log.info("retrieve.start", extra={"k": k})

    rewrite_input = _build_user_message(state)
    rewrite_response = llm.generate(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=rewrite_input),
        ],
    )
    query = rewrite_response.content.strip()
    used_fallback = False
    if not query:
        query = state.ticket.text
        used_fallback = True

    results = retriever.retrieve(query, k=k)
    top_files = [r.chunk.source_file for r in results[:3]]

    rationale = f"Query: {query!r}; k={k}; top_files={top_files}" + (
        " (rewrite empty; fell back to ticket text)" if used_fallback else ""
    )
    step = ReasoningStep(
        actor="retrieve",
        action="hybrid_search",
        evidence_refs=[r.chunk.chunk_id for r in results],
        rationale=rationale,
    )
    log.info(
        "retrieve.done",
        extra={"hits": len(results), "top_files": top_files, "fallback": used_fallback},
    )
    return append_step(state, step, retrievals=results)


def _build_user_message(state: TicketState) -> str:
    parts = [f"Ticket text: {state.ticket.text}"]
    if state.classification is not None:
        parts.append(f"Category hint: {state.classification.category}")
    parts.append(f"Metadata: {state.ticket.metadata.model_dump_json()}")
    return "\n".join(parts)
