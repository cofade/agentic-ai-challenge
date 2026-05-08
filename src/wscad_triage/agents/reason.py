"""Reason agent (Phase 3 — issue #21).

Drafts a solution with a claim-to-evidence map. Two cheap, deterministic
defences run before the verifier (ADR-008 layer-1):

(a) Every claim's ``chunk_id`` must appear in ``state.retrievals``;
    fabricated ids raise ``ValueError``.
(b) Every claim's ``quote`` must be a substring of the cited chunk's
    text; fabricated quotes raise ``ValueError``.

The verifier agent (issue #23) is the deeper line: it judges whether
each claim is *actually entailed* by the cited chunk. Reason rejects
"made-up chunk_id" and "made-up quote"; verify rejects "real chunk that
doesn't actually say that."

State update:

- ``state.draft_solution`` = ``DraftSolution`` produced by the LLM
  (after layer-1 validation).
- ``state.proposed_solution`` is NOT mirrored from the draft — PR3's
  renderer derives it from the supervisor's resolution_kind so the
  downgrade-to-clarify path doesn't accidentally surface a draft the
  verifier rejected.
- One ReasoningStep with ``evidence_refs`` = the cited chunk ids.
"""

from __future__ import annotations

from wscad_triage.agents._state import append_step
from wscad_triage.llm import LLMClient, Message, ToolSpec
from wscad_triage.observability import get_logger
from wscad_triage.schemas import DraftSolution, ReasoningStep, TicketState

_SYSTEM_PROMPT = """You are the reason agent.

You are given a support ticket and a list of retrieved KB chunks (each \
with chunk_id, source_file, and full text). Draft a concise, actionable \
solution.

For every factual claim in your solution, emit a ClaimEvidence object \
that cites the supporting chunk_id verbatim and quotes the exact \
substring of the chunk text that supports the claim. If the retrievals \
do not support a claim, omit it.

Return your draft by calling the emit_draft tool exactly once.
"""


def run(state: TicketState, llm: LLMClient) -> TicketState:
    """Draft a solution + claim-to-evidence map; reject fabricated chunk ids."""
    log = get_logger(state.ticket.ticket_id)
    log.info("reason.start", extra={"retrievals": len(state.retrievals)})

    user_message = _build_user_message(state)
    spec = ToolSpec(
        name="emit_draft",
        description="Emit the proposed solution with claim-to-evidence map.",
        input_schema=DraftSolution.model_json_schema(),
    )
    response = llm.generate(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_message),
        ],
        tools=[spec],
        response_format=DraftSolution,
    )

    if not response.tool_calls:
        raise ValueError("reason agent: LLM returned no tool calls; expected emit_draft")
    draft = DraftSolution.model_validate(response.tool_calls[0].arguments)

    chunk_text_by_id = {r.chunk.chunk_id: r.chunk.text for r in state.retrievals}

    fabricated = [c.chunk_id for c in draft.claims if c.chunk_id not in chunk_text_by_id]
    if fabricated:
        raise ValueError(
            f"reason agent: claims cite chunk_ids not in state.retrievals: {fabricated}"
        )

    # Layer-1 (b) of the groundedness gate per ADR-008: the cited quote must
    # appear verbatim in the cited chunk's text. Cheap, deterministic, and
    # protects the verifier (layer 2) from spending an LLM call on a quote
    # that already fails mechanically.
    bad_quotes = [
        (c.chunk_id, c.quote) for c in draft.claims if c.quote not in chunk_text_by_id[c.chunk_id]
    ]
    if bad_quotes:
        raise ValueError(
            f"reason agent: claims cite quotes that are not substrings of the "
            f"cited chunks: {bad_quotes}"
        )

    step = ReasoningStep(
        actor="reason",
        action="drafted_solution",
        evidence_refs=[c.chunk_id for c in draft.claims],
        rationale=f"Drafted with {len(draft.claims)} cited claim(s).",
    )
    log.info("reason.done", extra={"claims": len(draft.claims)})
    return append_step(state, step, draft_solution=draft)


def _build_user_message(state: TicketState) -> str:
    chunk_blocks: list[str] = []
    for r in state.retrievals:
        chunk_blocks.append(f"[{r.chunk.chunk_id} from {r.chunk.source_file}]\n{r.chunk.text}")
    chunks_section = "\n\n".join(chunk_blocks) if chunk_blocks else "(no retrievals)"
    return (
        f"Ticket id: {state.ticket.ticket_id}\n"
        f"Ticket text: {state.ticket.text}\n"
        f"Metadata: {state.ticket.metadata.model_dump_json()}\n\n"
        f"Retrieved chunks:\n{chunks_section}\n"
    )
