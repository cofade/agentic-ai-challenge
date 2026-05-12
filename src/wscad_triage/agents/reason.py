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

from typing import Any

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
    raw_claims_in = response.tool_calls[0].arguments.get("claims", []) or []
    raw_args = _normalize_draft_args(response.tool_calls[0].arguments)
    normalised_count = len(raw_args.get("claims", []))
    if normalised_count != len(raw_claims_in):
        # Visibility into the resilience path: a drop here lowers the
        # grounding score and can move a ticket from solve to clarify.
        # Logging it keeps the trace honest for anyone debugging the
        # downgrade.
        log.info(
            "reason.normalize.dropped",
            extra={"in": len(raw_claims_in), "out": normalised_count},
        )
    draft = DraftSolution.model_validate(raw_args)

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


def _normalize_draft_args(args: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion of the LLM's emit_draft arguments before validation.

    Local models (gpt-oss:20b observed) occasionally emit malformed claim
    objects: ``evidence`` instead of ``quote``, ``chunk`` instead of
    ``chunk_id``, or a claim missing the ``claim`` field entirely. Rather
    than failing the whole ticket on a single bad claim, this normaliser
    renames common-synonym keys and drops claims that still cannot pass
    the schema. The reason agent's layer-1 defences (fabricated chunk_id,
    quote-not-substring) still run on whatever survives, and the verifier
    still judges the surviving claims, so the layer-2 grounding gate is
    preserved.
    """
    if not isinstance(args, dict):
        return args
    claims = args.get("claims")
    if not isinstance(claims, list):
        return args
    normalised_claims: list[dict[str, Any]] = []
    for raw in claims:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        # Rename common synonym keys to the canonical schema fields.
        if "quote" not in item and "evidence" in item:
            item["quote"] = item.pop("evidence")
        if "quote" not in item and "supporting_text" in item:
            item["quote"] = item.pop("supporting_text")
        if "chunk_id" not in item and "chunk" in item:
            item["chunk_id"] = item.pop("chunk")
        if "chunk_id" not in item and "source_chunk" in item:
            item["chunk_id"] = item.pop("source_chunk")
        if "claim" not in item and "statement" in item:
            item["claim"] = item.pop("statement")
        if "claim" not in item and "assertion" in item:
            item["claim"] = item.pop("assertion")
        # Drop unrecognised extras so ``ClaimEvidence.extra="forbid"``
        # doesn't reject an otherwise-well-shaped claim because the LLM
        # tacked on a stray key.
        allowed = {"claim", "chunk_id", "quote"}
        item = {k: v for k, v in item.items() if k in allowed}
        # If ``claim`` is missing but the LLM gave us a substantive
        # ``quote``, fall back to using the quote as the claim. The
        # verifier still judges groundedness, and the layer-1 quote-
        # substring check still fires, so this is a recovery rather
        # than a fabrication. Empirically gpt-oss:20b often emits the
        # quote without restating the higher-level claim; treating
        # that as a single-claim citation is materially better than
        # losing the citation entirely.
        if "claim" not in item and isinstance(item.get("quote"), str) and item["quote"].strip():
            item["claim"] = item["quote"]
        # Only keep claims that have all three required fields (and
        # non-empty strings). Skipping is safer than fabricating: the
        # verifier still has the surviving claims, and a dropped bad
        # claim is at worst a missed citation.
        if all(isinstance(item.get(k), str) and item[k].strip() for k in allowed):
            normalised_claims.append(item)
    new_args = dict(args)
    new_args["claims"] = normalised_claims
    return new_args


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
