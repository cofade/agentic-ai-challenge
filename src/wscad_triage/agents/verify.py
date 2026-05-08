"""Verify agent — groundedness safety gate (Phase 3 — issue #23, ADR-008).

LLM-as-judge over the reason agent's claims. Returns a ``VerifierVerdict``
with a 0-1 ``grounding_score`` and per-claim ``ClaimVerdict`` entries.
The supervisor (PR3) downgrades a "solve" outcome to "clarify" when
``grounding_score`` falls below the threshold encoded in ADR-008.

Precondition handling: the agent raises ``ValueError`` if
``state.draft_solution`` is not set — calling verify without a draft is
a supervisor routing bug, and silently short-circuiting would let that
bug ship. The empty-claims case is handled deterministically (no LLM
call, ``grounding_score=0.0``) because there is nothing for an LLM to
judge; the supervisor can still read the verdict and downgrade as
usual.

Postcondition: the LLM's ``per_claim`` list MUST contain one verdict
for every claim in the draft; a misbehaving LLM that returns a high
score with an empty ``per_claim`` array would otherwise sneak past the
supervisor's threshold gate.
"""

from __future__ import annotations

from wscad_triage.agents._state import append_step
from wscad_triage.llm import LLMClient, Message, ToolSpec
from wscad_triage.observability import get_logger
from wscad_triage.schemas import ReasoningStep, TicketState, VerifierVerdict

_SYSTEM_PROMPT = """You are the verify agent.

You are given a draft solution with citations and the cited KB chunks. \
For each claim, judge whether the cited chunk text actually supports the \
claim. Output a per-claim verdict (grounded: true/false plus a one-sentence \
rationale) and an aggregate grounding_score in [0, 1] reflecting overall \
groundedness.

Return your judgment by calling the emit_verdict tool exactly once.
"""


def run(state: TicketState, llm: LLMClient) -> TicketState:
    """Score the draft solution's groundedness, claim by claim."""
    log = get_logger(state.ticket.ticket_id)

    if state.draft_solution is None:
        raise ValueError(
            "verify agent: state.draft_solution is None; reason must run first "
            "(supervisor must not route to verify without a draft)"
        )

    if not state.draft_solution.claims:
        # No claims to judge - record an explicit zero-score verdict so the
        # supervisor's downgrade gate fires, no LLM call required.
        log.info("verify.no_claims", extra={"reason": "draft has zero claims"})
        verdict = VerifierVerdict(grounding_score=0.0, per_claim=[])
        step = ReasoningStep(
            actor="verify",
            action="judged_grounding",
            evidence_refs=[],
            rationale="Draft has zero claims; grounding_score = 0.0.",
        )
        new_components = {**state.confidence_components, "verifier": 0.0}
        return append_step(
            state,
            step,
            verifier_verdict=verdict,
            confidence_components=new_components,
        )

    log.info("verify.start", extra={"claims": len(state.draft_solution.claims)})

    user_message = _build_user_message(state)
    spec = ToolSpec(
        name="emit_verdict",
        description="Emit per-claim grounding verdicts and an aggregate score.",
        input_schema=VerifierVerdict.model_json_schema(),
    )
    response = llm.generate(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_message),
        ],
        tools=[spec],
        response_format=VerifierVerdict,
    )

    if not response.tool_calls:
        raise ValueError("verify agent: LLM returned no tool calls; expected emit_verdict")
    verdict = VerifierVerdict.model_validate(response.tool_calls[0].arguments)

    expected = len(state.draft_solution.claims)
    if len(verdict.per_claim) != expected:
        raise ValueError(
            f"verify agent: per_claim length {len(verdict.per_claim)} does not match "
            f"draft.claims length {expected}; LLM must emit one verdict per claim"
        )

    step = ReasoningStep(
        actor="verify",
        action="judged_grounding",
        evidence_refs=[c.chunk_id for c in state.draft_solution.claims],
        rationale=(
            f"Grounding score {verdict.grounding_score:.2f}; "
            f"{sum(1 for v in verdict.per_claim if v.grounded)}/"
            f"{len(verdict.per_claim)} claims grounded."
        ),
    )
    new_components = {
        **state.confidence_components,
        "verifier": verdict.grounding_score,
    }
    log.info(
        "verify.done",
        extra={
            "grounding_score": verdict.grounding_score,
            "per_claim_count": len(verdict.per_claim),
        },
    )
    return append_step(
        state,
        step,
        verifier_verdict=verdict,
        confidence_components=new_components,
    )


def _build_user_message(state: TicketState) -> str:
    """Render draft + claims + cited chunk text for the LLM judge.

    Every ``claim.chunk_id`` is guaranteed to be in ``state.retrievals`` —
    the reason agent's layer-1 (a) check raises before reaching here. We
    let the dict lookup raise on a stale state rather than carry a
    defensive fallback for an impossible case.
    """
    assert state.draft_solution is not None  # narrowed by caller
    chunk_text_by_id = {r.chunk.chunk_id: r.chunk.text for r in state.retrievals}
    claim_blocks: list[str] = []
    for c in state.draft_solution.claims:
        cited_text = chunk_text_by_id[c.chunk_id]
        claim_blocks.append(
            f"Claim: {c.claim}\n"
            f"Cited chunk_id: {c.chunk_id}\n"
            f"Quote: {c.quote}\n"
            f"Cited chunk text:\n{cited_text}\n"
        )
    claims_section = "\n---\n".join(claim_blocks)
    return (
        f"Ticket text: {state.ticket.text}\n\n"
        f"Draft solution:\n{state.draft_solution.solution}\n\n"
        f"Claims to verify:\n{claims_section}\n"
    )
