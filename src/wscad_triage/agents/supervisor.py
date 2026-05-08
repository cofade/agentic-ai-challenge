"""Supervisor — deterministic routing + finalize sinks (issue #24, ADR-005).

Phase 3 ships a deterministic supervisor: pure functions on ``TicketState``
encode the two real branch points (after triage, after verify). LangGraph
is the executor; conditional edges read these functions to pick the next
node. ADR-005 documents the choice.

The "agentic routing in tool-call decisions" CLAUDE.md principle is upheld
by the *workers* — each worker is itself an LLM agent that emits a
structured tool call (``emit_classification``, ``emit_draft``,
``emit_questions``, ``emit_verdict``). The supervisor is the deterministic
glue between them. A future LLM-mediated decision swaps in by replacing
the body of one routing function — the LangGraph topology stays the same.
"""

from __future__ import annotations

from typing import Literal

from wscad_triage.agents._state import append_step
from wscad_triage.schemas import ReasoningStep, TicketState

GROUNDING_THRESHOLD = 0.4
"""ADR-008 gate: groundedness scores below this force the clarify path."""

CLARIFY_CONFIDENCE_PLACEHOLDER = 0.3
"""Placeholder for clarify outcomes until #28 lands ``min(rubric, verifier)``.

Below the 0.5 threshold by construction so a clarify outcome never
masquerades as a confident solve. # TODO(#28): replace with
``min(rubric, verifier)`` per ADR-007.
"""


def route_after_triage(state: TicketState) -> Literal["clarify", "retrieve"]:
    """Critical metadata gaps -> clarify; otherwise proceed to retrieval."""
    if state.classification is None:
        raise ValueError(
            "supervisor.route_after_triage: state.classification is None; "
            "triage must run before this routing decision"
        )
    if state.classification.missing_critical_fields:
        return "clarify"
    return "retrieve"


def route_after_verify(
    state: TicketState,
) -> Literal["finalize_solve", "finalize_clarify_downgrade"]:
    """ADR-008 gate: ``grounding_score < 0.4`` forces clarify-downgrade."""
    if state.verifier_verdict is None:
        raise ValueError(
            "supervisor.route_after_verify: state.verifier_verdict is None; "
            "verify must run before this routing decision"
        )
    if state.verifier_verdict.grounding_score < GROUNDING_THRESHOLD:
        return "finalize_clarify_downgrade"
    return "finalize_solve"


def finalize_solve(state: TicketState) -> TicketState:
    """Solve sink: promote draft.solution to proposed_solution."""
    if state.draft_solution is None:
        raise ValueError(
            "supervisor.finalize_solve: state.draft_solution is None; the solve "
            "path requires reason+verify to have populated a draft"
        )
    if state.verifier_verdict is None:
        raise ValueError(
            "supervisor.finalize_solve: state.verifier_verdict is None; the "
            "solve path requires verify to have run"
        )
    score = state.verifier_verdict.grounding_score
    step = ReasoningStep(
        actor="supervisor",
        action="finalize_solve",
        evidence_refs=[c.chunk_id for c in state.draft_solution.claims],
        rationale=(
            f"grounding_score={score:.2f} >= {GROUNDING_THRESHOLD}; "
            f"emitting solution from reason agent's draft."
        ),
    )
    return append_step(
        state,
        step,
        resolution_kind="solve",
        proposed_solution=state.draft_solution.solution,
        # TODO(#28): replace with min(rubric, verifier) per ADR-007.
        final_confidence=score,
    )


def finalize_clarify(state: TicketState) -> TicketState:
    """Missing-fields sink: clarify already populated followup_questions.

    The preliminary_assessment is a short rubric-style summary of the
    classification rationale + the gaps, so the renderer (Phase 4 #29)
    has something to print regardless of resolution_kind.
    """
    if state.classification is None:
        raise ValueError(
            "supervisor.finalize_clarify: state.classification is None; "
            "the clarify path requires triage to have populated it"
        )
    gaps = state.classification.missing_critical_fields
    assessment = (
        f"{state.classification.rationale} "
        f"Awaiting clarification on: {', '.join(gaps) if gaps else '(no gaps recorded)'}."
    )
    step = ReasoningStep(
        actor="supervisor",
        action="finalize_clarify",
        evidence_refs=[],
        rationale=(
            f"Triage flagged {len(gaps)} critical gap(s); "
            f"emitting follow-up questions instead of a solution."
        ),
    )
    return append_step(
        state,
        step,
        resolution_kind="clarify",
        preliminary_assessment=assessment,
        # TODO(#28): replace with min(rubric, verifier) per ADR-007.
        final_confidence=CLARIFY_CONFIDENCE_PLACEHOLDER,
    )


def finalize_clarify_downgrade(state: TicketState) -> TicketState:
    """Ungrounded-claim sink: synthesise an assessment from verdicts.

    The clarify worker is NOT called here -- its precondition (non-empty
    ``missing_critical_fields``) does not hold on this branch, and its
    questions wouldn't make sense for a groundedness failure. Instead,
    the supervisor itself emits a fixed-shape preliminary_assessment
    listing the ungrounded claims; followup_questions stays empty.
    """
    if state.verifier_verdict is None:
        raise ValueError("supervisor.finalize_clarify_downgrade: state.verifier_verdict is None")
    ungrounded = [v.claim for v in state.verifier_verdict.per_claim if not v.grounded]
    if ungrounded:
        assessment = (
            "Could not confidently propose a solution: "
            f"these claims were not grounded in the cited KB chunks: {ungrounded}."
        )
    else:
        # Defensive branch; not expected to fire in practice. The verifier
        # LLM emits both numbers in the same call and is unlikely to
        # contradict itself (low aggregate, every per-claim grounded).
        # Phase 5 eval will probe whether this disagreement actually
        # surfaces; until then we surface it loudly rather than silently
        # default to "no claims" messaging that would suggest the wrong
        # cause.
        assessment = (
            "Could not confidently propose a solution: aggregate grounding "
            f"score {state.verifier_verdict.grounding_score:.2f} fell below "
            f"the {GROUNDING_THRESHOLD} threshold despite per-claim agreement."
        )
    step = ReasoningStep(
        actor="supervisor",
        action="finalize_clarify_downgrade",
        evidence_refs=[],
        rationale=(
            f"grounding_score={state.verifier_verdict.grounding_score:.2f} "
            f"< {GROUNDING_THRESHOLD}; downgrading to clarify."
        ),
    )
    return append_step(
        state,
        step,
        resolution_kind="clarify",
        preliminary_assessment=assessment,
        # TODO(#28): replace with min(rubric, verifier) per ADR-007.
        final_confidence=CLARIFY_CONFIDENCE_PLACEHOLDER,
    )
