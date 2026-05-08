"""Triage agent (Phase 3 — issue #19).

Input:  ``Ticket`` + current ``TicketState``.
Output: a new ``TicketState`` with ``classification`` populated and one
reasoning step appended.

The agent does two things:

1. **Deterministic metadata pass.** Walk ``ticket.metadata``; any of
   ``product`` / ``version`` / ``os`` that is ``None`` or empty becomes a
   gap in ``missing_critical_fields``. This pass is the contract; the LLM
   cannot remove a field from the gap list.
2. **One LLM call.** The model returns a structured ``Classification``
   via the ToolSpec convention (ADR-004): one tool whose ``input_schema``
   is ``Classification.model_json_schema()``; the agent reads the first
   tool call's ``arguments``. The deterministic gap list is merged with
   anything the LLM adds (e.g. ``"steps_to_reproduce"``) and de-duplicated
   in input order.

Why deterministic-then-LLM and not LLM-only: the LLM forgetting a
missing OS in 1% of runs would be silently wrong. The contract is
enforced before the LLM ever sees the request.

The system prompt opens with ``"You are the triage agent."`` — the
unique trigger for ``MockLLMClient`` per the convention in ADR-004.
"""

from __future__ import annotations

from wscad_triage.agents._state import append_step
from wscad_triage.llm import LLMClient, Message, ToolSpec
from wscad_triage.observability import get_logger
from wscad_triage.schemas import Classification, ReasoningStep, TicketState

_CRITICAL_FIELDS: tuple[str, ...] = ("product", "version", "os")

_SYSTEM_PROMPT = """You are the triage agent.

Classify the ticket into one of these categories: Licensing, Installation, \
Errors, Performance, Other.
Pick a priority: Low, Medium, High, Critical.
List any critical missing fields the support engineer would need to \
resolve the ticket. The deterministic pre-pass already added these gaps; \
you may add domain-specific gaps (e.g. "steps_to_reproduce", "log_excerpt") \
but do not remove any.

Return your decision by calling the emit_classification tool exactly once.
"""


def run(state: TicketState, llm: LLMClient) -> TicketState:
    """Classify the ticket and identify metadata gaps."""
    log = get_logger(state.ticket.ticket_id)
    log.info("triage.start")

    deterministic_gaps = _missing_critical_fields(state)

    user_message = (
        f"Ticket id: {state.ticket.ticket_id}\n"
        f"Text: {state.ticket.text}\n"
        f"Metadata: {state.ticket.metadata.model_dump_json()}\n"
        f"Deterministic gaps already detected: {deterministic_gaps}\n"
    )

    spec = ToolSpec(
        name="emit_classification",
        description="Emit the ticket classification.",
        input_schema=Classification.model_json_schema(),
    )
    response = llm.generate(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_message),
        ],
        tools=[spec],
        response_format=Classification,
    )

    if not response.tool_calls:
        raise ValueError("triage agent: LLM returned no tool calls; expected emit_classification")
    args = response.tool_calls[0].arguments
    llm_classification = Classification.model_validate(args)

    merged_gaps = _merge_gaps(deterministic_gaps, llm_classification.missing_critical_fields)
    classification = llm_classification.model_copy(update={"missing_critical_fields": merged_gaps})

    step = ReasoningStep(
        actor="triage",
        action="classified",
        rationale=classification.rationale,
        evidence_refs=[],
    )
    log.info(
        "triage.done",
        extra={
            "category": classification.category,
            "priority": classification.priority,
            "gaps": merged_gaps,
        },
    )
    return append_step(state, step, classification=classification)


def _missing_critical_fields(state: TicketState) -> list[str]:
    """Deterministic gap detection on the ticket's structured metadata."""
    gaps: list[str] = []
    for field in _CRITICAL_FIELDS:
        value = getattr(state.ticket.metadata, field, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            gaps.append(field)
    return gaps


def _merge_gaps(deterministic: list[str], llm_gaps: list[str]) -> list[str]:
    """Union, preserving deterministic-first order then LLM additions."""
    seen = set(deterministic)
    extras: list[str] = []
    for gap in llm_gaps:
        if gap not in seen and gap.strip():
            seen.add(gap)
            extras.append(gap)
    return [*deterministic, *extras]
