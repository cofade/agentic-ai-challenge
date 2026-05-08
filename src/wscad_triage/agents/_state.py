"""Shared helpers for worker agents.

Kept tiny on purpose. Anything that grows beyond a one-liner gets promoted
to its own module; the goal is to centralise the
``state.model_copy(update={...})`` boilerplate without inventing a heavy
abstraction.
"""

from __future__ import annotations

from typing import Any

from wscad_triage.schemas import ReasoningStep, TicketState


def append_step(state: TicketState, step: ReasoningStep, **updates: Any) -> TicketState:
    """Return a new ``TicketState`` with one extra reasoning step.

    ``updates`` is forwarded to ``model_copy(update=...)`` for any other
    fields the agent populated. The reasoning trace is rebuilt as a fresh
    list so the original state's list is not aliased.
    """
    return state.model_copy(
        update={
            **updates,
            "reasoning_trace": [*state.reasoning_trace, step],
        }
    )
