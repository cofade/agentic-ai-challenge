"""LLMClient Protocol + MockLLMClient.

The Protocol is provider-agnostic: every agent in :mod:`wscad_triage.agents`
depends on this interface, never on a concrete backend. ``MockLLMClient``
powers the unit and integration tests deterministically — no scripted call
ever falls through silently, so logic bugs surface immediately.

Signature notes:

- ``response_format`` (a Pydantic model class, optional) signals "structured
  output expected." Anthropic does not have an equivalent native parameter;
  callers wanting structured output declare a :class:`ToolSpec` whose
  ``input_schema`` is the target model's JSON schema and force a tool call.
  See ADR-004.
- ``max_tokens`` and other vendor-specific knobs are configured on the
  backend (constructor) rather than per-call, keeping the Protocol's
  surface narrow.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from wscad_triage.llm.types import LLMResponse, Message, ToolSpec, Usage

if TYPE_CHECKING:
    from pydantic import BaseModel


@runtime_checkable
class LLMClient(Protocol):
    """Interface every backend (real or mock) implements."""

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        """Run one model call and return the structured response."""
        ...


class MockLLMClient:
    """Deterministic LLM stand-in for tests.

    The ``script`` maps a *trigger substring* to one or more
    :class:`LLMResponse` objects. When the trigger first appears in the
    concatenated content of a call's messages the next response from that
    trigger's queue is returned; subsequent calls with the same trigger
    consume the next entry. The first matching key (insertion order) wins.
    If no key matches, ``generate`` raises :class:`RuntimeError` so the
    test fails loudly rather than silently masking a routing bug.

    Per-trigger lists support multi-turn flows (e.g., supervisor → tool_use
    → tool_result → end_turn against a stable system prompt). A single
    :class:`LLMResponse` is wrapped into a one-element queue.

    Substring matching is shallow: two prompts that share a trigger word
    will both match the first registered entry. Tests must pick triggers
    that uniquely identify the calling agent — a role marker like
    ``"You are the triage agent"`` is the recommended convention. ADR-004
    documents the limitation; PR2 may replace this with prefix-matching if
    the substring approach starts producing test flakes.

    The :attr:`calls` list captures every invocation as a triple
    ``(messages, tools, response_format)`` so tests can assert
    tool-advertisement and structured-output requests, not only message
    text.
    """

    def __init__(self, script: dict[str, LLMResponse | Iterable[LLMResponse]]) -> None:
        if not script:
            raise ValueError("MockLLMClient requires a non-empty script")
        self._queues: dict[str, list[LLMResponse]] = {}
        for trigger, value in script.items():
            queue = [value] if isinstance(value, LLMResponse) else list(value)
            if not queue:
                raise ValueError(f"MockLLMClient: trigger {trigger!r} has an empty response list")
            self._queues[trigger] = queue
        self.calls: list[tuple[list[Message], list[ToolSpec] | None, type[BaseModel] | None]] = []

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), tools, response_format))
        haystack = "\n".join(m.content for m in messages)
        for trigger, queue in self._queues.items():
            if trigger in haystack:
                if not queue:
                    raise RuntimeError(f"MockLLMClient: trigger {trigger!r} script is exhausted")
                return queue.pop(0)
        raise RuntimeError(
            f"MockLLMClient: no scripted response matched. "
            f"Triggers: {list(self._queues.keys())!r}; "
            f"haystack snippet: {haystack[:200]!r}"
        )


def empty_usage() -> Usage:
    """Convenience zero-usage object for mock responses that don't care."""
    return Usage(input_tokens=0, output_tokens=0)
