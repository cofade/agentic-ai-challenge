"""Provider-agnostic data models for the LLM layer.

Every value crossing the :class:`LLMClient` boundary is one of these models —
``Message`` in, ``LLMResponse`` out. Backends translate to/from provider-native
formats internally; nothing else in the codebase imports the Anthropic or
OpenAI SDKs.

ADR-002 conventions: closed ``Literal`` enums, ``extra="forbid"``, bounded
numeric fields. Mirrors the schema discipline of :mod:`wscad_triage.schemas`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One message in an LLM conversation.

    ``cache=True`` marks this message as a prompt-caching breakpoint for
    backends that support it (Anthropic). Backends without caching ignore the
    flag — it is advisory, not a contract.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    cache: bool = False
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    """Declaration of a tool the LLM may call.

    ``input_schema`` is a JSON schema (dict). The schema's shape is the
    backend's responsibility; we do not validate it here because every
    provider's tool API accepts arbitrary JSON-schema fragments.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """One tool invocation requested by the LLM in its response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    """Token-usage accounting for a single ``generate`` call.

    Cache fields are zero on backends without prompt caching.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)


StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "pause_turn",
]

# Runtime-introspectable mirror of ``StopReason``. Backends use this to
# coerce unrecognised SDK literals (newer Anthropic versions may add stop
# reasons) without silently accepting them. Keep in sync with the Literal
# above; mypy does not enforce that, so a unit test asserts equality.
VALID_STOP_REASONS: frozenset[StopReason] = frozenset(
    {"end_turn", "tool_use", "max_tokens", "stop_sequence", "refusal", "pause_turn"}
)


class LLMResponse(BaseModel):
    """The result of a single ``LLMClient.generate`` call.

    ``content`` may be empty when ``stop_reason == "tool_use"``: the model's
    decision was to call tools, and the textual portion is whatever
    accompanies the tool calls (often nothing).
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: StopReason
    usage: Usage
