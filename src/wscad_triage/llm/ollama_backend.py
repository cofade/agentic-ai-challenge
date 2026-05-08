"""Ollama backend for the LLMClient Protocol (issue #55).

Translates the provider-agnostic :class:`Message` / :class:`ToolSpec` /
:class:`LLMResponse` types in :mod:`wscad_triage.llm.types` to and from the
Ollama Python SDK's wire format. Together with :class:`AnthropicBackend`
this is the second working backend; the Azure adapter is still a stub.

Two structural deltas from the Anthropic adapter:

1. **Tool-call IDs are synthesised.** Ollama's ``Message.ToolCall`` does
   not carry an ``id`` field — only ``function.{name, arguments}``. The
   Pydantic :class:`ToolCall` requires a non-empty string ``id``, so we
   mint a UUID4 per call. The id is opaque within a request and stable
   for the lifetime of one ``LLMResponse``.
2. **No prompt-cache support.** Ollama has no equivalent of Anthropic's
   ``cache_control`` ephemeral marker. The :attr:`Message.cache` flag is
   silently ignored, mirroring the documented stub behaviour for the
   Azure backend. ADR-004 §6 calls this out as advisory-only at the
   Protocol boundary.

``response_format``: same convention as :class:`AnthropicBackend` — the
parameter is preserved for ROADMAP fidelity but routed through the
ToolSpec/JSON-schema pattern documented in ADR-004.

Tool-use fidelity is model-dependent: ``gpt-oss:20b`` is the project's
tested choice. Smaller open-weight models can emit malformed JSON
arguments; the Pydantic round-trip in :meth:`generate` raises
:class:`pydantic.ValidationError` in that case so the failure is loud
rather than silent. See ``docs/11-risks-and-technical-debt/`` for the
broader consequence.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import ollama

from wscad_triage.llm.types import (
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

if TYPE_CHECKING:
    from ollama._types import ChatResponse
    from pydantic import BaseModel


_OLLAMA_TO_STOP_REASON: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_use": "tool_use",
    "tool_calls": "tool_use",
}
"""Map Ollama's ``done_reason`` strings onto the Protocol's ``StopReason``.

Ollama uses ``"stop"`` for normal completion and ``"length"`` when a token
cap is hit. Newer Ollama servers may return ``"tool_use"`` or
``"tool_calls"`` when the model emits tool calls; both forms are accepted
to track upstream additions without a release-coupled patch.
"""


class OllamaBackend:
    """Ollama SDK adapter implementing the :class:`LLMClient` Protocol.

    ``__init__`` is permissive — it does not connect to the server. Network
    failure surfaces on the first :meth:`generate` call, mirroring
    :class:`AnthropicBackend`. This lets unit tests construct the backend
    without a live daemon.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: float = 120.0,
    ) -> None:
        self._client = ollama.Client(host=base_url, timeout=timeout)
        self._model = model
        self._base_url = base_url

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,  # noqa: ARG002 — see module docstring
    ) -> LLMResponse:
        payload_messages = [_message_to_ollama(m) for m in messages]
        payload_tools = [_tool_to_ollama(t) for t in (tools or [])]

        response = self._client.chat(
            model=self._model,
            messages=payload_messages,
            tools=payload_tools or None,
            stream=False,
            options={"temperature": 0.0},
        )
        return _decode_response(response)


def _message_to_ollama(msg: Message) -> dict[str, Any]:
    """Translate one :class:`Message` to Ollama's chat-message dict shape.

    Ollama supports ``system`` / ``user`` / ``assistant`` / ``tool`` roles
    natively; no system-message extraction step is needed (unlike Anthropic).

    ``Message.tool_call_id`` is a precondition for ``role="tool"`` (we
    raise if it's missing — that's a caller-side contract bug regardless
    of which backend) but we deliberately do NOT propagate it onto the
    wire: Ollama's :class:`ollama._types.Message` model has no
    ``tool_call_id`` field; the SDK silently drops it. Ollama correlates
    tool results to the prior assistant turn's tool calls by **ordering**.
    For Phase 3 every worker emits exactly one tool call per turn, so
    ordering is sufficient. If a future phase ships multi-tool turns on
    Ollama, the :class:`ollama._types.Message` ``tool_name`` field is
    where the tool's name (not id) would go — see ADR-004 §6 future work.
    """
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.role == "tool" and msg.tool_call_id is None:
        raise ValueError("tool messages require tool_call_id")
    return out


def _tool_to_ollama(tool: ToolSpec) -> dict[str, Any]:
    """Translate :class:`ToolSpec` to Ollama's function-tool dict shape.

    The wire format is OpenAI-style: a ``{type, function: {name,
    description, parameters}}`` envelope. Ollama validates the parameters
    block as JSON Schema; our :attr:`ToolSpec.input_schema` already is
    one, so it passes through unchanged.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _decode_response(response: ChatResponse) -> LLMResponse:
    """Translate Ollama's :class:`ChatResponse` to :class:`LLMResponse`.

    Defensive on every numeric field (Ollama omits counters when the request
    short-circuits) and on the optional ``tool_calls`` list.
    """
    msg = response.message
    raw_calls = msg.tool_calls or []
    tool_calls: list[ToolCall] = [
        ToolCall(
            id=str(uuid.uuid4()),
            name=tc.function.name,
            # ``dict(Mapping[str, Any])`` already gives us ``dict[str, Any]``;
            # no cast needed.
            arguments=dict(tc.function.arguments),
        )
        for tc in raw_calls
    ]

    raw_reason = response.done_reason or ("tool_use" if tool_calls else "stop")
    stop_reason: StopReason = _OLLAMA_TO_STOP_REASON.get(raw_reason, "end_turn")

    usage = Usage(
        input_tokens=int(response.prompt_eval_count or 0),
        output_tokens=int(response.eval_count or 0),
    )
    return LLMResponse(
        content=msg.content or "",
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )
