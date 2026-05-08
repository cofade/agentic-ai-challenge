"""Anthropic backend for the LLMClient Protocol (issue #17).

Translates the provider-agnostic :class:`Message` / :class:`ToolSpec` /
:class:`LLMResponse` types in :mod:`wscad_triage.llm.types` to and from the
Anthropic SDK's wire format. The only place in the codebase that touches the
``anthropic`` SDK is this module.

Prompt caching: any :class:`Message` with ``cache=True`` is sent with
``cache_control={"type": "ephemeral"}`` on its content block. Anthropic
caches the prefix; agents put the (large, stable) KB context behind the
cache marker and append the (small, per-request) ticket text after it.

``response_format``: Anthropic does not have a structured-output parameter
equivalent to OpenAI's ``response_format``. Callers that need structured
output must declare a :class:`ToolSpec` whose ``input_schema`` is the
target Pydantic model's JSON schema and force a tool call. ADR-004
documents the convention.

Content-block coverage is intentionally narrow today: ``text`` and
``tool_use`` only. Anthropic also emits ``thinking``, ``redacted_thinking``,
``image``, and ``document`` blocks; none are enabled by current agent
prompts. When extended thinking lands, widen :func:`_decode_response`
explicitly rather than via a catch-all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import anthropic

from wscad_triage.llm.types import (
    VALID_STOP_REASONS,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
)

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from pydantic import BaseModel


class AnthropicBackend:
    """Anthropic SDK adapter implementing the :class:`LLMClient` Protocol.

    ``max_tokens`` is a backend-level setting (constructor arg) rather than
    a per-call Protocol parameter — keeps the provider-agnostic surface
    in :mod:`wscad_triage.llm.client` from baking in vendor-specific knobs.
    Override per-call by constructing a fresh backend.
    """

    def __init__(self, *, api_key: str, model: str, max_tokens: int = 1024) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,  # noqa: ARG002 — see module docstring
    ) -> LLMResponse:
        system_blocks, conversation = _split_system(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": conversation,
        }
        if system_blocks:
            request["system"] = system_blocks
        if tools is not None and len(tools) > 0:
            request["tools"] = [_tool_to_anthropic(t) for t in tools]

        response = self._client.messages.create(**request)
        return _decode_response(response)


def _split_system(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anthropic accepts ``system`` separately from the message list."""
    system_blocks: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            block: dict[str, Any] = {"type": "text", "text": msg.content}
            if msg.cache:
                block["cache_control"] = {"type": "ephemeral"}
            system_blocks.append(block)
            continue
        conversation.append(_message_to_anthropic(msg))
    return system_blocks, conversation


def _message_to_anthropic(msg: Message) -> dict[str, Any]:
    if msg.role == "tool":
        if msg.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id,
            "content": msg.content,
        }
        if msg.cache:
            block["cache_control"] = {"type": "ephemeral"}
        return {"role": "user", "content": [block]}

    text_block: dict[str, Any] = {"type": "text", "text": msg.content}
    if msg.cache:
        text_block["cache_control"] = {"type": "ephemeral"}
    return {"role": msg.role, "content": [text_block]}


def _tool_to_anthropic(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _decode_response(response: AnthropicMessage) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=cast(dict[str, Any], block.input),
                )
            )
        # Other block types (thinking, image, document, ...) are intentionally
        # dropped; widen here when an agent prompt enables them.

    raw_stop = response.stop_reason or "end_turn"
    stop_reason: StopReason = raw_stop if raw_stop in VALID_STOP_REASONS else "end_turn"

    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
    )
    return LLMResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        usage=usage,
    )
