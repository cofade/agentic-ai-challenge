"""Tests for ``AnthropicBackend`` — message translation, tool wiring, decoding.

The Anthropic SDK boundary is mocked via ``monkeypatch`` so these tests run
without network access. The live integration test lives at
``tests/integration/test_anthropic_live.py`` and is gated by the
``live_api`` marker plus ``ANTHROPIC_API_KEY``.

Pinned behaviour:

- ``role="system"`` messages are extracted into the request's ``system``
  field (Anthropic's wire convention).
- ``cache=True`` propagates to a ``cache_control={"type": "ephemeral"}``
  block on whichever message carries the flag.
- ``ToolSpec`` translates verbatim to Anthropic's ``{name, description,
  input_schema}`` shape.
- ``tool_use`` content blocks decode into :class:`ToolCall` entries on the
  response; ``text`` blocks concatenate into ``content``.
- Cache-usage counters (``cache_creation_input_tokens``,
  ``cache_read_input_tokens``) are surfaced on :class:`Usage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from wscad_triage.llm import AnthropicBackend, Message, ToolSpec

# ---- Fakes that mimic the Anthropic SDK return surface --------------------


@dataclass
class _FakeText:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUse:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_creation_input_tokens: int | None = 0
    cache_read_input_tokens: int | None = 0


@dataclass
class _FakeAnthropicResponse:
    content: list[Any]
    stop_reason: str | None = "end_turn"
    usage: _FakeUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _FakeUsage()


class _FakeMessages:
    """Captures the kwargs passed to ``messages.create`` for assertions."""

    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.last_kwargs = kwargs
        return self.response


class _FakeAnthropicClient:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self.messages = _FakeMessages(response)


@pytest.fixture
def backend_with_response() -> tuple[AnthropicBackend, _FakeAnthropicClient]:
    """Build a backend whose internal SDK client is a fake."""
    response = _FakeAnthropicResponse(content=[_FakeText(text="hello")])
    fake = _FakeAnthropicClient(response)

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend._client = fake  # type: ignore[assignment]
    backend._model = "claude-test"
    backend._max_tokens = 1024
    return backend, fake


def _replace_response(fake: _FakeAnthropicClient, response: _FakeAnthropicResponse) -> None:
    fake.messages.response = response


# ---- Tests ----------------------------------------------------------------


def test_text_only_response_decodes_to_content(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, _fake = backend_with_response
    out = backend.generate([Message(role="user", content="hi")])
    assert out.content == "hello"
    assert out.tool_calls == []
    assert out.stop_reason == "end_turn"


def test_system_message_extracted_to_system_field(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    backend.generate(
        [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="hi"),
        ]
    )
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    assert kwargs["system"] == [{"type": "text", "text": "You are helpful"}]
    # Conversation only carries the non-system message.
    assert len(kwargs["messages"]) == 1
    assert kwargs["messages"][0]["role"] == "user"


def test_cache_flag_emits_ephemeral_cache_control(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    backend.generate(
        [
            Message(role="system", content="big KB context", cache=True),
            Message(role="user", content="ticket"),
        ]
    )
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    # Non-cache message must NOT have cache_control.
    user_block = kwargs["messages"][0]["content"][0]
    assert "cache_control" not in user_block


def test_toolspec_translates_to_anthropic_shape(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    spec = ToolSpec(
        name="retrieve",
        description="Search KB",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    backend.generate([Message(role="user", content="hi")], tools=[spec])
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    assert kwargs["tools"] == [
        {
            "name": "retrieve",
            "description": "Search KB",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]


def test_tool_use_blocks_decode_to_tool_calls(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeAnthropicResponse(
            content=[
                _FakeText(text="thinking..."),
                _FakeToolUse(id="t1", name="retrieve", input={"q": "license"}),
            ],
            stop_reason="tool_use",
        ),
    )
    out = backend.generate([Message(role="user", content="?")])
    assert out.content == "thinking..."
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "t1"
    assert out.tool_calls[0].name == "retrieve"
    assert out.tool_calls[0].arguments == {"q": "license"}
    assert out.stop_reason == "tool_use"


def test_cache_usage_counters_surfaced(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeAnthropicResponse(
            content=[_FakeText(text="ok")],
            usage=_FakeUsage(
                input_tokens=20,
                output_tokens=4,
                cache_creation_input_tokens=400,
                cache_read_input_tokens=100,
            ),
        ),
    )
    out = backend.generate([Message(role="user", content="x")])
    assert out.usage.input_tokens == 20
    assert out.usage.output_tokens == 4
    assert out.usage.cache_creation_input_tokens == 400
    assert out.usage.cache_read_input_tokens == 100


def test_none_cache_counters_coerced_to_zero(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeAnthropicResponse(
            content=[_FakeText(text="ok")],
            usage=_FakeUsage(
                input_tokens=1,
                output_tokens=1,
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
            ),
        ),
    )
    out = backend.generate([Message(role="user", content="x")])
    assert out.usage.cache_creation_input_tokens == 0
    assert out.usage.cache_read_input_tokens == 0


def test_unknown_stop_reason_falls_back_to_end_turn(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeAnthropicResponse(content=[_FakeText(text="ok")], stop_reason="weird"),
    )
    out = backend.generate([Message(role="user", content="x")])
    assert out.stop_reason == "end_turn"


def test_tool_role_message_emits_tool_result_block(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    backend.generate(
        [
            Message(role="user", content="hi"),
            Message(role="tool", content="result data", tool_call_id="t1"),
        ]
    )
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    tool_result = kwargs["messages"][1]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "t1"
    assert tool_result["content"][0]["content"] == "result data"


def test_tool_role_without_tool_call_id_raises(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, _fake = backend_with_response
    with pytest.raises(ValueError, match="tool_call_id"):
        backend.generate([Message(role="tool", content="result")])


def test_max_tokens_uses_constructor_value(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    backend._max_tokens = 42
    backend.generate([Message(role="user", content="x")])
    assert fake.messages.last_kwargs is not None
    assert fake.messages.last_kwargs["max_tokens"] == 42


def test_response_format_is_no_op_for_anthropic(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    """Anthropic has no native response_format equivalent; param is ignored."""
    from pydantic import BaseModel

    class _Schema(BaseModel):
        x: int

    backend, fake = backend_with_response
    backend.generate([Message(role="user", content="x")], response_format=_Schema)
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    assert "response_format" not in kwargs


def test_empty_tools_list_does_not_send_tools_kwarg(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    backend, fake = backend_with_response
    backend.generate([Message(role="user", content="x")], tools=[])
    kwargs = fake.messages.last_kwargs
    assert kwargs is not None
    assert "tools" not in kwargs


def test_pause_turn_and_refusal_stop_reasons_pass_through(
    backend_with_response: tuple[AnthropicBackend, _FakeAnthropicClient],
) -> None:
    """Newer Anthropic stop reasons must surface, not get coerced to end_turn.

    A 'refusal' reaching agents as 'end_turn' would silently misroute a
    refused request as a normal stop with empty content.
    """
    for reason in ("pause_turn", "refusal"):
        backend, fake = backend_with_response
        _replace_response(
            fake,
            _FakeAnthropicResponse(content=[_FakeText(text="")], stop_reason=reason),
        )
        out = backend.generate([Message(role="user", content="x")])
        assert out.stop_reason == reason
