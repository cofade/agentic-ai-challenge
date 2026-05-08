"""Tests for ``OllamaBackend`` — message translation, tool wiring, decoding.

The ``ollama`` SDK boundary is mocked via ``monkeypatch`` so these tests
run without network access. The live integration test lives at
``tests/integration/test_ollama_live.py`` and is gated by the ``live_api``
marker plus a reachability + model-presence two-stage check.

Pinned behaviour (mirrors ``test_anthropic_backend.py`` shape):

- Roles pass through verbatim — Ollama supports ``system`` natively, no
  extraction step needed.
- ``cache=True`` is silently ignored (Ollama has no prompt-cache equivalent).
  Pinned because regressing this to "raise" would break agents that set
  the flag for the Anthropic path.
- :class:`ToolSpec` translates to Ollama's OpenAI-style ``{type: function,
  function: {name, description, parameters}}`` envelope.
- Tool-call response items get a synthesised UUID id (the ``ollama`` SDK
  doesn't emit one); arguments pass through as a plain dict.
- ``done_reason`` mapping: ``stop`` → ``end_turn``, ``length`` →
  ``max_tokens``, ``tool_use``/``tool_calls`` → ``tool_use``, unknown →
  ``end_turn``.
- ``None``-valued token counters coerce to zero (Ollama omits them on
  short-circuit responses).
- Constructor is permissive: no network call at ``__init__`` time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from wscad_triage.llm import Message, OllamaBackend, ToolSpec

# ---- Fakes that mimic the ollama SDK return surface ------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class _FakeOllamaToolCall:
    function: _FakeFunction


@dataclass
class _FakeOllamaMessage:
    content: str = ""
    tool_calls: list[_FakeOllamaToolCall] | None = None
    role: str = "assistant"


@dataclass
class _FakeChatResponse:
    """Subset of ollama.ChatResponse — only the fields ``_decode_response`` reads."""

    message: _FakeOllamaMessage = field(default_factory=_FakeOllamaMessage)
    done_reason: str | None = "stop"
    prompt_eval_count: int | None = 10
    eval_count: int | None = 5


class _FakeOllamaClient:
    """Captures the kwargs passed to ``client.chat`` for assertions."""

    def __init__(self, response: _FakeChatResponse) -> None:
        self.response = response
        self.last_kwargs: dict[str, Any] | None = None

    def chat(self, **kwargs: Any) -> _FakeChatResponse:
        self.last_kwargs = kwargs
        return self.response


@pytest.fixture
def backend_with_response() -> tuple[OllamaBackend, _FakeOllamaClient]:
    """Build an OllamaBackend whose internal SDK client is a fake."""
    response = _FakeChatResponse(message=_FakeOllamaMessage(content="hello"))
    fake = _FakeOllamaClient(response)

    backend = OllamaBackend.__new__(OllamaBackend)
    backend._client = fake  # type: ignore[assignment]
    backend._model = "gpt-oss:20b"
    backend._base_url = "http://localhost:11434"
    return backend, fake


def _replace_response(fake: _FakeOllamaClient, response: _FakeChatResponse) -> None:
    fake.response = response


# ---- Tests -----------------------------------------------------------------


def test_text_response_round_trip(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    backend, _fake = backend_with_response
    out = backend.generate([Message(role="user", content="hi")])
    assert out.content == "hello"
    assert out.tool_calls == []
    assert out.stop_reason == "end_turn"
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5


def test_system_role_passes_through_without_extraction(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    backend, fake = backend_with_response
    backend.generate(
        [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="hi"),
        ]
    )
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hi"},
    ]


def test_message_cache_flag_silently_ignored(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Ollama has no prompt-cache equivalent; the flag must not appear in the wire payload."""
    backend, fake = backend_with_response
    backend.generate([Message(role="system", content="big context", cache=True)])
    kwargs = fake.last_kwargs
    assert kwargs is not None
    msg = kwargs["messages"][0]
    assert "cache" not in msg
    assert "cache_control" not in msg


def test_toolspec_translates_to_openai_function_shape(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    backend, fake = backend_with_response
    spec = ToolSpec(
        name="emit_classification",
        description="Emit the ticket classification.",
        input_schema={"type": "object", "properties": {"category": {"type": "string"}}},
    )
    backend.generate([Message(role="user", content="classify")], tools=[spec])
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "emit_classification",
                "description": "Emit the ticket classification.",
                "parameters": {"type": "object", "properties": {"category": {"type": "string"}}},
            },
        }
    ]


def test_tool_call_extracted_from_response(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeChatResponse(
            message=_FakeOllamaMessage(
                content="",
                tool_calls=[
                    _FakeOllamaToolCall(
                        function=_FakeFunction(
                            name="emit_classification",
                            arguments={"category": "Licensing", "priority": "High"},
                        )
                    )
                ],
            ),
            done_reason="tool_calls",
        ),
    )
    out = backend.generate([Message(role="user", content="?")])
    assert out.content == ""
    assert len(out.tool_calls) == 1
    tc = out.tool_calls[0]
    assert tc.name == "emit_classification"
    assert tc.arguments == {"category": "Licensing", "priority": "High"}
    assert tc.id  # synthesised UUID, non-empty
    assert out.stop_reason == "tool_use"


def test_multiple_tool_calls_each_get_unique_uuid(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Two tool calls in one response must NOT share an id."""
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeChatResponse(
            message=_FakeOllamaMessage(
                content="",
                tool_calls=[
                    _FakeOllamaToolCall(function=_FakeFunction(name="a", arguments={})),
                    _FakeOllamaToolCall(function=_FakeFunction(name="b", arguments={})),
                ],
            ),
            done_reason="tool_calls",
        ),
    )
    out = backend.generate([Message(role="user", content="?")])
    ids = [tc.id for tc in out.tool_calls]
    assert len(ids) == 2
    assert ids[0] != ids[1]
    assert all(ids)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_use", "tool_use"),
        ("tool_calls", "tool_use"),
        ("weird-future-value", "end_turn"),
    ],
)
def test_done_reason_mapping(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
    raw: str,
    expected: str,
) -> None:
    """Ollama's ``done_reason`` strings map onto the Protocol's ``StopReason``.

    Parametrized so each case shows up as a distinct entry in CI when one
    regresses.
    """
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeChatResponse(message=_FakeOllamaMessage(content="ok"), done_reason=raw),
    )
    out = backend.generate([Message(role="user", content="x")])
    assert out.stop_reason == expected


def test_zero_token_counts_when_response_omits_them(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Ollama omits eval counters on short-circuited responses; coerce to 0."""
    backend, fake = backend_with_response
    _replace_response(
        fake,
        _FakeChatResponse(
            message=_FakeOllamaMessage(content="ok"),
            prompt_eval_count=None,
            eval_count=None,
        ),
    )
    out = backend.generate([Message(role="user", content="x")])
    assert out.usage.input_tokens == 0
    assert out.usage.output_tokens == 0


def test_tool_role_requires_tool_call_id(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    backend, _fake = backend_with_response
    with pytest.raises(ValueError, match="tool_call_id"):
        backend.generate([Message(role="tool", content="result")])


def test_tool_role_does_not_propagate_tool_call_id_to_payload(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Ollama's :class:`ollama._types.Message` model has no ``tool_call_id``
    field — the SDK silently drops it on serialisation. Pin that we don't
    pretend to propagate something the SDK won't carry; correlation is by
    ordering on this backend. See the ``_message_to_ollama`` docstring +
    ADR-004 §6.
    """
    backend, fake = backend_with_response
    backend.generate(
        [
            Message(role="user", content="hi"),
            Message(role="tool", content="result", tool_call_id="abc"),
        ]
    )
    kwargs = fake.last_kwargs
    assert kwargs is not None
    tool_msg = kwargs["messages"][1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "result"
    # Crucially: tool_call_id is NOT on the wire payload.
    assert "tool_call_id" not in tool_msg


def test_response_format_is_no_op_for_ollama(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Same as Anthropic — response_format is honoured via the ToolSpec convention."""
    from pydantic import BaseModel

    class _Schema(BaseModel):
        x: int

    backend, fake = backend_with_response
    backend.generate([Message(role="user", content="x")], response_format=_Schema)
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert "format" not in kwargs  # we never forward it; stays on the Protocol


def test_empty_tools_list_sends_none_not_empty_list(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """``tools=[]`` and ``tools=None`` both send ``tools=None`` to ollama.

    Ollama's SDK rejects an empty list at validation time on some server
    versions; ``None`` is the safe default for "no tools advertised."
    """
    backend, fake = backend_with_response
    backend.generate([Message(role="user", content="x")], tools=[])
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert kwargs["tools"] is None


def test_options_pin_temperature_to_zero(
    backend_with_response: tuple[OllamaBackend, _FakeOllamaClient],
) -> None:
    """Reproducibility: every request goes out with ``temperature=0.0``."""
    backend, fake = backend_with_response
    backend.generate([Message(role="user", content="x")])
    kwargs = fake.last_kwargs
    assert kwargs is not None
    assert kwargs["options"] == {"temperature": 0.0}


def test_constructor_does_not_perform_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror AnthropicBackend: __init__ must be permissive.

    A failing import or a probe call at construction would block every
    unit test from running without a live Ollama server.
    """

    def _no_request(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("OllamaBackend.__init__ must not make HTTP calls")

    # Patch the underlying httpx send call so any HTTP I/O during __init__
    # would loudly fail. URL parsing happens during construction (cheap,
    # local) but no socket should open.
    import httpx

    monkeypatch.setattr(httpx.Client, "send", _no_request, raising=False)
    backend = OllamaBackend(base_url="http://localhost:1", model="gpt-oss:20b")
    assert backend._model == "gpt-oss:20b"
