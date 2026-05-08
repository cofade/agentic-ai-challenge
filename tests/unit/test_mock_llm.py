"""Tests for ``MockLLMClient`` (issue #17).

Pins the contract every Phase 3 agent test relies on:

- A scripted trigger that matches anywhere in the concatenated message
  contents returns the matching response.
- An unmatched trigger raises ``RuntimeError`` — silent fallthrough is the
  bug class this client exists to surface.
- Per-trigger lists support multi-turn flows; a single response is wrapped.
- Each call captures ``(messages, tools, response_format)`` so tests can
  assert tool advertisement and structured-output requests, not only text.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from wscad_triage.llm import LLMResponse, Message, MockLLMClient, ToolSpec, Usage
from wscad_triage.llm.client import empty_usage


def _resp(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=empty_usage(),
    )


def test_substring_match_returns_scripted_response() -> None:
    mock = MockLLMClient(script={"triage": _resp("classified")})
    out = mock.generate([Message(role="user", content="please triage this")])
    assert out.content == "classified"


def test_unmatched_call_raises_runtime_error() -> None:
    mock = MockLLMClient(script={"triage": _resp("x")})
    with pytest.raises(RuntimeError, match="no scripted response matched"):
        mock.generate([Message(role="user", content="something irrelevant")])


def test_first_matching_trigger_wins() -> None:
    mock = MockLLMClient(
        script={
            "license": _resp("first"),
            "activation": _resp("second"),
        }
    )
    # "license activation" contains both triggers; insertion order picks the first.
    out = mock.generate([Message(role="user", content="license activation issue")])
    assert out.content == "first"


def test_match_against_concatenated_messages() -> None:
    mock = MockLLMClient(script={"verifier": _resp("ok")})
    out = mock.generate(
        [
            Message(role="system", content="You are the verifier agent"),
            Message(role="user", content="some claim"),
        ]
    )
    assert out.content == "ok"


def test_empty_script_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty script"):
        MockLLMClient(script={})


def test_per_trigger_list_supports_multi_turn() -> None:
    mock = MockLLMClient(script={"agent": [_resp("turn1"), _resp("turn2")]})
    msgs = [Message(role="user", content="agent")]
    assert mock.generate(msgs).content == "turn1"
    assert mock.generate(msgs).content == "turn2"


def test_exhausted_trigger_raises() -> None:
    mock = MockLLMClient(script={"agent": [_resp("only")]})
    msgs = [Message(role="user", content="agent")]
    mock.generate(msgs)
    with pytest.raises(RuntimeError, match="script is exhausted"):
        mock.generate(msgs)


def test_empty_response_list_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="empty response list"):
        MockLLMClient(script={"agent": []})


def test_calls_capture_tools_and_response_format() -> None:
    class _Schema(BaseModel):
        x: int

    spec = ToolSpec(name="t", description="d", input_schema={})
    mock = MockLLMClient(script={"x": _resp("ok")})
    msgs = [Message(role="user", content="x")]
    mock.generate(msgs, tools=[spec], response_format=_Schema)
    assert len(mock.calls) == 1
    captured_msgs, captured_tools, captured_format = mock.calls[0]
    assert captured_msgs == msgs
    assert captured_tools == [spec]
    assert captured_format is _Schema


def test_calls_capture_none_when_args_omitted() -> None:
    mock = MockLLMClient(script={"x": _resp("ok")})
    mock.generate([Message(role="user", content="x")])
    _msgs, tools, fmt = mock.calls[0]
    assert tools is None
    assert fmt is None


def test_empty_usage_helper_zeroes() -> None:
    u = empty_usage()
    assert u == Usage(input_tokens=0, output_tokens=0)
