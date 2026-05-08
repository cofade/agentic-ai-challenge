"""Round-trip + validation tests for the LLM-layer Pydantic models.

Mirrors the discipline of ``tests/unit/test_schemas.py`` (issue #8): every
boundary type is constructed, serialised, reparsed, and asserted equal.
``extra="forbid"`` is exercised with negative cases.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import BaseModel, ValidationError

from wscad_triage.llm import (
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from wscad_triage.llm.types import VALID_STOP_REASONS, StopReason


def assert_round_trip(model: BaseModel) -> None:
    parsed = type(model).model_validate_json(model.model_dump_json())
    assert parsed == model


def test_message_round_trip_minimal() -> None:
    msg = Message(role="user", content="hello")
    assert_round_trip(msg)
    assert msg.cache is False
    assert msg.tool_call_id is None


def test_message_round_trip_with_cache_and_tool_id() -> None:
    msg = Message(role="tool", content="result", cache=True, tool_call_id="call_123")
    assert_round_trip(msg)


def test_message_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="captain", content="anything")  # type: ignore[arg-type]


def test_message_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "x", "extra": 1})


def test_toolspec_round_trip() -> None:
    spec = ToolSpec(
        name="retrieve",
        description="Search KB",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert_round_trip(spec)


def test_toolcall_round_trip() -> None:
    call = ToolCall(id="t1", name="retrieve", arguments={"q": "license"})
    assert_round_trip(call)


def test_usage_round_trip_with_cache_fields() -> None:
    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=200,
        cache_read_input_tokens=100,
    )
    assert_round_trip(usage)


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)


def test_llmresponse_round_trip_text_only() -> None:
    resp = LLMResponse(
        content="hello",
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    assert_round_trip(resp)


def test_llmresponse_round_trip_with_tool_calls() -> None:
    resp = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name="retrieve", arguments={"q": "x"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=2, output_tokens=3),
    )
    assert_round_trip(resp)


def test_valid_stop_reasons_mirrors_literal() -> None:
    """Mypy can't enforce that the constant matches the Literal — pin it here."""
    literal_values = set(typing.get_args(StopReason))
    assert literal_values == VALID_STOP_REASONS


def test_llmresponse_rejects_unknown_stop_reason() -> None:
    with pytest.raises(ValidationError):
        LLMResponse(
            content="",
            tool_calls=[],
            stop_reason="bored",  # type: ignore[arg-type]
            usage=Usage(input_tokens=0, output_tokens=0),
        )
