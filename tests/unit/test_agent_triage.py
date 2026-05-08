"""Tests for the triage agent (issue #19).

Pinned behaviour:

- Deterministic metadata pass detects missing critical fields regardless
  of what the LLM returns.
- The LLM's classification is parsed via the ``emit_classification`` tool
  call; missing tool calls raise.
- The agent appends exactly one ReasoningStep with ``actor="triage"``.
- The two provided tickets (``tickets/tickets.json``) classify
  consistently across calls (acceptance gate for #19).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wscad_triage.agents import triage
from wscad_triage.llm import LLMResponse, MockLLMClient, ToolCall, Usage
from wscad_triage.schemas import Ticket, TicketMetadata, TicketState

REPO_ROOT = Path(__file__).resolve().parents[2]


def _classification_response(args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name="emit_classification", arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _state(ticket_id: str, text: str, **meta: str | None) -> TicketState:
    return TicketState(
        ticket=Ticket(
            ticket_id=ticket_id,
            text=text,
            metadata=TicketMetadata(**meta),
        )
    )


def test_classifies_with_full_metadata_no_gaps() -> None:
    state = _state(
        "T-001",
        "Application fails to start after update. Error 504 appears.",
        product="WSCAD Suite",
        version="2.3",
        os="Windows 11",
    )
    mock = MockLLMClient(
        script={
            "triage agent": _classification_response(
                {
                    "category": "Errors",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "Error 504 with full metadata.",
                }
            )
        }
    )

    new_state = triage.run(state, mock)

    assert new_state.classification is not None
    assert new_state.classification.category == "Errors"
    assert new_state.classification.priority == "High"
    assert new_state.classification.missing_critical_fields == []
    assert len(new_state.reasoning_trace) == 1
    assert new_state.reasoning_trace[0].actor == "triage"


def test_deterministic_pass_detects_missing_metadata() -> None:
    """LLM cannot remove gaps the deterministic pass found."""
    state = _state("T-002", "License stopped working on an offline machine.", product="WSCAD Suite")
    mock = MockLLMClient(
        script={
            "triage agent": _classification_response(
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": [],  # LLM tries to drop them
                    "rationale": "Offline license issue.",
                }
            )
        }
    )

    new_state = triage.run(state, mock)

    assert new_state.classification is not None
    # Deterministic gaps are preserved even when LLM omits them.
    gaps = new_state.classification.missing_critical_fields
    assert "version" in gaps
    assert "os" in gaps


def test_llm_extra_gaps_merged_after_deterministic_ones() -> None:
    state = _state("T-005", "App crashes silently.", product="WSCAD Suite")
    mock = MockLLMClient(
        script={
            "triage agent": _classification_response(
                {
                    "category": "Errors",
                    "priority": "Medium",
                    "missing_critical_fields": ["steps_to_reproduce", "log_excerpt"],
                    "rationale": "Vague crash; need repro + logs.",
                }
            )
        }
    )

    new_state = triage.run(state, mock)
    assert new_state.classification is not None

    gaps = new_state.classification.missing_critical_fields
    # Deterministic gaps come first, in their declaration order.
    assert gaps[:2] == ["version", "os"]
    # LLM extras follow.
    assert gaps[2:] == ["steps_to_reproduce", "log_excerpt"]


def test_no_tool_call_raises() -> None:
    state = _state("T-001", "ok", product="WSCAD Suite", version="2.3", os="Windows 11")
    mock = MockLLMClient(
        script={
            "triage agent": LLMResponse(
                content="here is some text",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        }
    )
    with pytest.raises(ValueError, match="no tool calls"):
        triage.run(state, mock)


def test_classification_validation_propagates() -> None:
    """Bad LLM args (e.g. unknown category) must not be silently swallowed."""
    from pydantic import ValidationError

    state = _state("T-001", "ok", product="WSCAD Suite", version="2.3", os="Windows 11")
    mock = MockLLMClient(
        script={
            "triage agent": _classification_response(
                {
                    "category": "TotallyMadeUp",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "x",
                }
            )
        }
    )
    with pytest.raises(ValidationError):
        triage.run(state, mock)


def test_classifies_provided_tickets_against_expected() -> None:
    """Acceptance gate for #19: each provided ticket produces the expected Classification.

    Asserts against an explicit ``expected`` table rather than re-running
    the same mocked call twice — equality of two identical mock outputs
    is a tautology, not a consistency check. With a deterministic mock,
    the meaningful contract is "given this ticket and this scripted
    response, the agent's *post-processing* (deterministic gap pass +
    LLM-extras merge) produces the expected final Classification."
    """
    tickets = json.loads((REPO_ROOT / "tickets" / "tickets.json").read_text())
    assert len(tickets) == 2, "tickets.json must hold the two provided tickets"

    # T-001 has full metadata; T-002 is missing version + os, so triage's
    # deterministic pass adds those gaps regardless of the LLM response.
    expected = {
        "T-001": {
            "input": {
                "category": "Errors",
                "priority": "High",
                "missing_critical_fields": [],
                "rationale": "Error 504 with full metadata.",
            },
            "want_category": "Errors",
            "want_priority": "High",
            "want_gaps": [],
        },
        "T-002": {
            "input": {
                "category": "Licensing",
                "priority": "High",
                "missing_critical_fields": [],
                "rationale": "Offline license issue.",
            },
            "want_category": "Licensing",
            "want_priority": "High",
            "want_gaps": ["version", "os"],  # deterministic gap pass adds these
        },
    }

    for entry in tickets:
        ticket = Ticket.model_validate(entry)
        state = TicketState(ticket=ticket)
        spec = expected[ticket.ticket_id]
        mock = MockLLMClient(
            script={"triage agent": _classification_response(spec["input"])}  # type: ignore[arg-type]
        )

        new_state = triage.run(state, mock)

        assert new_state.classification is not None
        assert new_state.classification.category == spec["want_category"]
        assert new_state.classification.priority == spec["want_priority"]
        assert new_state.classification.missing_critical_fields == spec["want_gaps"]


def test_advertises_emit_classification_toolspec() -> None:
    """Refactor-resistance: a refactor that drops the structured-output
    tool call should fail this test."""
    from wscad_triage.schemas import Classification

    state = _state("T-001", "ok", product="WSCAD Suite", version="2.3", os="Windows 11")
    mock = MockLLMClient(
        script={
            "triage agent": _classification_response(
                {
                    "category": "Errors",
                    "priority": "Low",
                    "missing_critical_fields": [],
                    "rationale": "ok",
                }
            )
        }
    )

    triage.run(state, mock)

    _messages, tools, response_format = mock.calls[0]
    assert tools is not None and len(tools) == 1
    assert tools[0].name == "emit_classification"
    assert tools[0].input_schema == Classification.model_json_schema()
    assert response_format is Classification
