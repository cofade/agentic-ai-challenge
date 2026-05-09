"""Tests for the clarify agent (issue #22).

Pinned behaviour:

- Generates between 2 and 4 follow-up questions; counts outside this range
  raise (Pydantic ``min_length``/``max_length`` on the structured output
  + a defensive re-check).
- Each question must reference one of the gaps from
  ``classification.missing_critical_fields`` (case-insensitive whole-word
  match against the gap name, its registered synonyms, or the
  underscore->space form for snake_case gaps).
- Boilerplate / one-word "questions" raise (the ``_MIN_QUESTION_LEN``
  rule).
- Exactly one ReasoningStep with ``actor="clarify"`` is appended.
"""

from __future__ import annotations

from typing import Any

import pytest

from wscad_triage.agents import clarify
from wscad_triage.llm import LLMResponse, MockLLMClient, ToolCall, Usage
from wscad_triage.schemas import (
    Classification,
    Ticket,
    TicketMetadata,
    TicketState,
)


def _questions_response(args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name="emit_questions", arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _state_with_gaps(*gaps: str) -> TicketState:
    return TicketState(
        ticket=Ticket(
            ticket_id="T-002",
            text="License stopped working on an offline machine.",
            metadata=TicketMetadata(product="WSCAD Suite"),
        ),
        classification=Classification(
            category="Licensing",
            priority="High",
            rationale="License issue with missing metadata.",
            missing_critical_fields=list(gaps),
        ),
    )


def test_question_count_in_2_to_4() -> None:
    """Acceptance gate for #22 part 1: 2-4 questions allowed."""
    state = _state_with_gaps("version", "os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite are you running?",
                        "Which operating system and OS version is the affected machine running?",
                    ]
                }
            )
        }
    )

    new_state = clarify.run(state, mock)
    assert len(new_state.followup_questions) == 2
    assert new_state.reasoning_trace[-1].actor == "clarify"


def test_three_questions_accepted() -> None:
    state = _state_with_gaps("version", "os", "log_excerpt")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite is installed?",
                        "Which operating system and os version are you running?",
                        "Can you share the log_excerpt around the failure?",
                    ]
                }
            )
        }
    )
    new_state = clarify.run(state, mock)
    assert len(new_state.followup_questions) == 3


def test_one_question_raises_at_pydantic_layer() -> None:
    from pydantic import ValidationError

    state = _state_with_gaps("version", "os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response({"questions": ["Which version is installed?"]})
        }
    )
    with pytest.raises(ValidationError):
        clarify.run(state, mock)


def test_five_questions_raises_at_pydantic_layer() -> None:
    from pydantic import ValidationError

    state = _state_with_gaps("version", "os", "log_excerpt", "steps")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite are you running?",
                        "Which os and version is in play?",
                        "Can you share the log_excerpt?",
                        "What were the steps before the failure?",
                        "Anything else helpful here?",
                    ]
                }
            )
        }
    )
    with pytest.raises(ValidationError):
        clarify.run(state, mock)


def test_each_question_references_a_gap() -> None:
    """Acceptance gate for #22 part 2: questions must address a listed gap."""
    state = _state_with_gaps("version", "os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite is installed?",
                        "What is the favourite colour of the support engineer?",
                    ]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="does not reference any listed gap"):
        clarify.run(state, mock)


def test_boilerplate_short_question_rejected() -> None:
    state = _state_with_gaps("os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite is installed and which os is running?",
                        "OS?",  # too short
                    ]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="too short"):
        clarify.run(state, mock)


def test_no_classification_raises() -> None:
    state = TicketState(ticket=Ticket(ticket_id="T", text="x"))
    mock = MockLLMClient(script={"clarify agent": _questions_response({"questions": []})})
    with pytest.raises(ValueError, match=r"state\.classification is None"):
        clarify.run(state, mock)


def test_no_gaps_raises() -> None:
    """Supervisor mis-routes: clarify shouldn't run when triage found no gaps."""
    state = _state_with_gaps()  # no gaps
    mock = MockLLMClient(script={"clarify agent": _questions_response({"questions": []})})
    with pytest.raises(ValueError, match="missing_critical_fields is empty"):
        clarify.run(state, mock)


def test_substring_collision_does_not_count_as_referencing_a_gap() -> None:
    """Regression for the reviewer's observation: 'os' substring matches
    inside 'diagnose'/'impossible'/etc. Whole-word matching must reject those.
    """
    state = _state_with_gaps("os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which operating system is the affected machine running?",
                        "Is it impossible to diagnose this without more info?",
                    ]
                }
            )
        }
    )
    with pytest.raises(ValueError, match="does not reference any listed gap"):
        clarify.run(state, mock)


def test_snake_case_gap_matches_human_shaped_question() -> None:
    """Triage emits LLM-extra gaps in snake_case (``log_excerpt``); the LLM
    paraphrases them as natural English (``"log excerpt"``) in the
    question. The matcher must accept both forms.
    """
    state = _state_with_gaps("log_excerpt", "steps_to_reproduce")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Can you share the log excerpt around the failure?",
                        "What were the exact steps to reproduce the issue?",
                    ]
                }
            )
        }
    )
    new_state = clarify.run(state, mock)
    assert len(new_state.followup_questions) == 2


def test_split_phrase_gap_matches_natural_phrasing() -> None:
    """Regression: gpt-oss:20b writes "steps you followed to reproduce" which
    splits the exact phrase "steps to reproduce". Component-word matching
    must accept it.
    """
    state = _state_with_gaps("log_excerpt", "steps_to_reproduce")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Could you describe the exact steps you followed to reproduce the failure?",
                        "Can you share any log output from when this happened?",
                    ]
                }
            )
        }
    )
    new_state = clarify.run(state, mock)
    assert len(new_state.followup_questions) == 2


def test_synonym_counts_as_gap_reference() -> None:
    """A question that uses 'operating system' references the 'os' gap."""
    state = _state_with_gaps("os", "version")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which operating system are you on?",
                        "Which release version of WSCAD is installed?",
                    ]
                }
            )
        }
    )
    new_state = clarify.run(state, mock)
    assert len(new_state.followup_questions) == 2


def test_advertises_emit_questions_toolspec() -> None:
    """Refactor-resistance: a refactor that drops the structured-output
    tool call should fail this test."""
    state = _state_with_gaps("version", "os")
    mock = MockLLMClient(
        script={
            "clarify agent": _questions_response(
                {
                    "questions": [
                        "Which version of WSCAD Suite are you running?",
                        "Which operating system version is in play?",
                    ]
                }
            )
        }
    )
    clarify.run(state, mock)

    _messages, tools, _response_format = mock.calls[0]
    assert tools is not None and len(tools) == 1
    assert tools[0].name == "emit_questions"
    # Schema must include the questions array constraint.
    schema = tools[0].input_schema
    assert "questions" in schema["properties"]


def test_no_tool_call_raises() -> None:
    state = _state_with_gaps("version")
    mock = MockLLMClient(
        script={
            "clarify agent": LLMResponse(
                content="text",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        }
    )
    with pytest.raises(ValueError, match="no tool calls"):
        clarify.run(state, mock)
