"""Clarify agent (Phase 3 — issue #22).

Generates 2-4 specific follow-up questions tied to the gaps identified by
triage. Hard-asserts the count and that each question references a real
gap (whole-word match against the gap's canonical name OR one of its
registered synonyms); otherwise raises. The acceptance gate (ROADMAP
#22) is "no boilerplate" — questions like "Please provide more info" do
not name a gap and so are rejected.

State update: ``state.followup_questions`` is replaced; one ReasoningStep
records the gaps the questions targeted.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wscad_triage.agents._state import append_step
from wscad_triage.llm import LLMClient, Message, ToolSpec
from wscad_triage.observability import get_logger
from wscad_triage.schemas import ReasoningStep, TicketState

_MIN_QUESTIONS = 2
_MAX_QUESTIONS = 4

# Defends against degenerate questions like "OS?" (3 chars; whole-word
# matches the "os" gap, but isn't actually asking anything useful). 12
# chars admits "Which OS now?" (13) and rejects "What OS?" (8). This is
# the kind of knob that moves to config.yaml in #27.
_MIN_QUESTION_LEN = 12

# Canonical critical-field name -> additional whole-word synonyms that
# also count as a reference to that gap. Without this map the substring
# match for short canonical names like "os" spuriously matched inside
# "diagnose", "impossible", "across", etc. — boilerplate slipped through.
# Synonyms are case-insensitive whole-word matches; multi-word synonyms
# match the phrase verbatim. Single-word synonyms must be at least 3
# characters to avoid trivial collisions.
_GAP_SYNONYMS: dict[str, tuple[str, ...]] = {
    "os": ("operating system", "windows", "linux", "macos"),
    "version": ("release", "build"),
    "product": ("product line", "edition"),
    "log_excerpt": ("logs", "log file", "log files", "error log", "event log", "application log"),
    "steps_to_reproduce": (
        "actions",
        "sequence",
        "sequence of actions",
        "procedure",
        "process",
        "workflow",
        "reproduction steps",
        "what you did",
        "what happened",
        "what you performed",
        "actions you performed",
    ),
    "error_message": (
        "error text",
        "error string",
        "exception",
        "exception message",
        "error displayed",
    ),
    "license_key": (
        "license id",
        "license number",
        "activation key",
        "license code",
    ),
}

_SYSTEM_PROMPT = """You are the clarify agent.

You are given a ticket and a list of critical gaps the triage agent \
identified. Generate between 2 and 4 specific follow-up questions, each \
addressing one listed gap. Do NOT use generic phrasing like \
"please provide more information"; name the gap explicitly.

Return your questions by calling the emit_questions tool exactly once.
"""


class _ClarifyOutput(BaseModel):
    """Structured output schema; intentionally local — only this agent reads it."""

    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(min_length=_MIN_QUESTIONS, max_length=_MAX_QUESTIONS)

    @field_validator("questions", mode="before")
    @classmethod
    def _flatten_pair_format(cls, v: object) -> object:
        """Coerce gap↔question pair dicts back to plain question strings.

        The tool's ``input_schema`` declares ``questions: array of string``,
        but instruction-tuned models (observed on ``gpt-oss:120b``) sometimes
        ignore it and emit ``[{"gap": "...", "question": "..."}, ...]`` to
        make the gap-to-question mapping explicit. ``gpt-oss:20b`` follows
        the schema as written. We normalize the pair format defensively so a
        single tool call doesn't drop the whole ticket on the floor.

        Candidate question-bearing keys are tried in order ``("question",
        "q", "text", "content")`` — first string-valued match wins. The
        ``question``/``gap`` shape is the only one observed in practice;
        the remaining three are anticipatory. Dicts that match none of the
        four fall through to downstream string-list validation, which
        raises — we never silently drop or fabricate.

        After flattening, an over-long list is truncated to ``_MAX_QUESTIONS``
        rather than raising. Some local models (gpt-oss:20b observed)
        occasionally emit 5+ questions even though the system prompt and
        tool schema both cap at 4; dropping the tail is materially better
        UX than failing the whole ticket because the model gave us *more*
        information than asked for. Under-emission (fewer than
        ``_MIN_QUESTIONS``) is still surfaced as a validation error
        because that case usually indicates the model misunderstood.
        """
        if not isinstance(v, list):
            return v
        normalized: list[object] = []
        for item in v:
            if isinstance(item, dict):
                for key in ("question", "q", "text", "content"):
                    candidate = item.get(key)
                    if isinstance(candidate, str):
                        normalized.append(candidate)
                        break
                else:
                    # Unknown dict shape — let downstream string validation raise.
                    normalized.append(item)
            else:
                normalized.append(item)
        if len(normalized) > _MAX_QUESTIONS:
            normalized = normalized[:_MAX_QUESTIONS]
        return normalized


def run(state: TicketState, llm: LLMClient) -> TicketState:
    """Generate 2-4 gap-specific follow-up questions."""
    log = get_logger(state.ticket.ticket_id)

    if state.classification is None:
        raise ValueError("clarify agent: state.classification is None; triage must run first")
    gaps = state.classification.missing_critical_fields
    if not gaps:
        raise ValueError(
            "clarify agent: classification.missing_critical_fields is empty; "
            "supervisor should not route to clarify when no gaps were detected"
        )

    log.info("clarify.start", extra={"gaps": gaps})

    user_message = f"Ticket text: {state.ticket.text}\nDetected gaps: {gaps}\n"
    spec = ToolSpec(
        name="emit_questions",
        description="Emit 2-4 follow-up questions, each tied to a listed gap.",
        input_schema=_ClarifyOutput.model_json_schema(),
    )
    response = llm.generate(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=user_message),
        ],
        tools=[spec],
        response_format=_ClarifyOutput,
    )

    if not response.tool_calls:
        raise ValueError("clarify agent: LLM returned no tool calls; expected emit_questions")
    output = _ClarifyOutput.model_validate(response.tool_calls[0].arguments)

    _validate_questions(output.questions, gaps)

    step = ReasoningStep(
        actor="clarify",
        action="generated_questions",
        evidence_refs=gaps,
        rationale=f"Generated {len(output.questions)} question(s) targeting gaps {gaps}.",
    )
    log.info("clarify.done", extra={"count": len(output.questions)})
    return append_step(state, step, followup_questions=output.questions)


def _validate_questions(questions: list[str], gaps: list[str]) -> None:
    """Reject empty, too-short, or off-topic questions.

    The Pydantic ``min_length``/``max_length`` already cover the count,
    but we re-check defensively in case the agent's call site grows.

    Gap matching is whole-word against the canonical gap name or any of
    its registered synonyms (case-insensitive). This avoids the
    substring-collision trap where short canonical names like ``"os"``
    spuriously match inside ``"diagnose"``, ``"impossible"``, etc.
    """
    if not _MIN_QUESTIONS <= len(questions) <= _MAX_QUESTIONS:
        raise ValueError(
            f"clarify agent: expected {_MIN_QUESTIONS}-{_MAX_QUESTIONS} questions, "
            f"got {len(questions)}"
        )
    for question in questions:
        stripped = question.strip()
        if len(stripped) < _MIN_QUESTION_LEN:
            raise ValueError(f"clarify agent: question too short to be specific: {question!r}")
        if not _references_any_gap(stripped, gaps):
            raise ValueError(
                f"clarify agent: question does not reference any listed gap ({gaps}): {question!r}"
            )


def _references_any_gap(question: str, gaps: list[str]) -> bool:
    """Whole-word match (case-insensitive) of any gap or its synonyms.

    Snake_case canonical gap names (``log_excerpt``,
    ``steps_to_reproduce``, ...) are matched in their human-shaped form
    too — the regex tries both the literal underscore form and the
    space-separated form. This lets the LLM write natural English ("Can
    you share the log excerpt around the failure?") without the matcher
    falsely rejecting it.
    """
    q_lower = question.lower()
    for gap in gaps:
        gap_lower = gap.lower()
        registered_synonyms = _GAP_SYNONYMS.get(gap_lower, ())
        terms: list[str] = [gap_lower]
        if "_" in gap_lower:
            spaced = gap_lower.replace("_", " ")
            terms.append(spaced)
            # Individual words as fallback for natural rephrasing that splits
            # the phrase (e.g. "steps you followed to reproduce" for
            # "steps_to_reproduce"). Words shorter than 3 chars ("to", "of")
            # are excluded to avoid matching trivial tokens.
            terms.extend(w for w in spaced.split() if len(w) >= 3)
        terms.extend(registered_synonyms)
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", q_lower):
                return True
    return False
