"""Human-readable text renderers for the pipeline Output.

- :func:`render_text` (Phase 4 — issue #29) mirrors the full format in
  ``tickets/Sample_Output.txt``. Used by the batch CLI and by the chat
  REPL's ``/details`` slash command.
- :func:`render_text_compact` (Phase 8 — issue #65) is a chat-friendly
  per-turn summary: header line + body + questions. No reasoning trace
  or source list (those live one ``/details`` away).
"""

from __future__ import annotations

import io
from pathlib import Path

from wscad_triage.schemas import Output, ReasoningStep

_CLARIFY_STATUS = "Insufficient information to determine root cause with acceptable confidence."


def render_text(output: Output) -> str:
    """Format *output* as a human-readable string.

    Both resolution paths share a common header (ticket ID, category,
    priority) and footer (confidence, reasoning trace, follow-up questions).
    The body section differs: ``solve`` emits "Proposed Solution:";
    ``clarify`` emits a fixed "Status:" preamble followed by
    "Preliminary Assessment:".
    """
    buf = io.StringIO()
    w = buf.write

    # --- header ---
    w(f"Ticket ID: {output.ticket_id}\n")
    w("\n")
    w(f"Category: {output.category}\n")
    w(f"Priority: {output.priority}\n")
    w("\n")

    # --- body ---
    if output.resolution_kind == "solve":
        w("Proposed Solution:\n")
        w(f"{output.proposed_solution}\n")
    else:
        w("Status:\n")
        w(f"{_CLARIFY_STATUS}\n")
        w("\n")
        w("Preliminary Assessment:\n")
        w(f"{output.preliminary_assessment}\n")

    # --- confidence ---
    w("\n")
    w("Confidence:\n")
    w(f"{output.confidence:.2f}\n")

    # --- reasoning trace ---
    w("\n")
    w("Reasoning Trace:\n")
    for step in output.reasoning_trace:
        w(_format_step(step))

    # --- cited sources (evidence_refs are chunk IDs — JSON-only; cited_sources are file-level) ---
    if output.cited_sources:
        w("\n")
        w("Cited Sources:\n")
        for src in output.cited_sources:
            w(f"- {src}\n")

    # --- follow-up questions ---
    if output.followup_questions:
        w("\n")
        if output.resolution_kind == "solve":
            w("Follow-up Questions:\n")
            for q in output.followup_questions:
                w(f"- {q}\n")
        else:
            w("Required Follow-up Questions:\n")
            for i, q in enumerate(output.followup_questions, start=1):
                w(f"{i}. {q}\n")

    return buf.getvalue()


def write_text(output: Output, path: Path) -> None:
    """Render *output* and write the result to *path* (UTF-8)."""
    path.write_text(render_text(output), encoding="utf-8")


def render_text_compact(output: Output) -> str:
    """Chat-friendly per-turn summary.

    Drops the reasoning trace and cited-source list relative to
    :func:`render_text` so the REPL stays readable when the user fires
    off five turns in a row. The same information is one ``/details``
    away (which re-renders via :func:`render_text`).
    """
    buf = io.StringIO()
    w = buf.write

    w(
        f"[Agent] Category: {output.category}  |  Priority: {output.priority}"
        f"  |  Confidence: {output.confidence:.2f}\n"
    )
    if output.resolution_kind == "solve":
        w("Proposed solution:\n")
        w(f"{output.proposed_solution}\n")
        if output.followup_questions:
            w("\nFollow-up questions:\n")
            for q in output.followup_questions:
                w(f"- {q}\n")
    else:
        w(f"Status: {_CLARIFY_STATUS}\n")
        w(f"Preliminary assessment: {output.preliminary_assessment}\n")
        if output.followup_questions:
            w("\nPlease answer one or more of:\n")
            for i, q in enumerate(output.followup_questions, start=1):
                w(f"  {i}. {q}\n")
    return buf.getvalue()


def _format_step(step: ReasoningStep) -> str:
    return f"- [{step.actor}] {step.action}: {step.rationale}\n"


__all__ = ["render_text", "render_text_compact", "write_text"]
