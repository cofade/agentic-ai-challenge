"""Human-readable text renderer for the pipeline Output (Phase 4 — issue #29).

Mirrors the format shown in ``tickets/Sample_Output.txt``.
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


def _format_step(step: ReasoningStep) -> str:
    lines = [f"- [{step.actor}] {step.action}: {step.rationale}\n"]
    for ref in step.evidence_refs:
        lines.append(f"  - {ref}\n")
    return "".join(lines)


__all__ = ["render_text", "write_text"]
