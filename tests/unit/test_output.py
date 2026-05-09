"""Unit tests for json_writer and text_renderer (Phase 4 — issue #29)."""

from __future__ import annotations

import json
from pathlib import Path

from wscad_triage.output.json_writer import write_json
from wscad_triage.output.text_renderer import render_text, write_text
from wscad_triage.schemas import Output, ReasoningStep

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_output(**overrides: object) -> Output:
    base: dict[str, object] = {
        "ticket_id": "T-001",
        "category": "Errors",
        "priority": "High",
        "resolution_kind": "solve",
        "proposed_solution": "Check the license server connectivity and re-activate the offline license via the License Manager.",
        "confidence": 0.85,
        "confidence_breakdown": {
            "rubric": 1.0,
            "retrieval_quality": 1.0,
            "metadata_completeness": 1.0,
            "verifier": 0.85,
        },
        "reasoning_trace": [
            ReasoningStep(
                actor="triage",
                action="classify",
                evidence_refs=[],
                rationale="Errors category; High priority; no missing fields.",
            ),
            ReasoningStep(
                actor="retrieve",
                action="query_kb",
                evidence_refs=[
                    "kb/original/Common_Errors.md",
                    "kb/original/Licensing_Offline_Activation.md",
                ],
                rationale="Retrieved 2 chunks matching error 504 and license activation.",
            ),
            ReasoningStep(
                actor="reason",
                action="draft_solution",
                evidence_refs=["kb/original/Licensing_Offline_Activation.md"],
                rationale="Error 504 maps to an expired or missing offline license.",
            ),
            ReasoningStep(
                actor="verify",
                action="ground_claims",
                evidence_refs=[],
                rationale="All claims grounded; score=0.85.",
            ),
            ReasoningStep(
                actor="supervisor",
                action="finalize_solve",
                evidence_refs=[],
                rationale="Grounding score 0.85 >= threshold; resolved as solve.",
            ),
        ],
        "cited_sources": [
            "kb/original/Common_Errors.md",
            "kb/original/Licensing_Offline_Activation.md",
        ],
    }
    base.update(overrides)
    return Output.model_validate(base)


def _clarify_output(**overrides: object) -> Output:
    base: dict[str, object] = {
        "ticket_id": "T-002",
        "category": "Installation",
        "priority": "Medium",
        "resolution_kind": "clarify",
        "preliminary_assessment": (
            "The application crash may have multiple root causes. "
            "Missing OS and version information prevents a definitive diagnosis."
        ),
        "confidence": 0.25,
        "confidence_breakdown": {
            "rubric": 0.25,
            "retrieval_quality": 0.0,
            "metadata_completeness": 0.5,
        },
        "reasoning_trace": [
            ReasoningStep(
                actor="triage",
                action="classify",
                evidence_refs=[],
                rationale="Installation category; Medium priority; missing critical fields: os, version.",
            ),
            ReasoningStep(
                actor="clarify",
                action="generate_questions",
                evidence_refs=[],
                rationale="Generated 2 follow-up questions targeting the detected gaps.",
            ),
            ReasoningStep(
                actor="supervisor",
                action="finalize_clarify",
                evidence_refs=[],
                rationale="Missing fields [os, version] prevent retrieval; routed to clarify.",
            ),
        ],
        "followup_questions": [
            "Which operating system and version is the application installed on?",
            "Which version of WSCAD Suite was installed?",
        ],
    }
    base.update(overrides)
    return Output.model_validate(base)


# ---------------------------------------------------------------------------
# render_text — structure
# ---------------------------------------------------------------------------


def test_render_solve_contains_proposed_solution() -> None:
    out = _solve_output()
    rendered = render_text(out)
    assert "Check the license server connectivity" in rendered
    assert "Proposed Solution:" in rendered


def test_render_solve_no_preliminary_assessment() -> None:
    out = _solve_output()
    assert "Preliminary Assessment:" not in render_text(out)


def test_render_clarify_contains_status_block() -> None:
    out = _clarify_output()
    rendered = render_text(out)
    assert "Status:" in rendered
    assert "Insufficient information" in rendered
    assert "Preliminary Assessment:" in rendered


def test_render_clarify_no_proposed_solution() -> None:
    out = _clarify_output()
    assert "Proposed Solution:" not in render_text(out)


def test_render_confidence_two_decimal_places() -> None:
    out = _solve_output(confidence=0.812)
    rendered = render_text(out)
    assert "0.81" in rendered
    assert "0.812" not in rendered


def test_render_empty_followup_omits_section() -> None:
    out = _solve_output(followup_questions=[])
    assert "Follow-up Questions:" not in render_text(out)


def test_render_clarify_numbered_questions() -> None:
    out = _clarify_output(followup_questions=["Q one?", "Q two?", "Q three?"])
    rendered = render_text(out)
    assert "1. Q one?" in rendered
    assert "2. Q two?" in rendered
    assert "3. Q three?" in rendered


def test_render_reasoning_trace_actor_shown() -> None:
    out = _solve_output()
    assert "[triage]" in render_text(out)
    assert "[retrieve]" in render_text(out)


def test_render_evidence_refs_indented() -> None:
    out = _solve_output()
    rendered = render_text(out)
    assert "  - kb/original/Common_Errors.md" in rendered
    assert "  - kb/original/Licensing_Offline_Activation.md" in rendered


def test_render_downgrade_no_questions() -> None:
    out = _clarify_output(followup_questions=[])
    assert "Required Follow-up Questions:" not in render_text(out)


# ---------------------------------------------------------------------------
# write_text / write_json
# ---------------------------------------------------------------------------


def test_write_text_creates_file(tmp_path: Path) -> None:
    out = _solve_output()
    dest = tmp_path / "T-001.txt"
    write_text(out, dest)
    content = dest.read_text(encoding="utf-8")
    assert content == render_text(out)


def test_write_json_round_trips(tmp_path: Path) -> None:
    out = _solve_output()
    dest = tmp_path / "T-001.json"
    write_json(out, dest)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    restored = Output.model_validate(raw)
    assert restored == out


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_solve() -> None:
    fixture = (_FIXTURES / "render_solve.txt").read_text(encoding="utf-8")
    assert render_text(_solve_output()) == fixture


def test_snapshot_clarify() -> None:
    fixture = (_FIXTURES / "render_clarify.txt").read_text(encoding="utf-8")
    assert render_text(_clarify_output()) == fixture
