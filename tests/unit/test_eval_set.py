"""Issue #32: validate the structure and coverage of the hand-labeled eval set.

Checks that ``tickets/eval_set.json`` is well-formed, meets the minimum size
requirement, and exercises all eight planned coverage cases (seven from the
Phase 5 roadmap plus ``"release notes grounded"`` added by issue #64).
"""

from __future__ import annotations

import json
from pathlib import Path

from wscad_triage.schemas import EvalTicket

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_PATH = REPO_ROOT / "tickets" / "eval_set.json"

REQUIRED_COVERAGE_CASES: frozenset[str] = frozenset(
    {
        "resolvable EN",
        "resolvable DE",
        "clarify missing OS",
        "clarify vague crash",
        "licensing vs installation",
        "ungrounded claim trap",
        "multilingual mixed",
        "release notes grounded",
    }
)


def _load() -> list[EvalTicket]:
    raw = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    return [EvalTicket.model_validate(item) for item in raw]


def test_eval_set_parses() -> None:
    tickets = _load()
    assert tickets, "eval_set.json must not be empty"


def test_eval_set_minimum_size() -> None:
    assert len(_load()) >= 15, "eval set must contain at least 15 tickets"


def test_all_coverage_cases_represented() -> None:
    cases = {t.coverage_case for t in _load()}
    missing = REQUIRED_COVERAGE_CASES - cases
    assert not missing, f"missing coverage cases: {sorted(missing)}"


def test_balanced_clarify_distribution() -> None:
    tickets = _load()
    clarify_count = sum(1 for t in tickets if t.should_clarify)
    solve_count = len(tickets) - clarify_count
    assert clarify_count >= 5, f"need ≥5 clarify tickets, got {clarify_count}"
    assert solve_count >= 5, f"need ≥5 solve tickets, got {solve_count}"


def test_ticket_ids_are_unique() -> None:
    ids = [t.ticket_id for t in _load()]
    assert len(ids) == len(set(ids)), "duplicate ticket_id values in eval_set.json"
