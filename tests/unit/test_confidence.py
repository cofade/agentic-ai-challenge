"""Unit tests for confidence.py (issues #27 + #28).

Acceptance criteria verified here:
  #27: boundary cases — zero retrieval, zero metadata, full evidence.
  #28: ungrounded-claim ticket lands below 0.5; aggregation is min(rubric, verifier).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wscad_triage.confidence import compute_confidence, compute_rubric
from wscad_triage.config import AppConfig, RubricWeightsConfig, load_app_config
from wscad_triage.schemas import (
    Classification,
    KBChunk,
    RetrievalResult,
    Ticket,
    TicketMetadata,
    TicketState,
)

_DEFAULT_WEIGHTS = RubricWeightsConfig()


# ---------------------------------------------------------------------------
# State builders


def _ticket() -> Ticket:
    return Ticket(
        ticket_id="T-test",
        text="test",
        metadata=TicketMetadata(product="WSCAD Suite", version="2.3", os="Windows 11"),
    )


def _classification(missing: list[str] | None = None) -> Classification:
    return Classification(
        category="Licensing",
        priority="High",
        missing_critical_fields=missing or [],
        rationale="test",
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id="c1", source_file="f.md", text="text"),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


def _state(
    *,
    retrievals: list[RetrievalResult] | None = None,
    missing: list[str] | None = None,
    verifier_score: float | None = None,
) -> TicketState:
    components: dict[str, float] = {}
    if verifier_score is not None:
        components["verifier"] = verifier_score
    return TicketState(
        ticket=_ticket(),
        classification=_classification(missing),
        retrievals=retrievals or [],
        confidence_components=components,
    )


# ---------------------------------------------------------------------------
# compute_rubric — component scores


def test_rubric_no_retrievals_no_missing() -> None:
    """Zero retrievals, no gaps: retrieval=0.0, metadata=1.0 → rubric=0.5."""
    state = _state(retrievals=[], missing=[])
    result = compute_rubric(state, _DEFAULT_WEIGHTS)
    assert result["retrieval_quality"] == 0.0
    assert result["metadata_completeness"] == 1.0
    assert result["rubric"] == pytest.approx(0.5)


def test_rubric_full_solve_path() -> None:
    """One or more retrievals, no gaps → rubric=1.0."""
    state = _state(retrievals=[_retrieval()], missing=[])
    result = compute_rubric(state, _DEFAULT_WEIGHTS)
    assert result["retrieval_quality"] == 1.0
    assert result["metadata_completeness"] == 1.0
    assert result["rubric"] == pytest.approx(1.0)


def test_rubric_two_missing_fields() -> None:
    """Retrievals present, 2 missing: metadata = max(0, 1 - 2*0.25) = 0.5 → rubric=0.75."""
    state = _state(retrievals=[_retrieval()], missing=["version", "os"])
    result = compute_rubric(state, _DEFAULT_WEIGHTS)
    assert result["retrieval_quality"] == 1.0
    assert result["metadata_completeness"] == pytest.approx(0.5)
    assert result["rubric"] == pytest.approx(0.75)


def test_rubric_many_missing_fields_clamped() -> None:
    """Five missing fields: metadata = max(0, 1 - 5*0.25) = 0.0 → rubric=0.5 (retrieval carries it)."""
    state = _state(retrievals=[_retrieval()], missing=["a", "b", "c", "d", "e"])
    result = compute_rubric(state, _DEFAULT_WEIGHTS)
    assert result["metadata_completeness"] == pytest.approx(0.0)
    assert result["rubric"] == pytest.approx(0.5)


def test_rubric_no_retrievals_many_missing() -> None:
    """Zero retrievals + 4 missing → both components low → rubric=0.0."""
    state = _state(retrievals=[], missing=["a", "b", "c", "d"])
    result = compute_rubric(state, _DEFAULT_WEIGHTS)
    assert result["retrieval_quality"] == 0.0
    assert result["metadata_completeness"] == pytest.approx(0.0)
    assert result["rubric"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_confidence — aggregation


def test_confidence_solve_min_takes_verifier() -> None:
    """rubric=1.0, verifier=0.85 → final = min(1.0, 0.85) = 0.85."""
    state = _state(retrievals=[_retrieval()], missing=[], verifier_score=0.85)
    final, _ = compute_confidence(state, _DEFAULT_WEIGHTS)
    assert final == pytest.approx(0.85)


def test_confidence_solve_min_takes_rubric() -> None:
    """rubric is lower than verifier → final = rubric.

    Uses custom weights to produce rubric=0.55: two missing fields reduce
    metadata_completeness to 0.5; retrieval_quality=1.0 → rubric=0.75.
    Override penalty to 0.9 to get a lower rubric.
    weights: ret=0.1, meta=0.9, penalty=0.5 → ret_q=1, meta=0.5 → rubric=0.55.
    """
    weights = RubricWeightsConfig(
        retrieval_quality=0.1,
        metadata_completeness=0.9,
        penalty_per_field=0.5,
    )
    state = _state(retrievals=[_retrieval()], missing=["os"], verifier_score=0.90)
    final, breakdown = compute_confidence(state, weights)
    assert breakdown["rubric"] == pytest.approx(0.55)
    assert final == pytest.approx(0.55)


def test_confidence_clarify_no_verifier() -> None:
    """No verifier score (clarify path): final = rubric only."""
    state = _state(retrievals=[], missing=["version", "os"], verifier_score=None)
    final, breakdown = compute_confidence(state, _DEFAULT_WEIGHTS)
    # rubric = 0.5 * 0 + 0.5 * 0.5 = 0.25
    assert final == pytest.approx(0.25)
    assert "verifier" not in breakdown


def test_confidence_breakdown_keys_solve() -> None:
    """Breakdown on the solve path includes all four expected keys."""
    state = _state(retrievals=[_retrieval()], missing=[], verifier_score=0.7)
    _, breakdown = compute_confidence(state, _DEFAULT_WEIGHTS)
    assert set(breakdown.keys()) == {
        "retrieval_quality",
        "metadata_completeness",
        "rubric",
        "verifier",
    }


def test_confidence_breakdown_keys_clarify() -> None:
    """Breakdown on the clarify path has three keys; no 'verifier'."""
    state = _state(retrievals=[], missing=["os"], verifier_score=None)
    _, breakdown = compute_confidence(state, _DEFAULT_WEIGHTS)
    assert set(breakdown.keys()) == {"retrieval_quality", "metadata_completeness", "rubric"}


def test_ungrounded_lands_below_half() -> None:
    """Acceptance criterion #28: ungrounded-claim ticket must produce final < 0.5.

    rubric = 1.0 (full metadata, retrievals present); verifier = 0.30
    (below the 0.4 grounding threshold) → final = min(1.0, 0.30) = 0.30 < 0.5.
    """
    state = _state(retrievals=[_retrieval()], missing=[], verifier_score=0.30)
    final, _ = compute_confidence(state, _DEFAULT_WEIGHTS)
    assert final < 0.5


def test_confidence_boundary_verifier_zero() -> None:
    """Verifier score 0.0 → final 0.0 regardless of rubric."""
    state = _state(retrievals=[_retrieval()], missing=[], verifier_score=0.0)
    final, _ = compute_confidence(state, _DEFAULT_WEIGHTS)
    assert final == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_app_config


def test_load_config_defaults_when_missing(tmp_path: Path) -> None:
    """Missing file returns AppConfig with all defaults — no exception."""
    cfg = load_app_config(tmp_path / "nonexistent.yaml")
    assert cfg == AppConfig()


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Custom weights in YAML are loaded correctly."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "confidence": {
                    "rubric_weights": {
                        "retrieval_quality": 0.3,
                        "metadata_completeness": 0.7,
                        "penalty_per_field": 0.2,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = load_app_config(config_file)
    assert cfg.confidence.rubric_weights.retrieval_quality == pytest.approx(0.3)
    assert cfg.confidence.rubric_weights.metadata_completeness == pytest.approx(0.7)
    assert cfg.confidence.rubric_weights.penalty_per_field == pytest.approx(0.2)
    # Unspecified section uses defaults.
    assert cfg.confidence.thresholds.grounding == pytest.approx(0.4)


def test_load_config_partial_overrides(tmp_path: Path) -> None:
    """Only one weight overridden; unspecified fields take defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.dump({"confidence": {"thresholds": {"grounding": 0.5}}}),
        encoding="utf-8",
    )
    cfg = load_app_config(config_file)
    assert cfg.confidence.thresholds.grounding == pytest.approx(0.5)
    # Rubric weights still default.
    assert cfg.confidence.rubric_weights.retrieval_quality == pytest.approx(0.5)
