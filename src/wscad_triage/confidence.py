"""Rubric-based confidence scoring (issues #27 + #28, ADR-007).

Two public functions:

* ``compute_rubric`` — weighted combination of retrieval quality and metadata
  completeness; returns a component breakdown dict plus the ``"rubric"`` total.
* ``compute_confidence`` — applies ``min(rubric, verifier)`` on the solve path
  and rubric-only on the clarify path; returns ``(final_confidence, breakdown)``.

The confidence bands that emerge from the defaults are documented in
``config.yaml``:
  < 0.5   clarify (by construction — see plan for the proof)
  0.5-0.7 solution with caveats
  > 0.7   confident solution
"""

from __future__ import annotations

from wscad_triage.config import RubricWeightsConfig
from wscad_triage.schemas import TicketState


def _retrieval_quality(state: TicketState) -> float:
    """1.0 when at least one chunk was retrieved; 0.0 on the clarify short-circuit."""
    return 1.0 if state.retrievals else 0.0


def _metadata_completeness(state: TicketState, penalty_per_field: float) -> float:
    """Linear penalty per missing critical field, clamped to [0, 1]."""
    if state.classification is None:
        return 0.0
    n_missing = len(state.classification.missing_critical_fields)
    return max(0.0, 1.0 - n_missing * penalty_per_field)


def compute_rubric(
    state: TicketState,
    weights: RubricWeightsConfig,
) -> dict[str, float]:
    """Compute rubric component scores.

    Returns a dict with keys ``retrieval_quality``, ``metadata_completeness``,
    and ``rubric`` (the weighted total, clamped to ``[0.0, 1.0]``).
    """
    rq = _retrieval_quality(state)
    mc = _metadata_completeness(state, weights.penalty_per_field)
    rubric = max(0.0, min(1.0, weights.retrieval_quality * rq + weights.metadata_completeness * mc))
    return {
        "retrieval_quality": rq,
        "metadata_completeness": mc,
        "rubric": rubric,
    }


def compute_confidence(
    state: TicketState,
    weights: RubricWeightsConfig,
) -> tuple[float, dict[str, float]]:
    """Compute final confidence and its breakdown.

    * **Solve path** (``state.confidence_components`` contains ``"verifier"``):
      ``final = min(rubric, verifier)``.  The conservative minimum prevents a
      high rubric score from masking a poorly-grounded solution.
    * **Clarify path** (no verifier score): ``final = rubric``.  The rubric is
      naturally below 0.5 on this path because ``retrieval_quality = 0.0``
      (no retrieval ran) with the default 0.5/0.5 weight split.

    Returns ``(final_confidence, breakdown)`` where ``breakdown`` includes all
    component scores plus ``"verifier"`` when present.
    """
    breakdown = compute_rubric(state, weights)
    rubric = breakdown["rubric"]

    verifier = state.confidence_components.get("verifier")
    if verifier is not None:
        final = min(rubric, verifier)
        breakdown["verifier"] = verifier
    else:
        final = rubric

    return max(0.0, min(1.0, final)), breakdown


__all__ = ["compute_confidence", "compute_rubric"]
