# ADR-007: Confidence quantification — rubric + min-aggregation with config-driven weights

- **Date:** 2026-05-09
- **Status:** Accepted

## Context

Phase 3 shipped three finalize nodes in `supervisor.py` with hard-coded
placeholder confidence values (`final_confidence = verifier_score` on the
solve path; `0.3` on the clarify paths). The placeholder approach was
explicitly labelled `TODO(#28)` pending a principled formula.

Phase 4 requires:

1. A formula that is **observable** — every component contributing to the
   final score must be named and logged so operators can understand why a
   ticket landed in the clarify band.
2. **Config-driven weights** so the formula can be tuned from `config.yaml`
   without touching Python (Phase 5 will grid-search over weight combinations
   against the held-out eval set).
3. **Construction-time invariants** — the confidence band boundaries
   (clarify < 0.5; confident > 0.7) must hold by construction, not by a
   post-hoc clamp, so that a misconfigured weight set does not silently
   produce contradictory routing.
4. **No new routing logic** — the formula outputs a float; actual routing is
   determined by the LangGraph topology (ADR-005) and the groundedness
   threshold (ADR-008). Confidence values are informational labels.

Constraints from existing artefacts:
- The verifier's `grounding_score` already lives in
  `state.confidence_components["verifier"]` (ADR-008).
- The clarify path never calls the retriever, so `retrieval_quality` is
  structurally zero on that path.
- The clarify path always has at least one entry in
  `missing_critical_fields`, so `metadata_completeness` is always < 1.0.

## Decision

### Formula

```
retrieval_quality(state)     = 1.0 if state.retrievals else 0.0
metadata_completeness(state) = max(0.0, 1.0 − |missing_critical_fields| × penalty_per_field)
rubric_score                 = clamp(w_ret × retrieval_quality + w_meta × metadata_completeness)

# solve path (verifier ran):
final_confidence = min(rubric_score, verifier_score)

# clarify path (verifier short-circuited):
final_confidence = rubric_score
```

Implementation lives in `src/wscad_triage/confidence.py`:
`compute_rubric(state, weights) -> dict[str, float]` and
`compute_confidence(state, weights) -> tuple[float, dict[str, float]]`.
The breakdown dict (keys: `retrieval_quality`, `metadata_completeness`,
`rubric`, and optionally `verifier`) is stored in
`state.confidence_components` and passed through to `Output.confidence_breakdown`.

### Default weights (shipped in `config.yaml`)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `retrieval_quality` weight | 0.5 | Equal weight: retrieval presence is as important as field completeness |
| `metadata_completeness` weight | 0.5 | Symmetric; the two weights sum to 1.0 to keep rubric ∈ [0, 1] |
| `penalty_per_field` | 0.25 | Four missing fields → completeness = 0.0; three → 0.25 |
| `grounding` threshold | 0.4 | Documented in ADR-008 |

### Construction-time invariants

| Path | Why the confidence band holds |
|------|-------------------------------|
| `finalize_clarify` (missing fields) | ≥1 missing field + no retrievals → `rubric ≤ 0.375 < 0.5` |
| `finalize_clarify_downgrade` (verifier) | verifier < 0.4 → `final = min(rubric, verifier) < 0.4 < 0.5` |
| `finalize_solve` | rubric = 1.0 (retrievals present, no missing fields on solve path); verifier ≥ 0.4 |

### Informational thresholds

The following bands appear in documentation and `Output.confidence` rendering
but do **not** gate any routing decision:

| Confidence | Band label |
|-----------|------------|
| < 0.5 | clarify |
| 0.5 – 0.7 | solution with caveats |
| > 0.7 | confident solution |

Routing is determined entirely by the LangGraph topology (triage gaps →
clarify; verifier score < 0.4 → downgrade). These labels are surfaced to the
operator for observability only.

### Configuration loading

`config.yaml` (optional; defaults match the shipped YAML) is loaded once in
`pipeline.build_graph()` via `load_app_config()` (`src/wscad_triage/config.py`)
and the `RubricWeightsConfig` object is passed as a closure argument to each
finalize node lambda. This means:

- Config is read once per graph compilation, not once per ticket.
- The `ConfidenceThresholdsConfig.grounding` field is also injected into the
  `route_after_verify` conditional edge so the groundedness gate is
  equally config-driven.
- Tests that call finalize functions directly pass a `RubricWeightsConfig`
  fixture instead of loading `config.yaml`.

## Consequences

- **Positive.** Formula is fully observable — the `confidence_breakdown` field
  in `Output` shows every contributing component. The JSON output carries
  the full breakdown; the text renderer shows the scalar `confidence` score
  only (the breakdown is available via the JSON artefact).
- **Positive.** Config-driven — Phase 5 can grid-search over weight
  combinations by editing `config.yaml` without touching Python.
- **Positive.** Invariants hold by construction — no routing code depends on
  the confidence value, so a misconfigured weight set cannot produce a
  contradictory solve/clarify split.
- **Negative — weights are not eval-tuned.** The defaults (0.5/0.5, penalty
  0.25) are defensible but not derived from a held-out ticket set. Phase 5
  issue #34 will revisit via grid search.
- **Negative — rubric is binary on retrieval.** `retrieval_quality` is 0 or 1;
  there is no partial credit for retrieving a few low-relevance chunks versus
  many high-relevance ones. A retrieval-score-weighted variant is possible
  but adds a second tunable parameter before the eval harness exists.

## Cross-references

- ADR-005 — supervisor topology (routing decisions that consume the
  confidence value as an output label, not a gate).
- ADR-008 — groundedness gate (the `verifier_score` that caps final
  confidence on the solve path via `min(rubric, verifier)`).
- `config.yaml` / `src/wscad_triage/config.py` — the Pydantic models and
  YAML schema that expose these weights.
- `src/wscad_triage/confidence.py` — the implementation.
- `tests/unit/test_confidence.py` — 14 unit tests covering boundary cases
  and the construction-time invariants above.
