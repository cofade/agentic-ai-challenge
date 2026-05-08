# ADR-008: Groundedness safety gate (verifier agent) at the end of the pipeline

- **Date:** 2026-05-08
- **Status:** Accepted

## Context

The system is RAG-grounded by design (ADR-006: BM25 + multilingual
embeddings + RRF). Retrieval surfaces relevant chunks, but a retrieved
chunk being relevant does not guarantee that the LLM's drafted solution
*actually says what the chunk says*. Two distinct hallucination failure
modes exist:

1. **Trivial hallucination — fabricated citation.** The LLM cites a
   ``chunk_id`` that was never retrieved. Cheap to catch: compare
   the cited id against the set of retrieved ids.
2. **Subtle hallucination — misrepresented citation.** The LLM cites a
   real, retrieved chunk but claims a fact the chunk does not contain.
   Not catchable by id comparison; requires reading both texts.

The challenge brief's "production mindset" rubric rewards explicitly
defending against both. The system's only authoritative source is the KB
(see ``CLAUDE.md`` Architecture Principles); a solution that escapes the
pipeline misrepresenting that source is the worst-case failure mode.

Constraints:

- Architecture Principle (``CLAUDE.md``): "Knowledge base is the only
  authoritative source. Agents may not introduce facts that are not
  present in retrieved KB chunks. The verifier agent enforces this; the
  rubric penalises it."
- Acceptance gate (ROADMAP issue #23): "ungrounded-claim test ticket
  triggers `< 0.4` grounding score and forces a clarify outcome."
- Confidence formula (ADR-007 / Phase 4 #28): final confidence is
  ``min(rubric_score, verifier_score)`` — the verifier's score caps
  the system's overall confidence.

## Decision

A dedicated **verify agent** runs after the **reason agent** as the last
step before the supervisor finalises an output. Two layers of defence:

### Layer 1 — deterministic structural checks (in the reason agent)

The reason agent emits ``DraftSolution`` containing a list of
``ClaimEvidence(claim, chunk_id, quote)``. Two cheap, deterministic
defences run after the LLM call but before the verifier ever sees the
claim:

1. **(a) Fabricated-id check.** Every ``chunk_id`` MUST be present in
   ``state.retrievals``; if any aren't, the agent raises ``ValueError``.
   Pinned by ``tests/unit/test_agent_reason.py::test_rejects_fabricated_chunk_id``
   and ``test_one_fabricated_claim_among_valid_ones_still_raises``.
2. **(b) Quote-substring check.** Every ``quote`` MUST be a substring of
   the cited chunk's text (``quote in chunk.text``). A quote the chunk
   does not literally contain is a fail-fast case — no LLM needed.
   Pinned by ``test_quote_not_in_chunk_text_raises``.

Together these protect layer 2 from spending an LLM call on claims that
already fail mechanically.

### Layer 2 — LLM-as-judge groundedness verdict (the verifier agent)

For each claim in ``DraftSolution.claims``, the verifier runs an LLM
call with:

- The claim text.
- The cited chunk's full text.

The LLM emits a ``VerifierVerdict``:

- ``grounding_score: float`` in ``[0, 1]`` (Pydantic ``Field(ge=0, le=1)``).
- ``per_claim: list[ClaimVerdict]`` with one verdict (``grounded: bool`` +
  ``rationale: str``) per claim.

The verifier writes ``state.verifier_verdict`` and
``state.confidence_components["verifier"] = grounding_score``. The
supervisor (PR3) reads ``confidence_components["verifier"]`` and:

- If the score is ``< 0.4``, **downgrades** the outcome to
  ``resolution_kind="clarify"`` regardless of how confident the rubric
  was. The unverified claims appear in the reasoning trace as gaps so
  the user can see what was rejected.
- Otherwise, the score caps the final confidence via
  ``min(rubric_score, verifier_score)`` (ADR-007 / Phase 4 #28).

The 0.4 threshold is chosen so that "one of three claims ungrounded"
(score ~ 0.33) trips the gate while "one of five claims questionable"
(score ~ 0.8) does not. Phase 5's eval harness will revisit the cutoff
once labelled ungrounded-trap tickets exist.

### Why LLM-as-judge and not deterministic-only

Layer 1's substring check (``quote in chunk.text``) catches verbatim
quoting but says nothing about whether the *claim* is actually entailed
by the quote. "Error 504 indicates a problem" quotes "Error 504
indicates a licensing problem" verbatim — but the dropped word changes
the meaning. Entailment is what we want; the LLM is the cheapest
available entailment oracle.

This is a known cost: the verifier itself can hallucinate verdicts. The
mitigation is the layered design — layer 1 catches the trivial cases
without an LLM, and Phase 5's eval harness measures how often the
verifier disagrees with hand labels.

### Verifier precondition handling

The verifier hard-asserts that ``state.draft_solution`` is set;
otherwise it raises. Calling verify before reason is a supervisor
routing bug, and silently degrading would let that bug ship. When the
draft has zero claims, the verifier short-circuits without an LLM call
and returns ``VerifierVerdict(grounding_score=0.0, per_claim=[])`` —
the supervisor's ``< 0.4`` downgrade gate catches that case naturally.
Pinned by ``test_no_draft_solution_raises`` and
``test_empty_claims_score_zero_without_llm_call``.

## Consequences

- **Positive.** No solution leaves the pipeline without an explicit
  groundedness pass. The supervisor's downgrade path means an LLM that
  hallucinates confidently still produces a "clarify" outcome with the
  unverified claims surfaced — never a confidently-wrong solution.
- **Positive.** The two-layer design means the cheap structural check
  (layer 1) protects the verifier itself: a verifier that hallucinates
  a verdict on a fabricated claim cannot escape, because the reason
  agent rejects the fabricated claim before the verifier ever sees it.
- **Positive.** The ``ClaimEvidence`` triple (claim, chunk_id, quote) is
  a typed boundary artefact — easy to log, easy to render in the
  reasoning trace, easy for the eval harness to score.
- **Negative — cost.** Adds one LLM call per ticket. Mitigated by
  Anthropic prompt caching (the static system prompt + chunks are
  cache-marked); per-ticket marginal cost is the per-claim diff plus
  output. Phase 5 measures.
- **Negative — verifier hallucination.** LLM-as-judge is not infallible.
  Documented; Phase 5's eval set includes hand-labelled ungrounded-trap
  tickets that exercise this failure mode.
- **Negative — threshold tuning.** 0.4 is a defensible default but not
  empirically grounded yet. ROADMAP #28 / #34 will revisit once the
  eval set is in place.
- **Locked-in.** ``DraftSolution.claims`` and ``VerifierVerdict`` are
  boundary types. Renaming fields is a refactor across the reason and
  verify agents plus the supervisor; mitigated by ADR-002's strict
  schemas (mypy + Pydantic catch every drift).

## Cross-references

- ADR-002 — schema conventions (the bounded ``[0, 1]`` ``grounding_score``).
- ADR-004 — LLM provider abstraction (the verifier uses the same
  ``LLMClient`` Protocol as every other agent; the structured-output
  convention is the ToolSpec/JSON-schema pattern documented there).
- ADR-006 — Hybrid RAG (the retriever the verifier reads its citations
  against).
- ADR-007 (Phase 4) — final confidence = ``min(rubric_score,
  verifier_score)`` is the consumer of this ADR's decision.
