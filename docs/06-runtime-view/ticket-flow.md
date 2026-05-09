# 6. Runtime View — ticket flow

Three canonical execution paths through the LangGraph state machine — the **resolvable** path, the **clarification-required (missing-fields)** path, and the **hallucination-trap (downgrade)** path. Routing decisions are deterministic functions on `TicketState` (see [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md)); the static state diagram lives in [`../05-building-block-view/`](../05-building-block-view/).

Phase 4 (#27, #28) ships `confidence.py`: `final_confidence = min(rubric_score, verifier_score)` on the solve path, and `final_confidence = rubric_score` (no verifier on the clarify short-circuit). The rubric combines two components (retrieval quality and metadata completeness) with weights from `config.yaml`; ADR-007 (#31) documents the design.

## Resolvable path (high confidence)

```
1.  Pipeline.run(ticket) compiles the graph and invokes it
2.  triage runs: classifies category + priority; deterministic
    metadata-gap pass yields no gaps
3.  Conditional edge route_after_triage -> "retrieve"
4.  retrieve runs: LLM rewrites the ticket into a focused query;
    HybridRetriever returns top-k KB chunks with provenance
5.  reason runs: drafts a solution; emits one ClaimEvidence per claim;
    layer-1 (a) verifies every chunk_id is in state.retrievals;
    layer-1 (b) verifies every quote is a substring of its chunk text
6.  verify runs: LLM-as-judge per claim; aggregate grounding_score >= 0.4
7.  Conditional edge route_after_verify -> "finalize_solve"
8.  finalize_solve sets resolution_kind="solve",
    proposed_solution=draft.solution, final_confidence=min(rubric_score, verifier_score)
9.  pipeline.state_to_output projects the final TicketState onto an Output
    (cited_sources derived from claim chunk_ids' source files)
10. Output (JSON + text rendering, Phase 4 #29) written
```

## Clarification-required path (missing critical fields)

```
1. Pipeline.run(ticket) compiles the graph and invokes it
2. triage runs: classifies category + priority; deterministic
   metadata-gap pass populates missing_critical_fields (e.g. ["os"])
3. Conditional edge route_after_triage -> "clarify"
4. clarify runs: generates 2-4 follow-up questions, each addressing
   a listed gap by whole-word match (or registered synonym)
5. Direct edge clarify -> "finalize_clarify"
6. finalize_clarify sets resolution_kind="clarify",
   preliminary_assessment=<rationale + gaps>,
   final_confidence=rubric_score (no verifier on this path; retrieval_quality=0.0
   since retrieval was short-circuited, so rubric < 0.5 by construction)
7. pipeline.state_to_output emits Output with followup_questions
```

The retriever is never called on this path — saving a vector-DB round-trip on tickets we can't answer until the user supplies the missing fields. The clarify worker's precondition (non-empty `missing_critical_fields`) is preserved by the routing decision.

## Hallucination-trap path (verifier downgrade)

```
1..5. Same as resolvable path: triage (no gaps) -> retrieve -> reason
6.    verify runs: aggregate grounding_score < 0.4 (ADR-008 threshold)
7.    Conditional edge route_after_verify -> "finalize_clarify_downgrade"
8.    finalize_clarify_downgrade sets resolution_kind="clarify",
      preliminary_assessment="Could not confidently propose a solution: ...",
      followup_questions stays empty,
      final_confidence=min(rubric_score, verifier_score)
9.    pipeline.state_to_output emits the clarify-shaped Output
```

The downgrade path **does not** re-route through the clarify worker. The clarify worker's gap-matcher would reject questions about ungrounded claims (those aren't `missing_critical_fields`), and synthesising fake gaps to satisfy the matcher would obscure what actually went wrong. Instead the supervisor's `finalize_clarify_downgrade` sink synthesises a fixed-shape `preliminary_assessment` listing the verifier's ungrounded claims directly.

The verifier is the safety gate that closes the loop on hallucination — even when retrieval looks adequate, an ungrounded solution triggers clarify-mode rather than shipping a confidently-wrong answer (ADR-008).
