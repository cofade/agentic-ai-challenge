# 6. Runtime View — ticket flow

The two canonical execution paths through the supervisor are the **resolvable** path and the **clarification-required** path. Both are described below as the supervisor's tool-call decisions, not as a hardcoded sequence — the supervisor may interleave calls (e.g., retrieve again after the verifier flags ungrounded claims).

This document is filled in fully during Phase 3 (issue #24); for now it captures the intended sequences so the supervisor's prompt can target them precisely.

## Resolvable path (high confidence)

```
1. Supervisor receives ticket → calls triage
2. Triage classifies tentatively, reports metadata completeness as adequate
3. Supervisor calls retrieve(query reformulated from ticket text)
4. Retrieve returns top-k chunks with provenance
5. Supervisor calls reason(chunks)
6. Reason drafts solution + claim-to-evidence map
7. Supervisor calls finalize → verify
8. Verify: all claims grounded → high grounding score
9. Confidence = min(rubric, verifier) ≥ 0.7 → solution emitted with brief caveats
10. Output (JSON + text) written
```

## Clarification path (low confidence)

```
1. Supervisor receives ticket → calls triage
2. Triage flags critical missing fields (e.g., OS, version)
3. Supervisor calls retrieve (still useful; returns weak hits)
4. Supervisor evaluates: rubric metadata-completeness component is low
5. Supervisor calls clarify
6. Clarify produces 2–4 targeted follow-up questions
7. Supervisor calls finalize → verify (no solution to verify; grounding score N/A)
8. Confidence < 0.5 → clarification-required output emitted (no solution)
9. Output (JSON + text) written; reasoning trace explicitly states the gaps
```

## Hallucination-trap path (verifier intervenes)

```
1..6. Same as resolvable path
7. Supervisor calls finalize → verify
8. Verify: ≥1 claim is ungrounded; grounding score drops below threshold
9. Final confidence = min(rubric, verifier) < 0.5
10. Supervisor downgrades to clarify-mode: emits a clarification request, lists the unverified claims as gaps
```

The verifier is the safety gate that closes the loop on hallucination — even if the rubric component is high, an ungrounded solution triggers clarify-mode.
