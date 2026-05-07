# 11. Risks and Technical Debt

Known risks, deliberate trade-offs, and items deferred to future iterations. Updated whenever a decision creates a known limitation.

## Known limitations

(Empty initially. Filled in as Phase 1+ work surfaces concrete trade-offs.)

## Deliberate non-goals (already decided)

- No UI / frontend (challenge brief: "Not Required").
- No model fine-tuning (brief: "Not Required").
- No external data sources at runtime — KB only.
- No coverage of legacy WSCAD product lines (SUITE, ELECTRIX non-AI, ELECTRIX ROCKET).
- No C# / .NET implementation (see ADR-001).

## Future work

- An evaluation pass with a held-out ticket set of 100+ items would let the rubric weights be tuned by gradient-free optimisation rather than by hand.
- Per-claim caching of grounding verdicts would reduce verifier latency on repeated KB chunks.
- A first-class production integration path (HTTP service or `Python.NET` host) is sketched in the README but not implemented.
