# 11. Risks and Technical Debt

Known risks, deliberate trade-offs, and items deferred to future iterations. Updated whenever a decision creates a known limitation.

## Known limitations

- **BM25 fake-hit vote inflation in hybrid RAG.** On an out-of-vocabulary query, `BM25` returns top-`k` chunks with `score == 0.0` in original corpus order. `HybridRetriever` folds those into RRF unfiltered, so they co-vote with any genuine embedding hits and can inflate the fused rank of an unrelated chunk. Documented in [ADR-006](../09-architecture-decisions/ADR-006-hybrid-rag.md) (negative-consequences section); pinned as a regression-watch test in `tests/unit/test_hybrid.py::test_bm25_oov_fake_hits_flow_through_unfiltered`. Mitigation (score-threshold filter on BM25 results before RRF) is deferred to issue #16 — the eval harness will quantify whether it actually regresses retrieval quality on the labelled set, and we don't want to hand-tune a threshold without that signal.

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
