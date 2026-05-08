# 11. Risks and Technical Debt

Known risks, deliberate trade-offs, and items deferred to future iterations. Updated whenever a decision creates a known limitation.

## Known limitations

- **BM25 fake-hit vote inflation in hybrid RAG.** On an out-of-vocabulary query, `BM25` returns top-`k` chunks with `score == 0.0` in original corpus order. `HybridRetriever` folds those into RRF unfiltered, so they co-vote with any genuine embedding hits and can inflate the fused rank of an unrelated chunk. Documented in [ADR-006](../09-architecture-decisions/ADR-006-hybrid-rag.md) (negative-consequences section); pinned as a regression-watch test in `tests/unit/test_hybrid.py::test_bm25_oov_fake_hits_flow_through_unfiltered`. Mitigation (score-threshold filter on BM25 results before RRF) is deferred to issue #16 — the eval harness will quantify whether it actually regresses retrieval quality on the labelled set, and we don't want to hand-tune a threshold without that signal.
- **Phase 2 KB is synthetic, not scraped.** The intended source is <https://www.wscad.com/electrix/release-notes/>, but the page is bot-protected and returned `HTTP 403` to `WebFetch`. Per the all-or-nothing fallback policy in ADR-003 (Phase 2 section), the corpus under `kb/electrix_ai_release_notes/` is six hand-authored notes tagged `synthetic_for_demo: true` rather than 14 scraped notes. Each file's `source_url` points to the canonical release-notes page with a synthetic anchor (`#v7-3-2-16` etc.). When the page becomes scrapable — or a one-time browser-based copy/paste is performed — the synthetic corpus should be replaced wholesale (single PR, all files swapped), and `test_kb_corpus.py::test_synthetic_for_demo_is_consistent_across_corpus` enforces that the swap leaves no half-real / half-synthetic state. Phase 3+ classifier and verifier behaviour evaluated against the synthetic corpus is therefore an estimate of post-extension grounding — not a measurement of real-world retrieval quality.
- **Retrieval is not version-aware.** The BM25 tokenizer (`src/wscad_triage/kb/retriever.py`) splits on Unicode `\w+`, so a version literal like `v7.3.2.16` shreds into the tokens `["v7", "3", "2", "16"]`. A query that names a version (e.g., "license activation in v7.3.2") therefore matches every release note that contains any of those numeric components, and the final ranking is decided by the surrounding lexical density — not by version semantics. `tests/unit/test_extended_corpus.py::test_version_string_tokenisation_is_dot_split` pins this tokenisation behaviour so the dependent retrieval test cannot silently start passing for a different reason. Genuine version-aware retrieval (metadata-conditioned filter on `KBChunk.metadata["version"]` before scoring) is future work; the Phase 3 supervisor or a Phase 4 ADR should claim it.
- **The all-or-nothing invariant is per-subcorpus, not project-wide.** `test_kb_corpus.py` enforces consistency of `synthetic_for_demo` across `kb/electrix_ai_release_notes/` only. Phase 3+ may add new sourced subcorpora (e.g., `kb/manuals/`, `kb/forum_posts/`); each will need its own consistency test, or the existing test should be parametrised over a registry of sourced subdirectories. The current per-directory hardcoding keeps the contract scoped and unambiguous, but it does not generalise without a refactor when the second sourced subcorpus lands.
- **Verifier groundedness threshold is not empirically grounded yet.** The 0.4 cutoff in [ADR-008](../09-architecture-decisions/ADR-008-groundedness-gate.md) is a defensible default ("one of three claims ungrounded" trips it; "one of five questionable" does not) but has no eval-set support. Phase 5's labelled ungrounded-trap tickets (#34) will revisit. Related: the verifier itself is an LLM and can hallucinate verdicts; layer-1 defences in `src/wscad_triage/agents/reason.py` (fabricated chunk_ids → ValueError; quote-not-in-chunk → ValueError) cap the worst-case escape, but a verifier that wrongly accepts an entailment can still let a subtle misread through.

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
