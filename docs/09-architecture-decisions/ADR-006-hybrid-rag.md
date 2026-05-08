# ADR-006: Hybrid RAG — BM25 + multilingual embeddings + Reciprocal Rank Fusion

- **Date:** 2026-05-08
- **Status:** Accepted

## Context

Issues #10, #11, and #12 build the retrieval stack that ranks `KBChunk`s against a query string. The agentic pipeline downstream (Phase 3) treats this stack as a single contract — `retrieve(query: str, k: int) -> list[RetrievalResult]` — and is agnostic to which retriever produced a hit. Three forces shape the design:

1. **Single-retriever failure modes are asymmetric.** Sparse retrievers (BM25) miss paraphrases and cross-lingual queries; dense retrievers (embeddings) drift from exact-match queries (error codes, version numbers, product names) where token overlap is the strongest signal. Support tickets contain both: free-form prose ("can't activate license after update") and exact tokens ("Error 504", "v7.3.2.16").
2. **The corpus is bilingual.** Phase 2 (#14) imports German and English ELECTRIX AI release notes. A sparse retriever alone cannot bridge the language gap when a German query references an English-language chunk; an embedding model can.
3. **The decision is load-bearing across three issues.** #10 (BM25), #11 (embeddings + cache), and #12 (RRF + `HybridRetriever`) all depend on this choice. Documenting it in a single ADR — even though only #10 implements code in the same PR — keeps the rationale co-located and avoids three smaller ADRs that would each restate the same context.

The alternatives considered were:

| Approach | Pros | Cons |
|---|---|---|
| **BM25 only** | Trivial; deterministic; zero cold-start cost. | Fails on paraphrases and cross-lingual queries; eval set will hit the ceiling fast. |
| **Embeddings only** | Strong on paraphrases; cross-lingual via multilingual model. | Drifts on exact-match (error codes, version strings); large cold-start cost; cache invalidation complicates reviewer reproduction. |
| **BM25 + embeddings + score normalization** | Combines both signals. | Score scales differ wildly between BM25 (unbounded log-likelihoods) and cosine similarity (bounded `[-1, 1]`); normalization choice (min-max, z-score, ...) is itself a hyperparameter without a clear default. |
| **BM25 + embeddings + Reciprocal Rank Fusion** *(chosen)* | Combines both signals; RRF needs no score normalization (operates on ranks); textbook default `k=60` is well-attested. | Still two retrievers to maintain; RRF discards score magnitude, which can hide unanimous high-confidence hits. |

User-confirmed during the issue-#10 plan: roll #13 (which is purely "write ADR-006") into the same PR and pin the choice for all three retriever issues now. The implementation lands across three PRs (#10/#11/#12); this ADR is the contract that ties them.

## Decision

- **Sparse layer (issue #10, this PR).** `rank-bm25>=0.2.2` (`BM25Okapi`, default parameters `k1=1.5`, `b=0.75`). Tokenizer: lowercase + Unicode `\w+` regex split + curated EN/DE stopword frozensets (~40 entries each, embedded in `kb/retriever.py`). No stemming. Numeric tokens preserved (the canonical `"Error 504"` acceptance test depends on `"504"` surviving tokenization). Stopwords are union-applied to all chunks regardless of language; per-chunk language-aware filtering is rejected as overengineering for a 5-chunk Phase-1 corpus.

- **Dense layer (issue #11).** `sentence-transformers` with `paraphrase-multilingual-MiniLM-L12-v2`. Embeddings cached at `.cache/embeddings/`, keyed by KB content hash so the cache invalidates automatically when chunks change. Cold-start computes and writes; warm-start loads without recompute. Cross-lingual unit test (DE query against EN docs) is the explicit acceptance criterion.

- **Fusion (issue #12).** Reciprocal Rank Fusion per Cormack et al. 2009: `score(d) = sum_r 1/(k + rank_r(d))` summed across retrievers `r`, with `k=60` (the value used in the original paper and most subsequent literature). RRF operates on ranks rather than scores, so no normalization is needed. The `HybridRetriever` composes a `BM25` and an `Embedding` instance and delegates `retrieve(query, k)` to both, then fuses.

- **API surface.** `kb.retriever.BM25` (this PR), `kb.retriever.Embedding` (#11), `kb.retriever.HybridRetriever` (#12). All three implement the same signature: `retrieve(query: str, k: int) -> list[RetrievalResult]`. Each emits `RetrievalResult` instances tagged with the appropriate `RetrieverName` literal (`"bm25"`, `"embedding"`, `"rrf"`) so downstream agents can audit provenance.

## Consequences

### Positive

- **Complementary failure modes.** BM25 nails exact-match queries (error codes, product names, version strings) where embeddings drift; embeddings nail paraphrases and cross-lingual queries where BM25 misses entirely. RRF preserves the strongest signal from either side.
- **No score normalization.** RRF operates on ranks, eliminating an entire class of hyperparameter decisions (min-max vs z-score vs sigmoid) and the subtle bugs that come from getting them wrong. The only hyperparameter is `k=60`, which is the literature default.
- **Determinism.** All three layers are deterministic on stable inputs — BM25 by construction, MiniLM by frozen weights, RRF by the rank computation. Eval results are reproducible across runs and reviewer machines, which matters for issue #16 (eval harness) and senior-reviewer passes.
- **Hot-swap of layers is local.** The `retrieve(query, k)` contract is what the supervisor agent depends on; the constructor and internal layer choices are implementation detail. Replacing `paraphrase-multilingual-MiniLM-L12-v2` with a different multilingual model or swapping `BM25Okapi` for `BM25Plus` is a one-file change.

### Negative

- **Three retrievers + fusion = more code paths and slower than a single retriever.** The dense layer alone is the dominant cost — both at cold-start (model load + corpus encode) and per-query (one forward pass for the query embedding). Acceptable because (a) the eval set is small enough that cold-start runs once per CI invocation, (b) per-query latency is still well under one second on a 5-chunk corpus, (c) the on-disk cache amortizes the corpus-encode cost across runs.
- **`paraphrase-multilingual-MiniLM-L12-v2` is ~470 MB on disk and in memory at runtime.** Larger than the BM25 layer's near-zero footprint. Acceptable because (a) it's the smallest multilingual model with reliable EN+DE quality, (b) sentence-transformers caches it under `~/.cache/huggingface/`, (c) the eval and CI environments load it once per run.
- **Stopwords are language-agnostic in the BM25 tokenizer.** `die` is German "the" but also an English noun; both meanings are filtered as a stopword. Accepted because (a) the cost is per-query token-level, not per-chunk-level, (b) the eval set will catch any retrieval regression, (c) per-language tokenizer dispatch requires reliable per-token language detection that lingua does not offer (it operates at sentence granularity).
- **RRF `k=60` is the textbook default; our corpus is currently tiny (5 chunks).** The constant matters less when the rank denominator is dominated by the additive `k` rather than the rank itself. Revisit at issue #16 (eval) if hybrid underperforms a sparse-only baseline on the labeled set.
- **RRF discards score magnitude.** A retriever returning a chunk at rank 1 with overwhelming confidence contributes `1/(60+1)` whether the second-place gap is 0.001 or 0.5. Documented limitation; if it bites we can layer reranking on top in a future ADR.
- **BM25 emits zero-score "fake hits" on out-of-vocabulary queries.** When no query token matches any chunk, every BM25 score is `0.0` and the retriever still returns top-`k` results in original-corpus order. RRF treats these as rank-1 votes — meaning a fully-OOV BM25 result can co-vote with a genuine embedding hit and inflate its rank. Pinned by `tests/unit/test_hybrid.py::test_bm25_oov_fake_hits_flow_through_unfiltered` and listed in `docs/11-risks-and-technical-debt/README.md`. Mitigation deferred to #16 if the eval set shows the fusion regresses on OOV queries; the obvious fix is filtering BM25 results below an absolute score threshold before RRF.
- **Lock-in is real.** Swapping the embedding model invalidates the `.cache/embeddings/` cache (the content hash includes the model identifier). Swapping BM25 changes the BM25 ranks the test suite asserts. The cost of swapping any layer is bounded but non-trivial; this ADR is where reviewers can challenge each layer before that cost compounds.

## References

- Issue #10 — BM25 retriever (this PR).
- Issue #11 — Multilingual embedding retriever with on-disk cache.
- Issue #12 — Hybrid retriever with Reciprocal Rank Fusion.
- Issue #13 — ADR-006 (this document; rolled into the #10 PR).
- Cormack, Clarke, Büttcher (2009). *Reciprocal rank fusion outperforms Condorcet and individual rank learning methods.* SIGIR 2009. — source of `k=60`.
- ADR-002 — Schema conventions; the `RetrieverName = Literal["bm25", "embedding", "rrf"]` literal is what each layer's `RetrievalResult` instances are tagged with.
- ADR-003 — KB NLP toolchain; the chunker that feeds all three retrievers.
- [`docs/05-building-block-view/README.md`](../05-building-block-view/README.md) — names `kb.retriever.BM25` and `kb.retriever.HybridRetriever` as separate building blocks under the same module.
