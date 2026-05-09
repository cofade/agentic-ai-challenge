# ADR-009: Multilingual KB content strategy — single unified index, language-tagged chunks

- **Date:** 2026-05-09
- **Status:** Accepted

## Context

The KB currently contains documents in both English and German:

- `kb/original/` — English documentation (user manuals, error guides,
  installation references).
- `kb/electrix_ai_release_notes/` — 4 EN + 2 DE synthetic release notes.

Support tickets may arrive in either language. Two architectural options
exist:

**Option A — separate language-specific indices.** Run BM25 and the
embedding retriever separately over EN and DE corpora; route the ticket
to the appropriate index based on detected language; merge results before
scoring.

**Option B — single unified index with multilingual embeddings.** All
chunks (EN and DE) are indexed together. A single BM25 instance and a
single multilingual embedding model service both languages. Language is
stored as metadata on each chunk for observability but does not gate
retrieval routing.

Phase 1 (ADR-003) already chose `paraphrase-multilingual-MiniLM-L12-v2`
as the embedding model precisely because it maps semantically related text
from different languages into the same vector space. BM25 uses a Unicode
`\w+` tokenizer with curated EN and DE stopword lists rather than a
language-specific stemmer — it is already language-agnostic by design.

Constraints:
- Phase 3's supervisor topology (ADR-005) has no language-conditioned
  routing — workers receive `TicketState` regardless of ticket language.
- The KB size is small (< 50 documents in Phase 2). Splitting into separate
  indices adds complexity with negligible performance benefit at this scale.
- Phase 5's eval set must include at least one DE ticket (ROADMAP #32) to
  measure cross-lingual retrieval quality before a more complex strategy
  is warranted.

## Decision

**Option B — single unified hybrid retriever, language-tagged chunks.**

Specifically:

1. **No language-conditioned routing.** Every ticket flows through the same
   retrieve → reason → verify path regardless of detected language. The
   multilingual embedding space handles semantic alignment transparently.

2. **Language detection at chunk level.** `lingua-language-detector`
   (ADR-003) detects the language of each sentence during chunking and stores
   it in `KBChunk.language`. This tag remains on the `KBChunk` for future
   observability; it is not currently surfaced in `Output` (`cited_sources`
   and `reasoning_trace` carry file paths and chunk IDs respectively, not
   language metadata). A future extension could surface language in
   `cited_sources` or the text renderer if operator demand arises.

3. **BM25 with joint EN/DE stopwords.** The `tokenize()` function in
   `retriever.py` removes entries from both `STOPWORDS_EN` and `STOPWORDS_DE`.
   A DE query token that is also an EN stopword will be correctly removed, and
   vice versa. Numeric tokens (version strings, error codes) survive in both
   languages — the "Error 504" acceptance test exercises this.

4. **Embedding model is multilingual by design.** `paraphrase-multilingual-MiniLM-L12-v2`
   was evaluated in ADR-003 on the bilingual corpus and confirmed to return a
   plausible EN chunk for a DE query in the cross-lingual test
   (`tests/unit/test_embedding.py::test_cross_lingual_query`). The per-chunk
   embedding cache (`.cache/embeddings/`) is shared across languages — no
   separate warm-start step needed.

5. **No translation at runtime.** The triage, reason, clarify, and verify
   agents receive the ticket text as-is. The LLM (provider-selected) is
   expected to handle both EN and DE input natively; empirically,
   `claude-sonnet-4-6` and `gpt-oss:20b` both do.

## Consequences

- **Positive.** Single retrieval path — no branching logic in the pipeline,
  no language classifier at inference time, no separate warm-start step.
- **Positive.** Embedding cache is shared — a cold start encodes all chunks
  once; subsequent queries in any language hit the warm cache.
- **Positive.** The `KBChunk.language` field provides observability without
  adding routing complexity — a future operator can filter by language if
  needed without changing the retrieval contract.
- **Positive.** Adding a new language (e.g., French KB documents) requires
  only adding the language to `lingua`'s detection scope and extending the
  stopword list; the retriever and pipeline need no changes.
- **Negative — BM25 keyword recall degrades on mixed-language queries.**
  A DE query for `Lizenzaktivierung` will not match EN chunks that say
  "license activation" via BM25 (no shared tokens after stopword removal).
  The embedding layer compensates, but BM25's contribution to the RRF fused
  score is limited to language-aligned hits. On the current synthetic corpus
  this is acceptable; Phase 5 will measure the gap.
- **Negative — retrieval quality for low-resource language pairs is
  unvalidated.** The cross-lingual test in Phase 1 uses a single DE query
  against a known EN chunk. Phase 5's eval set (#32) includes one DE ticket;
  that will produce the first quantitative estimate.
- **Negative — BM25 version-string tokenisation is language-agnostic.**
  The `v7.3.2.16` tokenisation issue documented in the risks README
  (`docs/11-risks-and-technical-debt/README.md`) applies equally to EN and
  DE version references. This is a known limitation of the shared BM25
  approach, not a multilingual-specific issue.

## Cross-references

- ADR-003 — KB NLP toolchain: the `lingua-language-detector` choice and
  the `paraphrase-multilingual-MiniLM-L12-v2` selection that makes Option B
  viable.
- ADR-006 — Hybrid RAG: the RRF fusion that combines BM25 and embedding
  results; the BM25 OOV/fake-hit caveat that applies to DE queries over
  EN chunks.
- `src/wscad_triage/kb/retriever.py` — `STOPWORDS_EN`, `STOPWORDS_DE`,
  `tokenize()`, `BM25`, `Embedding`, `HybridRetriever`.
- `tests/unit/test_embedding.py::test_cross_lingual_query` — Phase 1 test
  that pins cross-lingual retrieval correctness.
