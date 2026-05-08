# ADR-003: KB NLP toolchain — `pysbd` for sentence segmentation, `lingua-language-detector` for per-sentence language tagging

- **Date:** 2026-05-07
- **Status:** Accepted

## Context

Issue #9 builds the bottom of the retrieval stack: turning Markdown KB files into the `KBChunk`s that BM25 (#10), embeddings (#11), and the RRF hybrid (#12) will rank. Two NLP capabilities are needed and both have meaningful library choices:

1. **Sentence segmentation.** The project glossary and the building-block view both define a chunk as one sentence. The KB will become bilingual at #14 (German + English release notes), so the segmenter must work on at least those two languages.
2. **Per-sentence language detection.** Issue #9's acceptance criterion is explicit: "language detection on bilingual docs." A single document can contain both German and English sentences, so the language tag must be assigned per sentence, not per file. Detection must be deterministic so that test runs and reviewer reproductions produce identical chunk lists.

Both decisions are stable for the rest of Phase 1, will affect every retriever in #10–#12, and add new dependencies — which is one of the four ADR triggers in `CLAUDE.md`. Folding both into a single ADR keeps the decisions co-located and avoids ADR sprawl for two changes that ship in one PR.

The alternatives considered were:

| Approach | Pros | Cons |
|---|---|---|
| Hand-rolled regex segmenter + stopword detector | Zero new deps; trivial dependency surface | Real KB content (release notes) has abbreviations, version strings, and multilingual punctuation that defeat naive regex. Reads as amateurish for a senior reviewer. |
| `nltk` (Punkt) + `langdetect` | Well-known; multilingual | Punkt requires a runtime data download (CI flakiness); `langdetect` is stochastic — different runs can disagree on short text. |
| `spaCy` | High quality | Heavy install (>200 MB with models); language models are per-language and per-version; overkill for sentence-only segmentation. |
| **`pysbd` + `lingua-language-detector`** *(chosen)* | Both pure-Python; both deterministic; both MIT; `lingua` accurate on short text; no runtime data downloads. | `lingua`'s wheel is ~160 MB on disk because it bundles all-language models, even though we only ask it to discriminate `{en, de}` at runtime. |

## Decision

- **Sentence segmentation:** `pysbd>=0.3.4`. The segmenter is instantiated in English mode (`Segmenter(language="en", clean=False)`) and reused on German text. The English profile segments German prose acceptably because the punctuation conventions overlap; the *language tag* still comes from the dedicated detector below, so the segmenter's choice of language profile does not leak into the output. We revisit this if Phase 2 release-notes content (#14) breaks German-specific punctuation cases (e.g., quotation marks `„…"`).
- **Per-sentence language detection:** `lingua-language-detector>=2.0`, constrained to `{Language.ENGLISH, Language.GERMAN}` at builder time. `LanguageDetector.detect_language_of(...)` returns `None` for non-linguistic inputs (numbers, punctuation only) — the chunker emits `language=None` for those rather than guessing.
- **Frontmatter parsing:** `pyyaml` (already a dep). Implemented inline in `kb/loader.py` rather than via `python-frontmatter` to avoid a third dependency for ~15 lines of logic. Frontmatter values are stringified to fit `KBChunk.metadata: dict[str, str]`; lists and nested mappings raise `ValueError`.

## Consequences

### Positive

- **Determinism.** Both libraries return the same output across runs and platforms; chunk IDs (`{source_file}#{idx}`) are stable, which matters for test fixtures, retrieval caches, and reviewer reproductions.
- **Multilingual quality on short text.** `lingua` is specifically designed to be accurate on single sentences, where statistical detectors like `langdetect` are unreliable.
- **Pure-Python install.** No native extensions; CI on Linux + Windows behaves identically. No runtime data download (`nltk.download`) to fail in airgapped environments.
- **Bounded scope.** Restricting `lingua` to `{en, de}` matches the KB scope and keeps the in-memory model small even if the on-disk wheel is large.

### Negative

- **`lingua-language-detector` wheel is ~160 MB on disk.** Bigger than the ~5 MB we hoped for; `lingua` ships every language model in one wheel and the detector picks at runtime. Acceptable because (a) we only install it once per environment, (b) there's no equivalent deterministic alternative at this quality on short text, (c) the runtime memory footprint with `from_languages(EN, DE)` is small. If the install size becomes a deployment problem we can prebuild a slim wheel via `lingua`'s `low_accuracy_mode` API.
- **`pysbd`'s English-rule segmenter is used on German.** Deliberate compromise — works for the existing KB and acceptable for the bilingual fixture. Will be revisited at #16 when release-notes content lands.
- **Known segmentation glitch on `.NET`-style tokens.** A leading-period token like `.NET` is misinterpreted by `pysbd` as a sentence boundary (e.g., `Installation_Requirements.md` produces 2 chunks instead of 1). Documented here so reviewers and downstream issues (#10–#12) know not to rely on the chunk count being identical to the manually-counted sentence count. We accept the imperfect split because: (i) the affected fragments are still retrievable; (ii) BM25 still ranks the relevant chunk first on the matching query; (iii) the alternative (custom abbreviation lists in `pysbd`'s internal grammar) is fragile.
- **`pysbd` does not ship type stubs.** Imports it with `# type: ignore[import-untyped]` in `chunker.py`. `lingua` ships `py.typed`, so it type-checks cleanly under `mypy --strict`.
- **Lock-in.** Switching segmenters later changes chunk IDs, which invalidates any cached embeddings (#11) and any test fixtures that pin chunk counts. The cost of a future swap is real but bounded; this ADR is the place where reviewers can challenge the choice before that cost compounds.

## Phase 2 — corpus extension methodology (issues #14, #15, #16)

Phase 2 widens the KB beyond the three Phase-0 files by sourcing ELECTRIX AI release notes from <https://www.wscad.com/electrix/release-notes/>. Two procedural decisions ship with the data:

- **One-shot scrape, no committed scraper.** The release-notes scrape is performed once at implementation time by a subagent invocation of `WebFetch`; the markdown files under `kb/electrix_ai_release_notes/` are the deliverable. No scraping script, parser, or HTTP dependency lands in the repository — keeping the runtime closure (and `bandit` surface) unchanged. If the page changes shape and the corpus needs a refresh, the procedure is re-run, not rebuilt.
- **All-or-nothing fallback.** When the live scrape fails (the initial Phase-2 attempt hit a 403 on the bot-protected page), the corpus ships as 6–8 hand-authored synthetic notes tagged `synthetic_for_demo: true`. A mixed corpus (some real, some synthetic) is explicitly forbidden — `tests/unit/test_kb_corpus.py::test_synthetic_for_demo_is_consistent_across_corpus` enforces a single corpus-wide value for the flag. The current Phase-2 corpus is synthetic; this is recorded in `docs/11-risks-and-technical-debt/README.md`.
- **Filename and language convention.** Files are named `v<version>.md` (or `v<version>.<lang>.md` when the same version publishes both EN and DE bodies). The frontmatter `language` field is single-valued and string-typed (a list would be rejected by `parse_frontmatter`); per-sentence language detection in the chunker still tags individual sentences regardless. `parse_frontmatter` defensively quotes string values in the generated files to keep YAML 1.1 numeric coercion (e.g. `7.3` → `float`) from corrupting version strings.

## References

- Issue #9 — KB loader + sentence chunker.
- Issues #14, #15, #16 — Phase 2 corpus extension and retrieval acceptance.
- [`docs/05-building-block-view/README.md`](../05-building-block-view/README.md) — names `kb.loader` and `kb.chunker` as separate building blocks.
- [`docs/12-glossary/README.md`](../12-glossary/README.md) — defines a chunk as one sentence with provenance.
- ADR-002 — Schema conventions; `KBChunk.metadata: dict[str, str]` is the constraint that motivates the frontmatter stringification rule.
- ADR-006 — Hybrid RAG strategy; the `HybridRetriever` is the production retrieval path that the Phase-2 multi-version acceptance test exercises.
