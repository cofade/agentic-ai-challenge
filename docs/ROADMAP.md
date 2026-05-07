# Roadmap

This document is the **source of truth** for what is built and in what order. Each numbered item below corresponds to one GitHub Issue tracked on the project board.

The system being built is an agentic AI ticket-triage tool that processes technical support tickets via local-only RAG, classifies category and priority, decides whether to propose a solution or ask follow-up questions, and emits a structured, explainable result with confidence and a reasoning trace. The architecture and design decisions are described in [`docs/01-introduction-and-goals/`](01-introduction-and-goals/), [`docs/05-building-block-view/`](05-building-block-view/), and [`docs/09-architecture-decisions/`](09-architecture-decisions/).

Phases run sequentially; items inside a phase can be parallelised when dependencies allow but default to sequential execution. Each item ends with a senior-reviewer pass, a small PR (`Closes #N`), and a merge.

---

## Phase 0 — Bootstrap

Set up the project skeleton and the six engineering layers (context, quality gates, docs, CI, review workflow, entropy management) before any feature code.

- **#1 Author `docs/ROADMAP.md` and create issues / project items for the full plan.** Acceptance: roadmap committed; one GitHub Issue per item below; all issues added to the project board with status "Backlog" except #1 marked "In progress" / closing.
- **#2 Apply Layer 1 — `CLAUDE.md` adapted from the agentic-SE template.** Acceptance: file at repo root, under ~200 lines, Quick Reference / Workflow / Quality Checks / Architecture Principles / Known AI Pitfalls populated with project-specifics.
- **#3 Apply Layer 2 — `pyproject.toml` (ruff, mypy, bandit, pytest, coverage) + `.pre-commit-config.yaml`.** Acceptance: `uv sync` works; `uv run ruff check src/`, `uv run mypy src/`, `uv run bandit -r src/`, `uv run pytest` all run cleanly on the empty scaffold; `pre-commit install` succeeds.
- **#4 Apply Layer 3 — arc42 documentation skeleton.** Acceptance: every used arc42 section is its own folder with a `README.md` (`docs/01-introduction-and-goals/`, `docs/05-building-block-view/`, `docs/06-runtime-view/`, `docs/09-architecture-decisions/`, `docs/11-risks-and-technical-debt/`, `docs/12-glossary/`); a `docs/README.md` indexes them; each section has at least one paragraph of project-specific content (no placeholders); the ADR template is in place with ADR-001 accepted.
- **#5 Apply Layer 4 — `.github/workflows/ci.yml` with three parallel jobs.** Acceptance: lint, test, security jobs declared; each is independent (no `needs:` chain between them); workflow uses `uv` for dependency resolution; pushes trigger CI; CI green on the bootstrap commit.
- **#6 Apply Layer 5 — copy `senior-reviewer.md` into `.claude/agents/`.** Acceptance: agent file present and unmodified from template; CLAUDE.md "Pre-PR quality gates" section names the senior-reviewer as mandatory.
- **#7 Move provided challenge artifacts into target locations.** Acceptance: KB at `kb/original/`, sample tickets at `tickets/`, brief PDF preserved under `docs/`. (Already done — closes when committed.)

## Phase 1 — Schemas & KB foundation

Type-safe data contracts and the retrieval primitives the agents will share. No LLM calls yet; everything in this phase is deterministic and unit-testable.

- **#8 Pydantic schemas + ADR-002 (schema conventions).** `TicketMetadata`, `Ticket`, `KBChunk`, `RetrievalResult`, `Classification`, `ReasoningStep`, `Output`, `TicketState`. Acceptance: schemas in `src/wscad_triage/schemas.py`; round-trip JSON test for every schema; mypy clean; ADR-002 documents the closed-`Literal`/`extra="forbid"`/cross-field-validator conventions.
- **#9 KB loader + sentence chunker.** Reads `.md` files from `kb/`, parses optional YAML frontmatter, sentence-tokenises, returns `list[KBChunk]` with provenance (`source_path`, `chunk_index`, `language`). Acceptance: unit tests verify chunk counts, frontmatter extraction, language detection on bilingual docs.
- **#10 BM25 retriever.** `rank_bm25` over chunks; `retrieve(query, k) -> list[RetrievalResult]`. Acceptance: unit tests assert "Error 504" retrieves the licensing chunk top-1.
- **#11 Multilingual embedding retriever.** `paraphrase-multilingual-MiniLM-L12-v2` (or equivalent); cached embeddings in `.cache/embeddings/`. Acceptance: cold-start computes & caches; warm-start loads from cache; cross-lingual query test (DE query against EN docs returns plausible match).
- **#12 Hybrid retriever with Reciprocal Rank Fusion.** `HybridRetriever` composes BM25 and embeddings; RRF combines ranks. Acceptance: unit test asserts RRF returns top-k that consistently includes the strongest result from either single retriever.
- **#13 ADR-005 — Hybrid RAG strategy.** Acceptance: ADR documents the choice with context, decision, consequences; cross-referenced from `docs/09-architecture-decisions/README.md`.

## Phase 2 — KB extension via ELECTRIX AI release notes

Extend the corpus with one Markdown file per ELECTRIX AI version sourced from <https://www.wscad.com/electrix/release-notes/>. Scope is **ELECTRIX AI only** — no legacy WSCAD SUITE, no ELECTRIX (non-AI), no ELECTRIX ROCKET.

- **#14 Subagent scrape: ELECTRIX AI release notes → `kb/electrix_ai_release_notes/`.** Versions: `v7.4.0.17 (AI 2026)`, `v7.3.2.16`, `v7.3.2.14`, `v7.3.2.10`, `v7.3.2.8`, `v7.3.2.4`, `v7.3.2.3`, `v7.3.2.2`, `v7.3.1.4`, `v7.3.0.24`, `v7.3.0.23`, `v7.3.0.20`, `v7.3.0.19`, `v7.3.0.16`. Each file has YAML frontmatter (`source_url`, `version`, `product_line: "ELECTRIX AI"`, `language`, `synthetic_for_demo: false`).
- **#15 Validate scrape; fall back to hand-curated notes if parsing fails.** Acceptance: every file in `kb/electrix_ai_release_notes/` has valid frontmatter; if scrape fails, 6–8 plausible synthetic notes labelled `synthetic_for_demo: true`.
- **#16 Re-run retriever tests against extended corpus; tune chunking if needed.** Acceptance: existing tests still pass; one new test exercises retrieval over a multi-version query (e.g., "what changed for license activation in v7.3.2?").

## Phase 3 — LLM abstraction & agents

Provider-agnostic LLM client and the worker agents themselves. Tests use a deterministic mock backend; live providers are gated behind environment-loaded API keys.

- **#17 LLMClient interface + Anthropic backend.** Methods: `generate(messages, tools=None, response_format=None)`. Anthropic backend uses the SDK with prompt caching for KB context. Acceptance: integration test against the live API gated by `ANTHROPIC_API_KEY`; unit tests use a `MockLLMClient`.
- **#18 Azure OpenAI backend.** Full implementation if `AZURE_OPENAI_*` keys present, otherwise a documented stub raising `ConfigurationError` with a clear message. Acceptance: backend selectable via CLI flag; README documents Azure as the production target.
- **#19 Triage agent.** Input: `Ticket` + `TicketState`. Output: initial `Classification` + metadata-completeness assessment + critical-missing-fields list. Acceptance: integration test with mock LLM produces consistent classification for the two provided tickets.
- **#20 Retrieve agent.** Wraps the hybrid retriever as a worker. Inputs include a query reformulation step (LLM rewrites ticket text into a retrieval query). Acceptance: returns top-k chunks with provenance; reasoning step logged.
- **#21 Reason agent.** Drafts a `Solution` citing retrieved chunks; produces a claim-to-evidence map for the verifier. Acceptance: every claim in the proposed solution is mapped to at least one retrieved `KBChunk` ID.
- **#22 Clarify agent.** Generates 2–4 targeted follow-up questions tied to detected gaps. Acceptance: questions are specific (no boilerplate "please provide more info"); referenced gaps appear in the reasoning trace.
- **#23 Verifier agent (groundedness safety gate).** LLM-as-judge; for each claim in the proposed solution, decides whether it is grounded in the cited chunk; returns a 0–1 grounding score and per-claim verdicts. Acceptance: ungrounded-claim test ticket triggers `< 0.4` grounding score and forces a clarify outcome.
- **#24 Supervisor + LangGraph state graph + `pipeline.py`.** Supervisor is an LLM agent with tool definitions for each worker; supervisor's tool calls drive the next node; graph terminates when supervisor returns `Finalize`. Acceptance: `pipeline.run(ticket)` returns a complete `Output`; LangGraph state diagram exported to `docs/05-building-block-view/`.
- **#25 Integration tests with deterministic mocked LLM.** `tests/integration/` covers: resolvable ticket (high-confidence solution), clarify-required ticket, multilingual ticket, ungrounded-claim trap, missing-OS metadata. Acceptance: all integration tests pass without network access.
- **#26 ADRs 003 / 004 / 007 — provider abstraction, supervisor topology, groundedness gate.** Acceptance: three ADRs committed; cross-referenced from README.

## Phase 4 — Confidence, output, CLI

Make the system runnable end-to-end and produce the deliverable artifacts.

- **#27 Rubric confidence formula + config-driven weights.** `confidence.py` exposes `compute_rubric(state) -> float`; weights live in `config.yaml`. Acceptance: unit tests cover boundary cases (zero retrieval, zero metadata, full evidence).
- **#28 Final-confidence aggregation: `min(rubric_score, verifier_score)`.** Threshold defaults: `< 0.5` clarify, `0.5–0.7` solution-with-caveats, `> 0.7` solution. Acceptance: unit test asserts ungrounded-claim ticket lands below 0.5.
- **#29 JSON writer + Sample_Output.txt-style text renderer.** Pydantic `model_dump_json` for the canonical artifact; renderer formats human-readable text mirroring `tickets/Sample_Output.txt`. Acceptance: snapshot test compares rendered text format against an approved fixture.
- **#30 CLI (`wscad-triage`).** `wscad-triage <tickets.json> [--out out/] [--provider anthropic|azure]`. Acceptance: running on `tickets/tickets.json` produces `out/T-001.json`, `out/T-001.txt`, `out/T-002.json`, `out/T-002.txt`.
- **#31 ADRs 006 / 008 — confidence quantification, multilingual KB.** Acceptance: two ADRs committed.

## Phase 5 — Evaluation harness

Quantitative ground truth for the README's claims.

- **#32 Author 15–20 hand-labeled tickets.** Coverage: resolvable EN, resolvable DE, clarify (missing OS), clarify (vague crash), licensing-vs-installation ambiguity, ungrounded-claim trap, multilingual mixed. Stored at `tickets/eval_set.json` with `expected_category`, `expected_priority`, `should_clarify` fields. Acceptance: ≥ 15 tickets; balance across the seven listed cases.
- **#33 `eval/runner.py` and `eval/metrics.py`.** Runs pipeline over the eval set; emits per-ticket results plus aggregated metrics (category accuracy, priority accuracy, clarification precision/recall, mean grounded confidence, calibration data). Acceptance: `uv run python -m eval.runner` runs end-to-end against the eval set without errors and writes `eval/results/latest.json`.
- **#34 Run eval; record baseline numbers in README.** Acceptance: README includes a "Baseline metrics" subsection citing the latest run and noting that numbers are illustrative, not promises.
- **#35 ADR-009 — evaluation harness.** Acceptance: ADR documents the rubric, dataset construction, known limitations.

## Phase 6 — Documentation polish

Make the reviewer's first 5 minutes excellent.

- **#36 Author the reviewer-facing README.** Sections: 30-second mental model, quickstart (clone → `uv sync` → `cp .env.example .env` → `uv run wscad-triage tickets/tickets.json`), architecture diagram (Mermaid), trade-offs and what we deliberately did NOT do, baseline metrics, links to ADRs and arc42. Acceptance: a new user can run the system in under 5 minutes.
- **#37 Fill the arc42 docs.** Concretise `01-introduction-and-goals/README.md`, `05-building-block-view/README.md` (Mermaid of the LangGraph), `06-runtime-view/ticket-flow.md` (happy + clarify sequences), `11-risks-and-technical-debt/README.md`, `12-glossary/README.md`. Acceptance: each file has at least one paragraph of non-placeholder content; cross-references resolve.
- **#38 Cross-reference ADRs from README; index in `docs/09-architecture-decisions/README.md`.** Acceptance: ADR index lists all nine ADRs with one-line summaries; README links into the ADR index from the architecture section.

## Phase 7 — Final review & submission

Close every loop before sending the link.

- **#39 Senior-reviewer pass on the full main branch; address all P0 / P1.** Re-run until verdict is "mergeable as-is." Acceptance: senior-reviewer report archived in `docs/11-risks-and-technical-debt/` (or attached to the closing PR); all P0s resolved, all P1s either resolved or explicitly deferred with rationale.
- **#40 Verify CI green on `main`.** Acceptance: latest CI run shows lint, test, security jobs all passing.
- **#41 Reviewer-clone smoke test.** Acceptance: from a clean clone in a fresh directory, `cp .env.example .env` (set one API key), `uv sync && uv run wscad-triage tickets/tickets.json` succeeds and produces both JSON and text outputs.
- **#42 Toggle repo visibility per WSCAD's submission instructions; send the link.** Acceptance: visibility flipped at submission time; submission email includes a one-paragraph orientation pointing at the README quickstart.

---

## Out of scope

Documented here so we don't regret leaving them out.

- UI / frontend (PDF: "Not Required").
- Model fine-tuning (PDF: "Not Required").
- Coverage of non-AI ELECTRIX product variants (legacy SUITE, ELECTRIX, ELECTRIX ROCKET).
- C# / .NET implementation — out of scope per ADR-001.
- LLM-as-judge evaluation of solution quality beyond grounding (would require significantly larger eval budget).
- Real customer data — every example used is provided by WSCAD or hand-authored.
