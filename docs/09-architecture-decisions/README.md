# 9. Architecture Decision Records

ADRs capture decisions whose rationale is not obvious from the code. Each ADR has the format: title, date, status, context, decision, consequences. They are numbered and immutable once accepted; superseding decisions are recorded as new ADRs that link to the predecessor.

## Index

| # | Title | Status | Phase |
|---|-------|--------|-------|
| 001 | [Language and runtime: Python 3.11+ with uv](ADR-001-language-runtime.md) | Accepted | Phase 0 |
| 002 | [Schema conventions for the typed-contract layer](ADR-002-schema-conventions.md) | Accepted | Phase 1 |
| 003 | [KB NLP toolchain: pysbd + lingua-language-detector](ADR-003-kb-nlp-toolchain.md) | Accepted | Phase 1 |
| 004 | [Provider-agnostic LLM client; Anthropic default, Azure OpenAI as production target](ADR-004-llm-provider-abstraction.md) | Accepted | Phase 3 |
| 005 | [Supervisor + workers topology in LangGraph (over linear pipeline / single ReAct)](ADR-005-supervisor-topology.md) | Accepted | Phase 3 |
| 006 | [Hybrid RAG: BM25 + multilingual embeddings + Reciprocal Rank Fusion](ADR-006-hybrid-rag.md) | Accepted | Phase 1 |
| 007 | [Confidence = min(rubric_score, verifier_score) with config-driven rubric weights](ADR-007-confidence-quantification.md) | Accepted | Phase 4 |
| 008 | [Groundedness safety gate (verifier agent) at the end of the pipeline](ADR-008-groundedness-gate.md) | Accepted | Phase 3 |
| 009 | [Multilingual KB content strategy — single unified index, language-tagged chunks](ADR-009-multilingual-kb-strategy.md) | Accepted | Phase 4 |
| 010 | [Evaluation harness — 16-ticket hand-labeled set + runner with provenance envelope](ADR-010-evaluation-harness.md) | Accepted | Phase 5 |

## ADR template

```markdown
# ADR-NNN: Title

- **Date:** YYYY-MM-DD
- **Status:** Pending | Accepted | Superseded by ADR-MMM

## Context
What is the situation? What forces are at play? What constraints?

## Decision
What did we decide? Be concrete enough to be falsifiable.

## Consequences
What follows from this decision — both positive and negative? What does it lock us into?
```
