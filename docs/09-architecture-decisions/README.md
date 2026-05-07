# 9. Architecture Decision Records

ADRs capture decisions whose rationale is not obvious from the code. Each ADR has the format: title, date, status, context, decision, consequences. They are numbered and immutable once accepted; superseding decisions are recorded as new ADRs that link to the predecessor.

## Index

| # | Title | Status | Phase |
|---|-------|--------|-------|
| 001 | [Language and runtime: Python 3.11+ with uv](ADR-001-language-runtime.md) | Accepted | Phase 0 |
| 002 | [Schema conventions for the typed-contract layer](ADR-002-schema-conventions.md) | Accepted | Phase 1 |
| 003 | Provider-agnostic LLM client; Anthropic default, Azure OpenAI as production target | Pending | Phase 3 |
| 004 | Supervisor + workers topology in LangGraph (over linear pipeline / single ReAct) | Pending | Phase 3 |
| 005 | Hybrid RAG: BM25 + multilingual embeddings + Reciprocal Rank Fusion | Pending | Phase 1 |
| 006 | Confidence = min(rubric_score, verifier_score) with thresholds 0.5 / 0.7 | Pending | Phase 4 |
| 007 | Groundedness safety gate (verifier agent) at the end of the pipeline | Pending | Phase 3 |
| 008 | Multilingual KB; multilingual MiniLM embeddings | Pending | Phase 4 |
| 009 | Hand-labeled 15–20 ticket eval set + rubric metrics | Pending | Phase 5 |

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
