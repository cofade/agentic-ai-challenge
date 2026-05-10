# 1. Introduction and Goals

## Purpose

This system processes technical support tickets and produces a structured, explainable triage result. For each ticket it:

1. Understands the problem (free-text plus optional, possibly incomplete metadata).
2. Retrieves relevant knowledge from a local knowledge base (RAG).
3. Decides whether enough information exists to propose a solution, or whether to ask for clarification.
4. Emits a structured result containing classification (category + priority), proposed solution or clarification request, confidence, reasoning trace, and follow-up questions.

The knowledge base is the **only authoritative source** of facts. Solutions that are not grounded in retrieved KB chunks are penalised by the verifier agent and either flagged or replaced with a clarification request.

## Quality goals (in priority order)

| # | Goal | Concrete expression |
|---|------|---------------------|
| 1 | **Reasoning transparency** | Every output includes a step-by-step reasoning trace; every claim in the proposed solution is mapped to the KB chunks that support it. |
| 2 | **Agentic design quality** | Supervisor routing makes the next-action decision visible on each turn — deterministic in Phase 3 (two pure routing functions on `TicketState` + three finalize sinks), with a per-edge LLM swap-in documented in [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md). The system is not a hardcoded chain. |
| 3 | **Hallucination resistance** | The verifier agent rejects ungrounded claims; rubric-based confidence drops to clarify-mode below the threshold. |
| 4 | **Engineering clarity** | Pydantic-typed contracts at every boundary; deterministic core (chunking, retrieval, scoring) is unit-testable; LLM calls are mocked in tests. |
| 5 | **Trade-off awareness** | Every non-trivial decision is captured in an ADR with context, decision, and consequences. |

## Stakeholders

| Role | Concern |
|------|---------|
| Reviewer (CEO + Software Development Manager) | "Is the design genuinely agentic and production-aware?" |
| Future maintainer | "Can I add a worker agent without rewriting routing?" |
| End user (a triage operator) | "Is the recommendation clearly justified, with the evidence I'd need to verify it?" |

## Constraints

- The provided knowledge base (3 short Markdown files) is the only authoritative source; the corpus is extended only with content sourced from public WSCAD ELECTRIX AI release notes.
- No external data sources at runtime; no model fine-tuning; no UI / frontend.
- Multilingual support: KB content may be German, English, or mixed; embeddings must bridge the languages.
- LLM calls happen via a provider-agnostic interface so the system can run against Ollama (default, local), Anthropic (cloud), or Azure OpenAI (production target) without code changes.
