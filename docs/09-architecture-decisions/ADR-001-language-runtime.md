# ADR-001: Language and runtime — Python 3.11+ with uv

- **Date:** 2026-05-07
- **Status:** Accepted

## Context

The challenge calls for an agentic AI system with RAG and LLM tool use, deliverable in roughly one part-time week. WSCAD's ELECTRIX product line is a C#/.NET/WPF desktop application on Azure — a future production integration would likely happen via either an in-process .NET adapter or a service boundary calling out to a Python implementation.

Three options were on the table: pure Python, pure C# / .NET 8 with Microsoft Semantic Kernel, or a Python core with a thin C# wrapper.

## Decision

The system is implemented in Python 3.11+ using `uv` for dependency management and the standard agentic-software-engineering tooling chain (ruff, mypy, bandit, pytest, pre-commit).

The Python ecosystem (LangGraph, sentence-transformers, the official Anthropic and Azure OpenAI SDKs) gives the fastest path to a polished, testable system inside the available time budget. C# / .NET integration is documented as a future-work path: an HTTP service boundary or an embedded `Python.NET` host would make the existing implementation callable from ELECTRIX without changes to the agentic core.

## Consequences

- **Positive.** The richest LLM/RAG ecosystem; the paper's bootstrap protocol applies directly; the deliverable is in the language WSCAD reviewers expect for an AI take-home.
- **Positive.** All design decisions remain language-agnostic at the architectural level — the LangGraph state machine and the retrieval primitives translate cleanly to a Semantic Kernel implementation if the production team wants to port them.
- **Negative.** No direct demonstration of C# / .NET familiarity; this is mitigated by an explicit "Production integration paths" section in the README and by the provider-agnostic LLM abstraction (which keeps the Azure OpenAI backend a first-class concern).
- **Locked-in.** The CLI, package layout, and tooling configuration assume Python; switching languages would be a rewrite, not a migration.
