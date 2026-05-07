# Documentation

The project's architecture is documented in the [arc42](https://arc42.org/) template style. Each numbered section is a folder with its own `README.md` — open the folder to read the section.

The full arc42 template has twelve sections; this project uses **six** of them, plus a roadmap and the original challenge brief.

## Sections

| # | Section | What lives here |
|---|---------|-----------------|
| 1 | [Introduction and Goals](01-introduction-and-goals/) | System purpose, quality goals (prioritised), stakeholders, constraints. |
| 5 | [Building Block View](05-building-block-view/) | Static component structure, package layout, Mermaid diagram of the LangGraph. |
| 6 | [Runtime View](06-runtime-view/) | Dynamic behaviour: the canonical ticket-flow paths through the supervisor. |
| 9 | [Architecture Decisions](09-architecture-decisions/) | ADRs (Architecture Decision Records) — one file per non-trivial decision. |
| 11 | [Risks and Technical Debt](11-risks-and-technical-debt/) | Known limitations, deliberate non-goals, future work. |
| 12 | [Glossary](12-glossary/) | Domain terms (agentic, RAG, RRF, BM25, supervisor, verifier, ...). |

## Other documents in this folder

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Source of truth for the build plan; one entry per GitHub Issue, organised by phase. |
| [WSCAD AI Challenge 2026.pdf](WSCAD%20AI%20Challenge%202026.pdf) | The original challenge brief from WSCAD. Read-only reference. |

## Sections we deliberately do **not** use

The arc42 sections **02 Constraints** (folded into section 1), **03 Context and Scope**, **04 Solution Strategy**, **07 Deployment View**, **08 Cross-cutting Concepts**, and **10 Quality Requirements** are intentionally omitted: their content is either captured elsewhere (constraints in 01, quality requirements as goals in 01, deployment touched on in the README's "Production integration paths") or out of scope for a take-home assignment.

If a section becomes load-bearing later, it is added as its own folder with a `README.md` and listed in the table above.
