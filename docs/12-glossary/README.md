# 12. Glossary

| Term | Definition |
|------|------------|
| **Agentic** | An LLM-driven system in which the LLM (the *agent*) decides the next action — including which tool to call — based on the current state. Distinguished from a hardcoded "prompt chain" where the developer specifies the call sequence in advance. |
| **ADR** | Architecture Decision Record. A short, immutable document capturing the context, decision, and consequences of a non-trivial architectural choice. See [`../09-architecture-decisions/`](../09-architecture-decisions/). |
| **arc42** | A documentation template for software architectures with twelve numbered sections (introduction, constraints, context, solution strategy, building blocks, runtime view, deployment view, concepts, decisions, quality requirements, risks, glossary). This project uses sections 1, 5, 6, 9, 11, 12. |
| **BM25** | Best-Matching 25, a classical sparse-retrieval ranking function based on term-frequency / inverse-document-frequency with length normalisation. Strong on keyword anchors (e.g., error codes). |
| **Chunk** | A unit of retrievable text. In this project, chunks are individual sentences with provenance metadata (source path, position, language). |
| **Confidence** | A 0–1 score on the system's own output. Computed as `min(rubric_score, verifier_score)`; thresholds (0.5, 0.7) decide between clarify-mode and solution-mode. See ADR-006 (pending). |
| **Grounding** | The property that every claim in a proposed solution is supported by an explicitly cited KB chunk. Enforced by the verifier agent. |
| **KB** | Knowledge Base. The local Markdown corpus that is the *only* authoritative source of facts the system may rely on. |
| **LangGraph** | A library that builds stateful agent graphs over LangChain primitives. Used here for the supervisor + workers topology with explicit state transitions. |
| **RAG** | Retrieval-Augmented Generation. The pattern of retrieving relevant context from a corpus and injecting it into an LLM prompt to ground generation. |
| **Reasoning trace** | An ordered list of `ReasoningStep` records, one per agent action, recording the actor, the action, the evidence references, and the rationale. Visible in both JSON and text outputs. |
| **RRF** | Reciprocal Rank Fusion. A method to combine multiple ranked retrieval lists by summing the reciprocals of their ranks. Avoids the pitfalls of score normalisation across heterogeneous retrievers (e.g., BM25 vs cosine similarity). |
| **Rubric** | The weighted-component formula that produces `rubric_score`: retrieval strength, metadata completeness, evidence alignment, LLM self-report. Weights live in `config.yaml`. |
| **Supervisor** | The LLM agent at the centre of the LangGraph state machine. On each turn it inspects the current `TicketState` and chooses the next worker (or `Finalize`). The graph terminates when it returns `Finalize`. |
| **Verifier** | A specialised LLM-as-judge agent that evaluates whether each claim in a proposed solution is grounded in its cited KB chunk. Returns a 0–1 grounding score and per-claim verdicts. |
