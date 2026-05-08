# 5. Building Block View

## High-level structure

The system is a single Python package, `wscad_triage`, organised into orthogonal subpackages.

```mermaid
graph TD
    CLI[cli.py] --> Pipeline[pipeline.py]
    Pipeline --> Graph[LangGraph state graph]
    Graph --> Supervisor[agents.supervisor routing fns]
    Graph --> Triage[agents.triage]
    Graph --> Retrieve[agents.retrieve]
    Graph --> Reason[agents.reason]
    Graph --> Clarify[agents.clarify]
    Graph --> Verify[agents.verify]
    Retrieve --> KB[kb.retriever HybridRetriever]
    KB --> BM25[kb.retriever.BM25]
    KB --> Embed[kb.retriever.Embedding]
    Pipeline --> Output[output.json_writer / text_renderer]
    Triage & Retrieve & Reason & Clarify & Verify -.->|LLM calls| Client[llm.client.LLMClient]
    Client --> Anthropic[llm.anthropic_backend]
    Client --> Azure[llm.azure_openai_backend]
```

The supervisor module exposes routing functions and finalize sinks; LangGraph wires them as conditional edges and terminal nodes. See [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md) for why the supervisor is deterministic in Phase 3 (vs. an LLM-as-supervisor) and how a future PR can swap LLM-mediated decisions in per-edge.

## LangGraph state diagram

This is the actual graph compiled by `pipeline.build_graph(...)`. Solid arrows are direct edges; dashed arrows are conditional edges driven by `supervisor.route_after_triage` and `supervisor.route_after_verify`.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	triage(triage)
	retrieve(retrieve)
	reason(reason)
	clarify(clarify)
	verify(verify)
	finalize_solve(finalize_solve)
	finalize_clarify(finalize_clarify)
	finalize_clarify_downgrade(finalize_clarify_downgrade)
	__end__([<p>__end__</p>]):::last
	__start__ --> triage;
	clarify --> finalize_clarify;
	reason --> verify;
	retrieve --> reason;
	triage -.-> clarify;
	triage -.-> retrieve;
	verify -.-> finalize_clarify_downgrade;
	verify -.-> finalize_solve;
	finalize_clarify --> __end__;
	finalize_clarify_downgrade --> __end__;
	finalize_solve --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Regenerate after a topology change:

```bash
uv run python -c "from unittest.mock import Mock; from wscad_triage.pipeline import build_graph; print(build_graph(Mock(), Mock()).get_graph().draw_mermaid())"
```

Paste the output above the regeneration command. The graph is committed (not auto-rendered at doc-build time) so changes are visible in PR diffs.

## Building blocks

| Block | Responsibility | Lives in |
|-------|----------------|----------|
| `cli` | Click entry point; loads tickets, runs the pipeline, writes output. | [`src/wscad_triage/cli.py`](../../src/wscad_triage/cli.py) |
| `pipeline` | Builds and compiles the LangGraph state graph; runs a single ticket end-to-end. | `src/wscad_triage/pipeline.py` (Phase 3) |
| `agents.supervisor` | Deterministic routing fns + three finalize sinks; LangGraph wires them via conditional edges. See [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md). | `src/wscad_triage/agents/supervisor.py` (Phase 3) |
| `agents.triage` | Initial classification + metadata-completeness assessment. | `src/wscad_triage/agents/triage.py` (Phase 3) |
| `agents.retrieve` | Wraps `HybridRetriever`; reformulates ticket text into a retrieval query. | `src/wscad_triage/agents/retrieve.py` (Phase 3) |
| `agents.reason` | Drafts the proposed solution; produces a claim-to-evidence map. | `src/wscad_triage/agents/reason.py` (Phase 3) |
| `agents.clarify` | Generates 2–4 targeted follow-up questions tied to detected gaps. | `src/wscad_triage/agents/clarify.py` (Phase 3) |
| `agents.verify` | Groundedness safety gate; per-claim verdict + 0–1 grounding score. | `src/wscad_triage/agents/verify.py` (Phase 3) |
| `kb.loader` | Loads `.md` files, parses YAML frontmatter, returns `RawDoc` objects. | `src/wscad_triage/kb/loader.py` (Phase 1) |
| `kb.chunker` | Sentence-level chunking with provenance metadata. | `src/wscad_triage/kb/chunker.py` (Phase 1) |
| `kb.retriever` | `BM25` (sparse), `Embedding` (multilingual MiniLM with on-disk cache), and `HybridRetriever` (RRF, Phase 1 / issue #12). All three implement `retrieve(query, k) -> list[RetrievalResult]`. | `src/wscad_triage/kb/retriever.py` (Phase 1) |
| `llm.client` | Provider-agnostic LLM interface; deterministic mock for tests. | `src/wscad_triage/llm/client.py` (Phase 3) |
| `llm.anthropic_backend` | Anthropic SDK adapter with prompt caching for KB context. | `src/wscad_triage/llm/anthropic_backend.py` (Phase 3) |
| `llm.azure_openai_backend` | Azure OpenAI adapter (production target). | `src/wscad_triage/llm/azure_openai_backend.py` (Phase 3) |
| `confidence` | Rubric formula + verifier integration; `compute_confidence(state)`. | `src/wscad_triage/confidence.py` (Phase 4) |
| `output.json_writer` | Serialises the canonical `Output` model. | `src/wscad_triage/output/json_writer.py` (Phase 4) |
| `output.text_renderer` | Renders the JSON to the human-readable Sample_Output.txt-style text. | `src/wscad_triage/output/text_renderer.py` (Phase 4) |
| `observability` | Structured JSON logging with per-ticket trace IDs. | `src/wscad_triage/observability.py` (Phase 3) |
| `eval.runner` / `eval.metrics` | Evaluation harness over the labeled ticket set. | [`eval/`](../../eval/) (Phase 5) |

The package boundaries are deliberately narrow: each subpackage exposes a small, typed surface area; cross-subpackage imports go through that surface area only.
