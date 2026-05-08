# agentic-ai-challenge

An agentic AI ticket-triage system: processes technical support tickets, retrieves grounding from a local knowledge base (hybrid RAG), decides whether to propose a solution or ask follow-up questions, and emits a structured, explainable result with confidence and a reasoning trace.

> Submission for the **WSCAD Code Challenge** (AI Technology Lead, take-home assignment). The full reviewer-facing README — quickstart, architecture diagram, baseline metrics, trade-offs — is authored as the deliverable for [issue #36](https://github.com/cofade/agentic-ai-challenge/issues/36) once the system is end-to-end runnable. This document orients early visitors to a repo that is still under construction.

## Status

**Phase 0 — Bootstrap: complete.** The repo's six engineering layers (context, quality gates, docs, CI, review workflow, entropy management) are in place. The agentic core itself is under construction; build progress is tracked on [project board #2](https://github.com/users/cofade/projects/2).

| Phase | What it delivers | State |
|-------|------------------|-------|
| 0 | Bootstrap (this layer) | Done |
| 1 | Pydantic schemas + KB foundation (loader, chunker, BM25, embeddings, hybrid retriever) | Done |
| 2 | KB extension (ELECTRIX AI release notes; synthetic — see [risks doc](docs/11-risks-and-technical-debt/README.md)) | Done |
| 3 | LLM abstraction + supervisor and worker agents in LangGraph | In progress |
| 4 | Confidence aggregation, output renderers, CLI | Open |
| 5 | Hand-labeled eval set + metrics harness | Open |
| 6 | Reviewer-facing README, arc42 fill-in, ADR cross-refs | Open |
| 7 | Senior-reviewer pass, CI green on main, submission | Open |

## Quickstart (works today; covers Phase 0 only)

```bash
# Clone and create a project-local venv with uv inside it
git clone https://github.com/cofade/agentic-ai-challenge.git
cd agentic-ai-challenge
python -m venv .venv
.\.venv\Scripts\python -m pip install uv     # PowerShell on Windows
# or:  ./.venv/bin/python -m pip install uv  # bash on macOS/Linux

# Sync dependencies and install pre-commit hooks
uv sync --all-extras
uv run pre-commit install

# Run the gates
uv run pytest tests/ -v --cov=src/wscad_triage
uv run ruff check src/ tests/ eval/
uv run mypy src/
uv run bandit -r src/ --severity-level high
```

The CLI itself (`uv run wscad-triage tickets/tickets.json`) is a stub today; it lands with [issue #30](https://github.com/cofade/agentic-ai-challenge/issues/30) at the close of Phase 4.

## Configuration

Phase 3 introduces a provider-agnostic LLM client (see [ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md)). Copy `.env.example` to `.env` and set the keys for the provider you want to use:

```bash
cp .env.example .env
# then edit .env to set ANTHROPIC_API_KEY (default provider)
```

Provider selection is `WSCAD_TRIAGE_PROVIDER=anthropic|azure`:

- **`anthropic`** (default) — production code path. Uses the Anthropic SDK with prompt caching for KB context.
- **`azure`** — Azure OpenAI is named as the production target in the brief but ships as a documented stub today; constructing the backend raises `ConfigurationError`. Full implementation is deferred ([ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md), "Future work").

Tests use a deterministic `MockLLMClient` and never hit the network. The single `live_api`-marked smoke test at `tests/integration/test_anthropic_live.py` is **deselected by default** via `addopts = ["-m", "not live_api"]` in `pyproject.toml`; opt in with `ANTHROPIC_API_KEY=... uv run pytest -m live_api`. If `-m live_api` is selected without an API key, the test calls `pytest.skip()` and exits cleanly.

## Where to read further

- [`docs/`](docs/) — architecture documentation in arc42 style. Start with [`docs/README.md`](docs/README.md).
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the build plan; one entry per GitHub Issue, organised by phase.
- [`docs/01-introduction-and-goals/`](docs/01-introduction-and-goals/) — system purpose, prioritised quality goals, stakeholders, constraints.
- [`docs/05-building-block-view/`](docs/05-building-block-view/) — component map and the LangGraph diagram of the supervisor + workers topology.
- [`docs/09-architecture-decisions/`](docs/09-architecture-decisions/) — ADRs. Five are Accepted today: ADR-001 (language and runtime), ADR-002 (schema conventions), ADR-003 (KB NLP toolchain), ADR-004 (LLM provider abstraction), and ADR-006 (hybrid RAG strategy); the rest are pending and land alongside the components they document.
- [`CLAUDE.md`](CLAUDE.md) — Claude Code instructions for anyone working on the repo with an AI agent.
- [`docs/WSCAD AI Challenge 2026.pdf`](docs/WSCAD%20AI%20Challenge%202026.pdf) — the original challenge brief.

## Layout

```
agentic-ai-challenge/
├── README.md                    ← you are here
├── CLAUDE.md                    ← agent operating instructions
├── pyproject.toml               ← uv + ruff + mypy strict + bandit + pytest
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml     ← three parallel jobs: lint, test, security
├── .claude/agents/              ← senior-reviewer pre-PR review agent
├── src/wscad_triage/            ← the package (currently scaffold only)
├── tests/                       ← unit + integration tests
├── eval/                        ← evaluation harness (Phase 5)
├── kb/                          ← local knowledge base (the only authoritative source)
│   ├── original/                ← the 3 KB files provided by WSCAD
│   └── electrix_ai_release_notes/  ← per-version notes (Phase 2; currently synthetic — see docs/11-risks-and-technical-debt/)
├── tickets/                     ← provided sample tickets + golden eval set (Phase 5)
└── docs/                        ← arc42-style architecture documentation
```
