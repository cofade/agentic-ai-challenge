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
| 3 | LLM abstraction + supervisor and worker agents in LangGraph | Done |
| 4 | Confidence aggregation, output renderers, CLI | Done |
| 5 | Hand-labeled eval set + metrics harness | Done |
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

The CLI (`uv run wscad-triage tickets/tickets.json --out out/`) is implemented as of Phase 4 ([issue #30](https://github.com/cofade/agentic-ai-challenge/issues/30)). It requires a running LLM backend; see "Quickstart with Ollama" below for the offline path.

## Configuration

Phase 3 introduces a provider-agnostic LLM client (see [ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md)). Copy `.env.example` to `.env` and set the keys for the provider you want to use:

```bash
cp .env.example .env
# .env defaults to WSCAD_TRIAGE_PROVIDER=ollama; no API key needed.
```

Provider selection is `WSCAD_TRIAGE_PROVIDER=ollama|anthropic|azure`:

- **`ollama`** (default) — self-hosted, runs against a local [Ollama](https://ollama.com) server. The pipeline is runnable offline with no API account. See "Quickstart with Ollama" below.
- **`anthropic`** — cloud, paid. Uses the Anthropic SDK with prompt caching for KB context. Requires `ANTHROPIC_API_KEY` in `.env`.
- **`azure`** — Azure OpenAI is named as the production target in the brief but ships as a documented stub today; constructing the backend raises `ConfigurationError`. Full implementation is deferred ([ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md), "Future work").

Tests use a deterministic `MockLLMClient` and never hit the network. Two `live_api`-marked smoke tests live at `tests/integration/test_anthropic_live.py` and `tests/integration/test_ollama_live.py`; both are **deselected by default** via `addopts = ["-m", "not live_api"]` in `pyproject.toml`. Opt in with `uv run pytest -m live_api`. The Ollama test skips cleanly if the local server is unreachable or the configured model is not pulled; the Anthropic test skips if `ANTHROPIC_API_KEY` is unset.

### Quickstart with Ollama

```bash
# 1. Install Ollama (https://ollama.com/download)
# 2. Pull the project's tested model
ollama pull gpt-oss:20b

# 3. Start the server (new terminal, if not already running)
ollama serve

# 4. Run the pipeline
uv sync --all-extras
uv run wscad-triage tickets/tickets.json --out out/
```

Tool-use fidelity is model-dependent — `gpt-oss:20b` is the project's tested choice. Smaller open-weight models can emit malformed tool-call JSON; see [`docs/11-risks-and-technical-debt/`](docs/11-risks-and-technical-debt/) for the recommended-model list and the Phase 5 fidelity-benchmark plan.

## Baseline metrics

> Numbers recorded on 2026-05-10 against the 16-ticket hand-labeled eval set using `ollama/gpt-oss:20b`.
> They are illustrative — the model is intentionally conservative — not a performance promise.

| Metric | Value |
|--------|-------|
| Tickets in eval set | 16 |
| Pipeline errors | 2 |
| Category accuracy (error-free tickets) | 57.1% (8 / 14) |
| Priority accuracy (error-free tickets) | 42.9% (6 / 14) |
| Clarification precision | 0.429 |
| Clarification recall | 1.000 |
| Clarification F1 | 0.600 |
| Mean confidence (overall) | 0.134 |
| Mean confidence (solve only) | n/a — zero solve outputs |
| Mean confidence (clarify only) | 0.134 |

**Key observations:**

- `gpt-oss:20b` routed every error-free ticket to `clarify` (zero `solve` outcomes). Each ticket either triggered the missing-fields branch in triage or was downgraded by the groundedness gate ([ADR-008](docs/09-architecture-decisions/ADR-008-groundedness-gate.md)), which forces `clarify` when `grounding_score < 0.4`. A better-calibrated or cloud model will produce a different solve/clarify split.
- The 2 pipeline errors (E-08, E-09 — "clarify vague crash") stem from a known gap in the clarify agent's gap-reference validator: it requires each generated question to contain the exact gap identifier string (e.g. `steps_to_reproduce`), but `gpt-oss:20b` paraphrases instead of quoting. This is a pre-existing agent constraint, not introduced by the harness.
- Category accuracy varies by coverage case: the model handles `licensing vs installation` perfectly (100%) but struggles with `resolvable EN` and `multilingual mixed` (both 33.3%).

**Per-coverage-case breakdown:**

| Coverage case | n | Errors | Cat acc | Prio acc | Mean conf |
|---------------|---|--------|---------|----------|-----------|
| clarify missing OS | 2 | 0 | 50.0% | 50.0% | 0.062 |
| clarify vague crash | 2 | 2 | n/a | n/a | n/a |
| licensing vs installation | 2 | 0 | 100.0% | 100.0% | 0.125 |
| multilingual mixed | 3 | 0 | 33.3% | 33.3% | 0.125 |
| resolvable DE | 2 | 0 | 100.0% | 50.0% | 0.125 |
| resolvable EN | 3 | 0 | 33.3% | 33.3% | 0.167 |
| ungrounded claim trap | 2 | 0 | 50.0% | 0.0% | 0.188 |

**Reproduce:**

```bash
# Requires a running Ollama server with gpt-oss:20b pulled (see Quickstart with Ollama above)
uv run python -m eval.runner
# Writes eval/results/latest.json and a timestamped sibling for archival.
```

The canonical baseline file is [`eval/results/latest.json`](eval/results/latest.json). Evaluation design and metrics definitions are in [ADR-010](docs/09-architecture-decisions/ADR-010-evaluation-harness.md).

## Where to read further

- [`docs/`](docs/) — architecture documentation in arc42 style. Start with [`docs/README.md`](docs/README.md).
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — the build plan; one entry per GitHub Issue, organised by phase.
- [`docs/01-introduction-and-goals/`](docs/01-introduction-and-goals/) — system purpose, prioritised quality goals, stakeholders, constraints.
- [`docs/05-building-block-view/`](docs/05-building-block-view/) — component map and the LangGraph diagram of the supervisor + workers topology.
- [`docs/09-architecture-decisions/`](docs/09-architecture-decisions/) — ADRs. Ten are Accepted today: ADR-001 (language and runtime), ADR-002 (schema conventions), ADR-003 (KB NLP toolchain), ADR-004 (LLM provider abstraction — three backends including Ollama), ADR-005 (supervisor topology), ADR-006 (hybrid RAG strategy), ADR-007 (confidence quantification formula), ADR-008 (groundedness safety gate), ADR-009 (multilingual KB strategy), and ADR-010 (evaluation harness); the rest are pending and land alongside the components they document.
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
