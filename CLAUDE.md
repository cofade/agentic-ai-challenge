# agentic-ai-challenge — Claude Code Instructions

An agentic AI ticket-triage system: classifies WSCAD support tickets, retrieves grounding from a local knowledge base, decides whether to propose a solution or ask follow-up questions, and emits a structured, explainable result.

## Quick Reference

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run tests with branch coverage
uv run pytest tests/ -v --cov=src/wscad_triage --cov-report=term-missing

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Security scan
uv run bandit -r src/ --severity-level high

# Pre-commit (all hooks against all files)
uv run pre-commit run --all-files

# Run the pipeline end-to-end against the provided sample
uv run wscad-triage tickets/tickets.json --out out/

# Run the evaluation harness over the labeled set
uv run python -m eval.runner
```

## Debugging Protocol

When a bug requires instrumentation:
1. Add temporary `print()` statements with a `[DEBUG]` prefix at the suspected location.
2. Add `[DEBUG]` statements at agent entry/exit to trace the supervisor's routing decisions.
3. Run the reproduction case (a single ticket via the CLI is usually enough).
4. Capture the output and identify the root cause.
5. Remove ALL `[DEBUG]` statements before committing.

For permanent instrumentation, use `src/wscad_triage/observability.py` which emits structured JSON logs with a per-ticket trace ID. Never commit `print()` calls.

## Documentation Map

| Need | Location |
|------|----------|
| The product roadmap (source of truth for issues) | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| System purpose, scope, evaluation criteria | [`docs/01-introduction-and-goals.md`](docs/01-introduction-and-goals.md) |
| Component structure and LangGraph diagram | [`docs/05-building-block-view/`](docs/05-building-block-view/) |
| Runtime sequences (happy path, clarify path) | [`docs/06-runtime-view/`](docs/06-runtime-view/) |
| Architecture decisions (ADRs) | [`docs/09-architecture-decisions/`](docs/09-architecture-decisions/) |
| Known limitations and technical debt | [`docs/11-risks-and-technical-debt/`](docs/11-risks-and-technical-debt/) |
| Domain glossary (RAG, RRF, BM25, agentic, ...) | [`docs/12-glossary.md`](docs/12-glossary.md) |
| The original challenge brief PDF | [`docs/WSCAD AI Challenge 2026.pdf`](docs/WSCAD%20AI%20Challenge%202026.pdf) |

## Workflow

| Step | Action | Notes |
|------|--------|-------|
| 1 | Enter Plan Mode | Produce a plan before writing code, especially for any non-trivial issue. |
| 2 | Read the issue | Understand acceptance criteria fully before starting; ask if unclear. |
| 3 | Create feature branch | `git checkout -b feature/issue-N-short-description`. |
| 4 | Implement | Run quality checks incrementally — `uv run pytest`, `ruff`, `mypy` after each meaningful change. |
| 5 | Pre-commit checks pass | All hooks green: ruff lint, ruff format, mypy, bandit, pytest. |
| 6 | Senior-reviewer agent runs and returns "mergeable" | See "Pre-PR quality gates" below. Loop until clean. |
| 7 | Commit and PR | Conventional commit: `feat(#N): short description` or `chore(#N): ...`. PR body includes `Closes #N`. |
| 8 | Clear context | `/clear` after merge — start each issue clean. |

**Never commit before pre-commit hooks pass. Never open a PR before the senior-reviewer verdict is clean.**

## Quality Checks (run before every commit)

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run bandit -r src/ --severity-level high
uv run pytest tests/ -v
```

All five must pass. Fix failures before proceeding.

## Pre-PR quality gates (mandatory)

1. Local quality checks pass (the five commands above).
2. The senior-reviewer agent (`.claude/agents/senior-reviewer.md`) runs against the current branch and returns either "mergeable as-is" or "mergeable with [minor changes]."
3. If senior-reviewer raises P0s: address them, re-run the agent, loop until clean.
4. PR is opened only after both 1 and 2 pass.

## Progress Tracking

**Current phase:** Phase 0 — Bootstrap
**Completed:** —
**In progress:** Phase 0 issues #1–#7
**Next up:** Phase 1 — Schemas & KB foundation (issues #8–#13)

(Update this section at the start of each session.)

## Architecture Principles

- **Pydantic at every boundary.** All ticket I/O, KB chunks, agent inputs/outputs, and the final `Output` are typed Pydantic models. The pipeline never passes raw `dict` between agents.
- **Knowledge base is the only authoritative source.** Agents may not introduce facts that are not present in retrieved KB chunks. The verifier agent enforces this; the rubric penalises it.
- **Supervisor decides; workers execute.** Agentic routing lives in the LangGraph supervisor's tool-call decisions, not in hardcoded transitions. New worker agents are added by registering tools, not by editing routing logic.
- **Configuration over code.** Confidence-rubric weights, thresholds, retrieval `k`, and provider selection live in `config.yaml`. Code reads from a typed `Settings` object; nothing is magically hardcoded.
- **Deterministic core; LLM at the edges.** Retrieval, chunking, scoring math, output rendering are deterministic and unit-tested. LLM calls are mocked in tests and gated behind a real-key environment for live integration.

## Known AI Pitfalls

(Empty initially. Add an entry every time an AI agent makes a mistake that took time to diagnose. Format: symptom → root cause → prevention.)

## ADR Triggers

Create an Architecture Decision Record in `docs/09-architecture-decisions/` when:
- Adding a new dependency (LangGraph, sentence-transformers, etc.).
- Choosing between two non-trivial implementation approaches (e.g., RRF vs score normalisation).
- Changing an established pattern in the codebase (e.g., switching the supervisor topology).
- Working around a constraint that is not obvious from the code (e.g., a model's tool-call quirk).

ADR format: title, date, status, context, decision, consequences. See `docs/09-architecture-decisions/README.md`.
