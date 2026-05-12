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

# Interactive chat REPL (primary reviewer surface)
uv run wscad-triage chat

# Batch-process a tickets file (legacy / automation form)
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

For permanent instrumentation, use `src/wscad_triage/observability.py` — `get_logger(trace_id)` returns a `LoggerAdapter` that tags every record with the per-ticket trace ID. Phase 5 swaps the formatter for full structured JSON; the call-site surface stays the same. Never commit `print()` calls.

## Documentation Map

| Need | Location |
|------|----------|
| All documentation (start here) | [`docs/`](docs/) |
| The product roadmap (source of truth for issues) | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| System purpose, scope, evaluation criteria | [`docs/01-introduction-and-goals/`](docs/01-introduction-and-goals/) |
| Component structure and LangGraph diagram | [`docs/05-building-block-view/`](docs/05-building-block-view/) |
| Runtime sequences (happy path, clarify path) | [`docs/06-runtime-view/`](docs/06-runtime-view/) |
| Architecture decisions (ADRs) | [`docs/09-architecture-decisions/`](docs/09-architecture-decisions/) |
| Known limitations and technical debt | [`docs/11-risks-and-technical-debt/`](docs/11-risks-and-technical-debt/) |
| Domain glossary (RAG, RRF, BM25, agentic, ...) | [`docs/12-glossary/`](docs/12-glossary/) |
| The original challenge brief PDF | [`docs/WSCAD AI Challenge 2026.pdf`](docs/WSCAD%20AI%20Challenge%202026.pdf) |

## Workflow

| Step | Action | Notes |
|------|--------|-------|
| 1 | Read the issue | Understand acceptance criteria fully before starting; ask if unclear. |
| 2 | Enter Plan Mode | Produce a plan before writing code, especially for any non-trivial issue. |
| 3 | Move issue to **Doing** | The moment the plan is approved, move the GitHub Project status from `To Do` → `Doing`. Do this *before* creating the branch. |
| 4 | Create feature branch | `git checkout -b feature/issue-N-short-description`. Always branch off `main`; never commit on `main` directly. |
| 5 | Implement | Run quality checks incrementally — `uv run pytest`, `ruff`, `mypy` after each meaningful change. |
| 6 | Pre-commit checks pass | All hooks green: ruff lint, ruff format, mypy, bandit, pytest. |
| 7 | Senior-reviewer agent runs and returns "mergeable" | See "Pre-PR quality gates" below. Loop until clean. |
| 8 | Push branch + open PR | Conventional commit: `feat(#N): short description` or `chore(#N): ...`. PR body uses `Refs #N` (**not** `Closes #N`) — the user's manual-test pass is what closes the issue, not PR merge. |
| 9 | Move issue to **Resolved** | Project status `Doing` → `Resolved` once the PR is open. Implementation is finished but awaiting the user's manual verification. |
| 10 | User manually tests | The user runs the change locally and confirms it works as intended. Do not self-mark anything as Closed before this. |
| 11 | Merge, close issue, move to **Closed** | After the user signs off: merge the PR, run `gh issue close N`, and set the project card to `Closed`. |
| 12 | Clear context | `/clear` after merge — start each issue clean. |

**Never push directly to `main`.** Every change ships through a PR from a feature branch — no exceptions, including docs-only or "trivial" edits. The only commit that ever lands on `main` outside this flow is the one bootstrap commit that already exists.

**Never commit before pre-commit hooks pass. Never open a PR before the senior-reviewer verdict is clean. Never close an issue before the user has manually verified the change.**

### Project board state machine

| State | Meaning |
|-------|---------|
| `To Do` | Issue is in the backlog, plan not yet approved. |
| `Doing` | Plan is approved and an agent is actively implementing. Exactly one issue per agent should be in `Doing` at a time. |
| `Resolved` | Implementation is complete and the PR is open. Awaiting the user's manual test. |
| `Closed` | User has manually verified the change; PR is merged; issue is closed. |

### Useful gh commands for the board

```bash
# Find the project item id for an issue
gh project item-list 2 --owner cofade --format json | jq '.items[] | select(.content.number == N)'

# Move an issue (replace ITEM_ID, FIELD_ID, OPTION_ID with the values from the field-list call)
gh project item-edit --id ITEM_ID --project-id PVT_kwHOAyyXvs4BXCTE \
  --field-id PVTSSF_lAHOAyyXvs4BXCTEzhSR_U8 --single-select-option-id OPTION_ID

# Status option ids:
#   To Do     f75ad846
#   Doing     61e4505c
#   Resolved  47fc9ee4
#   Closed    98236657
```

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

**Current phase:** Phase 8 — Interactive chat CLI (in progress)
**Completed:** Phase 0 — Bootstrap (#1–#7); Phase 1 — Schemas & KB foundation (#8–#13); Phase 2 — KB extension (#14–#16, real release notes via #48); Phase 3 — LLM client + Anthropic/Azure/Ollama backends + five worker agents + supervisor + LangGraph (#17–#26, PRs #50–#52, #54, #56); Phase 4 — confidence formula + output rendering + CLI + ADRs (#27–#31, PRs #57–#58); Phase 5 — hand-labeled eval set + eval harness + baseline metrics + ADR-010 (#32–#35, PRs #60–#61); Phase 6 — reviewer-facing README + arc42 fill-in + ADR cross-refs (#36–#38, PR #62); Phase 7 — senior-reviewer pass + #39/#63 + release-notes-grounded eval tickets #64/#66
**In progress:** #65 — Interactive chat REPL (`wscad-triage chat`, ADR-011, Phase 8)
**Next up:** Phase 7 close-out remainder (#40 CI verify, #41 reviewer-clone smoke, #42 visibility + submission link)

(Update this section at the start of each session.)

## Architecture Principles

- **Pydantic at every boundary.** All ticket I/O, KB chunks, agent inputs/outputs, and the final `Output` are typed Pydantic models. The pipeline never passes raw `dict` between agents.
- **Knowledge base is the only authoritative source.** Agents may not introduce facts that are not present in retrieved KB chunks. The verifier agent enforces this; the rubric penalises it.
- **Supervisor routes deterministically; workers are the LLMs.** Each worker is itself an LLM agent emitting structured tool calls (`emit_classification`, `emit_draft`, `emit_questions`, `emit_verdict`); LangGraph wires them together. The supervisor — two pure routing functions on `TicketState` plus three finalize sinks — is *not* an LLM in Phase 3 ([ADR-005](docs/09-architecture-decisions/ADR-005-supervisor-topology.md)). LLM-mediated routing is a per-edge swap-in if eval shows the static thresholds underperform. New workers are added by registering a node + edge — not by editing routing logic.
- **Configuration over code.** A typed `Settings` object is the single source of runtime config. Provider selection and credentials come from environment variables / `.env` (Phase 3, see [ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md)); confidence-rubric weights, thresholds, and retrieval `k` move into `config.yaml` with Phase 4 issue #27. Nothing is magically hardcoded.
- **Deterministic core; LLM at the edges.** Retrieval, chunking, scoring math, output rendering are deterministic and unit-tested. LLM calls are mocked in tests and gated behind a real-key environment for live integration. The LLM provider is configurable per [ADR-004](docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md): `ollama` (default, self-hosted, runnable offline against a local server), `anthropic` (cloud), `azure` (production target, stub today).

## Known AI Pitfalls

(Empty initially. Add an entry every time an AI agent makes a mistake that took time to diagnose. Format: symptom → root cause → prevention.)

## ADR Triggers

Create an Architecture Decision Record in `docs/09-architecture-decisions/` when:
- Adding a new dependency (LangGraph, sentence-transformers, etc.).
- Choosing between two non-trivial implementation approaches (e.g., RRF vs score normalisation).
- Changing an established pattern in the codebase (e.g., switching the supervisor topology).
- Working around a constraint that is not obvious from the code (e.g., a model's tool-call quirk).

ADR format: title, date, status, context, decision, consequences. See `docs/09-architecture-decisions/README.md`.
