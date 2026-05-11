# Baseline sample outputs

Pre-rendered outputs for the brief's two sample tickets (`tickets/tickets.json`) — `T-001` and `T-002` — so a reviewer can see what the system actually produces without cloning and running it.

## Files

```
docs/baseline-outputs/
└── gpt-oss-20b/        ← Ollama gpt-oss:20b, the tested local backend (ADR-004)
    ├── T-001.json      ← canonical Pydantic Output, JSON form
    ├── T-001.txt       ← human-readable rendering (mirrors Sample_Output.txt format)
    ├── T-002.json
    └── T-002.txt
```

Captured on 2026-05-11 against `main` HEAD [`2409ad9`](https://github.com/cofade/agentic-ai-challenge/commit/2409ad9) plus the `feature/issue-64-release-notes-tickets` branch (which adds the clarify-agent pair-dict coercion fix discussed in the PR). The model is `ollama/gpt-oss:20b`. Re-running the same command on a fresh clone today should produce semantically equivalent outputs (model temperature is fixed at 0; minor tokenizer-driven variation between Ollama versions is possible).

## What both tickets route to

Both `T-001` and `T-002` route to the **clarify-missing-fields** sink under `gpt-oss:20b`. The triage agent classifies cleanly (T-001 → Errors/High, T-002 → Licensing/High) but flags multiple critical fields as missing — so the supervisor short-circuits to the clarify worker and the retriever never runs. This matches the brief's `Sample_Output #2` shape (the clarification-required case in `tickets/Sample_Output.txt`), not `Sample_Output #1` (the resolvable case).

A stronger model (`gpt-oss:120b`, Anthropic Claude Sonnet, a cloud frontier model) would more readily commit to the Licensing classification for T-001, find the *Error 504 → license re-activation* path via the KB chunks in `Common_Errors.md` + `Licensing_Offline_Activation.md`, and route to `finalize_solve`. The architecture supports this path; the test suite at [`tests/integration/test_pipeline.py:79`](../../tests/integration/test_pipeline.py) (`test_resolvable_ticket_returns_solve_output`) exercises it end-to-end with a deterministic `MockLLMClient`. The model is the variable — not the pipeline.

## Reproduce

```powershell
# Assumes a running Ollama server with gpt-oss:20b pulled (see top-level README).
uv run wscad-triage tickets/tickets.json --out docs/baseline-outputs/gpt-oss-20b/
```

The CLI emits one JSON + one text file per ticket, named after `ticket_id`. The text rendering mirrors the format of `tickets/Sample_Output.txt`.

## How to read the JSON

Each `*.json` file is one [`Output`](../../src/wscad_triage/schemas.py) Pydantic model — strictly typed, identical across providers, identical between solve and clarify paths (the path-specific fields are `proposed_solution` / `preliminary_assessment`). Fields the brief mandates at the top level:

| Brief field | JSON path |
|---|---|
| Proposed solution | `proposed_solution` (solve path) **or** `preliminary_assessment` (clarify-downgrade path) |
| Classification | `category` + `priority` |
| Confidence | `confidence` + `confidence_breakdown` (every contributing component named) |
| Reasoning trace | `reasoning_trace` (ordered list of `{actor, action, evidence_refs, rationale}`) |
| Optional follow-up questions | `followup_questions` (non-empty on clarify paths; empty on solve) |

For the full per-ticket eval-harness numbers (21 hand-labelled tickets, 8 coverage cases), see [`eval/results/latest.json`](../../eval/results/latest.json) and the **Baseline metrics** section of the top-level [`README.md`](../../README.md).
