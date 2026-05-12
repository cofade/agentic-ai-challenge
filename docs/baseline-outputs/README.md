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

Re-captured on 2026-05-12 against the Phase 8 branch (issue #65) after the triage-prompt rewrite and reason-agent normaliser shipped. The model is `ollama/gpt-oss:20b`. Re-running the same command on a fresh clone today produces semantically equivalent outputs (model temperature is fixed at 0; sample-to-sample variation in claim emission is possible — see "What both tickets route to" below).

## What both tickets route to

`T-001` and `T-002` both end at a clarify sink under `gpt-oss:20b` on the artefacts committed here, **but the failure modes are now distinct**:

- **`T-001`** is classified `Licensing / High` (the brief's expected category) and the retriever **does run**, fetching `Common_Errors.md` and the release-notes chunks. The reason agent emits zero grounded claims on this particular run, so the verifier scores grounding 0.0 and the supervisor routes to `finalize_clarify_downgrade` per ADR-008. Earlier ad-hoc runs in the same PR observed `T-001` producing a fully grounded Licensing solve at confidence 1.0 against the same model — the variable is the reason agent's claim emission, which is sample-to-sample stochastic with `gpt-oss:20b`. The architecture works end-to-end (retrieval surfaces the right chunks, verifier is a real layer-2 gate); the local model's tool-use fidelity is the bottleneck.
- **`T-002`** has only `product` populated; the deterministic gap pass flags `version` + `os` as missing, so the supervisor short-circuits to `finalize_clarify` exactly as the brief's `Sample_Output #2` shape expects.

A stronger model (`gpt-oss:120b`, Anthropic Claude Sonnet, a cloud frontier model) would more reliably emit grounded claims on T-001 and route to `finalize_solve`. The architecture supports this path; the test suite at [`tests/integration/test_pipeline.py:79`](../../tests/integration/test_pipeline.py) (`test_resolvable_ticket_returns_solve_output`) exercises it end-to-end with a deterministic `MockLLMClient`. The model is the variable — not the pipeline.

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
