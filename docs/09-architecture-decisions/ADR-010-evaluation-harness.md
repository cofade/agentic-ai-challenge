# ADR-010: Evaluation harness — hand-labeled set + runner with provenance envelope

- **Date:** 2026-05-10
- **Status:** Accepted
- **Update 2026-05-11 (issue #64):** Eval set grew from 16 → 21 tickets and from 7 → 8 coverage cases (added `"release notes grounded"` — tickets E-17..E-21 in [`tickets/eval_set.json`](../../tickets/eval_set.json)). The numbers cited below (16 tickets, 7 cases) describe the Phase 5 snapshot at which this ADR was originally accepted; the architectural decisions (provenance envelope, per-coverage-case aggregation, single-author labelling, deliberate non-goal of population-level accuracy) all carry forward unchanged. The new coverage case fits the same scaffolding — only the `CoverageCase` Literal in [`src/wscad_triage/schemas.py`](../../src/wscad_triage/schemas.py) and the `REQUIRED_COVERAGE_CASES` set in [`tests/unit/test_eval_set.py`](../../tests/unit/test_eval_set.py) needed updating.

## Context

The pipeline is end-to-end runnable as of Phase 4 (ADRs 007 / 008 / 009),
but every quality claim about it is qualitative until something measures
it. Phase 5 closes that gap. The challenge brief asks for an explainable
result; reviewers will want a concrete signal that "explainable" means
"correct often enough to be useful," not just "well-typed."

What needs measuring, derived from the rubric implicit in the brief and
the four explicit metrics in roadmap item #33:

- **Category accuracy** — does the triage agent (ADR-005) pick the right
  one of `Licensing | Installation | Errors | Performance | Other`?
- **Priority accuracy** — does it return the right `Low | Medium | High |
  Critical`?
- **Clarification precision / recall** — when the ticket is genuinely
  underspecified, does the supervisor route to the clarify path? When the
  ticket is well-specified, does it stay on the solve path?
- **Mean grounded confidence** — for the tickets that produced a
  `proposed_solution`, how confident is the system on average? "Grounded"
  here is shorthand for "passed the verifier's groundedness gate per
  ADR-008," which is the only path to a ``solve`` outcome.

Why this is not a learned/large eval set:

- There are no production tickets to learn from. The brief provides two
  sample tickets; the KB documents the answers; everything else is
  hand-authored.
- One author (the project owner) labels every ticket. A larger set would
  not improve label quality and would amplify single-author bias.
- The seven coverage cases listed in roadmap item #32 are the deliberate
  stress-test surface — `resolvable EN`, `resolvable DE`, `clarify
  missing OS`, `clarify vague crash`, `licensing vs installation`,
  `ungrounded claim trap`, `multilingual mixed`. The eval set's job is
  to span these scenarios, not to estimate population-level accuracy.

Constraints from earlier ADRs:

- ADR-007 / ADR-008 fix the confidence formula and the groundedness
  gate. The eval harness must be able to read `Output.confidence` and
  `Output.resolution_kind` without re-implementing either; the harness
  scores existing outputs, it does not re-score them.
- ADR-002 ("Pydantic at every boundary") applies — every value the
  harness emits is a typed Pydantic model with `extra="forbid"`.
- ADR-004's provider abstraction means the harness must be
  provider-agnostic. A baseline run on Ollama must use the same code
  path as a future run on Anthropic or Azure.

## Decision

Ship a deterministic, provider-agnostic eval harness with a
reproducibility envelope. Three subsystems:

### 1. Dataset

`tickets/eval_set.json` — **16 hand-labeled tickets** (issue #32, merged
in PR #60). Each ticket extends the runtime `Ticket` schema with four
ground-truth fields plus free-text labeller notes:

```python
class EvalTicket(Ticket):
    expected_category: Category
    expected_priority: Priority
    should_clarify: bool
    coverage_case: CoverageCase   # one of the seven listed above
    notes: str = ""
```

Distribution invariants are pinned by `tests/unit/test_eval_set.py`:
≥15 tickets total, all seven `coverage_case` values represented, ≥5
clarify and ≥5 solve tickets, unique `ticket_id` values. The set is
intentionally bilingual (EN + DE) to exercise ADR-009's single-index
multilingual strategy.

### 2. Rubric

Implemented in `eval/metrics.py`; pure-Python, no I/O, unit-tested
without the LLM.

- **Category / priority accuracy.** Direct equality check against the
  `expected_*` fields. Computed over rows where the pipeline did not
  raise.
- **Clarification confusion matrix.** Treat `should_clarify=True` as the
  positive label and `Output.resolution_kind == "clarify"` as the
  positive prediction. The four cells are TP / FP / TN / FN; precision
  is TP/(TP+FP); recall is TP/(TP+FN); F1 is the harmonic mean.
- **Mean grounded confidence.** Mean of `Output.confidence` over rows
  where `Output.resolution_kind == "solve"`. The supervisor's only path
  to a `solve` outcome runs through the verifier (ADR-008), so any
  confidence on a solve output is grounded confidence by definition.
- **Undefined ratios return `None`.** When the denominator is zero
  (no positive predictions, no positive labels, no solve outputs, etc.)
  the metric is `None` rather than `0.0` or `nan` so a downstream reader
  cannot misinterpret "no data" as "perfect zero."
- **Per-coverage-case roll-up.** The same accuracy and confidence
  metrics, partitioned by `coverage_case`, surface which scenarios the
  pipeline degrades on. A category-level confusion matrix
  (`expected → actual → count`) accompanies the aggregate row.

### 3. Reproducibility envelope

Every run writes a `RunMetadata` block alongside the metrics:

```python
class RunMetadata(BaseModel):
    timestamp_utc: str
    provider: str          # "ollama" | "anthropic" | "azure"
    model: str             # resolved per-provider model identifier
    kb_chunk_count: int
    eval_set_path: str
    eval_set_sha256: str   # bytes of the eval set file
    git_commit: str        # best-effort `git rev-parse HEAD`
    wscad_triage_version: str
```

Without this, "reproduce the baseline" is guesswork — the same code on
different KB content, eval-set revisions, or model versions can produce
materially different numbers. The envelope makes drift attributable.

### 4. Tooling

`eval/runner.py` exposes two entry points:

- `run_eval(eval_tickets, llm, retriever, metadata, results_dir,
  pipeline_run=pipeline.run)` — testable kernel with `pipeline_run` as
  a keyword-only injection point so unit/integration tests can exercise
  the loop without scripting all five agents.
- `main(argv)` — argparse layer for `python -m eval.runner`. Mirrors
  `src/wscad_triage/cli.py` for argument shape, Settings construction
  (the alias-kwarg pattern), and KB / LLM-client construction.

Per-ticket exceptions are caught and recorded as `error` rows so one
malformed pipeline run cannot abort the whole eval. Construction errors
(missing eval set, missing KB directory, missing credentials) fail
fast before the loop. The runner exits 0 only when `n_errors == 0`;
otherwise it exits 1, so CI can detect regressions without parsing
`latest.json`.

Output is written to `eval/results/latest.json` (the canonical baseline,
committed) and `eval/results/{timestamp}.json` (timestamped local
history, gitignored). A plain-ASCII summary table prints to stdout.

## Consequences

- **Positive — measurable baseline.** The four acceptance-criteria
  numbers and the per-case breakdown are now in `eval/results/latest.json`
  and rendered into the README. A future change that regresses
  classification quality is visible in the diff of those numbers.
- **Positive — reproducibility envelope makes drift attributable.**
  When a re-run produces different numbers, the metadata block tells the
  reviewer whether the eval set, KB content, code, or model version
  moved.
- **Positive — pure-Python metrics layer.** The scoring functions are
  unit-tested independently of the LLM. The pipeline-bound integration
  test injects a fake `pipeline_run` so it stays fast and deterministic.
- **Positive — provider-agnostic by construction.** The same harness
  produces a baseline against Ollama, Anthropic, or a future Azure
  backend; comparing them is a matter of running the harness twice and
  diffing the metadata.
- **Negative — n=16 is too small for statistical confidence intervals.**
  The README's "Baseline metrics" section flags numbers as illustrative,
  not promises. A single ticket flipping from correct to wrong moves
  category accuracy by ~6.25 percentage points; a Wilson-interval CI
  would span double-digit ranges. The numbers are useful for spotting
  regressions, not for claiming absolute quality.
- **Negative — single-author labels carry rubric bias.** The
  `licensing vs installation` and `clarify vague crash` cases sit on
  judgment calls where a different labeller could plausibly disagree.
  The `notes` field on each `EvalTicket` records the labeller's
  reasoning so a second reviewer can audit individual rows; that does
  not eliminate the bias, only documents it.
- **Negative — LLM nondeterminism produces drift on identical code.**
  Ollama and Anthropic both apply temperature defaults; re-running the
  harness against the same commit yields slightly different numbers.
  The committed `latest.json` is one snapshot; a future ADR could pin
  seed/temperature for both providers if reproducibility-on-identical-code
  becomes important.
- **Negative — first committed baseline is degenerate.** The `ollama/gpt-oss:20b` run recorded in `eval/results/latest.json` produced zero `solve` outputs: every error-free ticket was routed to `clarify`, making `mean_confidence_solve` null and grounded-confidence comparisons between providers impossible from this snapshot alone. The numbers are still useful as a regression baseline (any future run that produces any `solve` outputs will be immediately comparable) and the degenerate outcome is itself diagnostic — it reveals that `gpt-oss:20b` cannot clear the 0.4 groundedness gate on this KB, which is a signal worth knowing. A cloud-provider run (Anthropic or Azure) is expected to produce a richer baseline.
- **Negative — no calibration metric.** Confidence calibration (does
  `confidence == 0.7` actually correspond to ~70% accuracy on those
  rows?) is meaningful only at larger n; ECE / reliability diagrams at
  n=16 are noise. Deferred. The mean confidence is reported per
  resolution kind so a gross miscalibration (high confidence on
  consistently wrong answers) would still be visible.

## Cross-references

- ADR-002 — Schema conventions: `extra="forbid"`, closed `Literal`
  enums; the result models in `eval/metrics.py` follow the same rules.
- ADR-005 — Supervisor topology: the `clarify` vs `solve` routing
  decision is what the clarification confusion matrix scores.
- ADR-007 — Confidence quantification: defines what
  `mean_confidence_*` is averaging.
- ADR-008 — Groundedness gate: defines why "mean confidence on solve
  outputs" is "mean grounded confidence."
- ADR-009 — Multilingual KB strategy: the `resolvable DE` and
  `multilingual mixed` coverage cases stress-test the single-index
  decision.
- `src/wscad_triage/schemas.py` — `EvalTicket`, `CoverageCase`.
- `eval/metrics.py` — scoring functions, result models.
- `eval/runner.py` — runner kernel, argparse layer, `RunMetadata`.
- `tickets/eval_set.json` — the labeled set itself.
- `tests/unit/test_eval_set.py` — distribution invariants on the eval
  set.
- `tests/unit/test_eval_metrics.py` — metric-correctness unit tests.
- `tests/integration/test_eval_runner.py` — runner end-to-end test
  with injected fake `pipeline_run`.
