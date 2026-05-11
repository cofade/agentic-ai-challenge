# Final senior-reviewer pass — pre-submission archive (issue #39)

- **Date:** 2026-05-11
- **Scope:** entire `main` HEAD (commit `829f1b2`) as the final pre-submission quality gate. Not a branch-vs-main diff.
- **Reviewer:** `.claude/agents/senior-reviewer.md` (Opus 4.7, "senior staff engineer, 20 years, in a bad mood today" persona)
- **Initial verdict:** mergeable. No P0 blockers; 6 P1s and 7 P2s; 4 architectural smells.
- **Post-fix verdict:** see [Re-review verdict](#re-review-verdict) at the bottom of this file.

## Why this archive exists

Phase 7's acceptance gate (per `docs/ROADMAP.md` #39) requires a senior-reviewer pass on the full `main` branch and the resulting report archived in this directory. This file is that archive. Anything called out and *not* fixed is captured under [Deferred items](#deferred-items) with rationale, so a reviewer auditing the gap between the report and the shipped code can see why each item was deferred rather than addressed.

## Initial review — full output

### Overall verdict (initial)

> This is shippable. It would not embarrass anyone — quite the opposite, it is materially above the bar I expect from take-home submissions, and it leaves an unusually small attack surface for a hostile reviewer. Test count is 354 with branch coverage, the local quality gates are green, CI on `main` was green on the most recent push, baseline metrics match the artifact byte-for-byte, and the architecture decisions are documented honestly enough that the candid divergence from a literal "agentic supervisor" reading is *itself* an ADR (ADR-005), which is the right call. There are no P0 blockers. There is a small but real set of P1s and P2s that I would still want fixed before submission — none of them threaten the verdict, but each one is the kind of thing a reviewer in a bad mood will quote when they want to push back. Fix the cheap ones; the rest you can defend.

### Things the initial review genuinely endorsed

- [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md) owns the deterministic-supervisor framing instead of papering over it. Buys credibility on the agentic-design axis.
- Layer-1 defences in [`src/wscad_triage/agents/reason.py:73-92`](../../src/wscad_triage/agents/reason.py) (fabricated `chunk_id` raises, quote-not-substring raises) — cheap deterministic safety the verifier doesn't have to spend an LLM call to catch.
- Supervisor test pins the 0.4 threshold on both sides in [`tests/unit/test_pipeline_supervisor.py:138-155`](../../tests/unit/test_pipeline_supervisor.py).
- Confidence + metrics computation is pure and I/O-free; the eval runner's metadata envelope (commit hash, eval-set sha256, KB chunk count) makes the README baseline reproducible to the byte.

### Concrete problems (initial — ranked by severity)

#### P0 (must fix before merge)

*None.*

#### P1 (should fix before submission)

1. **`README.md:62`, `src/wscad_triage/llm/anthropic_backend.py:8-12`** — README claimed `cache_control` wiring as a Configuration-section feature; the agent call sites do not actually exercise it.
2. **`pyproject.toml:16`** — `click>=8.1` declared as runtime dependency; zero imports anywhere in the codebase (`grep -r "import click\|from click"` empty). The CLI uses `argparse`.
3. **`docs/05-building-block-view/README.md:78`** — claimed "Click entry point"; same drift as #2.
4. **`docs/09-architecture-decisions/ADR-007-confidence-quantification.md:49-51`, `src/wscad_triage/confidence.py:64-66`, `docs/12-glossary/README.md:12`** — three places said "clarify path: final = rubric_score." Incorrect for the downgrade-clarify branch where the verifier ran and `compute_confidence` returns `min(rubric, verifier)`. ADR-007's own invariants table (line 74) already correctly noted `final = min(rubric, verifier)` for the downgrade case — the ADR contradicted itself.
5. **`docs/11-risks-and-technical-debt/README.md:14`** — cited a `[tool.mypy] packages = ["wscad_triage"]` block that does not exist in `pyproject.toml`. The substantive claim (eval/ is outside the mypy gate) is correct; the citation was wrong.
6. **`eval/results/latest.json:9`, `README.md` baseline section** — eval artifact's `git_commit` field is `b66373293b7f2e07952fba52a7e74fa17cdd7570`, which is the Phase 5 feature-branch HEAD. The diff between that commit and current `main` HEAD is doc-only plus an `eval/runner.py` refactor (`match/case` replacing `if/elif`) — so the metrics are still valid, but a reviewer cross-referencing `git log` would not find the commit on main.

#### P2 (nits, would be nice)

1. **`out/T-001.txt:6-13`** — the CLI run against the provided `tickets.json` routes to clarify under the default Ollama model. The README's mental model implies T-001 hits the solve path.
2. **`src/wscad_triage/confidence.py:12-14`** — the "0.5/0.7 caveat" bands documented in the module docstring are not consumed anywhere downstream; advertises a contract no one fulfills.
3. **`src/wscad_triage/cli.py:30-79`** — no preflight for the Ollama backend; first generate call surfaces network errors as a stack trace from inside the SDK.
4. **`tests/integration/test_pipeline.py`** — mocked-LLM scenarios test wiring, not prompt-elicitation behaviour. Defensible but a reviewer may ask the question.
5. **`docs/11-risks-and-technical-debt/README.md:13`** — LangSmith env-var caveat reads as "ships a privacy footgun, documents it" to a take-home reviewer.
6. **`pyproject.toml:8`** — project description says "decides between solving and clarifying" but the system emits three sinks (solve, missing-fields clarify, downgrade clarify).
7. **`eval/results/latest.json:23-38`** — category confusion matrix shows the model defaults to `Errors` regardless of true label; this is a *finding* the README didn't surface.

#### Architectural smells (initial)

1. **CLAUDE.md "Architecture Principles" block** had pre-ADR-005 wording ("Supervisor decides; workers execute. ... the supervisor stitches them together via a LangGraph state graph"). Technically true but a reviewer reading CLAUDE.md first then ADR-005 would notice the framing shift.
2. **Three-way confidence-prose drift** (covered in P1 #4 above).
3. **Retrieve agent rewrites DE tickets into English-tokenised queries** (`src/wscad_triage/agents/retrieve.py:46-58`). Works in the hybrid design because the embedding side compensates, but the trade-off was not surfaced as a Phase 5 finding.
4. **`out/` and `.VSCodeCounter/` working-tree clutter** — would shape an impression for a reviewer running `git status` first.

## Fixes applied this pass

Each fix lists the file(s) touched and the resolved finding.

| Finding | Fix | File(s) |
|---|---|---|
| P1 #1 (cache_control claim) | Reworded README Configuration entry; moved the "plumbing exists but not exercised" detail to a one-line aside linking to the risks doc. Also clarified that Anthropic subscriptions do not include API keys. | [README.md](../../README.md) |
| P1 #2 (dead `click` dep) | Removed `click>=8.1` from `[project].dependencies`. | [pyproject.toml](../../pyproject.toml) |
| P1 #3 ("Click entry point") | Changed to "`argparse` entry point" in the building-block table. | [docs/05-building-block-view/README.md](../05-building-block-view/README.md) |
| P1 #4 (confidence prose, 3 places) | Rewrote all three locations to distinguish "verifier ran (solve + downgrade) → `min(rubric, verifier)`" from "missing-fields clarify (verifier never ran) → rubric only." Added an explicit note in ADR-007 that the branch is decided by whether `state.confidence_components["verifier"]` is set. | [src/wscad_triage/confidence.py](../../src/wscad_triage/confidence.py), [docs/09-architecture-decisions/ADR-007-confidence-quantification.md](../09-architecture-decisions/ADR-007-confidence-quantification.md), [docs/12-glossary/README.md](../12-glossary/README.md) |
| P1 #5 (mypy citation) | Replaced the fictional `[tool.mypy] packages = ["wscad_triage"]` reference with the real gate (`uv run mypy src/` in `.github/workflows/ci.yml`). | [docs/11-risks-and-technical-debt/README.md](README.md) |
| P1 #6 (eval commit ref) | Added a one-line note in the README baseline section explaining that `git_commit` references the Phase 5 PR head and that only doc updates have landed on main since (verified via `git diff --stat b663732 HEAD -- ":(exclude)docs/*" ":(exclude)*.md"` — only `eval/results/latest.json`, `eval/runner.py`, and `tests/integration/test_eval_runner.py` changed). The metrics remain valid against current pipeline behaviour. Re-running the eval against `829f1b2` is a one-command refresh if desired. | [README.md](../../README.md) |
| P2 #1 (T-001 mental model) | Added an explicit note in the README Key Observations: the sample tickets exhibit the same clarify pattern under the Ollama default. | [README.md](../../README.md) |
| P2 #2 (unused band docstring) | Removed the 0.5/0.7-caveat band lines from `confidence.py`'s module docstring; tightened the module summary to describe only what the function returns. | [src/wscad_triage/confidence.py](../../src/wscad_triage/confidence.py) |
| P2 #3 (Ollama preflight) | Added `_preflight_ollama(base_url, model)` to the CLI. Two-stage gate (server reachable, then model present) raises `ConfigurationError` with an actionable message. Four new unit tests cover success, server unreachable, model not pulled, and the CLI exit-code-1 integration. Stdlib `urllib.request` only — no new runtime dependency. | [src/wscad_triage/cli.py](../../src/wscad_triage/cli.py), [tests/test_cli.py](../../tests/test_cli.py) |
| P2 #7 (category-confusion observation) | Added a bullet in the README Key Observations naming the "model defaults to `Errors`" failure mode and what a production deployment would do about it. | [README.md](../../README.md) |
| Smell #1 (CLAUDE.md wording) | Rewrote the "Supervisor decides; workers execute" principle to "Supervisor routes deterministically; workers are the LLMs" and anchored it to ADR-005's language. | [CLAUDE.md](../../CLAUDE.md) |

## Deferred items

Each entry includes the original finding, the rationale for not fixing it now, and where the risk is tracked if it persists.

- **P2 #4 (mocked-LLM scenarios pin wiring, not prompt-elicitation).** Defensible by design — the live Ollama smoke test (`tests/integration/test_ollama_live.py`) covers the tool-use roundtrip and the eval harness exercises the full pipeline against `gpt-oss:20b`. Splitting mocked tests into a "prompt fidelity" tier without a ground-truth fidelity benchmark is build-for-the-sake-of-it; the Phase 5 per-model fidelity benchmark (tracked at [docs/11-risks-and-technical-debt/README.md](README.md) "Ollama tool-use fidelity is model-dependent") is the proper resolution.
- **P2 #5 (LangSmith env-var caveat).** Left as documented. Scrubbing `LANGSMITH_*` / `LANGCHAIN_TRACING_V2` in the CLI startup is a behavior change; the current caveat is documented in the risks doc and the env vars are inert by default. A defensive code-level scrub at submission time would itself be an unexpected behavior that a reviewer could read as "candidate is patching around langgraph's transitive behavior at the application layer" — net-neutral. Keep as-is.
- **P2 #6 (pyproject description wording).** Cosmetic; "decides between solving and clarifying" is technically accurate at the user-visible Output level (`resolution_kind ∈ {solve, clarify}`); the three-sinks distinction is internal routing, surfaced by ADR-005.
- **Smell #3 (retrieve-agent English rewrite for DE tickets).** Recorded behaviour but not yet a finding worth a code change — the hybrid design tolerates it by construction (the embedding side carries the load on language-mixed queries) and the eval set's `multilingual mixed` coverage case is the leading indicator if it ever regresses. A future per-language re-ranker is the proper resolution; out of scope for Phase 7.
- **Smell #4 (`out/` and `.VSCodeCounter/` clutter).** `out/` is in `.gitignore` (verified — does not appear in `git status`) so it never ships to a reviewer's clone. `.VSCodeCounter/` was previously untracked-but-not-gitignored; added to `.gitignore` in this fix pass so it stays out of a reviewer's `git status` as well. Neither directory ships to a fresh clone.
- **Re-running the eval against current `main` HEAD.** Considered — Ollama and `gpt-oss:20b` are available locally. Skipped because the only code change since `b663732` is an `eval/runner.py` `match/case` refactor that does not affect pipeline behaviour; the existing artifact's numbers are reproducible byte-for-byte against the current code. The README note now states this explicitly. If a re-run is desired post-submission: `uv run python -m eval.runner`.

## Re-review verdict

Run date: 2026-05-11. Reviewer: senior-reviewer agent (Opus 4.7, fresh-eyes pass on the post-fix branch).

**Status:** mergeable with minor changes. No P0s, two new P1s, five new P2s.

**Previous P0/P1/P2 status (judged on current state, not on follow-through):**
- P1 #1 (cache_control claim) — resolved in `README.md`. (New finding: same drift survives in `docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md:106-108`; logged as new P1 below.)
- P1 #2 (dead `click` dep) — resolved cleanly in `pyproject.toml` and `uv.lock`.
- P1 #3 (Click→argparse) — resolved in `docs/05-building-block-view/README.md`.
- P1 #4 (3-way confidence prose) — resolved in `src/wscad_triage/confidence.py`, ADR-007, and the glossary. (New finding: same drift survives in `docs/11-risks-and-technical-debt/README.md:14`; logged as new P1 below.)
- P1 #5 (mypy citation) — resolved; CI gate is correctly cited.
- P1 #6 (eval commit ref) — resolved; README footnote accurate.
- P2 #1, #2, #3, #7 and Smell #1 — all resolved.

**New findings raised by the re-review:**

*P1 (should fix before submission):*
1. `docs/09-architecture-decisions/ADR-004-llm-provider-abstraction.md:106-108` retains the cache-marker overclaim. Same drift the README fix addressed. No agent in `src/wscad_triage/agents/*.py` sets `cache=True`. Rewrite to match the corrected README framing.
2. `docs/11-risks-and-technical-debt/README.md:14` still describes the formula as "`rubric_score` alone on the clarify path." Same drift the author already fixed in three other places. Rewrite to distinguish verifier-ran from missing-fields-clarify.

*P2 (nits):*
- `docs/06-runtime-view/ticket-flow.md:5` summary uses the older two-way framing; per-path detail at lines 38-41 and 53-56 is correct.
- `tests/test_cli.py:208-211`: orphan duplicate section banner from the Ollama preflight insert. Delete.
- `README.md:170` category-confusion bullet slightly overstates the failure mode (Licensing column mostly maps correctly).
- This archive's Smell #4 rationale incorrectly states `.VSCodeCounter/` is gitignored; it is not (it is untracked but absent from `.gitignore`). Either gitignore it or amend the rationale.
- `senior-reviewer-final.md` itself is currently untracked; `git add` it before opening the PR.

**Architectural smells:** the same doc-drift pattern (an overclaim corrected in N places but missed in N+1 and N+2) repeated twice across this pass. A `git grep` for the original problematic phrases after applying corrections would have caught both.

**Recommendation:** apply the two P1 fixes plus the four P2 nits (all five-minute edits), commit the archive in the same commit, then ship. No re-run of the senior-reviewer required after that — the remaining items are mechanical doc deletions and a `git add`, not behavioural changes.

## Resolution of re-review findings

All seven re-review findings were addressed in the same fix pass that produced this archive:

| Finding | Resolution |
|---|---|
| New P1 #1 (ADR-004 cache_control overclaim) | Reworded ADR-004 §3 Anthropic block to match the corrected README framing: plumbing exists, agents do not yet set `cache=True`, deferred. |
| New P1 #2 (risks doc confidence formula) | Rewrote the bullet at [`docs/11-risks-and-technical-debt/README.md`](README.md) line 14 to distinguish "verifier ran" from "missing-fields clarify," consistent with the now-corrected ADR-007. |
| P2 #1 (ticket-flow.md:5 summary) | Rewrote the summary intro to the three-way framing. |
| P2 #2 (orphan banner in `tests/test_cli.py:208-211`) | Banner deleted. |
| P2 #3 (README:170 category-confusion overstatement) | Rewritten as "asymmetric — Licensing mostly maps correctly, Installation and Other default to Errors." |
| P2 #4 (`.VSCodeCounter/` gitignore claim) | Added `.VSCodeCounter/` to `.gitignore` and amended this archive's Smell #4 rationale to match. |
| P2 #5 (this archive untracked) | This file ships in the issue #39 PR alongside the fix pass; `git add` is part of the commit. |

A grep sweep was run after these fixes against the original problematic phrases ("`rubric_score` alone on the clarify path", "cache_control markers for KB", "on the clarify short-circuit") to verify no remaining drift. The re-review recommendation was explicit that no further senior-reviewer run is required after these mechanical doc fixes; archive is therefore considered final for issue #39.
