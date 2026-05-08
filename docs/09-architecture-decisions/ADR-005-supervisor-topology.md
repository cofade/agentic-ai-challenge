# ADR-005: Supervisor + workers topology in LangGraph (over linear pipeline / single ReAct)

- **Date:** 2026-05-08
- **Status:** Accepted

## Context

The system is a five-worker triage pipeline (triage, retrieve, reason,
clarify, verify — issues #19-#23). Three structurally different
ways to wire it:

1. **Linear pipeline.** Plain function composition: every ticket runs
   triage -> retrieve -> reason -> verify -> finalise. Cheapest to
   implement; loses any ability to skip work for tickets that triage
   already knows can't be answered, and forces every clarify-only
   ticket through reason+verify wastefully.
2. **Supervisor + workers in a state graph.** Workers are nodes; a
   thin "supervisor" layer decides routing between them. Splits into
   two sub-options based on how the supervisor decides:
   - **(b1) Deterministic edges.** Routing is a small set of pure
     functions on `TicketState`; no LLM calls in routing.
   - **(b2) LLM-as-supervisor.** Each "tick", an LLM call advertises
     a tool surface (`run_triage`, `run_retrieve`, ...,
     `finalize_solve`, `finalize_clarify`) and the tool call picks
     the next node.
3. **Single ReAct agent.** One LLM with all five workers exposed as
   tools; the LLM decides everything in one loop. Most flexible;
   hardest to make deterministic and to reason about.

Constraints we have to satisfy regardless of choice:

- **Acceptance gate (ROADMAP #25).** Integration tests cover five
  scenarios end-to-end and pass without network access. The choice
  must be testable with `MockLLMClient` and a deterministic stub
  retriever.
- **Acceptance gate (ROADMAP #24).** `pipeline.run(ticket)` returns a
  complete `Output`; a LangGraph state diagram is exported to
  `docs/05-building-block-view/`.
- **Architecture Principle (`CLAUDE.md`).** "Supervisor decides;
  workers execute. Agentic routing lives in the LangGraph
  supervisor's tool-call decisions, not in hardcoded transitions.
  New worker agents are added by registering tools, not by editing
  routing logic."
- **ADR-008 groundedness gate.** When the verifier returns a score
  below 0.4, the outcome must be downgraded to clarify. Whatever
  routing surface we pick has to compose cleanly with this gate.
- **Future flexibility.** Phase 5+ may eval that LLM-mediated routing
  beats the static thresholds for some edge cases. The choice should
  let us swap LLM-mediated decisions in *per-edge* without
  re-architecting the executor.

## Decision

**Option 2-(b1): Supervisor + workers in LangGraph with deterministic
conditional edges.**

Concretely:

- **Executor.** LangGraph's `StateGraph[TicketState]`. Each worker is
  a node; the graph has three terminal sinks (`finalize_solve`,
  `finalize_clarify`, `finalize_clarify_downgrade`) that each set
  `state.resolution_kind` and the appropriate body field
  (`proposed_solution` xor `preliminary_assessment`) before END.
- **Routing.** Two pure functions in
  `src/wscad_triage/agents/supervisor.py`:
  - `route_after_triage(state) -> "clarify" | "retrieve"`. Returns
    `"clarify"` iff `classification.missing_critical_fields` is
    non-empty.
  - `route_after_verify(state) -> "finalize_solve" | "finalize_clarify_downgrade"`.
    Returns `"finalize_clarify_downgrade"` iff
    `verifier_verdict.grounding_score < 0.4` (the ADR-008 threshold).
- **Forced transitions.** `triage -> retrieve -> reason -> verify`
  and `clarify -> finalize_clarify` are direct edges; no decision is
  needed.
- **Pipeline entry-point.** `pipeline.run(ticket, llm, retriever) ->
  Output` compiles the graph, invokes it, and projects the final
  state onto an `Output`. The compiled graph's
  `.get_graph().draw_mermaid()` is the source diagram exported to
  the building-block view.

### Why this and not (b2) LLM-as-supervisor

The two routing decisions in this pipeline are not ambiguous: they
are boolean tests on typed state (`missing_critical_fields` non-empty;
`grounding_score < 0.4`). Spending an LLM call to make a deterministic
decision is dead weight — it adds latency, adds cost, makes tests
harder, and *every wrong call is a regression*. The
"supervisor as LLM agent" framing in the ROADMAP (issue #24) reads
literally as (b2); we intentionally diverge and document the
divergence here.

The CLAUDE.md "agentic routing in tool-call decisions" principle is
preserved by the *workers*, not by the supervisor: each worker (triage,
reason, clarify, verify) is itself an LLM agent that emits a structured
tool call (`emit_classification`, `emit_draft`, `emit_questions`,
`emit_verdict`). The supervisor is the deterministic glue between
them — and the right place for that glue is exactly where it can be
unit-tested without a mock LLM.

### Why this and not (b2) is also not "we never use an LLM here"

The two routing functions live in `agents/supervisor.py` *as functions*
precisely so a future PR can replace one body with an LLM call without
touching the LangGraph topology, the worker contracts, or the conditional
edge wiring. "Add a worker" is still: register a node + add an edge
(per CLAUDE.md). "Make routing smarter" is still: replace the body of
one function. The principle survives; only the current implementation
is deterministic.

### Why three finalize sinks, not two

The clarify worker's precondition (non-empty
`missing_critical_fields`) does not hold on the ungrounded-claim
downgrade path: there are no metadata gaps, only a verifier verdict
saying the LLM's claims aren't supported. Routing the downgrade
through the clarify worker would either (a) violate the worker's
contract or (b) require synthesising fake metadata gaps, which would
in turn confuse the gap-matcher in the question validator.

The third sink (`finalize_clarify_downgrade`) keeps the clarify
worker's contract clean: it synthesises a fixed-shape
`preliminary_assessment` from the verifier's per-claim rationales
and emits no follow-up questions. The Output's cross-field validator
(ADR-002) enforces the solve/clarify mutual-exclusion regardless of
which sink ran.

### Why LangGraph and not a hand-rolled dispatcher

LangGraph adds one runtime dependency. In return:

- A drawable state diagram (`graph.get_graph().draw_mermaid()`) —
  the artefact that issue #24 acceptance criterion explicitly asks
  for.
- Persistence and streaming hooks already in place for any future
  long-running run.
- Industry-familiar shape: reviewers reading "supervisor + workers
  in LangGraph" know roughly what to expect.

A hand-rolled dispatcher would be ~30 lines today, but every
extension (checkpoint, retry, async branching) is more bespoke code
to maintain.

## Consequences

- **Positive — testable.** Both routing functions and all three
  finalize sinks are pure; they unit-test without LangGraph in the
  loop. The integration tests run the full graph end-to-end through
  `MockLLMClient` and a stub retriever — no network, no flake.
- **Positive — predictable.** The two boolean routing decisions are
  the only branch points; everything else is a direct edge. A
  clarify-required ticket short-circuits before retrieval (saves a
  vector-DB round trip on tickets we can't answer); an
  ungrounded-claim trap downgrades safely without re-running clarify
  on synthetic gaps.
- **Positive — extensible.** Adding a new worker is: register a
  LangGraph node + add an edge from / to the right neighbours.
  Replacing a routing decision with an LLM call is: change the body
  of one function. The rest of the topology stays put.
- **Negative — runtime dependency.** LangGraph + transitive deps
  (`langchain-core`, `langgraph-checkpoint`, ...). Pinned at
  `>=0.2.0` and locked via `uv.lock`. The transitive surface is
  larger than a hand-rolled dispatcher; the persistence/streaming
  hooks earn their keep when Phase 5 eval starts long-running
  parallel runs.
- **Negative — three terminal nodes.** Slightly more topology than
  a two-sink design. Justified above; the alternative (synthetic
  gaps in the downgrade path) was strictly worse.
- **Negative — divergence from CLAUDE.md "tool-call decisions"
  literal reading.** Documented in this ADR; the spirit
  ("supervisor decides; workers execute"; "new agents register, not
  edit routing logic") is preserved by the function-per-decision
  layout.
- **Locked-in.** `TicketState` carries one extra field
  (`preliminary_assessment`) so the downgrade sink has somewhere to
  put its synthesised text without piggybacking on
  `proposed_solution`. The clarify finalize sink also writes this
  field for renderer symmetry (#29).

## Cross-references

- ADR-002 — schema conventions (the `Output` cross-field validator
  enforces solve/clarify body exclusivity that the finalize sinks
  must satisfy).
- ADR-004 — LLM provider abstraction (every worker is an LLM agent
  through the `LLMClient` Protocol; the supervisor isn't).
- ADR-008 — groundedness safety gate (the 0.4 threshold consumed by
  `route_after_verify`).
- ADR-007 (Phase 4 #28) — `min(rubric_score, verifier_score)` will
  replace the verifier-score-as-confidence placeholder used by the
  finalize sinks.
- `docs/05-building-block-view/README.md` — exported state diagram.
- `docs/06-runtime-view/ticket-flow.md` — the three end-to-end
  sequences (resolvable, clarify, hallucination-trap) implemented
  by this topology.
