# ADR-002: Schema conventions for the typed-contract layer

- **Date:** 2026-05-07
- **Status:** Accepted

## Context

`CLAUDE.md` declares **"Pydantic at every boundary"** as one of five Architecture Principles: every value that crosses a component boundary in this codebase is a Pydantic model, never a raw `dict`. That principle decides *that* we type our contracts; it does not decide *how*.

Several non-trivial design choices have to be made up front because the schemas in `src/wscad_triage/schemas.py` will be referenced by every downstream component (KB loader, retrievers, agents, supervisor, output renderer, eval harness). Choosing a convention now and getting one ADR for the whole module is cheaper than retro-explaining it later.

The choices in question:

1. **Closed `Literal` enums vs open `str`** for fields like ticket category and priority — does the schema enforce the allowed value set, or just document it?
2. **`extra="forbid"` vs `extra="allow"`** — does an unknown JSON key raise `ValidationError`, or pass through silently?
3. **Cross-field validation on `Output`** — `proposed_solution` and `preliminary_assessment` are mutually exclusive depending on `resolution_kind`; should the schema enforce that, or do we trust callers?
4. **`frozen=True` for immutability** — should models be hashable / immutable, or not?

## Decision

The conventions in `src/wscad_triage/schemas.py` are:

- **Closed `Literal` for enumerated fields.** `Category`, `Priority`, `ResolutionKind`, and `RetrieverName` are `Literal[...]` aliases. The LLM-facing prompt receives the closed list; the schema rejects anything outside it. `Category` includes a deliberate `"Other"` catch-all so the LLM has a legal escape valve when no listed category fits — this is cheaper than dropping the schema-level guarantee.
- **`extra="forbid"` everywhere except `TicketMetadata`.** Strict-by-default; unknown keys are bugs and should fail loud. `TicketMetadata` is the one exception: real tickets carry vendor-specific keys (`vendor_internal_id`, ticketing-system fields, etc.) and rejecting them at the boundary would force every caller to pre-clean inputs. `TicketMetadata` uses `extra="allow"` so unknown keys round-trip intact; the triage agent decides what is critical.
- **Cross-field validation on `Output` via `model_validator(mode="after")`.** When `resolution_kind="solve"`, `proposed_solution` must be set and `preliminary_assessment` must be `None`; vice versa for `clarify`. Encoding the invariant in the schema means a malformed `Output` cannot escape the pipeline silently — the verifier agent or a buggy renderer would otherwise be the only line of defence.
- **No `frozen=True`.** Pydantic's `frozen=True` freezes the model wrapper but leaves contained `list` / `dict` containers mutable, which is a deceptive guarantee. The cost of full immutability (replacing every list with a tuple, every dict with a frozen mapping) is not justified for an issue scoped to data contracts. Revisit if a future caller relies on hashable models.
- **Bounded numeric fields use `Field(ge=..., le=...)`.** `Output.confidence` is constrained to `[0.0, 1.0]` so an out-of-range probability is a `ValidationError`, not a logic bug discovered during rendering.

## Consequences

- **Positive.** Every cross-component boundary has machine-checked invariants; reviewers and downstream agents can read `schemas.py` and trust the types. The `Literal` enums document the legal value set in a way the LLM prompt can paste verbatim. Round-trip JSON tests are mechanical and exhaustive.
- **Positive.** `extra="forbid"` catches typos in agent return values at construction time — a worker agent that returns `{"catagory": "Licensing"}` fails immediately rather than producing a silent default.
- **Positive.** The cross-field validator on `Output` makes it impossible to ship a self-contradictory result; this is exactly the kind of safety the brief's "production mindset" rubric rewards.
- **Negative.** `Literal` enums are extension-by-edit: adding a category requires touching `schemas.py` and re-running tests. This is the right trade-off for now (the WSCAD brief lists a stable set of categories) but is worth revisiting if categories become data-driven.
- **Negative.** `TicketMetadata`'s `extra="allow"` carve-out is a small inconsistency in the otherwise-strict module. It is documented here and in the model docstring; new schemas default to `extra="forbid"` unless they have the same vendor-passthrough requirement.
- **Negative — known follow-up.** `KBChunk.metadata: dict[str, str]` is the loose-metadata pattern handled with a typed dict rather than `extra="allow"`. KB frontmatter routinely carries non-string values (e.g., `synthetic_for_demo: false`); the loader (issue #9) is expected to coerce values to strings at load time, or this type widens to `dict[str, object]` with the trade-off documented in #9's PR. Parked here so the loader author does not need to re-derive the choice.
- **Locked-in.** Every downstream module imports these models; renaming a field is a refactor across `agents/`, `output/`, and `eval/`. Anticipated; the alternative (loose contracts) would be worse.
- **Eval harness extension.** `EvalTicket(Ticket)` (issue #32) extends `Ticket` with three ground-truth fields (`expected_category`, `expected_priority`, `should_clarify`) and a closed `CoverageCase` Literal for the seven Phase 5 test cases. The same conventions apply: closed `Literal` for enumerated fields, `extra="forbid"`, no cross-field validators beyond what inheritance already provides. The `notes` field is `str` with a default of `""` — it is human-readable rationale, not a typed contract, so no `Literal` constraint is appropriate.
