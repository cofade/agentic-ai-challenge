# 6. Runtime View

This section describes how the system behaves at runtime — the dynamic interplay between the supervisor and the worker agents as a single ticket flows through the LangGraph state machine.

## Documents in this section

- [`ticket-flow.md`](ticket-flow.md) — The three canonical paths: resolvable, clarification-required, and the hallucination-trap path where the verifier intervenes.

Additional runtime sequences (e.g., batch processing, retry/backoff patterns, the eval harness loop) are documented here as they are added in Phases 3–5.

## Cross-references

- The static structure of the components named in these sequences lives in [`../05-building-block-view/`](../05-building-block-view/).
- Decisions about the supervisor topology and the verifier safety gate are captured in [`../09-architecture-decisions/`](../09-architecture-decisions/): [ADR-005](../09-architecture-decisions/ADR-005-supervisor-topology.md) (Accepted; supervisor topology in LangGraph); [ADR-008](../09-architecture-decisions/ADR-008-groundedness-gate.md) (Accepted; verifier safety gate).
