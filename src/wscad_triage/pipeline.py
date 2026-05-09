"""End-to-end ticket-triage pipeline (Phase 3 — issue #24, ADR-005).

Compiles the LangGraph state graph and exposes ``run(ticket, llm,
retriever) -> Output``. The CLI (Phase 4 #30) wraps this; integration
tests (issue #25) script the LLM with ``MockLLMClient`` and assert on the
returned ``Output``.

Topology (ADR-005): five worker nodes (triage, retrieve, reason, clarify,
verify) plus three terminal sinks (finalize_solve, finalize_clarify,
finalize_clarify_downgrade). Two conditional edges encode the routing
decisions; everything else is a direct edge. The conditional functions
live in :mod:`wscad_triage.agents.supervisor` so a future LLM-mediated
decision swaps in by replacing one function -- the LangGraph topology
stays the same.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from wscad_triage.agents import clarify, reason, retrieve, supervisor, triage, verify
from wscad_triage.config import load_app_config
from wscad_triage.kb import HybridRetriever
from wscad_triage.llm import LLMClient
from wscad_triage.schemas import Output, Ticket, TicketState


def build_graph(llm: LLMClient, retriever: HybridRetriever) -> Any:
    """Compile the supervisor state graph.

    Worker callables are bound with the ``llm`` / ``retriever`` at compile
    time so the LangGraph nodes are 1-arg ``state -> state`` functions.
    Returns the compiled graph (LangGraph's ``CompiledStateGraph``); the
    return type is ``Any`` because LangGraph does not export a stable
    public type alias.
    """
    app_config = load_app_config()
    weights = app_config.confidence.rubric_weights
    grounding_threshold = app_config.confidence.thresholds.grounding

    g: StateGraph[TicketState, Any, TicketState, TicketState] = StateGraph(TicketState)

    g.add_node("triage", lambda s: triage.run(s, llm))
    g.add_node("retrieve", lambda s: retrieve.run(s, llm, retriever=retriever))
    g.add_node("reason", lambda s: reason.run(s, llm))
    g.add_node("clarify", lambda s: clarify.run(s, llm))
    g.add_node("verify", lambda s: verify.run(s, llm))
    g.add_node("finalize_solve", lambda s: supervisor.finalize_solve(s, weights))
    g.add_node("finalize_clarify", lambda s: supervisor.finalize_clarify(s, weights))
    g.add_node(
        "finalize_clarify_downgrade",
        lambda s: supervisor.finalize_clarify_downgrade(s, weights),
    )

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        supervisor.route_after_triage,
        {"retrieve": "retrieve", "clarify": "clarify"},
    )
    g.add_edge("retrieve", "reason")
    g.add_edge("reason", "verify")
    g.add_conditional_edges(
        "verify",
        lambda s: supervisor.route_after_verify(s, grounding_threshold=grounding_threshold),
        {
            "finalize_solve": "finalize_solve",
            "finalize_clarify_downgrade": "finalize_clarify_downgrade",
        },
    )
    g.add_edge("clarify", "finalize_clarify")
    g.add_edge("finalize_solve", END)
    g.add_edge("finalize_clarify", END)
    g.add_edge("finalize_clarify_downgrade", END)
    return g.compile()


def run(ticket: Ticket, llm: LLMClient, retriever: HybridRetriever) -> Output:
    """End-to-end: ticket -> Output. Compiles the graph, invokes it, and
    coerces the final state back to ``Output``.
    """
    graph = build_graph(llm, retriever)
    final_state_raw = graph.invoke(TicketState(ticket=ticket))
    # LangGraph returns the state as a dict; coerce back to the typed
    # model so the ``_state_to_output`` helper can rely on the schema.
    state = TicketState.model_validate(final_state_raw)
    return state_to_output(state)


def state_to_output(state: TicketState) -> Output:
    """Project the final ``TicketState`` onto an ``Output``.

    Public so callers (e.g. the CLI in Phase 4 #30) can render a
    pre-existing state without re-running the graph. The Output's
    cross-field validator enforces that ``proposed_solution`` and
    ``preliminary_assessment`` are mutually exclusive per
    ``resolution_kind`` -- the finalize nodes are responsible for setting
    exactly one.

    ``cited_sources`` is populated only for ``resolution_kind="solve"``.
    On the clarify-downgrade path the verifier explicitly rejected the
    draft's claims, so listing those source files would misrepresent
    them as supporting evidence for an outcome that has no proposed
    solution. On the missing-fields clarify path there is no draft at
    all. Empty in both clarify cases.
    """
    if state.classification is None:
        raise ValueError(
            "pipeline.state_to_output: state.classification is None; the "
            "graph must have run triage to completion before this point"
        )
    if state.resolution_kind is None:
        raise ValueError(
            "pipeline.state_to_output: state.resolution_kind is None; one "
            "finalize node must have run before this point"
        )
    if state.final_confidence is None:
        raise ValueError(
            "pipeline.state_to_output: state.final_confidence is None; the "
            "finalize node is responsible for setting it"
        )

    cited: list[str] = []
    if state.resolution_kind == "solve" and state.draft_solution is not None:
        # Look up source_file via state.retrievals rather than parsing
        # the chunk_id literal -- the chunk_id format is the chunker's
        # convention (kb/chunker.py), not a contract pipeline.py should
        # depend on. The reason agent's layer-1 (a) check guarantees
        # every claim.chunk_id is in state.retrievals.
        source_by_chunk = {r.chunk.chunk_id: r.chunk.source_file for r in state.retrievals}
        cited = sorted({source_by_chunk[c.chunk_id] for c in state.draft_solution.claims})
    return Output(
        ticket_id=state.ticket.ticket_id,
        category=state.classification.category,
        priority=state.classification.priority,
        resolution_kind=state.resolution_kind,
        proposed_solution=state.proposed_solution,
        preliminary_assessment=state.preliminary_assessment,
        followup_questions=list(state.followup_questions),
        confidence=state.final_confidence,
        confidence_breakdown=dict(state.confidence_components),
        reasoning_trace=list(state.reasoning_trace),
        cited_sources=cited,
    )


__all__ = ["build_graph", "run", "state_to_output"]
