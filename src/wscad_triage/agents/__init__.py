"""Worker agents (Phase 3 — issues #19-#23).

Each agent module exposes a single ``run(state, llm, **deps) -> TicketState``
function. The supervisor (PR3) imports the modules and calls
``triage.run(state, llm)``, ``retrieve.run(state, llm, retriever=hybrid)``,
and so on; agents themselves never call each other.

Convention pinned by ADR-008 and the Phase-3 super-plan:

- Pure functions: ``run`` returns a *new* ``TicketState``; never mutates
  the input. Use ``state.model_copy(update={...})`` plus an extended
  ``reasoning_trace`` list. Clean reducer semantics for PR3's LangGraph
  wiring.
- Exactly one ``ReasoningStep`` per agent invocation. The ``actor`` field
  matches the agent's module name (``"triage"``, ``"retrieve"``, ...).
- Structured outputs (Triage, Reason, Clarify, Verify) go through the
  ToolSpec convention documented in ADR-004 — the agent declares one tool
  whose ``input_schema`` is the target Pydantic model's
  ``model_json_schema()`` and reads the result from
  ``LLMResponse.tool_calls[0].arguments``.
- LLM-failure modes (malformed JSON, missing tool call, schema violation)
  surface as exceptions. Agents do NOT catch; the supervisor decides
  retry policy.
"""

from wscad_triage.agents import clarify, reason, retrieve, supervisor, triage, verify

__all__ = ["clarify", "reason", "retrieve", "supervisor", "triage", "verify"]
