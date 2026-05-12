# ADR-011: Interactive chat CLI on top of the one-shot pipeline

- **Date:** 2026-05-12
- **Status:** Accepted

## Context

Phase 4 shipped a batch CLI ([`src/wscad_triage/cli.py`](../../src/wscad_triage/cli.py)) that
consumes a JSON file of tickets and writes structured artefacts to disk —
one pass, one shot, no human in the loop. That format suits an automated
grader; it does not suit a WSCAD tester who wants to *experience* the
agentic loop. The clarify path is the most visibly "agentic" surface in
the system (the supervisor stops, asks 2–4 grounded questions, and waits
for an answer), but with a one-shot CLI those questions land on disk
and the loop never closes — the reviewer reads a JSON file instead of
having a conversation.

The challenge brief (§Not Required) lists *UI or frontend*; it does not
preclude a richer command-line surface, and it explicitly rewards
*agentic workflows*, *reasoning transparency*, and *production
mindset*. A REPL that lets a tester paste a ticket, see the agent's
response, answer follow-up questions in-place, and watch the pipeline
re-converge satisfies all three.

The forces:

1. The existing graph is a single-shot finite-state machine — every
   ``pipeline.run`` invocation starts at ``START`` and ends at one of
   three finalize sinks. There is no native pause/resume.
2. LangGraph supports checkpointers (``MemorySaver``, SQLite, Postgres)
   that snapshot graph state to a thread id, and supports
   ``interrupt`` nodes that pause mid-graph and resume on user input.
3. The clarify sink already emits a clean, structured ``Output`` with
   ``followup_questions``. Re-invoking the graph against an updated
   ``Ticket`` would re-triage with the new context — the same code
   path eval uses for every ticket — and is cheap enough offline against
   Ollama.
4. The triage agent merges a *deterministic* metadata gap pass with the
   LLM's missing-fields list ([`src/wscad_triage/agents/triage.py`](../../src/wscad_triage/agents/triage.py) line 84).
   That contract — the LLM may add gaps, never remove them — is
   load-bearing for clarify safety and must not be loosened for the
   chat surface.

## Decision

Build an **interactive REPL above the graph, not inside it.**

1. Add a ``wscad-triage chat`` subcommand wired to a new
   [`src/wscad_triage/chat.py`](../../src/wscad_triage/chat.py) module.
   The existing positional ``wscad-triage tickets.json`` form is
   preserved (routes to the batch subcommand) for backward
   compatibility.
2. The REPL re-invokes ``pipeline.run`` from scratch each turn.
   No LangGraph checkpointer; no ``interrupt`` node; no graph-internal
   chat state. Every turn is a complete pipeline pass on an evolving
   ``Ticket``.
3. Follow-up answers fold back into ``Ticket.text`` as a structured
   ``[Follow-up turn N]\nQ: ...\nA: ...`` block. The triage and reason
   agents read ``ticket.text`` directly, so the conversation history is
   visible to every agent without changing any prompt.
4. The renderer emits a **compact** per-turn summary
   (``render_text_compact``: category + priority + body + confidence;
   no reasoning trace, no source list); ``/details`` reprints the last
   turn in full via the existing ``render_text``.
5. Three artefacts saved per turn (idempotent overwrite by ticket id):
   ``<id>.json``, ``<id>.txt``, ``<id>.chat.md`` (markdown transcript
   of the whole session).
6. Slash commands ``/quit``, ``/exit``, ``/help``, ``/details``,
   ``/reset``. EOF is ``/quit``. Solve outputs do **not** auto-exit
   — the reviewer remains in control. A soft turn-cap warning fires
   around turn 8 (configurable via ``chat.soft_turn_warning_at`` in
   ``config.yaml``); the session never force-exits.

## Alternatives considered

- **MemorySaver checkpointer + ``interrupt`` at the clarify sink.**
  Pro: textbook "agentic" pattern. Con: meaningful graph-topology
  changes for a UX win that re-invoke covers; checkpoint state would
  need invalidation when the user's answer materially changes the
  classification (which is the common case). The graph itself is
  already cheap to re-run offline; checkpointing pays for itself only
  when each turn costs real money or wall time. Re-invoke wins on
  simplicity at the cost of one extra retrieval call per turn.
- **Structured-field extractor on the user's answer.** Pull
  ``os = "Windows 11"`` out of ``"Windows 11, .NET runtime missing"``
  and write it onto ``Ticket.metadata.os``. **Adopted in the same
  PR as this ADR** (``chat.extract_metadata_hints`` /
  ``chat.apply_metadata_hints``) after manual testing surfaced the
  obvious UX failure: without it, a chat user typing the answer to
  "Which OS?" gets asked again on the next turn because the deterministic
  gap pass reads ``ticket.metadata``, not ``ticket.text``. The
  implementation is regex-based with a curated product whitelist
  (``WSCAD Suite`` / ``ELECTRIX AI`` / ``ELECTRIX``); LLM-based
  extraction was rejected as over-engineered for an offline-Ollama
  demo. The extractor is conservative — it only fills empty fields
  and only matches well-anchored patterns — so an ambiguous reply
  still routes to clarify and never fabricates metadata.
- **Streaming token output per turn.** Pro: feels more chat-like.
  Con: no provider-portable streaming API yet (ADR-004 wraps three
  backends); block-print is sufficient for offline Ollama. Deferred.

## Consequences

**Positive.**

- Zero changes to the LangGraph topology, the agents, or the schemas.
  The REPL is a thin layer; existing tests for triage / clarify /
  reason / verify / supervisor all unchanged.
- The conversation is visible end-to-end in three places: the on-screen
  compact view, the ``.txt`` (full ``render_text``), and the
  ``.chat.md`` transcript. Reviewers can attach the markdown to a bug
  report or a slack message without parsing JSON.
- ``/details`` lets a curious reviewer see the reasoning trace and
  cited sources for the latest turn without cluttering the chat. The
  full transparency surface from Phase 4 is one keystroke away.
- The same ``pipeline.run`` entry point is exercised in batch CLI,
  chat REPL, and eval harness. One code path, three surfaces.

**Negative / locked in.**

- The regex-based metadata extractor (``chat.extract_metadata_hints``)
  is best-effort. It catches well-anchored patterns (``Windows 11``,
  ``version: 7.4.0.17``, ``WSCAD Suite``) but is silent on idiomatic
  variants (``"on Win11"``, ``"I'm running ver 7.4"``). When the
  extractor misses, the deterministic gap pass flags the field and the
  loop continues; we never fabricate metadata. The trade-off vs. an
  LLM-based extractor is documented above.
- Re-invoking the graph each turn means re-triage and re-retrieval per
  turn. Cheap offline; non-trivial against a paid cloud provider for
  long sessions. The soft turn-cap warning nudges users to ``/quit``
  if a session gets long.
- The renderer split — ``render_text`` (full) plus ``render_text_compact``
  (chat) — adds one entry point to ``wscad_triage.output``. Both
  share the same ``Output`` model; drift between them is a unit-test
  surface, not a runtime risk.
- The chat REPL writes artefacts after every turn with the same id,
  so the previous turn's artefacts are overwritten in place. The full
  history lives in ``<id>.chat.md`` for any session that wants to
  audit the path.

## Verification

- Unit tests at [`tests/unit/test_chat.py`](../../tests/unit/test_chat.py)
  cover the pure helpers (auto-id, JSON/free-text auto-detect, Q/A
  merging, transcript markdown, slash dispatch) and the REPL state
  machine (solve, clarify→answer→solve, EOF, pipeline error, reset,
  soft cap, help).
- Integration tests at
  [`tests/integration/test_chat_session.py`](../../tests/integration/test_chat_session.py)
  drive the REPL end-to-end against the real ``pipeline.run`` with a
  scripted ``MockLLMClient`` + ``StubRetriever``, covering one-turn
  solve and clarify→answer→solve.
- Manual smoke: ``uv run wscad-triage chat`` against a local Ollama
  server with ``gpt-oss:20b`` pulled, walking the clarify→answer→solve
  loop with a free-text ticket and inspecting the three saved artefacts.
