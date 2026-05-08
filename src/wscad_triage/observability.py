"""Per-trace logger factory.

Each agent calls ``get_logger(state.ticket.ticket_id).info(...)`` at
entry/exit so the reasoning of a single ticket can be reconstructed from
log output. Provides context-tagged logs (every record carries the
``trace_id`` extra) — but does NOT install a JSON formatter. CLAUDE.md's
Debugging Protocol still describes ``[DEBUG]`` ``print()`` for ad-hoc
debugging; this module is the durable replacement for *committed*
instrumentation.

This is a Phase-3 minimum. Full structured-JSON observability lands in
Phase 5 (one ``logging.Formatter`` subclass at the root logger); the
surface here keeps that upgrade local — every agent already calls
through ``get_logger``.
"""

from __future__ import annotations

import logging


def get_logger(trace_id: str) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger that prefixes every record with the ticket's trace id.

    Callers are expected to use the agent's module name as the action
    qualifier in the log message itself (``"triage.start"``,
    ``"retrieve.done"``, etc.) — keeps the LoggerAdapter's contextual
    state minimal.
    """
    base = logging.getLogger("wscad_triage")
    return logging.LoggerAdapter(base, {"trace_id": trace_id})
