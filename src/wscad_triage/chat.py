"""Interactive chat REPL for the ticket-triage pipeline (Phase 8 — issue #65).

The batch CLI in :mod:`wscad_triage.cli` processes a JSON file of tickets in
one pass. This module wraps the same :func:`wscad_triage.pipeline.run`
entry point in a REPL so a reviewer can paste a ticket description, see a
compact agent response, answer the follow-up questions in-place, and watch
the pipeline re-run with the updated ticket. See ADR-011 for the design
rationale (re-invoke whole graph per turn; no LangGraph checkpointer;
append Q/A blocks to ``ticket.text``).

Slash commands: ``/quit``, ``/exit``, ``/help``, ``/details``, ``/reset``.
EOF (Ctrl-D / Ctrl-Z) is treated as ``/quit``. Solve outputs do *not*
auto-exit so the user can keep iterating.

The REPL writes three artefacts to ``out_dir`` after every turn, named by
ticket id: ``<id>.json`` (canonical Output), ``<id>.txt`` (full
:func:`render_text`), and ``<id>.chat.md`` (full markdown transcript).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from wscad_triage import pipeline
from wscad_triage.config import AppConfig, load_app_config
from wscad_triage.kb import HybridRetriever
from wscad_triage.llm import LLMClient
from wscad_triage.output import render_text, render_text_compact, write_json, write_text
from wscad_triage.schemas import Output, Ticket

# ---------------------------------------------------------------------------
# Constants

_QUIT_COMMANDS = frozenset({"/quit", "/exit"})

# Whitelist for ticket_id when used as a filesystem path component. A
# user-controlled ``ticket_id`` (free via pasted JSON) could otherwise
# traverse out of ``out_dir`` (``"../escape"``) or land hidden dot-files.
# Schema-level validation on Ticket would have wider blast radius; the
# sanitiser lives at the only place the id touches disk. Dots are
# excluded so a hostile ``"../"`` cannot leave any dot in the stem and
# the artefact extension (``.json`` etc.) is the only place a dot
# appears in the final filename.
_TICKET_ID_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_\-]")

# Best-effort patterns for extracting OS / version / product hints from
# free-text ticket descriptions and follow-up answers. The goal is to
# let a chat user write naturally ("Windows 11, version 7.3.0.16") and
# have the triage agent's deterministic gap pass clear without forcing
# them to paste JSON. Conservative by design: each pattern only sets
# the corresponding ``TicketMetadata`` field if it is currently
# ``None``, so a pre-existing JSON value is never clobbered, and an
# ambiguous text leaves the field empty for the deterministic pass to
# flag honestly.

_OS_PATTERN = re.compile(
    r"\b(Windows\s*(?:Server\s*)?\d+(?:\.\d+)?|macOS|Mac\s*OS\s*X?|Ubuntu|Debian|"
    r"Fedora|CentOS|RHEL|SUSE|Arch\s*Linux|Linux)\b",
    re.IGNORECASE,
)

# Version literals like ``7.3.0.16``, ``2.3``, ``v7.4.0.17``. We anchor on
# either a ``version`` / ``ver`` / ``v`` label, or accept a bare
# dotted-numeric near the start of a follow-up answer. Anchoring avoids
# matching the ``504`` in ``Error 504``.
_VERSION_PATTERN = re.compile(
    r"\b(?:version|ver|v)\s*[:=\"]?\s*v?([0-9]+(?:\.[0-9]+){1,3})\b",
    re.IGNORECASE,
)

# WSCAD product names. Curated list rather than a generic noun-phrase
# extractor: known products in scope per the brief and the KB corpus.
_PRODUCT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bWSCAD\s+Suite\b", re.IGNORECASE),
    re.compile(r"\bELECTRIX\s+AI\b", re.IGNORECASE),
    re.compile(r"\bELECTRIX\b", re.IGNORECASE),
)

SlashResult = Literal["continue", "quit", "reset"]

_BANNER = (
    "WSCAD ticket-triage chat. Type /help for commands; /quit to exit.\n"
    "Each turn re-runs the full agentic pipeline against your configured LLM.\n\n"
)

_HELP_TEXT = """Slash commands:
  /help        Show this message.
  /details     Reprint the last turn in full (reasoning trace, cited sources).
  /reset       Clear the current ticket and start a new one without exiting.
  /quit, /exit End the chat session (EOF / Ctrl-D / Ctrl-Z work too).

Anything that does not start with "/" is folded into the current ticket as
either an answer to the agent's pending follow-up questions or as
additional context, then the pipeline re-runs.
"""

_CHAT_TRANSCRIPT_SUFFIX = ".chat.md"


# ---------------------------------------------------------------------------
# Session state


@dataclass
class Turn:
    """One entry in the chat transcript."""

    role: Literal["user", "agent"]
    text: str
    output: Output | None = None


@dataclass
class ChatSession:
    """Mutable per-session state owned by the REPL loop."""

    ticket: Ticket
    history: list[Turn] = field(default_factory=list)
    turn_count: int = 0
    last_output: Output | None = None
    soft_cap_warned: bool = False


# ---------------------------------------------------------------------------
# Pure helpers — exposed for unit tests


def auto_ticket_id(now: Callable[[], datetime] | None = None) -> str:
    """Mint a synthetic ticket id anchored to the current UTC timestamp."""
    _now = (now or _utcnow)()
    return f"chat-{_now.strftime('%Y%m%d-%H%M%S')}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def parse_initial_input(raw: str, *, now: Callable[[], datetime] | None = None) -> Ticket:
    """Parse the user's initial input as either a Ticket JSON or free text.

    Auto-detection is cheap: if *raw* starts with ``{`` or ``[``, try to
    parse it as JSON and validate against :class:`Ticket` (or pick the
    first element of a JSON array). Any failure — malformed JSON, missing
    required fields, ``extra`` keys rejected by ``model_config`` — falls
    through to the free-text path, which mints a synthetic ticket id
    via :func:`auto_ticket_id`.

    Raises :class:`ValueError` if *raw* is empty or whitespace-only.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError("initial input is empty")
    if stripped.startswith(("{", "[")):
        ticket = _try_parse_ticket_json(stripped)
        if ticket is not None:
            return ticket
    return Ticket(ticket_id=auto_ticket_id(now), text=stripped)


def _try_parse_ticket_json(raw: str) -> Ticket | None:
    """Return a :class:`Ticket` if *raw* parses as one; otherwise None."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        if isinstance(data, list) and data:
            return Ticket.model_validate(data[0])
        if isinstance(data, dict):
            return Ticket.model_validate(data)
    except ValidationError:
        return None
    return None


def append_followup(ticket: Ticket, questions: list[str], answer: str, turn: int) -> Ticket:
    """Return a copy of *ticket* with a Q/A block appended to ``text``.

    The triage and reason agents read ``ticket.text`` directly, so folding
    the prior follow-up questions back in lets them see what was already
    asked and avoid asking the same gap twice. Format::

        [Follow-up turn N]
        Q: <question 1>
        Q: <question 2>
        A: <user's answer>

    Raises :class:`ValueError` on empty *answer* so the REPL can prompt
    again instead of silently no-op'ing the turn.
    """
    answer = answer.strip()
    if not answer:
        raise ValueError("follow-up answer is empty")
    q_lines = "\n".join(f"Q: {q}" for q in questions)
    addendum = f"\n\n[Follow-up turn {turn}]\n{q_lines}\nA: {answer}"
    return ticket.model_copy(update={"text": ticket.text + addendum})


def append_more_context(ticket: Ticket, more: str, turn: int) -> Ticket:
    """Return a copy of *ticket* with an additional-context block appended.

    Used when the user adds context *without* a prior set of follow-up
    questions to answer (i.e. the previous turn produced a solve and the
    user is refining, or providing more detail unprompted).
    """
    more = more.strip()
    if not more:
        raise ValueError("additional context is empty")
    addendum = f"\n\n[Additional context turn {turn}]\n{more}"
    return ticket.model_copy(update={"text": ticket.text + addendum})


def transcript_markdown(session: ChatSession) -> str:
    """Render the full session as a markdown transcript.

    The agent-turn header carries the classification + confidence + kind
    so the document is scannable on its own (a reviewer reading the
    saved ``.chat.md`` doesn't need to open the JSON to know which
    turn was a solve and which was a clarify).
    """
    lines: list[str] = [f"# Chat session {session.ticket.ticket_id}", ""]
    turn_num = 0
    for entry in session.history:
        if entry.role == "user":
            turn_num += 1
            lines.append(f"## Turn {turn_num} -- user")
            lines.append("")
            lines.append(entry.text)
            lines.append("")
        else:
            output = entry.output
            if output is None:
                continue
            header = (
                f"## Turn {turn_num} -- agent "
                f"(category={output.category}, priority={output.priority}, "
                f"conf={output.confidence:.2f}, kind={output.resolution_kind})"
            )
            lines.append(header)
            lines.append("")
            lines.append(render_text_compact(output).rstrip("\n"))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_artefacts(session: ChatSession, out_dir: Path) -> None:
    """Write the three per-turn artefacts (idempotent overwrite).

    No-op if the session has not produced an agent turn yet. The ticket id
    is sanitised via :func:`safe_filename` before path concatenation so a
    user-controlled JSON like ``{"ticket_id": "../escape"}`` cannot land
    artefacts outside *out_dir*.
    """
    if session.last_output is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(session.ticket.ticket_id)
    write_json(session.last_output, out_dir / f"{stem}.json")
    write_text(session.last_output, out_dir / f"{stem}.txt")
    transcript_path = out_dir / f"{stem}{_CHAT_TRANSCRIPT_SUFFIX}"
    transcript_path.write_text(transcript_markdown(session), encoding="utf-8")


def safe_filename(ticket_id: str) -> str:
    """Return a filesystem-safe stem for *ticket_id*.

    Replaces any character outside ``[A-Za-z0-9_-]`` with ``_`` (dots are
    intentionally excluded so the stem cannot carry path-traversal
    sequences and the artefact extension is the only dot in the final
    filename). Empty input collapses to ``"_"`` so the resulting path is
    always well-formed.
    """
    sanitised = _TICKET_ID_SAFE_CHARS.sub("_", ticket_id)
    return sanitised or "_"


def extract_metadata_hints(text: str) -> dict[str, str]:
    """Scan *text* for OS / version / product hints. Best-effort.

    Returns a partial mapping suitable for merging into
    :class:`TicketMetadata`. Only well-anchored matches are returned;
    when a pattern misses, the corresponding key is absent from the
    result, so the deterministic gap pass continues to flag the field.

    This is the chat-mode workaround for the fact that the triage
    agent's deterministic gap pass reads ``Ticket.metadata.{product,
    version, os}`` rather than ``Ticket.text`` — without this, a user
    typing "Windows 11, version 7.3.0.16" would keep being asked for OS
    and version on every turn.
    """
    hints: dict[str, str] = {}

    os_match = _OS_PATTERN.search(text)
    if os_match:
        hints["os"] = _normalise_os(os_match.group(1))

    ver_match = _VERSION_PATTERN.search(text)
    if ver_match:
        hints["version"] = ver_match.group(1)

    for pattern in _PRODUCT_PATTERNS:
        prod_match = pattern.search(text)
        if prod_match:
            hints["product"] = prod_match.group(0).strip()
            break

    return hints


def _normalise_os(raw: str) -> str:
    """Tidy up an OS hint without overreaching.

    Collapses internal whitespace so ``"Windows   11"`` becomes
    ``"Windows 11"``; preserves casing for canonical names like
    ``"macOS"`` by recognising them; otherwise returns the matched
    span verbatim.
    """
    cleaned = re.sub(r"\s+", " ", raw).strip()
    lower = cleaned.lower()
    if lower in {"mac os", "macos", "mac os x"}:
        return "macOS"
    if lower == "linux":
        return "Linux"
    return cleaned


def apply_metadata_hints(ticket: Ticket) -> Ticket:
    """Return a copy of *ticket* with any free-text metadata hints folded in.

    Existing non-``None`` fields are preserved; only previously empty
    fields are populated from hints. The full ``ticket.text`` is scanned
    so follow-up answer blocks contribute to subsequent triage passes.
    """
    hints = extract_metadata_hints(ticket.text)
    if not hints:
        return ticket
    updates: dict[str, str] = {}
    if ticket.metadata.product is None and "product" in hints:
        updates["product"] = hints["product"]
    if ticket.metadata.version is None and "version" in hints:
        updates["version"] = hints["version"]
    if ticket.metadata.os is None and "os" in hints:
        updates["os"] = hints["os"]
    if not updates:
        return ticket
    new_metadata = ticket.metadata.model_copy(update=updates)
    return ticket.model_copy(update={"metadata": new_metadata})


# ---------------------------------------------------------------------------
# Slash command dispatch


def is_slash(line: str) -> bool:
    """True if *line* starts with ``/`` after stripping."""
    return line.strip().startswith("/")


def handle_slash(
    cmd: str,
    session: ChatSession | None,
    write: Callable[[str], None],
) -> SlashResult:
    """Dispatch a slash command and tell the loop whether to continue.

    ``/details`` before the first turn prints a polite no-op rather than
    raising — slash commands should never crash the REPL.
    """
    cmd_lower = cmd.strip().lower()
    if cmd_lower in _QUIT_COMMANDS:
        return "quit"
    if cmd_lower == "/help":
        write(_HELP_TEXT)
        return "continue"
    if cmd_lower == "/details":
        if session is None or session.last_output is None:
            write("[chat] No turn to show yet -- type a description to start.\n")
        else:
            write(render_text(session.last_output))
        return "continue"
    if cmd_lower == "/reset":
        return "reset"
    write(f"[chat] Unknown command: {cmd_lower!r}. Type /help for the list.\n")
    return "continue"


# ---------------------------------------------------------------------------
# I/O


def read_multiline(read: Callable[[str], str], write: Callable[[str], None]) -> str:
    """Read lines until a blank one. Returns the joined text (stripped).

    Used for the initial ticket so reviewers can paste multi-line prose
    or pretty-printed JSON without escaping newlines. Subsequent
    follow-up answers are single-line.

    Raises :class:`EOFError` on Ctrl-D / Ctrl-Z so the caller can treat
    EOF as ``/quit``.
    """
    write("Describe your issue (paste JSON or free text; submit with a blank line):\n")
    lines: list[str] = []
    while True:
        prompt = "> " if not lines else "... "
        line = read(prompt)
        if line.strip() == "":
            if not lines:
                continue
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# REPL loop


def run_repl(
    *,
    llm: LLMClient,
    retriever: HybridRetriever,
    out_dir: Path,
    app_config: AppConfig | None = None,
    read: Callable[[str], str] | None = None,
    write: Callable[[str], None] | None = None,
    run_pipeline: Callable[[Ticket, LLMClient, HybridRetriever], Output] | None = None,
) -> int:
    """Run the chat REPL until the user quits. Returns the exit code.

    The I/O surfaces and pipeline runner are injected so tests can drive
    the loop deterministically without touching stdin/stdout. The
    defaults wire stdin/stdout and :func:`pipeline.run`.
    """
    cfg = (app_config or load_app_config()).chat
    _read = read or _default_read
    _write = write or _default_write
    _run = run_pipeline or pipeline.run

    _write(_BANNER)

    session: ChatSession | None = None
    while True:
        # Track pre-turn state so the pipeline-error path can roll back
        # the ticket mutation completely. The chat keeps both stateless
        # pipeline.run and stateful ticket.text mutation in the same turn,
        # so they must commit or roll back together.
        is_new_session = session is None
        ticket_before: Ticket | None = None if session is None else session.ticket

        if session is None:
            raw = _read_initial(_read, _write)
            if raw is None:
                _write("\n[chat] Goodbye.\n")
                return 0
            if raw == "":
                continue
            if is_slash(raw):
                action = handle_slash(raw, None, _write)
                if action == "quit":
                    _write("[chat] Goodbye.\n")
                    return 0
                continue
            try:
                ticket = parse_initial_input(raw)
            except ValueError as exc:
                _write(f"[chat] {exc}\n")
                continue
            # Best-effort metadata extraction so a user describing
            # "Windows 11, version 7.3" in prose doesn't get asked for
            # OS/version on every turn (ADR-011, Consequences).
            ticket = apply_metadata_hints(ticket)
            session = ChatSession(ticket=ticket)
            session.history.append(Turn(role="user", text=raw))
        else:
            try:
                line = _read("> ")
            except EOFError:
                _write("\n[chat] Goodbye.\n")
                save_artefacts(session, out_dir)
                return 0
            stripped = line.strip()
            if not stripped:
                continue
            if is_slash(stripped):
                action = handle_slash(stripped, session, _write)
                if action == "quit":
                    _write("[chat] Goodbye.\n")
                    save_artefacts(session, out_dir)
                    return 0
                if action == "reset":
                    save_artefacts(session, out_dir)
                    session = None
                    _write("[chat] Session reset.\n")
                continue
            turn_no = session.turn_count + 1
            # ``stripped`` is non-empty (guard at line above), so
            # append_followup / append_more_context cannot raise ValueError
            # from their own empty-input checks; no try/except needed.
            if session.last_output is not None and session.last_output.followup_questions:
                session.ticket = append_followup(
                    session.ticket,
                    session.last_output.followup_questions,
                    stripped,
                    turn_no,
                )
            else:
                session.ticket = append_more_context(session.ticket, stripped, turn_no)
            # Re-extract metadata after the user's answer: replies like
            # "Windows 11" or "version 7.3.0.16" populate the deterministic
            # gap fields so the next triage pass clears them.
            session.ticket = apply_metadata_hints(session.ticket)
            session.history.append(Turn(role="user", text=stripped))

        # Pipeline turn -- shared by initial and follow-up branches above
        try:
            output = _run(session.ticket, llm, retriever)
        except Exception as exc:
            # The REPL must survive an agent error rather than crash the
            # whole session; surface a friendly message and roll back the
            # pre-pipeline state. Initial-turn errors drop the freshly
            # created session entirely (user re-enters from scratch);
            # follow-up errors restore the prior ticket so the rejected
            # Q/A block does not re-attach on the next try.
            _write(f"[chat] Pipeline error: {exc}\n")
            if is_new_session:
                session = None
            else:
                assert session is not None  # narrowing for mypy
                assert ticket_before is not None
                session.ticket = ticket_before
                if session.history and session.history[-1].role == "user":
                    session.history.pop()
            continue
        session.turn_count += 1
        session.last_output = output
        session.history.append(Turn(role="agent", text="", output=output))
        _write(render_text_compact(output))
        save_artefacts(session, out_dir)

        if not session.soft_cap_warned and session.turn_count >= cfg.soft_turn_warning_at:
            _write(
                f"[chat] This session has run {session.turn_count} turns. "
                "Consider /quit and restarting with a clearer description if you're stuck.\n"
            )
            session.soft_cap_warned = True


def _read_initial(read: Callable[[str], str], write: Callable[[str], None]) -> str | None:
    """Read the initial multi-line input. Returns None on EOF."""
    try:
        return read_multiline(read, write)
    except EOFError:
        return None


def _default_read(prompt: str) -> str:
    return input(prompt)


def _default_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


__all__ = [
    "ChatSession",
    "SlashResult",
    "Turn",
    "append_followup",
    "append_more_context",
    "apply_metadata_hints",
    "auto_ticket_id",
    "extract_metadata_hints",
    "handle_slash",
    "is_slash",
    "parse_initial_input",
    "read_multiline",
    "run_repl",
    "safe_filename",
    "save_artefacts",
    "transcript_markdown",
]
