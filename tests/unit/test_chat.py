"""Unit tests for the interactive chat REPL (Phase 8 — issue #65)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from wscad_triage import chat
from wscad_triage.config import AppConfig, ChatConfig
from wscad_triage.schemas import Output, ReasoningStep, Ticket

# ---------------------------------------------------------------------------
# Output fixtures


def _solve_output(ticket_id: str = "chat-T-1") -> Output:
    return Output(
        ticket_id=ticket_id,
        category="Licensing",
        priority="High",
        resolution_kind="solve",
        proposed_solution="Re-activate the offline license.",
        confidence=0.81,
        reasoning_trace=[
            ReasoningStep(
                actor="supervisor",
                action="finalize_solve",
                evidence_refs=[],
                rationale="Grounding passed.",
            )
        ],
        cited_sources=["Licensing_Offline_Activation.md"],
    )


def _clarify_output(
    ticket_id: str = "chat-T-1",
    questions: list[str] | None = None,
) -> Output:
    return Output(
        ticket_id=ticket_id,
        category="Installation",
        priority="Medium",
        resolution_kind="clarify",
        preliminary_assessment="Multiple plausible root causes.",
        followup_questions=questions
        or [
            "Which operating system and version is in use?",
            "Which WSCAD product version is installed?",
        ],
        confidence=0.32,
        reasoning_trace=[
            ReasoningStep(
                actor="triage",
                action="missing_critical_fields",
                evidence_refs=["os", "version"],
                rationale="OS and version absent.",
            )
        ],
    )


# ---------------------------------------------------------------------------
# auto_ticket_id


def test_auto_ticket_id_uses_injected_clock() -> None:
    fixed = datetime(2026, 5, 12, 14, 30, 0, tzinfo=UTC)
    assert chat.auto_ticket_id(now=lambda: fixed) == "chat-20260512-143000"


def test_auto_ticket_id_default_is_well_formed() -> None:
    result = chat.auto_ticket_id()
    assert result.startswith("chat-")
    # YYYYMMDD-HHMMSS = 15 chars after the prefix
    assert len(result) == len("chat-") + 15


# ---------------------------------------------------------------------------
# parse_initial_input


def test_parse_initial_input_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        chat.parse_initial_input("   \n  ")


def test_parse_initial_input_freetext_mints_synthetic_id() -> None:
    fixed = datetime(2026, 5, 12, 14, 30, 0, tzinfo=UTC)
    ticket = chat.parse_initial_input("App crashes on startup", now=lambda: fixed)
    assert ticket.ticket_id == "chat-20260512-143000"
    assert ticket.text == "App crashes on startup"
    assert ticket.metadata.product is None


def test_parse_initial_input_accepts_ticket_dict() -> None:
    raw = json.dumps({"ticket_id": "T-001", "text": "Error 504", "metadata": {"os": "Windows 11"}})
    ticket = chat.parse_initial_input(raw)
    assert ticket.ticket_id == "T-001"
    assert ticket.metadata.os == "Windows 11"


def test_parse_initial_input_accepts_array_uses_first_element() -> None:
    raw = json.dumps(
        [
            {"ticket_id": "T-001", "text": "First"},
            {"ticket_id": "T-002", "text": "Second"},
        ]
    )
    ticket = chat.parse_initial_input(raw)
    assert ticket.ticket_id == "T-001"


def test_parse_initial_input_malformed_json_falls_back_to_freetext() -> None:
    # Looks like JSON, isn't quite — should NOT crash, should be treated as prose.
    raw = "{not really json"
    ticket = chat.parse_initial_input(raw)
    assert ticket.ticket_id.startswith("chat-")
    assert ticket.text == raw


def test_parse_initial_input_json_with_extra_keys_falls_back_to_freetext() -> None:
    # Ticket has extra="forbid"; an unknown key should fall through.
    raw = json.dumps({"ticket_id": "T-001", "text": "x", "bogus_field": 42})
    ticket = chat.parse_initial_input(raw)
    assert ticket.ticket_id.startswith("chat-")
    assert ticket.text == raw


# ---------------------------------------------------------------------------
# append_followup / append_more_context


def test_append_followup_attaches_q_a_block() -> None:
    ticket = Ticket(ticket_id="T-1", text="App crashes.")
    questions = ["Which OS?", "Which version?"]
    updated = chat.append_followup(ticket, questions, "Windows 11, v2.3", turn=2)
    assert updated.ticket_id == "T-1"
    assert "App crashes." in updated.text
    assert "[Follow-up turn 2]" in updated.text
    assert "Q: Which OS?" in updated.text
    assert "Q: Which version?" in updated.text
    assert "A: Windows 11, v2.3" in updated.text


def test_append_followup_does_not_mutate_input() -> None:
    ticket = Ticket(ticket_id="T-1", text="Original")
    chat.append_followup(ticket, ["Q?"], "A", turn=1)
    assert ticket.text == "Original"


def test_append_followup_empty_answer_raises() -> None:
    ticket = Ticket(ticket_id="T-1", text="x")
    with pytest.raises(ValueError, match="answer is empty"):
        chat.append_followup(ticket, ["Q?"], "   ", turn=1)


def test_append_more_context_appends_block() -> None:
    ticket = Ticket(ticket_id="T-1", text="Original")
    updated = chat.append_more_context(ticket, "Extra detail", turn=3)
    assert "Original" in updated.text
    assert "[Additional context turn 3]" in updated.text
    assert "Extra detail" in updated.text


def test_append_more_context_empty_raises() -> None:
    ticket = Ticket(ticket_id="T-1", text="x")
    with pytest.raises(ValueError, match="empty"):
        chat.append_more_context(ticket, "", turn=1)


# ---------------------------------------------------------------------------
# transcript_markdown


def test_transcript_markdown_two_turn_session() -> None:
    ticket = Ticket(ticket_id="chat-T-1", text="initial")
    session = chat.ChatSession(ticket=ticket)
    session.history.append(chat.Turn(role="user", text="App crashes on startup."))
    session.history.append(chat.Turn(role="agent", text="", output=_clarify_output("chat-T-1")))
    session.history.append(chat.Turn(role="user", text="Windows 11"))
    session.history.append(chat.Turn(role="agent", text="", output=_solve_output("chat-T-1")))

    md = chat.transcript_markdown(session)
    assert md.startswith("# Chat session chat-T-1\n")
    assert "## Turn 1 -- user" in md
    assert "## Turn 1 -- agent (category=Installation," in md
    assert "## Turn 2 -- user" in md
    assert "## Turn 2 -- agent (category=Licensing," in md
    assert "kind=solve" in md
    assert "kind=clarify" in md


# ---------------------------------------------------------------------------
# save_artefacts


def test_save_artefacts_writes_three_files(tmp_path: Path) -> None:
    ticket = Ticket(ticket_id="chat-T-42", text="hello")
    session = chat.ChatSession(ticket=ticket)
    session.last_output = _solve_output("chat-T-42")
    session.history.append(chat.Turn(role="user", text="hello"))
    session.history.append(chat.Turn(role="agent", text="", output=session.last_output))

    chat.save_artefacts(session, tmp_path)
    assert (tmp_path / "chat-T-42.json").exists()
    assert (tmp_path / "chat-T-42.txt").exists()
    assert (tmp_path / "chat-T-42.chat.md").exists()

    # JSON round-trip
    restored = Output.model_validate(
        json.loads((tmp_path / "chat-T-42.json").read_text(encoding="utf-8"))
    )
    assert restored == session.last_output

    # Markdown transcript has the right shape
    md = (tmp_path / "chat-T-42.chat.md").read_text(encoding="utf-8")
    assert "Chat session chat-T-42" in md


def test_save_artefacts_noop_before_first_turn(tmp_path: Path) -> None:
    ticket = Ticket(ticket_id="chat-T-1", text="hello")
    session = chat.ChatSession(ticket=ticket)
    chat.save_artefacts(session, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_save_artefacts_sanitises_ticket_id_against_path_traversal(tmp_path: Path) -> None:
    """A user-pasted JSON with a hostile ticket_id must not escape out_dir."""
    ticket = Ticket(ticket_id="../escape", text="hello")
    session = chat.ChatSession(ticket=ticket)
    session.last_output = _solve_output("../escape")
    session.history.append(chat.Turn(role="user", text="hello"))
    session.history.append(chat.Turn(role="agent", text="", output=session.last_output))
    chat.save_artefacts(session, tmp_path)

    # Nothing should be written outside tmp_path
    parent = tmp_path.parent
    assert not (parent / "escape.json").exists()
    assert not (parent / "escape.txt").exists()
    # All artefacts land inside tmp_path under a sanitised stem (no dots in stem)
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["___escape.chat.md", "___escape.json", "___escape.txt"]


def test_extract_metadata_hints_pulls_os_version_product() -> None:
    text = "Application crashes on Windows 11. Product: WSCAD Suite version: 7.3.0.16"
    hints = chat.extract_metadata_hints(text)
    assert hints == {"os": "Windows 11", "version": "7.3.0.16", "product": "WSCAD Suite"}


def test_extract_metadata_hints_handles_label_free_text() -> None:
    text = "I am using ELECTRIX AI v7.4.0.17 on macOS"
    hints = chat.extract_metadata_hints(text)
    assert hints.get("os") == "macOS"
    assert hints.get("version") == "7.4.0.17"
    assert hints.get("product") == "ELECTRIX AI"


def test_extract_metadata_hints_does_not_match_error_codes() -> None:
    text = "Error 504 occurred. The application failed."
    hints = chat.extract_metadata_hints(text)
    # "504" must not become a version; no version label = no match
    assert "version" not in hints


def test_apply_metadata_hints_preserves_existing_values() -> None:
    ticket = Ticket(
        ticket_id="T-1",
        text="Windows 11, version 7.4.0.17, WSCAD Suite",
        metadata=Ticket.model_validate(
            {
                "ticket_id": "T-1",
                "text": "x",
                "metadata": {"product": "ELECTRIX AI", "os": "Linux"},
            }
        ).metadata,
    )
    updated = chat.apply_metadata_hints(ticket)
    # Existing product / os are not overwritten by hints
    assert updated.metadata.product == "ELECTRIX AI"
    assert updated.metadata.os == "Linux"
    # Previously empty version is filled from the text
    assert updated.metadata.version == "7.4.0.17"


def test_apply_metadata_hints_noop_when_no_hints() -> None:
    ticket = Ticket(ticket_id="T-1", text="It just doesn't work, please help")
    updated = chat.apply_metadata_hints(ticket)
    # No matches in the text -> no metadata changes, but a new Ticket may
    # still be returned for immutability convenience -- check the values.
    assert updated.metadata.product is None
    assert updated.metadata.version is None
    assert updated.metadata.os is None


def test_safe_filename_replaces_unsafe_chars() -> None:
    assert chat.safe_filename("T-001") == "T-001"
    assert chat.safe_filename("chat-20260512-143000") == "chat-20260512-143000"
    # Dots are excluded so the stem can't carry traversal sequences and the
    # only dots in the final filename come from the artefact extension.
    assert chat.safe_filename("../escape") == "___escape"
    assert chat.safe_filename("a/b\\c:d") == "a_b_c_d"
    assert chat.safe_filename("v7.4.0.17") == "v7_4_0_17"
    # Empty input collapses to a single underscore so the result is always
    # a non-empty filename. All-unsafe inputs of length N become N underscores.
    assert chat.safe_filename("") == "_"
    assert chat.safe_filename("///") == "___"


# ---------------------------------------------------------------------------
# Slash command parser


def test_is_slash_recognises_leading_slash() -> None:
    assert chat.is_slash("/quit")
    assert chat.is_slash("  /help  ")
    assert not chat.is_slash("hello /quit")
    assert not chat.is_slash("")


def test_handle_slash_quit_and_exit_both_quit() -> None:
    out: list[str] = []
    assert chat.handle_slash("/quit", None, out.append) == "quit"
    assert chat.handle_slash("/exit", None, out.append) == "quit"


def test_handle_slash_help_writes_text() -> None:
    out: list[str] = []
    result = chat.handle_slash("/help", None, out.append)
    assert result == "continue"
    written = "".join(out)
    assert "/quit" in written
    assert "/details" in written
    assert "/reset" in written


def test_handle_slash_details_without_session_is_polite() -> None:
    out: list[str] = []
    result = chat.handle_slash("/details", None, out.append)
    assert result == "continue"
    assert "No turn to show" in "".join(out)


def test_handle_slash_details_with_session_prints_full_render() -> None:
    ticket = Ticket(ticket_id="chat-T-1", text="x")
    session = chat.ChatSession(ticket=ticket, last_output=_solve_output())
    out: list[str] = []
    result = chat.handle_slash("/details", session, out.append)
    assert result == "continue"
    written = "".join(out)
    assert "Reasoning Trace:" in written
    assert "Cited Sources:" in written


def test_handle_slash_reset_returns_reset() -> None:
    out: list[str] = []
    assert chat.handle_slash("/reset", None, out.append) == "reset"


def test_handle_slash_unknown_command_is_not_fatal() -> None:
    out: list[str] = []
    result = chat.handle_slash("/banana", None, out.append)
    assert result == "continue"
    assert "Unknown command" in "".join(out)


def test_handle_slash_is_case_insensitive() -> None:
    assert chat.handle_slash("/QUIT", None, lambda _s: None) == "quit"


# ---------------------------------------------------------------------------
# read_multiline


def test_read_multiline_collects_until_blank_line() -> None:
    inputs = iter(["line one", "line two", ""])
    captured: list[str] = []

    def read(_p: str) -> str:
        return next(inputs)

    raw = chat.read_multiline(read, captured.append)
    assert raw == "line one\nline two"


def test_read_multiline_ignores_leading_blank_lines() -> None:
    inputs = iter(["", "", "actual content", ""])

    def read(_p: str) -> str:
        return next(inputs)

    raw = chat.read_multiline(read, lambda _: None)
    assert raw == "actual content"


def test_read_multiline_propagates_eof() -> None:
    def read(_p: str) -> str:
        raise EOFError

    with pytest.raises(EOFError):
        chat.read_multiline(read, lambda _: None)


# ---------------------------------------------------------------------------
# run_repl — driven by scripted input/output


class _ScriptedIO:
    """Drives the REPL with a queued input list and captures output."""

    def __init__(self, inputs: list[str | EOFError]) -> None:
        self._inputs = list(inputs)
        self.written: list[str] = []
        self.read_prompts: list[str] = []

    def read(self, prompt: str) -> str:
        self.read_prompts.append(prompt)
        if not self._inputs:
            raise EOFError
        value = self._inputs.pop(0)
        if isinstance(value, EOFError) or value is EOFError:
            raise EOFError
        return value

    def write(self, text: str) -> None:
        self.written.append(text)

    @property
    def output(self) -> str:
        return "".join(self.written)


def _make_app_config(soft_cap: int = 8) -> AppConfig:
    return AppConfig(chat=ChatConfig(soft_turn_warning_at=soft_cap))


def test_run_repl_solve_on_first_turn_then_quit(tmp_path: Path) -> None:
    io = _ScriptedIO(["App crashes, error 504", "", "/quit"])
    fake_output = _solve_output("chat-T-1")
    pipeline_calls: list[Ticket] = []

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        pipeline_calls.append(ticket)
        # Make ticket_id deterministic regardless of clock
        return fake_output.model_copy(update={"ticket_id": ticket.ticket_id})

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )

    assert exit_code == 0
    assert len(pipeline_calls) == 1
    assert "Re-activate the offline license" in io.output
    assert "Goodbye" in io.output
    # Artefacts saved per turn
    ticket_id = pipeline_calls[0].ticket_id
    assert (tmp_path / f"{ticket_id}.json").exists()
    assert (tmp_path / f"{ticket_id}.txt").exists()
    assert (tmp_path / f"{ticket_id}.chat.md").exists()


def test_run_repl_clarify_then_answer_then_solve(tmp_path: Path) -> None:
    io = _ScriptedIO(
        [
            "App crashes after install",  # initial line 1
            "",  # submit
            "Windows 11, WSCAD Suite 2.3",  # follow-up answer
            "/quit",
        ]
    )

    fake_outputs = iter(
        [
            _clarify_output("chat-T-1"),
            _solve_output("chat-T-1"),
        ]
    )
    seen_texts: list[str] = []

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        seen_texts.append(ticket.text)
        out = next(fake_outputs)
        return out.model_copy(update={"ticket_id": ticket.ticket_id})

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )

    assert exit_code == 0
    assert len(seen_texts) == 2
    # Turn 2 has the Q/A block appended
    assert "[Follow-up turn 2]" in seen_texts[1]
    assert "A: Windows 11, WSCAD Suite 2.3" in seen_texts[1]


def test_run_repl_eof_after_first_turn_saves_and_exits(tmp_path: Path) -> None:
    io = _ScriptedIO(["Error 504", "", EOFError()])

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        return _solve_output("chat-T-1").model_copy(update={"ticket_id": ticket.ticket_id})

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    assert exit_code == 0
    assert "Goodbye" in io.output
    assert any(p.suffix == ".json" for p in tmp_path.iterdir())


def test_run_repl_pipeline_error_does_not_crash_repl(tmp_path: Path) -> None:
    io = _ScriptedIO(["App crashes", "", "/quit"])

    def boom(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        raise RuntimeError("simulated agent failure")

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=boom,
    )

    assert exit_code == 0
    assert "Pipeline error" in io.output
    assert "simulated agent failure" in io.output


def test_run_repl_followup_error_rolls_back_ticket_text(tmp_path: Path) -> None:
    """Follow-up pipeline error must not leave the Q/A block stuck in ticket.text.

    Without the rollback, the next successful turn re-appends ``[Follow-up turn N]``
    on top of an already-appended one — the doubled block bug flagged by
    senior-reviewer P1.
    """
    initial = json.dumps(
        {
            "ticket_id": "T-RB",
            "text": "App crashes after install",
            "metadata": {"product": "WSCAD Suite", "version": "2.3", "os": "Windows 11"},
        }
    )
    io = _ScriptedIO(
        [
            initial,
            "",  # submit initial
            "first answer attempt (will error)",
            "second answer attempt (succeeds)",
            "/quit",
        ]
    )
    clarify = _clarify_output("T-RB", questions=["Can you share the log excerpt?"])
    solve = _solve_output("T-RB")
    seen_texts: list[str] = []
    raise_next = {"flag": False}

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        seen_texts.append(ticket.text)
        if raise_next["flag"]:
            raise_next["flag"] = False
            raise RuntimeError("transient failure")
        if len(seen_texts) == 1:
            return clarify
        if len(seen_texts) == 2:
            # First follow-up arrives -- arm the next call to fail.
            raise_next["flag"] = True
            raise RuntimeError("transient failure")
        return solve

    chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    # ``seen_texts`` records every pipeline call's ticket.text. Only one
    # ``[Follow-up turn 2]`` block should ever appear in any single call.
    for call_idx, text in enumerate(seen_texts):
        assert text.count("[Follow-up turn 2]") <= 1, (
            f"call {call_idx} has doubled follow-up block:\n{text}"
        )


def test_run_repl_initial_error_drops_session(tmp_path: Path) -> None:
    """Initial-turn pipeline error must drop the session so the user re-enters."""
    io = _ScriptedIO(["bad ticket", "", "/quit"])
    calls: list[Ticket] = []

    def boom(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        calls.append(ticket)
        raise RuntimeError("initial failure")

    chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=boom,
    )
    # Only one pipeline call attempted; nothing saved because no successful turn.
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_run_repl_reset_starts_a_fresh_ticket(tmp_path: Path) -> None:
    io = _ScriptedIO(
        [
            "First ticket text",
            "",
            "/reset",
            "Second ticket text",
            "",
            "/quit",
        ]
    )
    pipeline_calls: list[str] = []

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        pipeline_calls.append(ticket.ticket_id)
        return _solve_output(ticket.ticket_id)

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    assert exit_code == 0
    assert len(pipeline_calls) == 2
    # Two distinct synthetic IDs (or same if clock happened to tick same second)
    assert all(tid.startswith("chat-") for tid in pipeline_calls)
    assert "Session reset" in io.output


def test_run_repl_soft_cap_warning_fires_once(tmp_path: Path) -> None:
    # 4 turns total, soft cap at 2 -> warning fires after turn 2, not again.
    io = _ScriptedIO(["initial", "", "more 1", "more 2", "more 3", "/quit"])
    n = 0

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        nonlocal n
        n += 1
        return _solve_output(ticket.ticket_id)

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(soft_cap=2),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    assert exit_code == 0
    warning_count = io.output.count("has run")
    assert warning_count == 1, f"expected one soft-cap warning, got {warning_count}"


def test_run_repl_help_command_does_not_advance_turn(tmp_path: Path) -> None:
    io = _ScriptedIO(["initial", "", "/help", "/quit"])
    calls: list[Ticket] = []

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        calls.append(ticket)
        return _solve_output(ticket.ticket_id)

    chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    assert len(calls) == 1  # only the initial turn
    assert "/details" in io.output  # help text mentions /details


def test_run_repl_initial_quit_exits_cleanly(tmp_path: Path) -> None:
    io = _ScriptedIO(["/quit", ""])

    def fake_pipeline(ticket: Ticket, llm: Any, retriever: Any) -> Output:
        raise AssertionError("pipeline must not be called when user quits at start")

    exit_code = chat.run_repl(
        llm=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=_make_app_config(),
        read=io.read,
        write=io.write,
        run_pipeline=fake_pipeline,
    )
    assert exit_code == 0
    assert list(tmp_path.iterdir()) == []
