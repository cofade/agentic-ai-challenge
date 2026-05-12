"""Integration tests for the chat REPL driving the real pipeline (issue #65).

These exercise :func:`wscad_triage.chat.run_repl` against the real
:func:`wscad_triage.pipeline.run`, scripted via ``MockLLMClient`` and a
``StubRetriever`` (the same shape used by ``test_pipeline.py``). They
prove the wiring end-to-end without hitting the network.

Unit-level REPL state-machine behavior (slash commands, transcript
shape, soft cap, etc.) is covered by ``tests/unit/test_chat.py``; this
file focuses on the cross-module path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wscad_triage import chat
from wscad_triage.config import AppConfig, ChatConfig
from wscad_triage.llm import LLMResponse, ToolCall, Usage
from wscad_triage.schemas import KBChunk, Output, RetrievalResult

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers replicated from tests/integration/test_pipeline.py for parity


def _tool_response(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id="t1", name=name, arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=0, output_tokens=0),
    )


def _retrieval(chunk_id: str, source_file: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=KBChunk(chunk_id=chunk_id, source_file=source_file, text=text),
        score=0.9,
        rank=1,
        retriever="rrf",
    )


class _ScriptedIO:
    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self.written: list[str] = []

    def read(self, _prompt: str) -> str:
        if not self._inputs:
            raise EOFError
        return self._inputs.pop(0)

    def write(self, text: str) -> None:
        self.written.append(text)

    @property
    def output(self) -> str:
        return "".join(self.written)


# ---------------------------------------------------------------------------
# Scenario: solve on first turn, real pipeline, then /quit


def test_chat_one_turn_solve_end_to_end(
    tmp_path: Path, mock_llm: Any, stub_retriever_factory: Any
) -> None:
    chunk_text = (
        "Error code 504 indicates a licensing validation failure after an update. "
        "Re-activate the offline license via License Manager."
    )
    retriever = stub_retriever_factory(_retrieval("err-504", "Common_Errors.md", chunk_text))

    llm = mock_llm(
        **{
            "You are the triage agent.": _tool_response(
                "emit_classification",
                {
                    "category": "Licensing",
                    "priority": "High",
                    "missing_critical_fields": [],
                    "rationale": "Error code 504 with WSCAD Suite specified.",
                },
            ),
            "You are the retrieve agent.": _tool_response(
                "emit_query", {"query": "error 504 licensing offline"}
            ),
            "You are the reason agent.": _tool_response(
                "emit_draft",
                {
                    "solution": (
                        "Re-activate the offline license via the License Manager. "
                        "Error 504 indicates a licensing validation failure after update."
                    ),
                    "claims": [
                        {
                            "claim": "Error code 504 indicates a licensing validation failure after an update.",
                            "chunk_id": "err-504",
                            "quote": (
                                "Error code 504 indicates a licensing validation failure after an update."
                            ),
                        }
                    ],
                },
            ),
            "You are the verify agent.": _tool_response(
                "emit_verdict",
                {
                    "grounding_score": 0.95,
                    "per_claim": [
                        {
                            "claim": (
                                "Error code 504 indicates a licensing validation failure after an update."
                            ),
                            "grounded": True,
                            "rationale": "Quote is a literal substring of the chunk.",
                        }
                    ],
                },
            ),
        }
    )

    initial_ticket = json.dumps(
        {
            "ticket_id": "T-001",
            "text": "Application fails to start after update. Error 504 appears.",
            "metadata": {"product": "WSCAD Suite", "version": "2.3", "os": "Windows 11"},
        }
    )
    io = _ScriptedIO([initial_ticket, "", "/quit"])

    exit_code = chat.run_repl(
        llm=llm,
        retriever=retriever,  # type: ignore[arg-type] -- duck-typed; pipeline uses .retrieve only
        out_dir=tmp_path,
        app_config=AppConfig(chat=ChatConfig(soft_turn_warning_at=8)),
        read=io.read,
        write=io.write,
    )

    assert exit_code == 0
    assert "Re-activate the offline license" in io.output
    # Artefacts saved with the JSON-provided ticket id
    json_file = tmp_path / "T-001.json"
    txt_file = tmp_path / "T-001.txt"
    md_file = tmp_path / "T-001.chat.md"
    assert json_file.exists(), "T-001.json not produced"
    assert txt_file.exists(), "T-001.txt not produced"
    assert md_file.exists(), "T-001.chat.md not produced"

    output = Output.model_validate(json.loads(json_file.read_text(encoding="utf-8")))
    assert output.resolution_kind == "solve"
    assert output.category == "Licensing"
    assert output.cited_sources == ["Common_Errors.md"]

    # Transcript captures both the user input and the agent header
    md = md_file.read_text(encoding="utf-8")
    assert "Chat session T-001" in md
    assert "## Turn 1 -- user" in md
    assert "kind=solve" in md


# ---------------------------------------------------------------------------
# Scenario: clarify on first turn, user answers, solve on second turn


def test_chat_clarify_then_answer_then_solve(
    tmp_path: Path, mock_llm: Any, stub_retriever_factory: Any
) -> None:
    """Turn 1 routes to clarify on an LLM-added gap; turn 2 LLM drops the gap.

    Why an LLM-added gap and not a deterministic metadata gap: the triage
    agent's deterministic pre-pass reads ``ticket.metadata.{product, version, os}``,
    which a free-text follow-up answer cannot fill (the chat REPL appends to
    ``ticket.text``, not metadata — see ADR-011). To exercise the
    clarify→answer→solve loop the metadata is complete from the start; the
    *LLM* adds a non-metadata gap (here ``log_excerpt``) on turn 1 and
    drops it on turn 2 once the user's answer is in the text.
    """
    chunk_text = "Installation fails on missing .NET runtime. Install .NET 6.0."
    retriever = stub_retriever_factory(
        _retrieval("inst-1", "Installation_Requirements.md", chunk_text)
    )

    llm = mock_llm(
        **{
            "You are the triage agent.": [
                _tool_response(
                    "emit_classification",
                    {
                        "category": "Installation",
                        "priority": "Medium",
                        "missing_critical_fields": ["log_excerpt"],
                        "rationale": "Crash on install with no error log shared.",
                    },
                ),
                _tool_response(
                    "emit_classification",
                    {
                        "category": "Installation",
                        "priority": "Medium",
                        "missing_critical_fields": [],
                        "rationale": "Error log now provided in the follow-up.",
                    },
                ),
            ],
            "You are the clarify agent.": _tool_response(
                "emit_questions",
                {
                    "questions": [
                        "Can you share the log excerpt around the crash?",
                        "What text appears in the error log right before the crash?",
                    ]
                },
            ),
            "You are the retrieve agent.": _tool_response(
                "emit_query", {"query": "installation crash dotnet runtime"}
            ),
            "You are the reason agent.": _tool_response(
                "emit_draft",
                {
                    "solution": "Install .NET 6.0 runtime, then retry installation.",
                    "claims": [
                        {
                            "claim": "Install .NET 6.0.",
                            "chunk_id": "inst-1",
                            "quote": "Install .NET 6.0.",
                        }
                    ],
                },
            ),
            "You are the verify agent.": _tool_response(
                "emit_verdict",
                {
                    "grounding_score": 0.9,
                    "per_claim": [
                        {
                            "claim": "Install .NET 6.0.",
                            "grounded": True,
                            "rationale": "Quote is a literal substring of the chunk.",
                        }
                    ],
                },
            ),
        }
    )

    initial_ticket = json.dumps(
        {
            "ticket_id": "T-CHAT",
            "text": "Application crashes immediately after installation.",
            "metadata": {"product": "WSCAD Suite", "version": "2.3", "os": "Windows 11"},
        }
    )
    io = _ScriptedIO(
        [
            initial_ticket,
            "",  # submit
            "ERROR_NET_NOT_FOUND: .NET 6.0 runtime missing during install",  # follow-up answer
            "/quit",
        ]
    )

    exit_code = chat.run_repl(
        llm=llm,
        retriever=retriever,  # type: ignore[arg-type]
        out_dir=tmp_path,
        app_config=AppConfig(chat=ChatConfig(soft_turn_warning_at=8)),
        read=io.read,
        write=io.write,
    )

    assert exit_code == 0
    # Both turn outputs visible on screen
    assert "Preliminary assessment" in io.output
    assert "Install .NET 6.0" in io.output

    # Final saved artefact is the solve from turn 2 (latest overwrite of the same id)
    json_file = tmp_path / "T-CHAT.json"
    assert json_file.exists(), "T-CHAT.json not produced"
    final = Output.model_validate(json.loads(json_file.read_text(encoding="utf-8")))
    assert final.resolution_kind == "solve"
    assert final.category == "Installation"

    # Transcript shows both turns
    md = (tmp_path / "T-CHAT.chat.md").read_text(encoding="utf-8")
    assert "## Turn 1 -- user" in md
    assert "## Turn 2 -- user" in md
    assert "kind=clarify" in md
    assert "kind=solve" in md
