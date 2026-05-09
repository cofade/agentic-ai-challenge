"""Tests for the wscad-triage CLI (Phase 4 — issue #30)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wscad_triage.cli import main
from wscad_triage.kb import Embedding
from wscad_triage.schemas import Output, ReasoningStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_solve_output(ticket_id: str = "T-001") -> Output:
    return Output(
        ticket_id=ticket_id,
        category="Errors",
        priority="High",
        resolution_kind="solve",
        proposed_solution="Re-activate the offline license.",
        confidence=0.85,
        reasoning_trace=[
            ReasoningStep(
                actor="supervisor",
                action="finalize_solve",
                evidence_refs=[],
                rationale="Grounding passed.",
            )
        ],
    )


def _fake_kb_dir(tmp_path: Path) -> Path:
    """Return a non-empty directory that passes the is_dir() check."""
    d = tmp_path / "kb"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_returns_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "wscad-triage" in out


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_tickets_file_returns_two(tmp_path: Path) -> None:
    nonexistent = str(tmp_path / "does_not_exist.json")
    exit_code = main([nonexistent, "--kb-dir", str(_fake_kb_dir(tmp_path))])
    assert exit_code == 2


def test_kb_dir_not_found_returns_two(tmp_path: Path) -> None:
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text("[]", encoding="utf-8")
    exit_code = main([str(tickets_file), "--kb-dir", str(tmp_path / "no_kb")])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Happy path — files are written
# ---------------------------------------------------------------------------


def test_produces_json_and_text_files(tmp_path: Path) -> None:
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text(
        json.dumps([{"ticket_id": "T-001", "text": "Something broke.", "metadata": {}}]),
        encoding="utf-8",
    )
    kb_dir = _fake_kb_dir(tmp_path)
    out_dir = tmp_path / "out"
    fixed_output = _minimal_solve_output("T-001")

    with (
        patch("wscad_triage.cli.pipeline.run", return_value=fixed_output),
        patch("wscad_triage.cli.make_client", return_value=MagicMock()),
        patch("wscad_triage.cli.load_kb", return_value=[]),
        patch("wscad_triage.cli.BM25", return_value=MagicMock()),
        patch("wscad_triage.cli.Embedding", return_value=MagicMock()),
        patch("wscad_triage.cli.HybridRetriever", return_value=MagicMock()),
    ):
        exit_code = main([str(tickets_file), "--out", str(out_dir), "--kb-dir", str(kb_dir)])

    assert exit_code == 0

    json_file = out_dir / "T-001.json"
    txt_file = out_dir / "T-001.txt"
    assert json_file.exists(), "T-001.json not found"
    assert txt_file.exists(), "T-001.txt not found"

    # JSON round-trips cleanly
    restored = Output.model_validate(json.loads(json_file.read_text(encoding="utf-8")))
    assert restored == fixed_output

    # Text contains the ticket ID
    assert "Ticket ID: T-001" in txt_file.read_text(encoding="utf-8")


def test_print_uses_ascii_arrow(tmp_path: Path, capsys) -> None:
    """Progress line must be ASCII-safe (no Unicode arrows that break cp1252)."""
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text(
        json.dumps([{"ticket_id": "T-001", "text": "x", "metadata": {}}]),
        encoding="utf-8",
    )
    kb_dir = _fake_kb_dir(tmp_path)
    out_dir = tmp_path / "out"

    with (
        patch("wscad_triage.cli.pipeline.run", return_value=_minimal_solve_output("T-001")),
        patch("wscad_triage.cli.make_client", return_value=MagicMock()),
        patch("wscad_triage.cli.load_kb", return_value=[]),
        patch("wscad_triage.cli.BM25", return_value=MagicMock()),
        patch("wscad_triage.cli.Embedding", return_value=MagicMock()),
        patch("wscad_triage.cli.HybridRetriever", return_value=MagicMock()),
    ):
        main([str(tickets_file), "--out", str(out_dir), "--kb-dir", str(kb_dir)])

    out = capsys.readouterr().out
    # Must be encodable with cp1252 — no Unicode arrows
    out.encode("cp1252")
    assert "->" in out


# ---------------------------------------------------------------------------
# Provider flag
# ---------------------------------------------------------------------------


def test_provider_flag_forwarded(tmp_path: Path) -> None:
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text(json.dumps([]), encoding="utf-8")
    kb_dir = _fake_kb_dir(tmp_path)

    captured: dict[str, object] = {}

    def fake_make_client(settings):  # type: ignore[no-untyped-def]
        captured["provider"] = settings.llm_provider
        return MagicMock()

    with (
        patch("wscad_triage.cli.make_client", side_effect=fake_make_client),
        patch("wscad_triage.cli.load_kb", return_value=[]),
        patch("wscad_triage.cli.BM25", return_value=MagicMock()),
        patch("wscad_triage.cli.Embedding", return_value=MagicMock()),
        patch("wscad_triage.cli.HybridRetriever", return_value=MagicMock()),
    ):
        exit_code = main([str(tickets_file), "--provider", "anthropic", "--kb-dir", str(kb_dir)])

    assert exit_code == 0
    assert captured.get("provider") == "anthropic"


# ---------------------------------------------------------------------------
# Default output directory
# ---------------------------------------------------------------------------


def test_out_defaults_to_out_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text(
        json.dumps([{"ticket_id": "T-042", "text": "Test.", "metadata": {}}]),
        encoding="utf-8",
    )
    kb_dir = _fake_kb_dir(tmp_path)
    fixed_output = _minimal_solve_output("T-042")

    monkeypatch.chdir(tmp_path)
    with (
        patch("wscad_triage.cli.pipeline.run", return_value=fixed_output),
        patch("wscad_triage.cli.make_client", return_value=MagicMock()),
        patch("wscad_triage.cli.load_kb", return_value=[]),
        patch("wscad_triage.cli.BM25", return_value=MagicMock()),
        patch("wscad_triage.cli.Embedding", return_value=MagicMock()),
        patch("wscad_triage.cli.HybridRetriever", return_value=MagicMock()),
    ):
        exit_code = main([str(tickets_file), "--kb-dir", str(kb_dir)])

    assert exit_code == 0
    assert (tmp_path / "out" / "T-042.json").exists()


# ---------------------------------------------------------------------------
# KB dir integration — real kb/, fake Embedding encoder, empty tickets list
# ---------------------------------------------------------------------------


def test_kb_dir_loads_real_kb(tmp_path: Path) -> None:
    """Acceptance test for ROADMAP #30: real kb/ is loaded with --kb-dir.

    Uses an empty tickets list so pipeline.run is never called; the test
    verifies that --kb-dir resolves correctly, load_kb succeeds on the real
    corpus, and BM25 + HybridRetriever construction completes. A fake encoder
    is injected into Embedding to avoid downloading the sentence-transformers
    model during tests.
    """
    tickets_file = tmp_path / "tickets.json"
    tickets_file.write_text("[]", encoding="utf-8")

    rng = np.random.default_rng(42)

    def _fake_encoder(texts: list[str]) -> np.ndarray:
        return rng.random((len(texts), 384), dtype=np.float32)

    real_kb_dir = Path("kb")
    if not real_kb_dir.is_dir():
        pytest.skip("kb/ not found — run from the project root")

    def _embedding_with_fake_encoder(chunks):  # type: ignore[no-untyped-def]
        return Embedding(chunks, encoder=_fake_encoder)

    with (
        patch("wscad_triage.cli.make_client", return_value=MagicMock()),
        patch("wscad_triage.cli.Embedding", side_effect=_embedding_with_fake_encoder),
    ):
        exit_code = main([str(tickets_file), "--kb-dir", str(real_kb_dir)])

    assert exit_code == 0
