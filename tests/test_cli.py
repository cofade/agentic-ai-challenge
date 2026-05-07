"""Smoke tests for the CLI entry point.

The CLI is a stub until Phase 4 (issue #30); these tests pin both branches so
the bootstrap meets the coverage gate.
"""

from __future__ import annotations

from wscad_triage.cli import main


def test_help_returns_zero(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["--help"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "wscad-triage" in out


def test_default_invocation_returns_one_with_roadmap_pointer(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main([])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "ROADMAP" in out
