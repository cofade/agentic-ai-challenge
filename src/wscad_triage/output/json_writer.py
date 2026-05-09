"""JSON serialisation for the pipeline Output (Phase 4 — issue #29)."""

from __future__ import annotations

from pathlib import Path

from wscad_triage.schemas import Output


def write_json(output: Output, path: Path) -> None:
    """Write *output* as a pretty-printed JSON file at *path* (indent=2)."""
    path.write_text(output.model_dump_json(indent=2), encoding="utf-8")


__all__ = ["write_json"]
