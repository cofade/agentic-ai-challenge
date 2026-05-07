"""Read Markdown KB files and parse optional YAML frontmatter into ``RawDoc``.

The loader handles the *boundary* between disk and in-memory typed values: it
walks a KB directory, decodes UTF-8, splits any ``---``-delimited frontmatter
from the body, and produces deterministic :class:`RawDoc` objects sorted by
``source_file`` so that downstream chunking is stable across runs.

``RawDoc`` is intentionally local to this subpackage — it does not appear in
:mod:`wscad_triage.schemas` because no value of this type ever crosses out of
``kb.*``. The chunker consumes ``RawDoc`` and emits :class:`KBChunk`, which is
the boundary type for the rest of the pipeline.

Frontmatter parsing rules (kept narrow on purpose):

- The frontmatter block must start at the very first byte with ``---\\n`` and
  end with a line containing only ``---``.
- Values are coerced to ``str`` so they fit ``KBChunk.metadata: dict[str, str]``
  (``True`` → ``"True"``, ``42`` → ``"42"``).
- Lists and nested mappings raise :class:`ValueError`. The KB convention is
  flat scalars; the Phase 2 release-notes scrape (#14) follows that convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

_FRONTMATTER_DELIM = "---"


class RawDoc(BaseModel):
    """A parsed Markdown document before sentence-level chunking."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Split a file's content into (frontmatter, body).

    Returns ``({}, raw)`` unchanged if the file does not begin with a
    frontmatter block. Frontmatter scalars are stringified so the result
    fits :class:`KBChunk`'s ``dict[str, str]`` metadata field.
    """
    if not raw.startswith((_FRONTMATTER_DELIM + "\n", _FRONTMATTER_DELIM + "\r\n")):
        return {}, raw

    lines = raw.splitlines(keepends=True)
    closing_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FRONTMATTER_DELIM:
            closing_idx = idx
            break
    if closing_idx is None:
        return {}, raw

    yaml_block = "".join(lines[1:closing_idx])
    body = "".join(lines[closing_idx + 1 :])

    try:
        parsed: Any = yaml.safe_load(yaml_block) if yaml_block.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML frontmatter: {exc}") from exc
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Frontmatter must be a YAML mapping; got {type(parsed).__name__}.")

    metadata: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(value, list | dict):
            raise ValueError(
                f"Frontmatter value for {key!r} is a {type(value).__name__}; "
                "only flat scalars are supported."
            )
        metadata[str(key)] = str(value)

    return metadata, body


def load_documents(kb_dir: Path) -> list[RawDoc]:
    """Walk ``kb_dir`` recursively for ``*.md`` files and return ``RawDoc`` list.

    Paths in ``RawDoc.source_file`` are relative to ``kb_dir`` and use forward
    slashes regardless of OS. The result is sorted by ``source_file`` so the
    chunker produces stable ``chunk_id`` values across runs and platforms.
    Empty files and frontmatter-only files yield ``RawDoc(text="", ...)``;
    the chunker is responsible for skipping them.
    """
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory does not exist: {kb_dir}")

    docs: list[RawDoc] = []
    for path in sorted(kb_dir.rglob("*.md")):
        relative = path.relative_to(kb_dir).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{relative}: file is not valid UTF-8 ({exc})") from exc
        try:
            metadata, body = parse_frontmatter(raw)
        except ValueError as exc:
            raise ValueError(f"{relative}: {exc}") from exc
        docs.append(RawDoc(source_file=relative, text=body, metadata=metadata))
    return docs
