"""Issue #15: validate the frontmatter contract on the Phase 2 KB corpus.

Every file under ``kb/electrix_ai_release_notes/`` must declare the five
required keys, valid values, and a single corpus-wide value for the
``synthetic_for_demo`` flag (the all-or-nothing fallback invariant: either
every file is real, or every file is synthetic).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wscad_triage.kb.loader import parse_frontmatter

REPO_ROOT = Path(__file__).resolve().parents[2]
ELECTRIX_KB = REPO_ROOT / "kb" / "electrix_ai_release_notes"

REQUIRED_KEYS = frozenset(
    {"source_url", "version", "product_line", "language", "synthetic_for_demo"}
)
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
SYNTH_VALID = frozenset({"True", "False"})


def _all_files() -> list[Path]:
    """List every ``*.md`` directly under ``ELECTRIX_KB``.

    Non-recursive on purpose: the corpus contract for Phase 2 is one flat
    directory of ``v<version>[.<lang>].md`` files. If a future change
    nests a subdirectory, the validation contract needs to be revisited
    explicitly — silent recursion would let unintended files slip in.
    """
    files = sorted(ELECTRIX_KB.glob("*.md"))
    assert files, f"Phase 2 KB directory must contain at least one .md file: {ELECTRIX_KB}"
    return files


@pytest.mark.parametrize("path", _all_files(), ids=lambda p: p.name)
def test_frontmatter_required_keys_and_values(path: Path) -> None:
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - metadata.keys()
    assert not missing, f"{path.name}: missing keys {sorted(missing)}"
    assert metadata["product_line"] == "ELECTRIX AI"
    assert metadata["language"] in {"en", "de"}
    assert metadata["synthetic_for_demo"] in SYNTH_VALID
    assert VERSION_RE.match(metadata["version"]), (
        f"{path.name}: bad version {metadata['version']!r}"
    )
    assert metadata["source_url"].startswith("https://"), f"{path.name}: source_url must use https"
    assert body.strip(), f"{path.name}: empty body"


def test_synthetic_for_demo_is_consistent_across_corpus() -> None:
    """All-or-nothing fallback invariant: every file agrees on the flag."""
    flags = {
        parse_frontmatter(p.read_text(encoding="utf-8"))[0]["synthetic_for_demo"]
        for p in _all_files()
    }
    assert len(flags) == 1, f"mixed synthetic flags across corpus: {flags}"


def test_filename_matches_frontmatter_version() -> None:
    """``v<version>[.<lang>].md`` filenames must match their frontmatter version."""
    name_re = re.compile(r"^v(?P<version>\d+\.\d+\.\d+\.\d+)(?:\.(?P<lang>[a-z]{2}))?\.md$")
    for path in _all_files():
        match = name_re.match(path.name)
        assert match, f"{path.name}: does not match v<version>[.<lang>].md convention"
        metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert match["version"] == metadata["version"], (
            f"{path.name}: filename version {match['version']!r} != frontmatter "
            f"version {metadata['version']!r}"
        )
        if match["lang"] is not None:
            assert match["lang"] == metadata["language"], (
                f"{path.name}: filename lang {match['lang']!r} != frontmatter "
                f"language {metadata['language']!r}"
            )
