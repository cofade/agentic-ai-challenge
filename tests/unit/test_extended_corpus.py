"""Issue #16: retrieval over the extended (Phase 2) KB corpus.

Verifies that the hybrid retriever, when built over the union KB (Phase 0
``kb/original/`` + Phase 2 ``kb/electrix_ai_release_notes/``), surfaces a
version-appropriate release note for a query that names both a topic and a
version. Network-marked because it loads the real
``paraphrase-multilingual-MiniLM-L12-v2`` encoder.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENDED_KB = REPO_ROOT / "kb"  # union of original/ + electrix_ai_release_notes/

V7_3_2_PATTERN = re.compile(r"^electrix_ai_release_notes/v7\.3\.2\.\d+(\.[a-z]{2})?\.md$")


@pytest.mark.network
def test_multi_version_query_surfaces_v7_3_2_release_note(tmp_path: Path) -> None:
    """Hybrid retrieval over the union KB must surface a v7.3.2.* note.

    Uses the real ``paraphrase-multilingual-MiniLM-L12-v2`` encoder. Skipped
    in environments without HuggingFace access (matches the pattern in
    ``test_embedding.py``).
    """
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except (OSError, ImportError) as exc:  # pragma: no cover — network-gated
        pytest.skip(f"multilingual model unavailable in this environment: {exc}")

    chunks = load_kb(EXTENDED_KB)
    assert chunks, "extended corpus must load at least one chunk"
    hybrid = HybridRetriever(BM25(chunks), Embedding(chunks, cache_dir=tmp_path))

    results = hybrid.retrieve("license activation in v7.3.2", k=5)

    assert results, "extended corpus must produce hits for a v7.3.2 query"
    matches = [r.chunk.source_file for r in results if V7_3_2_PATTERN.match(r.chunk.source_file)]
    assert matches, (
        f"expected at least one v7.3.2.* release note in top-5; got "
        f"{[r.chunk.source_file for r in results]}"
    )
