"""Issue #16: retrieval over the extended (Phase 2) KB corpus.

Pins the corpus-extension behaviour: when the KB is the union of
``kb/original/`` + ``kb/electrix_ai_release_notes/``, a topical query
(``"license activation in v7.3.2"``) surfaces a Phase-2 v7.3.2 release
note in the top ranks. This is a behavioural property of the lexical and
semantic match between query and synthetic body — not version-aware
retrieval. Genuine version-aware retrieval (metadata-conditioned filter)
is future work; see ``docs/11-risks-and-technical-debt/README.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb, tokenize

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTENDED_KB = REPO_ROOT / "kb"  # union of original/ + electrix_ai_release_notes/

V7_3_2_PATTERN = re.compile(r"^electrix_ai_release_notes/v7\.3\.2\.\d+(\.[a-z]{2})?\.md$")


def test_version_string_tokenisation_is_dot_split() -> None:
    """Guard: the BM25 tokenizer shreds version dots into separate components.

    Pinned because the multi-version retrieval test below assumes this
    behaviour; if a future tokenizer change preserves dots (e.g. switches
    to ``v7.3.2.16`` as a single token), retrieval semantics for
    version-tagged queries change materially and the dependent test
    needs to be re-thought, not just rebased.
    """
    assert tokenize("license activation in v7.3.2.16") == [
        "license",
        "activation",
        "v7",
        "3",
        "2",
        "16",
    ]


@pytest.mark.network
def test_multi_version_query_surfaces_v7_3_2_release_note(tmp_path: Path) -> None:
    """Hybrid retrieval over the union KB must surface a v7.3.2.* note.

    Uses the real ``paraphrase-multilingual-MiniLM-L12-v2`` encoder.
    Skipped in environments without HuggingFace access (matches the
    pattern in ``test_embedding.py``). Asserts the v7.3.2.16.md note
    ranks top-1 and at least one v7.3.2.* note appears in top-3 — both
    properties are decided by lexical match, not version-aware
    retrieval; rebase the assertion if a deliberate change to the
    Phase-2 synthetic bodies shifts the top-1 file.
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
    assert results[0].chunk.source_file == "electrix_ai_release_notes/v7.3.2.16.md", (
        f"expected v7.3.2.16.md to rank top-1; got "
        f"{[(r.rank, r.chunk.source_file) for r in results]}"
    )
    v7_3_2_in_top3 = sum(1 for r in results[:3] if V7_3_2_PATTERN.match(r.chunk.source_file))
    assert v7_3_2_in_top3 >= 1, (
        f"expected at least one v7.3.2.* note in top-3; got "
        f"{[(r.rank, r.chunk.source_file) for r in results[:3]]}"
    )
