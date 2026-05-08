"""Unit tests for ``wscad_triage.kb.retriever.Embedding``.

Covers issue #11's acceptance criteria — cold-start computes & caches; warm-start
loads from cache without recompute; cross-lingual query (DE → EN docs) — plus
the retriever contract surface shared with :class:`BM25` (ranking invariants,
edge cases, JSON round-trip).

Most tests inject a deterministic fake encoder so the suite stays fast and the
real ~470 MB model is downloaded only by the cross-lingual acceptance test.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest

from wscad_triage.kb import Embedding, load_kb
from wscad_triage.schemas import KBChunk, RetrievalResult

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_KB = REPO_ROOT / "tests" / "fixtures" / "kb"

FAKE_DIM: Final[int] = 8


def _chunk(chunk_id: str, text: str, language: str | None = "en") -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        source_file=chunk_id.split("#")[0],
        text=text,
        language=language,
    )


def _fake_encoder(texts: list[str]) -> npt.NDArray[np.float32]:
    """Deterministic, content-derived embeddings: each token contributes to one slot.

    Two texts that share tokens end up close in cosine similarity; texts with
    no shared tokens are orthogonal. Good enough to exercise ranking + cache
    semantics without loading a real model.
    """
    vectors = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            slot = hash(token) % FAKE_DIM
            vectors[i, slot] += 1.0
    return vectors


# ---------------------------------------------------------------------------
# Acceptance criterion (issue #11): cross-lingual retrieval with the real model
# ---------------------------------------------------------------------------


def test_german_query_against_english_corpus_returns_plausible_match(tmp_path: Path) -> None:
    """A German query against an English-only corpus surfaces a plausible match.

    This is issue #11's headline acceptance criterion. Uses the real
    ``paraphrase-multilingual-MiniLM-L12-v2`` — slow on first run (model
    download + corpus encode), fast afterwards via the on-disk cache. Skipped
    when the model can't be loaded (e.g. sandboxed runs without HF access).
    """
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except (OSError, ImportError) as exc:  # pragma: no cover — network-gated
        pytest.skip(f"multilingual model unavailable in this environment: {exc}")

    chunks = [
        _chunk("en.md#0", "The application starts successfully on supported operating systems."),
        _chunk("en.md#1", "Users should verify their license status before launching."),
        _chunk("en.md#2", "Contact support if the application fails to start."),
    ]
    embedding = Embedding(chunks, cache_dir=tmp_path)
    results = embedding.retrieve("Lizenzstatus überprüfen", k=3)
    assert results, "expected non-empty result list"
    assert results[0].chunk.chunk_id == "en.md#1", (
        f"expected the license-status sentence to rank top, got {results[0].chunk.chunk_id}"
    )


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cold_start_writes_cache(tmp_path: Path) -> None:
    chunks = [_chunk("a.md#0", "alpha"), _chunk("b.md#0", "beta")]
    Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)
    npz_files = list(tmp_path.glob("*.npz"))
    meta_files = list(tmp_path.glob("*.meta.json"))
    assert len(npz_files) == 1
    assert len(meta_files) == 1


def test_warm_start_loads_cache_without_recomputing(tmp_path: Path) -> None:
    chunks = [_chunk("a.md#0", "alpha"), _chunk("b.md#0", "beta")]
    Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)

    spy = MagicMock(side_effect=_fake_encoder)
    warm = Embedding(chunks, cache_dir=tmp_path, encoder=spy)
    spy.assert_not_called()
    # Retrieval still works — the cache loaded the chunk-encode pass.
    assert warm.retrieve("alpha", k=1)[0].chunk.chunk_id == "a.md#0"
    # Now the encoder is called exactly once (for the query).
    assert spy.call_count == 1


def test_changing_chunks_invalidates_cache(tmp_path: Path) -> None:
    Embedding([_chunk("a.md#0", "alpha")], cache_dir=tmp_path, encoder=_fake_encoder)
    Embedding([_chunk("b.md#0", "beta")], cache_dir=tmp_path, encoder=_fake_encoder)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_changing_model_name_invalidates_cache(tmp_path: Path) -> None:
    chunks = [_chunk("a.md#0", "alpha")]
    Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder, model_name="model-A")
    Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder, model_name="model-B")
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_corrupt_cache_falls_back_to_recompute(tmp_path: Path) -> None:
    chunks = [_chunk("a.md#0", "alpha")]
    Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)
    cache_file = next(tmp_path.glob("*.npz"))
    cache_file.write_bytes(b"not a real npz")

    spy = MagicMock(side_effect=_fake_encoder)
    Embedding(chunks, cache_dir=tmp_path, encoder=spy)
    spy.assert_called_once()  # corpus was re-encoded


# ---------------------------------------------------------------------------
# Ranking + result-shape invariants
# ---------------------------------------------------------------------------


def _embedding_over_real_kb(tmp_path: Path) -> Embedding:
    return Embedding(load_kb(FIXTURE_KB), cache_dir=tmp_path, encoder=_fake_encoder)


def test_retrieve_returns_at_most_k_results(tmp_path: Path) -> None:
    assert len(_embedding_over_real_kb(tmp_path).retrieve("license", k=2)) == 2


def test_retrieve_with_k_larger_than_corpus_returns_all(tmp_path: Path) -> None:
    chunks = load_kb(FIXTURE_KB)
    embedding = Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)
    assert len(embedding.retrieve("license", k=100)) == len(chunks)


def test_ranks_are_one_indexed_and_consecutive(tmp_path: Path) -> None:
    results = _embedding_over_real_kb(tmp_path).retrieve("license", k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_scores_are_descending(tmp_path: Path) -> None:
    results = _embedding_over_real_kb(tmp_path).retrieve("license", k=5)
    assert all(a.score >= b.score for a, b in pairwise(results))


def test_retriever_field_is_embedding_literal(tmp_path: Path) -> None:
    results = _embedding_over_real_kb(tmp_path).retrieve("license", k=3)
    assert results
    assert all(r.retriever == "embedding" for r in results)


def test_retrieval_result_round_trips_through_json(tmp_path: Path) -> None:
    results = _embedding_over_real_kb(tmp_path).retrieve("license", k=3)
    assert results
    for r in results:
        assert RetrievalResult.model_validate_json(r.model_dump_json()) == r


# ---------------------------------------------------------------------------
# Edge cases — must mirror BM25's contract for HybridRetriever interchangeability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -3])
def test_retrieve_with_non_positive_k_returns_empty(tmp_path: Path, k: int) -> None:
    assert _embedding_over_real_kb(tmp_path).retrieve("license", k=k) == []


@pytest.mark.parametrize("query", ["", "   \n\t  "])
def test_retrieve_with_blank_query_returns_empty(tmp_path: Path, query: str) -> None:
    assert _embedding_over_real_kb(tmp_path).retrieve(query, k=5) == []


def test_retrieve_with_empty_corpus_returns_empty(tmp_path: Path) -> None:
    embedding = Embedding([], cache_dir=tmp_path, encoder=_fake_encoder)
    assert embedding.retrieve("anything", k=5) == []


def test_empty_corpus_does_not_invoke_encoder(tmp_path: Path) -> None:
    """Constructing over an empty corpus must not download / load the model."""
    spy = MagicMock(side_effect=_fake_encoder)
    Embedding([], cache_dir=tmp_path, encoder=spy)
    spy.assert_not_called()
