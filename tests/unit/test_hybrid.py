"""Unit tests for ``wscad_triage.kb.retriever.HybridRetriever``.

Covers the issue-#12 acceptance criterion ("RRF returns top-k that consistently
includes the strongest result from either single retriever") plus the RRF math
itself (per Cormack et al. 2009, ADR-006), the BM25-wins-on-RRF-tie contract,
the shared retriever contract (ranks, scores, edge cases, JSON round-trip),
and a regression-watch pin on BM25's out-of-vocabulary fake-hit fall-through
(documented in ADR-006's negative-consequences section; mitigation deferred
to issue #16).

Stub retrievers are used wherever the test asserts a precise RRF score or a
specific cross-retriever ordering — that pins the math without depending on
BM25 / Embedding scoring quirks. Acceptance tests use real ``BM25`` and
``Embedding`` (with the fake encoder from ``test_embedding.py``) to verify
end-to-end composition.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import pytest

from wscad_triage.kb import BM25, Embedding, HybridRetriever, load_kb
from wscad_triage.schemas import KBChunk, RetrievalResult, RetrieverName

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
    """Same deterministic, content-derived encoder as test_embedding.py.

    Two texts that share whitespace-split tokens end up close in cosine
    similarity; texts with no shared tokens are orthogonal.
    """
    vectors = np.zeros((len(texts), FAKE_DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in text.lower().split():
            slot = hash(token) % FAKE_DIM
            vectors[i, slot] += 1.0
    return vectors


class _StubRetriever:
    """Minimal duck-typed stand-in for :class:`BM25` / :class:`Embedding`.

    Returns a precomputed ranked list of chunks regardless of ``query`` /
    ``k`` (truncated to ``k``). Lets the RRF-math tests assert exact fused
    scores without depending on BM25 / cosine scoring.
    """

    def __init__(self, name: RetrieverName, ranked: list[KBChunk]) -> None:
        self._name = name
        self._ranked = ranked

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if k <= 0 or not query.strip() or not self._ranked:
            return []
        n = min(k, len(self._ranked))
        return [
            RetrievalResult(
                chunk=self._ranked[i],
                # Score values are arbitrary — RRF only consults rank.
                score=1.0 - 0.1 * i,
                rank=i + 1,
                retriever=self._name,
            )
            for i in range(n)
        ]


def _hybrid(bm25: object, embedding: object, **kwargs: int) -> HybridRetriever:
    """Build a HybridRetriever, casting duck-typed stubs to the declared types."""
    return HybridRetriever(
        cast(BM25, bm25),
        cast(Embedding, embedding),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion (issue #12)
# ---------------------------------------------------------------------------


def test_rrf_top_k_includes_top1_from_each_retriever_via_stubs() -> None:
    """Pinned: when the two retrievers disagree on top-1, hybrid surfaces both.

    The roadmap acceptance criterion ("top-k consistently includes the
    strongest result from either single retriever") is enforced here against
    stubs that *guarantee* divergence — a real-retriever version follows.

    Each retriever returns a disjoint two-element ranking; no chunk is a
    cross-retriever consensus pick (which would otherwise outscore each
    individual top-1 via summed RRF contributions and crowd it out of top-2).
    """
    a = _chunk("a.md#0", "alpha")
    b = _chunk("b.md#0", "beta")
    c = _chunk("c.md#0", "gamma")
    d = _chunk("d.md#0", "delta")
    bm25 = _StubRetriever("bm25", [a, c])  # BM25 top-1 = a
    emb = _StubRetriever("embedding", [b, d])  # Embedding top-1 = b

    hybrid = _hybrid(bm25, emb)
    results = hybrid.retrieve("any query", k=2)
    ids = {r.chunk.chunk_id for r in results}
    assert {"a.md#0", "b.md#0"} <= ids


def test_pool_widening_surfaces_consensus_picks() -> None:
    """A consensus pick at rank-2-in-both wins the hybrid top-1.

    Stubs are constructed so a chunk ranked 2 in both retrievers — fused
    score ``2/(60+2)`` — outranks chunks ranked 1 in only one — fused
    score ``1/(60+1)``. The widened pool (``RRF_POOL_MULTIPLIER * k``)
    is what makes the rank-2 chunk visible at all; with pool=k=1 it
    would not be in the candidate set.
    """
    a = _chunk("a.md#0", "alpha")  # BM25 rank 1 only
    b = _chunk("b.md#0", "beta")  # Embedding rank 1 only
    c = _chunk("c.md#0", "gamma")  # rank 2 in both — the consensus pick
    bm25 = _StubRetriever("bm25", [a, c])
    emb = _StubRetriever("embedding", [b, c])

    [top] = _hybrid(bm25, emb).retrieve("q", k=1)
    assert top.chunk.chunk_id == "c.md#0"
    assert top.score == pytest.approx(2.0 / 62)


# ---------------------------------------------------------------------------
# RRF math (Cormack et al. 2009: score = sum 1/(k + rank), k=60)
# ---------------------------------------------------------------------------


def test_rrf_score_matches_formula_for_chunk_in_both() -> None:
    """A chunk at rank 1 in both retrievers scores ``2 / (60 + 1)`` exactly."""
    a = _chunk("a.md#0", "alpha")
    bm25 = _StubRetriever("bm25", [a])
    emb = _StubRetriever("embedding", [a])

    [result] = _hybrid(bm25, emb).retrieve("q", k=1)
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert result.score == pytest.approx(expected)


def test_rrf_chunk_in_only_one_retriever_gets_single_contribution() -> None:
    a = _chunk("a.md#0", "alpha")
    b = _chunk("b.md#0", "beta")
    bm25 = _StubRetriever("bm25", [a, b])  # a:1, b:2
    emb = _StubRetriever("embedding", [b])  # b:1, a missing

    results = _hybrid(bm25, emb).retrieve("q", k=2)
    by_id = {r.chunk.chunk_id: r.score for r in results}

    # b: 1/(60+2) from BM25 + 1/(60+1) from Embedding
    assert by_id["b.md#0"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    # a: 1/(60+1) from BM25 only
    assert by_id["a.md#0"] == pytest.approx(1.0 / 61)
    # b's combined contribution outranks a's single contribution
    assert results[0].chunk.chunk_id == "b.md#0"


def test_custom_rrf_k_changes_scores() -> None:
    """The ``rrf_k`` constructor kwarg is plumbed through into the formula."""
    a = _chunk("a.md#0", "alpha")
    bm25 = _StubRetriever("bm25", [a])
    emb = _StubRetriever("embedding", [])

    [result] = _hybrid(bm25, emb, rrf_k=10).retrieve("q", k=1)
    assert result.score == pytest.approx(1.0 / 11)


@pytest.mark.parametrize("rrf_k", [0, -1, -100])
def test_constructor_rejects_non_positive_rrf_k(rrf_k: int) -> None:
    bm25 = _StubRetriever("bm25", [])
    emb = _StubRetriever("embedding", [])
    with pytest.raises(ValueError, match="rrf_k must be positive"):
        _hybrid(bm25, emb, rrf_k=rrf_k)


def test_bm25_wins_exact_rrf_tie() -> None:
    """Tie-break contract: a BM25-only and an embedding-only chunk at the
    same rank produce equal RRF scores; BM25 is folded in first so the
    BM25-favoured chunk lands ahead of the embedding-favoured one.

    Pins the docstring claim — without this, a refactor of the dict-loop
    into a comprehension could silently flip the order.
    """
    a = _chunk("a.md#0", "alpha")  # BM25 only
    b = _chunk("b.md#0", "beta")  # Embedding only
    bm25 = _StubRetriever("bm25", [a])
    emb = _StubRetriever("embedding", [b])

    results = _hybrid(bm25, emb).retrieve("q", k=2)
    assert [r.chunk.chunk_id for r in results] == ["a.md#0", "b.md#0"]
    assert results[0].score == pytest.approx(results[1].score)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_same_chunk_in_both_retrievers_appears_once() -> None:
    a = _chunk("a.md#0", "alpha")
    b = _chunk("b.md#0", "beta")
    bm25 = _StubRetriever("bm25", [a, b])
    emb = _StubRetriever("embedding", [a, b])

    results = _hybrid(bm25, emb).retrieve("q", k=5)
    ids = [r.chunk.chunk_id for r in results]
    assert len(ids) == len(set(ids)), f"duplicate chunk_ids: {ids}"
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# Ranking + result-shape invariants (mirrors BM25 / Embedding)
# ---------------------------------------------------------------------------


def _hybrid_over_fixture_kb(tmp_path: Path) -> HybridRetriever:
    chunks = load_kb(FIXTURE_KB)
    return HybridRetriever(
        BM25(chunks),
        Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder),
    )


def test_retrieve_returns_at_most_k_results(tmp_path: Path) -> None:
    assert len(_hybrid_over_fixture_kb(tmp_path).retrieve("license", k=2)) == 2


def test_retrieve_with_k_larger_than_corpus_returns_all_unique(tmp_path: Path) -> None:
    chunks = load_kb(FIXTURE_KB)
    hybrid = HybridRetriever(
        BM25(chunks), Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)
    )
    results = hybrid.retrieve("license", k=1000)
    # k larger than corpus → at most |corpus| unique chunks
    assert len(results) <= len(chunks)
    assert len({r.chunk.chunk_id for r in results}) == len(results)


def test_ranks_are_one_indexed_and_consecutive(tmp_path: Path) -> None:
    results = _hybrid_over_fixture_kb(tmp_path).retrieve("license", k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_scores_are_descending(tmp_path: Path) -> None:
    results = _hybrid_over_fixture_kb(tmp_path).retrieve("license", k=5)
    assert all(a.score >= b.score for a, b in pairwise(results))


def test_retriever_field_is_rrf_literal(tmp_path: Path) -> None:
    results = _hybrid_over_fixture_kb(tmp_path).retrieve("license", k=3)
    assert results
    assert all(r.retriever == "rrf" for r in results)


def test_retrieval_result_round_trips_through_json(tmp_path: Path) -> None:
    results = _hybrid_over_fixture_kb(tmp_path).retrieve("license", k=3)
    assert results
    for r in results:
        assert RetrievalResult.model_validate_json(r.model_dump_json()) == r


# ---------------------------------------------------------------------------
# Edge cases — match the BM25 / Embedding edge-case contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -3])
def test_retrieve_with_non_positive_k_returns_empty(tmp_path: Path, k: int) -> None:
    assert _hybrid_over_fixture_kb(tmp_path).retrieve("license", k=k) == []


@pytest.mark.parametrize("query", ["", "   \n\t  "])
def test_retrieve_with_blank_query_returns_empty(tmp_path: Path, query: str) -> None:
    assert _hybrid_over_fixture_kb(tmp_path).retrieve(query, k=5) == []


def test_retrieve_with_empty_corpus_returns_empty(tmp_path: Path) -> None:
    hybrid = HybridRetriever(BM25([]), Embedding([], cache_dir=tmp_path, encoder=_fake_encoder))
    assert hybrid.retrieve("anything", k=5) == []


def test_stopwords_only_query_falls_back_to_embedding_hits(tmp_path: Path) -> None:
    """All-stopword query → BM25 returns []; Embedding still returns hits.

    The ADR-006 contract: hybrid degrades gracefully when one retriever has
    nothing to contribute.
    """
    chunks = load_kb(FIXTURE_KB)
    bm25 = BM25(chunks)
    embedding = Embedding(chunks, cache_dir=tmp_path, encoder=_fake_encoder)
    hybrid = HybridRetriever(bm25, embedding)

    stopwords_query = "the and of"
    assert bm25.retrieve(stopwords_query, k=3) == []
    embedding_hits = embedding.retrieve(stopwords_query, k=3)
    hybrid_hits = hybrid.retrieve(stopwords_query, k=3)

    # Hybrid is non-empty iff embedding is (the only contributing layer).
    assert bool(hybrid_hits) == bool(embedding_hits)
    if embedding_hits:
        # Order follows embedding ranks (1/(60+rank) is monotonic).
        assert [r.chunk.chunk_id for r in hybrid_hits] == [r.chunk.chunk_id for r in embedding_hits]


def test_empty_embedding_layer_falls_back_to_bm25_hits(tmp_path: Path) -> None:
    """Symmetric to the all-stopwords case: when the embedding layer is built
    over an empty corpus and BM25 isn't, the BM25 hits flow through alone.
    """
    chunks = load_kb(FIXTURE_KB)
    bm25 = BM25(chunks)
    empty_embedding = Embedding([], cache_dir=tmp_path, encoder=_fake_encoder)
    hybrid = HybridRetriever(bm25, empty_embedding)

    query = "license"
    bm25_hits = bm25.retrieve(query, k=3)
    hybrid_hits = hybrid.retrieve(query, k=3)
    assert bm25_hits, "fixture sanity: BM25 should match 'license'"
    assert [r.chunk.chunk_id for r in hybrid_hits] == [r.chunk.chunk_id for r in bm25_hits]


# ---------------------------------------------------------------------------
# Regression-watch: BM25 OOV fake-hit fall-through (ADR-006 negative bullet)
# ---------------------------------------------------------------------------


def test_bm25_oov_fake_hits_flow_through_unfiltered(tmp_path: Path) -> None:
    """Pin the current degraded behaviour ADR-006 documents and defers to #16.

    On a fully out-of-vocabulary query, ``BM25.get_scores`` returns an
    all-zero vector. The BM25 wrapper still emits top-``k`` results in
    original-corpus order with ``score == 0.0``. RRF currently consumes
    those unfiltered — meaning a score-zero "fake hit" gets the same
    rank-1 vote as a genuine top-1 match.

    The pin is constructed against an empty embedding layer so the
    assertion is deterministic: only BM25 contributes, and its OOV hits
    appear verbatim in the hybrid output. When issue #16's mitigation
    lands (e.g. a score-threshold filter on BM25 results before RRF) the
    hybrid output will become ``[]`` here and this test will demand an
    update.
    """
    chunks = load_kb(FIXTURE_KB)
    bm25 = BM25(chunks)
    empty_embedding = Embedding([], cache_dir=tmp_path, encoder=_fake_encoder)
    hybrid = HybridRetriever(bm25, empty_embedding)

    oov_query = "xyzzy plover quux"
    bm25_oov = bm25.retrieve(oov_query, k=3)
    assert bm25_oov, "BM25 emits fake hits on OOV queries — the property under test"
    assert all(r.score == 0.0 for r in bm25_oov), (
        "OOV fake hits must score exactly 0.0; if BM25 scoring changes, "
        "this assertion needs to be revisited along with the RRF mitigation."
    )

    hybrid_hits = hybrid.retrieve(oov_query, k=3)
    assert {r.chunk.chunk_id for r in hybrid_hits} == {r.chunk.chunk_id for r in bm25_oov}, (
        "Current behaviour: BM25 zero-score OOV fake hits flow through into the "
        "hybrid result. Flipping this requires a deliberate change — see "
        "ADR-006 negative-consequences and issue #16."
    )
