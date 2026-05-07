"""Unit tests for ``wscad_triage.kb.retriever``.

Covers the issue-#10 acceptance criterion ("Error 504" retrieves the
licensing chunk top-1) plus the BM25 contract surface: tokenizer correctness,
ranking invariants, and edge cases (empty corpus, empty / stopword-only
query, ``k`` boundaries).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from wscad_triage.kb import BM25, load_kb, tokenize
from wscad_triage.schemas import KBChunk, RetrievalResult

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_KB = REPO_ROOT / "kb" / "original"
FIXTURE_KB = REPO_ROOT / "tests" / "fixtures" / "kb"


def _chunk(chunk_id: str, text: str, language: str | None = "en") -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        source_file=chunk_id.split("#")[0],
        text=text,
        language=language,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion (issue #10)
# ---------------------------------------------------------------------------


def test_error_504_query_retrieves_licensing_chunk_top1() -> None:
    """The canonical issue-#10 acceptance test."""
    bm25 = BM25(load_kb(REAL_KB))
    results = bm25.retrieve("Error 504", k=3)
    assert results, "expected non-empty result list"
    top = results[0]
    assert top.chunk.source_file == "Common_Errors.md"
    assert "504" in top.chunk.text


# ---------------------------------------------------------------------------
# Ranking + result-shape invariants
# ---------------------------------------------------------------------------


def test_retrieve_returns_at_most_k_results() -> None:
    bm25 = BM25(load_kb(REAL_KB))
    assert len(bm25.retrieve("license", k=2)) == 2


def test_retrieve_with_k_larger_than_corpus_returns_all() -> None:
    chunks = load_kb(REAL_KB)
    bm25 = BM25(chunks)
    results = bm25.retrieve("license", k=100)
    assert len(results) == len(chunks)


def test_ranks_are_one_indexed_and_consecutive() -> None:
    bm25 = BM25(load_kb(REAL_KB))
    results = bm25.retrieve("license", k=4)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_scores_are_descending() -> None:
    bm25 = BM25(load_kb(REAL_KB))
    results = bm25.retrieve("license", k=5)
    assert all(a.score >= b.score for a, b in pairwise(results))


def test_retriever_field_is_bm25_literal() -> None:
    bm25 = BM25(load_kb(REAL_KB))
    results = bm25.retrieve("Error 504", k=3)
    assert all(r.retriever == "bm25" for r in results)


def test_retrieval_result_round_trips_through_json() -> None:
    bm25 = BM25(load_kb(REAL_KB))
    results = bm25.retrieve("Error 504", k=3)
    for r in results:
        assert RetrievalResult.model_validate_json(r.model_dump_json()) == r


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_retrieve_with_k_zero_returns_empty() -> None:
    assert BM25(load_kb(REAL_KB)).retrieve("license", k=0) == []


def test_retrieve_with_k_negative_returns_empty() -> None:
    assert BM25(load_kb(REAL_KB)).retrieve("license", k=-3) == []


def test_retrieve_with_empty_query_returns_empty() -> None:
    assert BM25(load_kb(REAL_KB)).retrieve("", k=5) == []


def test_retrieve_with_whitespace_query_returns_empty() -> None:
    assert BM25(load_kb(REAL_KB)).retrieve("   \n\t  ", k=5) == []


def test_retrieve_with_only_stopwords_query_returns_empty() -> None:
    """A query made entirely of stopwords tokenises to ``[]``."""
    assert BM25(load_kb(REAL_KB)).retrieve("the and of", k=5) == []


def test_retrieve_with_empty_corpus_returns_empty() -> None:
    assert BM25([]).retrieve("anything", k=5) == []


def test_retrieve_with_all_empty_chunks_returns_empty() -> None:
    """When every chunk tokenises to ``[]`` the index is unbuildable."""
    chunks = [
        _chunk("a.md#0", "..."),
        _chunk("b.md#0", "the and of"),
    ]
    assert BM25(chunks).retrieve("anything", k=5) == []


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases() -> None:
    assert tokenize("Error 504") == ["error", "504"]


def test_tokenize_drops_english_stopwords() -> None:
    assert tokenize("the application starts") == ["application", "starts"]


def test_tokenize_drops_german_stopwords() -> None:
    assert tokenize("die Anwendung startet") == ["anwendung", "startet"]


def test_tokenize_preserves_unicode_umlauts() -> None:
    assert tokenize("Lizenzstatus überprüfen") == ["lizenzstatus", "überprüfen"]


def test_tokenize_preserves_numeric_tokens() -> None:
    assert "504" in tokenize("Error 504")


def test_tokenize_strips_punctuation() -> None:
    """Pure-punctuation input tokenises to ``[]``."""
    assert tokenize("...") == []
    assert tokenize("!?,.;:") == []


# ---------------------------------------------------------------------------
# Multilingual sanity
# ---------------------------------------------------------------------------


def test_german_query_finds_german_chunk() -> None:
    """A German query against a corpus that includes German chunks should
    plausibly surface a German chunk in the top-3.

    Not strict top-1: the German fixture has only one DE chunk and BM25 has
    no cross-lingual signal, so we just assert the DE chunk shows up.
    """
    bm25 = BM25(load_kb(FIXTURE_KB))
    results = bm25.retrieve("Lizenzstatus überprüfen", k=3)
    assert any(r.chunk.language == "de" for r in results)
