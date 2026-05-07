r"""BM25 sparse retriever over the KB chunk list.

This is the sparse layer of the hybrid RAG strategy documented in ADR-006.
The dense (embeddings) and fused (RRF) layers land in issues #11 and #12;
all three implement the same ``retrieve(query, k) -> list[RetrievalResult]``
contract so the supervisor agent can route to any of them interchangeably.

The tokenizer is intentionally simple: lowercase + Unicode ``\w+`` regex split
+ curated EN/DE stopword removal. Numeric tokens (e.g. ``"504"``) survive,
which is required by the canonical "Error 504" acceptance test. No stemming
or per-language dispatch — see ADR-006 for the rationale.
"""

from __future__ import annotations

import re
from typing import Final

from rank_bm25 import BM25Okapi

from wscad_triage.schemas import KBChunk, RetrievalResult

STOPWORDS_EN: Final[frozenset[str]] = frozenset(
    {
        "a", "about", "an", "and", "any", "are", "as", "at", "be", "been",
        "being", "but", "by", "can", "could", "did", "do", "does", "for",
        "from", "had", "has", "have", "if", "in", "into", "is", "it", "its",
        "no", "not", "of", "on", "or", "such", "than", "that", "the", "their",
        "then", "there", "these", "they", "this", "to", "was", "were", "will",
        "with", "would",
    }
)  # fmt: skip

STOPWORDS_DE: Final[frozenset[str]] = frozenset(
    {
        "aber", "alle", "als", "am", "an", "auch", "auf", "aus", "bei", "bin",
        "bis", "das", "dass", "dem", "den", "der", "des", "die", "doch", "ein",
        "eine", "einem", "einen", "einer", "eines", "er", "es", "für", "hat",
        "hatte", "ich", "im", "ist", "mit", "nach", "nicht", "noch", "nur",
        "oder", "sein", "sich", "sie", "sind", "über", "um", "und", "von",
        "vor", "war", "waren", "wenn", "werden", "wie", "wird", "wir", "zu",
        "zum", "zur",
    }
)  # fmt: skip

_STOPWORDS: Final[frozenset[str]] = STOPWORDS_EN | STOPWORDS_DE
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase, regex-split into Unicode word tokens, drop EN/DE stopwords.

    Numeric tokens (e.g. ``"504"``) are preserved — the canonical "Error 504"
    acceptance test depends on this.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """BM25Okapi retriever over a list of :class:`KBChunk`s.

    Build once with the full chunk list; call :meth:`retrieve` per query.
    """

    def __init__(self, chunks: list[KBChunk]) -> None:
        self._chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        # ``BM25Okapi`` divides by the average document length and raises on an
        # empty corpus. Short-circuit when there are no chunks at all or when
        # every chunk tokenises to nothing (only stopwords / punctuation).
        if not tokenized or not any(tokenized):
            self._bm25: BM25Okapi | None = None
        else:
            self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Return up to ``k`` :class:`RetrievalResult`s, ranked by BM25 score.

        Returns ``[]`` when ``k <= 0``, when the corpus is empty, or when the
        query tokenises to nothing (all stopwords or all punctuation).
        """
        if self._bm25 is None or k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        n = min(k, len(self._chunks))
        # ``sorted`` is stable on ties → ties resolve to the original chunk order.
        ranked_indices = sorted(range(len(self._chunks)), key=lambda i: -scores[i])[:n]
        return [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                rank=rank,
                retriever="bm25",
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]
