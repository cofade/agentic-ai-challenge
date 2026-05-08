r"""Retrievers over the KB chunk list — sparse (BM25) and dense (Embedding).

Both layers of the hybrid RAG strategy documented in ADR-006 live here. The
fused layer (RRF) lands in issue #12; all three implement the same
``retrieve(query, k) -> list[RetrievalResult]`` contract so the supervisor
agent can route to any of them interchangeably.

BM25's tokenizer is intentionally simple: lowercase + Unicode ``\w+`` regex
split + curated EN/DE stopword removal. Numeric tokens (e.g. ``"504"``)
survive, which is required by the canonical "Error 504" acceptance test. No
stemming or per-language dispatch — see ADR-006 for the rationale.

The Embedding retriever uses ``paraphrase-multilingual-MiniLM-L12-v2`` via
sentence-transformers and caches encoded chunk vectors in ``.cache/embeddings/``
keyed by a hash of the model name + chunk content. Cold-start computes and
writes; warm-start loads without recompute. Per-chunk vectors are L2-normalised
at write time so query-time similarity is a single cosine-equivalent dot product.
"""

from __future__ import annotations

import hashlib
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
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


DEFAULT_EMBEDDING_MODEL: Final[str] = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_CACHE_DIR: Final[Path] = Path(".cache/embeddings")

EncoderFn = Callable[[list[str]], "npt.NDArray[np.float32]"]


def _l2_normalise(matrix: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-wise L2 normalisation; zero rows survive as zero (no NaN)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _cache_key(model_name: str, chunks: list[KBChunk]) -> str:
    """Stable 16-hex-char digest over model name and (chunk_id, text) pairs.

    Order of ``chunks`` is part of the key — reordering invalidates the cache,
    which keeps stored embeddings positionally aligned with the chunk list.
    Including ``model_name`` means swapping models invalidates automatically.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\n")
    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode("utf-8"))
        hasher.update(b"\t")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:16]


def _default_encoder_factory(model_name: str) -> EncoderFn:
    """Lazily construct a SentenceTransformer-backed encoder.

    Imported inside the factory so test runs that inject a fake encoder never
    touch sentence-transformers (and therefore never download the ~470 MB model).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> npt.NDArray[np.float32]:
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vectors, dtype=np.float32)

    return encode


class Embedding:
    """Dense retriever over a list of :class:`KBChunk`s.

    Encodes chunk texts once with ``paraphrase-multilingual-MiniLM-L12-v2`` (or
    any sentence-transformers-compatible model passed via ``model_name``) and
    caches the resulting matrix at ``cache_dir/<key>.npz`` so warm-starts skip
    both the model load and the corpus encode. Per-query cost is one query
    encode + one matrix-vector product against the L2-normalised chunk matrix.

    Pass ``encoder=`` a deterministic fake in tests to avoid the model download.
    """

    def __init__(
        self,
        chunks: list[KBChunk],
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = DEFAULT_EMBEDDING_CACHE_DIR,
        encoder: EncoderFn | None = None,
    ) -> None:
        self._chunks = chunks
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._encoder: EncoderFn | None = encoder

        if not chunks:
            self._embeddings: npt.NDArray[np.float32] = np.zeros((0, 0), dtype=np.float32)
            return

        cache_path = cache_dir / f"{_cache_key(model_name, chunks)}.npz"
        cached = self._try_load_cache(cache_path)
        if cached is not None:
            self._embeddings = cached
            return

        # Cache miss: build (and memoise) the encoder for the corpus encode;
        # ``retrieve`` then reuses it for query encodes without reloading the
        # underlying ~470 MB SentenceTransformer.
        raw = self._get_encoder()([c.text for c in chunks])
        self._embeddings = _l2_normalise(np.asarray(raw, dtype=np.float32))
        self._write_cache(cache_path)

    def _get_encoder(self) -> EncoderFn:
        """Lazily build and memoise the encoder used for query and corpus encoding.

        Constructed at most once per ``Embedding`` instance — so a cache-hit
        ``__init__`` followed by many ``retrieve`` calls loads the model exactly
        once, on the first ``retrieve``.
        """
        if self._encoder is None:
            self._encoder = _default_encoder_factory(self._model_name)
        return self._encoder

    def _try_load_cache(self, cache_path: Path) -> npt.NDArray[np.float32] | None:
        if not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached_ids = list(data["chunk_ids"])
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            warnings.warn(
                f"embedding cache {cache_path.name} is corrupt; recomputing: {exc}",
                stacklevel=2,
            )
            return None
        expected_ids = [c.chunk_id for c in self._chunks]
        if cached_ids != expected_ids or embeddings.shape[0] != len(expected_ids):
            # The cache key already commits to (chunk_id, text) per chunk, so a
            # hash match implies an id match. Landing here means a sha256-prefix
            # collision or a bug in ``_cache_key`` — surface it loudly rather
            # than silently re-encoding and masking the cause.
            warnings.warn(
                f"embedding cache {cache_path.name} hash matched but chunk ids "
                "differ; this implies a hash collision or a bug in _cache_key. "
                "Recomputing.",
                stacklevel=2,
            )
            return None
        return embeddings

    def _write_cache(self, cache_path: Path) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_ids = np.array([c.chunk_id for c in self._chunks])
        # ``np.savez`` auto-appends ``.npz`` to a path argument; passing a file
        # handle pins the exact path so the atomic rename below works.
        tmp_npz = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_npz.open("wb") as fh:
            np.savez(fh, embeddings=self._embeddings, chunk_ids=chunk_ids)
        os.replace(tmp_npz, cache_path)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Return up to ``k`` :class:`RetrievalResult`s, ranked by cosine similarity.

        Returns ``[]`` when ``k <= 0``, when the corpus is empty, or when the
        query is empty/whitespace-only — the same edge-case contract as
        :class:`BM25` so the two retrievers are interchangeable in the
        :class:`HybridRetriever` (issue #12) wrapper.
        """
        if k <= 0 or self._embeddings.shape[0] == 0 or not query.strip():
            return []

        query_vec = np.asarray(self._get_encoder()([query]), dtype=np.float32)
        query_vec = _l2_normalise(query_vec.reshape(1, -1))[0]
        scores = self._embeddings @ query_vec

        n = min(k, self._embeddings.shape[0])
        # Stable sort on the negated scores → ties resolve to original chunk order.
        ranked_indices = np.argsort(-scores, kind="stable")[:n]
        return [
            RetrievalResult(
                chunk=self._chunks[int(idx)],
                score=float(scores[int(idx)]),
                rank=rank,
                retriever="embedding",
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]
