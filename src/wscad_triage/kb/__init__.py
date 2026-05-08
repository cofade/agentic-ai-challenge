"""Knowledge-base subpackage: loader + sentence-level chunker + retrievers.

Public surface:

- :func:`load_kb` — top-level entry point that walks a KB directory, parses
  each Markdown file's optional YAML frontmatter, sentence-tokenises the body,
  and returns a flat ``list[KBChunk]`` with per-sentence language tags.
- :class:`RawDoc` — model for a parsed-but-not-yet-chunked document. Used by
  the chunker and by tests; not re-exported at the top-level package because
  values of this type never cross out of ``kb.*`` (the boundary type for the
  rest of the pipeline is :class:`~wscad_triage.schemas.KBChunk`).
- :func:`build_detector` — factory for the en/de language detector; expose
  this so callers that drive ``chunk_document`` directly can reuse one
  detector across many documents.
- :class:`BM25` — sparse retriever over a chunk list.
- :class:`Embedding` — dense multilingual retriever with on-disk cache.
  Composes with BM25 via the RRF :class:`HybridRetriever` (issue #12).
- :func:`tokenize` — the BM25 tokenizer; exposed so retrieval-adjacent code
  (and tests) can reproduce the exact token stream BM25 sees.

The split between :mod:`wscad_triage.kb.loader`, :mod:`wscad_triage.kb.chunker`,
and :mod:`wscad_triage.kb.retriever` mirrors the building-block view at
``docs/05-building-block-view/README.md``. The NLP toolchain choice is
documented in ADR-003; the hybrid RAG strategy in ADR-006.
"""

from wscad_triage.kb.chunker import (
    build_detector,
    chunk_document,
    detect_language,
    load_kb,
    segment_sentences,
)
from wscad_triage.kb.loader import RawDoc, load_documents, parse_frontmatter
from wscad_triage.kb.retriever import BM25, Embedding, tokenize

__all__ = [
    "BM25",
    "Embedding",
    "RawDoc",
    "build_detector",
    "chunk_document",
    "detect_language",
    "load_documents",
    "load_kb",
    "parse_frontmatter",
    "segment_sentences",
    "tokenize",
]
