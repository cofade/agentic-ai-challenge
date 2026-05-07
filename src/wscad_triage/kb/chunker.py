"""Sentence-level chunking with per-sentence language detection.

Chunks the body text of every :class:`RawDoc` into individual sentences and
attaches:

- ``chunk_id`` — ``"{source_file}#{index}"``, the deterministic citation
  handle used throughout the pipeline. The 0-based index resets per document.
- ``source_file`` — copied through from the loader (forward-slashed relative
  path).
- ``language`` — per-sentence ISO 639-1 code (``"en"`` / ``"de"``) or ``None``
  if the language detector cannot decide on a short or non-linguistic input.
  Restricting the detector to English + German keeps memory low and matches
  the KB scope (German + English release notes).
- ``metadata`` — copied through from the document's frontmatter.

Library choices (``pysbd``, ``lingua-language-detector``) are recorded in
ADR-003. The segmenter is instantiated in English mode and reused on German
text — punctuation conventions overlap enough that quality is acceptable for
the current KB; the language tag still comes from a dedicated detector.
"""

from __future__ import annotations

from pathlib import Path

import pysbd
from lingua import IsoCode639_1, Language, LanguageDetector, LanguageDetectorBuilder

from wscad_triage.kb.loader import RawDoc, load_documents
from wscad_triage.schemas import KBChunk

_SUPPORTED_LANGUAGES = (Language.ENGLISH, Language.GERMAN)


def build_detector() -> LanguageDetector:
    """Construct a deterministic en/de :class:`LanguageDetector`.

    Public so callers (notably tests) can build one detector and reuse it
    rather than paying the construction cost on every call.
    """
    return LanguageDetectorBuilder.from_languages(*_SUPPORTED_LANGUAGES).build()


def _build_segmenter() -> pysbd.Segmenter:
    """Construct a pysbd Segmenter in English mode (used for both en/de)."""
    return pysbd.Segmenter(language="en", clean=False)


def segment_sentences(text: str, segmenter: pysbd.Segmenter | None = None) -> list[str]:
    """Split ``text`` into a list of sentences (whitespace-only inputs → ``[]``).

    If ``segmenter`` is omitted a fresh one is built; pass an explicit
    instance to amortise construction across many documents (this is what
    :func:`load_kb` does internally).
    """
    if not text.strip():
        return []
    seg = segmenter if segmenter is not None else _build_segmenter()
    return [s.strip() for s in seg.segment(text) if s.strip()]


def detect_language(sentence: str, detector: LanguageDetector) -> str | None:
    """Return ``"en"``/``"de"`` for ``sentence`` or ``None`` if undetectable."""
    language = detector.detect_language_of(sentence)
    if language is None:
        return None
    iso: IsoCode639_1 = language.iso_code_639_1
    return iso.name.lower()


def chunk_document(
    doc: RawDoc,
    detector: LanguageDetector,
    segmenter: pysbd.Segmenter | None = None,
) -> list[KBChunk]:
    """Chunk one ``RawDoc`` into a list of ``KBChunk``s.

    Empty bodies (and frontmatter-only files) produce an empty list. The
    document's frontmatter is copied verbatim into every chunk's ``metadata``.
    """
    sentences = segment_sentences(doc.text, segmenter=segmenter)
    return [
        KBChunk(
            chunk_id=f"{doc.source_file}#{idx}",
            source_file=doc.source_file,
            text=sentence,
            language=detect_language(sentence, detector),
            metadata=doc.metadata,
        )
        for idx, sentence in enumerate(sentences)
    ]


def load_kb(kb_dir: Path) -> list[KBChunk]:
    """Load and chunk every ``*.md`` under ``kb_dir`` into a flat ``list[KBChunk]``.

    Builds the language detector and the sentence segmenter once each and
    reuses them across every document, so per-call construction cost is
    paid only once per ``load_kb`` invocation.
    """
    detector = build_detector()
    segmenter = _build_segmenter()
    chunks: list[KBChunk] = []
    for doc in load_documents(kb_dir):
        chunks.extend(chunk_document(doc, detector, segmenter=segmenter))
    return chunks
