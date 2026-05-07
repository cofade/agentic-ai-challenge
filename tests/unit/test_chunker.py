"""Unit tests for ``wscad_triage.kb.chunker``.

Covers the three acceptance bullets from issue #9: chunk counts, frontmatter
extraction propagation, and language detection on bilingual documents. The
sentence-segmenter is third-party (``pysbd``); we assert outcomes (counts,
ids, language tags) rather than the exact split rules.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from wscad_triage.kb import (
    RawDoc,
    build_detector,
    chunk_document,
    load_kb,
    segment_sentences,
)
from wscad_triage.schemas import KBChunk

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_KB = REPO_ROOT / "tests" / "fixtures" / "kb"
REAL_KB = REPO_ROOT / "kb" / "original"


def _by_source(chunks: list[KBChunk]) -> dict[str, list[KBChunk]]:
    grouped: dict[str, list[KBChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.source_file].append(chunk)
    return grouped


def test_load_kb_real_corpus_chunk_counts() -> None:
    """Chunk counts on the shipped KB.

    ``Installation_Requirements.md`` contains the token ``.NET`` whose leading
    period is misinterpreted as a sentence boundary by ``pysbd``; we accept
    the resulting 2-chunk split for that file (documented in ADR-003 as a
    known limitation) and assert the deterministic total.
    """
    chunks = load_kb(REAL_KB)
    by_src = _by_source(chunks)
    assert len(by_src["Common_Errors.md"]) == 1
    assert len(by_src["Installation_Requirements.md"]) == 2
    assert len(by_src["Licensing_Offline_Activation.md"]) == 2
    assert len(chunks) == 5


def test_chunk_id_format_is_source_hash_index() -> None:
    chunks = load_kb(REAL_KB)
    seen_indices: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        prefix, _, suffix = chunk.chunk_id.partition("#")
        assert prefix == chunk.source_file
        idx = int(suffix)
        seen_indices[chunk.source_file].append(idx)
    for source, indices in seen_indices.items():
        assert indices == list(
            range(len(indices))
        ), f"chunk indices for {source} are not 0..N-1: {indices}"


def test_chunk_counts_per_fixture_doc() -> None:
    chunks = load_kb(FIXTURE_KB)
    by_src = _by_source(chunks)
    assert len(by_src["simple_en.md"]) == 3
    assert len(by_src["simple_de.md"]) == 3
    assert len(by_src["with_frontmatter.md"]) == 2
    assert len(by_src["bilingual.md"]) == 6
    assert len(by_src["nested/buried.md"]) == 1
    assert "empty.md" not in by_src
    assert "only_frontmatter.md" not in by_src


def test_language_detection_english_doc_all_en() -> None:
    chunks = load_kb(FIXTURE_KB)
    en_chunks = [c for c in chunks if c.source_file == "simple_en.md"]
    assert all(c.language == "en" for c in en_chunks)


def test_language_detection_german_doc_all_de() -> None:
    chunks = load_kb(FIXTURE_KB)
    de_chunks = [c for c in chunks if c.source_file == "simple_de.md"]
    assert all(c.language == "de" for c in de_chunks)


def test_bilingual_document_per_sentence_language_tags() -> None:
    """Acceptance criterion: language detection on bilingual docs.

    The fixture mixes EN/DE sentences and includes a German sentence that
    embeds the English brand name "WSCAD SUITE 2025" — a real-world case
    where a naive char-trigram detector would mis-tag the sentence as
    English. ``lingua`` is supposed to handle this; this test pins that.
    """
    chunks = load_kb(FIXTURE_KB)
    bilingual = [c for c in chunks if c.source_file == "bilingual.md"]
    assert [c.language for c in bilingual] == ["en", "de", "en", "de", "de", "en"]


def test_metadata_propagates_from_frontmatter_to_every_chunk() -> None:
    chunks = load_kb(FIXTURE_KB)
    fm_chunks = [c for c in chunks if c.source_file == "with_frontmatter.md"]
    expected = {
        "product_line": "ELECTRIX AI",
        "version": "7.3.2.16",
        "synthetic_for_demo": "True",
        "chunk_count": "2",
    }
    for chunk in fm_chunks:
        assert chunk.metadata == expected


def test_empty_doc_yields_no_chunks() -> None:
    detector = build_detector()
    empty_doc = RawDoc(source_file="empty.md", text="", metadata={})
    assert chunk_document(empty_doc, detector) == []


def test_whitespace_only_doc_yields_no_chunks() -> None:
    detector = build_detector()
    blank = RawDoc(source_file="blank.md", text="   \n\n  \t  \n", metadata={})
    assert chunk_document(blank, detector) == []


def test_chunks_round_trip_through_kbchunk_schema() -> None:
    """Every chunk emitted by the chunker must reparse cleanly."""
    chunks = load_kb(FIXTURE_KB)
    for chunk in chunks:
        reparsed = KBChunk.model_validate_json(chunk.model_dump_json())
        assert reparsed == chunk


def test_segment_sentences_handles_blank_input() -> None:
    assert segment_sentences("") == []
    assert segment_sentences("   \n\t  ") == []


def test_segment_sentences_strips_whitespace() -> None:
    sentences = segment_sentences("First sentence.   Second sentence.")
    assert sentences == ["First sentence.", "Second sentence."]


def test_detect_language_returns_none_for_undetectable_input() -> None:
    """``lingua`` returns no decision on numeric / punctuation-only inputs."""
    from wscad_triage.kb import detect_language

    detector = build_detector()
    assert detect_language("123", detector) is None
    assert detect_language("!!!", detector) is None


def test_chunk_document_carries_document_metadata_independent_of_text() -> None:
    detector = build_detector()
    doc = RawDoc(
        source_file="x.md",
        text="One. Two.",
        metadata={"version": "1.0"},
    )
    chunks = chunk_document(doc, detector)
    assert len(chunks) == 2
    assert all(c.metadata == {"version": "1.0"} for c in chunks)
    assert [c.chunk_id for c in chunks] == ["x.md#0", "x.md#1"]
