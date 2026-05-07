"""Unit tests for ``wscad_triage.kb.loader``.

Covers UTF-8 reading, recursive directory walks, frontmatter extraction,
type coercion of non-string scalars, and the empty / frontmatter-only edge
cases the chunker depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wscad_triage.kb.loader import RawDoc, load_documents, parse_frontmatter

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_KB = REPO_ROOT / "tests" / "fixtures" / "kb"
REAL_KB = REPO_ROOT / "kb" / "original"


def _doc_by_name(docs: list[RawDoc], name: str) -> RawDoc:
    return next(d for d in docs if d.source_file == name)


def test_load_real_kb_returns_three_docs() -> None:
    docs = load_documents(REAL_KB)
    assert [d.source_file for d in docs] == [
        "Common_Errors.md",
        "Installation_Requirements.md",
        "Licensing_Offline_Activation.md",
    ]
    for doc in docs:
        assert doc.text.strip()
        assert doc.metadata == {}


def test_load_documents_walks_subdirectories() -> None:
    docs = load_documents(FIXTURE_KB)
    paths = [d.source_file for d in docs]
    assert "nested/buried.md" in paths
    nested = _doc_by_name(docs, "nested/buried.md")
    assert nested.text.strip().startswith("A nested document")


def test_load_documents_returns_sorted_for_determinism() -> None:
    docs = load_documents(FIXTURE_KB)
    paths = [d.source_file for d in docs]
    assert paths == sorted(paths)


def test_empty_file_yields_empty_text() -> None:
    docs = load_documents(FIXTURE_KB)
    empty = _doc_by_name(docs, "empty.md")
    assert empty.text == ""
    assert empty.metadata == {}


def test_frontmatter_extraction_separates_metadata_from_body() -> None:
    docs = load_documents(FIXTURE_KB)
    doc = _doc_by_name(docs, "with_frontmatter.md")
    assert doc.metadata == {
        "product_line": "ELECTRIX AI",
        "version": "7.3.2.16",
        "synthetic_for_demo": "True",
        "chunk_count": "2",
    }
    assert "---" not in doc.text
    assert "license activation procedure" in doc.text


def test_frontmatter_stringifies_non_string_scalars() -> None:
    raw = "---\nflag: true\ncount: 42\nname: hello\n---\nBody.\n"
    metadata, body = parse_frontmatter(raw)
    assert metadata == {"flag": "True", "count": "42", "name": "hello"}
    assert body == "Body.\n"


def test_only_frontmatter_yields_empty_body() -> None:
    docs = load_documents(FIXTURE_KB)
    doc = _doc_by_name(docs, "only_frontmatter.md")
    assert doc.text.strip() == ""
    assert doc.metadata == {
        "product_line": "ELECTRIX AI",
        "version": "7.3.2.10",
    }


def test_no_frontmatter_passes_through_unchanged() -> None:
    raw = "Just a sentence without frontmatter.\n"
    metadata, body = parse_frontmatter(raw)
    assert metadata == {}
    assert body == raw


def test_frontmatter_with_no_closing_delimiter_falls_back_to_no_frontmatter() -> None:
    raw = "---\nproduct: WSCAD\nbut never closed...\nmore body text\n"
    metadata, body = parse_frontmatter(raw)
    assert metadata == {}
    assert body == raw


def test_frontmatter_with_list_value_raises() -> None:
    raw = "---\nproducts:\n  - WSCAD Suite\n  - ELECTRIX AI\n---\nBody.\n"
    with pytest.raises(ValueError, match="products"):
        parse_frontmatter(raw)


def test_frontmatter_with_nested_mapping_raises() -> None:
    raw = "---\nmeta:\n  inner: value\n---\nBody.\n"
    with pytest.raises(ValueError, match="meta"):
        parse_frontmatter(raw)


def test_frontmatter_top_level_must_be_mapping() -> None:
    raw = "---\n- one\n- two\n---\nBody.\n"
    with pytest.raises(ValueError, match="mapping"):
        parse_frontmatter(raw)


def test_frontmatter_with_explicit_null_yaml_treated_as_empty_metadata() -> None:
    """A frontmatter block whose YAML body parses to ``None`` (e.g. ``~``)."""
    raw = "---\n~\n---\nBody.\n"
    metadata, body = parse_frontmatter(raw)
    assert metadata == {}
    assert body == "Body.\n"


def test_unicode_handling_preserves_german_umlauts() -> None:
    docs = load_documents(FIXTURE_KB)
    de = _doc_by_name(docs, "simple_de.md")
    assert "ü" in de.text
    assert "Lizenzstatus" in de.text


def test_load_documents_raises_on_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_load_documents_wraps_non_utf8_with_filename(tmp_path: Path) -> None:
    bad = tmp_path / "broken.md"
    bad.write_bytes(b"\xff\xfe\xfd not valid utf-8")
    with pytest.raises(ValueError, match=r"broken\.md.*UTF-8"):
        load_documents(tmp_path)


def test_load_documents_wraps_malformed_yaml_with_filename(tmp_path: Path) -> None:
    import yaml

    bad = tmp_path / "yaml_broken.md"
    bad.write_text("---\nkey: value: nested\n  bad: indent\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"yaml_broken\.md.*Malformed YAML") as exc_info:
        load_documents(tmp_path)
    # Pin the `from exc` chain so a future refactor cannot silently drop it.
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert isinstance(exc_info.value.__cause__.__cause__, yaml.YAMLError)


def test_load_documents_wraps_list_value_with_filename(tmp_path: Path) -> None:
    bad = tmp_path / "list_value.md"
    bad.write_text("---\nproducts:\n  - one\n  - two\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"list_value\.md.*products"):
        load_documents(tmp_path)


def test_rawdoc_forbids_extra_fields() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RawDoc.model_validate({"source_file": "x.md", "text": "y", "metadata": {}, "stray": "no"})
