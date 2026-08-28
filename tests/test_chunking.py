"""Hierarchical chunking behaviour.

These tests use a synthetic document and a word-based token counter, so they
run without PDFs, models or Neo4j.
"""

from __future__ import annotations

import pytest

from rag.chunker import TokenCounter, chunk_document, chunk_section, split_sentences
from rag.schemas import Paragraph, ParsedDocument

from conftest import make_section


def test_short_section_is_not_split(parsed_document, counter, chunking_config):
    """A short coherent section stays whole even though it exceeds target_tokens.

    The first section holds 70 tokens against target=100, max=140.
    """
    section = parsed_document.sections[0]
    chunks = chunk_section(
        parsed_document, section, counter, chunking_config, start_chunk_index=0
    )
    assert len(chunks) == 1
    assert chunks[0].section_title == "What Is Matter?"


def test_long_section_is_split_into_multiple_chunks(
    parsed_document, counter, chunking_config
):
    section = parsed_document.sections[1]
    chunks = chunk_section(
        parsed_document, section, counter, chunking_config, start_chunk_index=0
    )
    assert len(chunks) > 1


def test_chunks_never_span_two_sections(parsed_document, counter, chunking_config):
    """Section boundaries are hard, which keeps the Section parent unambiguous."""
    chunks = chunk_document(parsed_document, counter, chunking_config)
    section_ids = {s.section_id for s in parsed_document.sections}
    for chunk in chunks:
        assert chunk.section_id in section_ids
    # Each chunk belongs to exactly one section, so grouping recovers both.
    produced = {chunk.section_id for chunk in chunks}
    assert produced == section_ids


def _shares_text(first, second) -> bool:
    """True when ``second`` begins with text that also ends ``first``."""
    shared_paragraphs = set(first.text.split("\n\n")) & set(second.text.split("\n\n"))
    if shared_paragraphs:
        return True
    lead = second.text.split("\n\n")[0]
    return bool(lead) and lead in first.text


def test_consecutive_chunks_overlap(parsed_document, counter, chunking_config):
    """Adjacent chunks in one section share boundary text."""
    section = parsed_document.sections[1]
    chunks = chunk_section(
        parsed_document, section, counter, chunking_config, start_chunk_index=0
    )
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:]):
        assert _shares_text(first, second), (
            f"no overlap between {first.chunk_id} and {second.chunk_id}"
        )


def test_overlap_survives_paragraphs_larger_than_the_overlap_budget(
    metadata, counter, chunking_config
):
    """A paragraph bigger than overlap_tokens still yields a text overlap.

    Whole-unit carry-back cannot apply here, so the packer must fall back to
    carrying the trailing sentences of the previous chunk.
    """
    paragraphs = [
        Paragraph(
            text=" ".join(f"Ice melts into water at step {i} of {n}." for i in range(12)),
            page_number=1,
        )
        for n in range(4)
    ]
    section = make_section(metadata.document_id, 0, "Melting", paragraphs)
    document = ParsedDocument(
        metadata=metadata, content_hash="h", page_count=1, sections=[section]
    )
    chunks = chunk_section(
        document, section, counter, chunking_config, start_chunk_index=0
    )
    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:]):
        assert _shares_text(first, second)


def test_chunk_ids_are_deterministic(parsed_document, counter, chunking_config):
    """Re-chunking the same input yields identical IDs, which is what makes
    ingestion idempotent."""
    first = chunk_document(parsed_document, counter, chunking_config)
    second = chunk_document(parsed_document, counter, chunking_config)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len(set(c.chunk_id for c in first)) == len(first), "chunk IDs must be unique"


def test_chunk_index_is_document_wide_and_ordered(
    parsed_document, counter, chunking_config
):
    chunks = chunk_document(parsed_document, counter, chunking_config)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_carries_full_lineage(parsed_document, counter, chunking_config):
    """Every chunk must be traceable up to Grade without extra lookups."""
    chunks = chunk_document(parsed_document, counter, chunking_config)
    assert chunks
    for chunk in chunks:
        assert chunk.section_id
        assert chunk.page_ids
        assert chunk.document_id == parsed_document.metadata.document_id
        assert chunk.unit_id == "grade_02:science:unit_01_matter"
        assert chunk.subject_id == "grade_02:science"
        assert chunk.grade_id == "grade_02"
        assert chunk.grade == 2
        assert chunk.subject == "science"
        assert chunk.page_start <= chunk.page_end


def test_page_range_matches_source_paragraphs(
    parsed_document, counter, chunking_config
):
    """Page ranges come from the paragraphs actually included, not the section."""
    chunks = chunk_document(parsed_document, counter, chunking_config)
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 3
        assert chunk.page_ids
        assert f":p{chunk.page_start:04d}" in chunk.page_ids[0]
        assert f":p{chunk.page_end:04d}" in chunk.page_ids[-1]


def test_oversized_paragraph_is_split_by_sentence(metadata, counter, chunking_config):
    """A single paragraph above max_tokens falls back to sentence boundaries."""
    sentences = " ".join(f"Water freezes at zero degrees number {i}." for i in range(40))
    section = make_section(
        metadata.document_id,
        0,
        "Freezing",
        [Paragraph(text=sentences, page_number=4)],
    )
    document = ParsedDocument(
        metadata=metadata, content_hash="h", page_count=4, sections=[section]
    )
    chunks = chunk_section(
        document, section, counter, chunking_config, start_chunk_index=0
    )
    assert len(chunks) > 1
    for chunk in chunks:
        # Sentence-level splitting keeps chunks near the target, not the ceiling.
        assert chunk.token_count <= chunking_config.max_tokens


def test_tiny_fragments_are_dropped(metadata, counter, chunking_config):
    """Page furniture (stray numbers, single words) must not become chunks."""
    section = make_section(
        metadata.document_id, 0, "Stub", [Paragraph(text="42", page_number=1)]
    )
    document = ParsedDocument(
        metadata=metadata, content_hash="h", page_count=1, sections=[section]
    )
    assert chunk_section(
        document, section, counter, chunking_config, start_chunk_index=0
    ) == []


def test_empty_document_produces_no_chunks(metadata, counter, chunking_config):
    document = ParsedDocument(
        metadata=metadata, content_hash="h", page_count=0, sections=[]
    )
    assert chunk_document(document, counter, chunking_config) == []


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two. Three.", 3),
        ("No terminator here", 1),
        ("Is it 2 cm? Yes. It is.", 3),
        ("", 0),
    ],
)
def test_split_sentences(text, expected):
    assert len(split_sentences(text)) == expected


def test_fallback_token_counter_is_used_without_tokenizer():
    """Chunking degrades to estimated counts rather than failing outright."""
    counter = TokenCounter(tokenizer_path=None)
    assert counter.is_exact is False
    assert counter.count("three little words") > 0
