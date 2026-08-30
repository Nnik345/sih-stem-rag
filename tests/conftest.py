"""Shared pytest fixtures.

Puts ``src/`` on the path so the tests import the same ``rag`` package the
scripts use, and builds small synthetic parsed documents so chunking and
metadata can be tested without touching PDFs, models or Neo4j.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.chunker import TokenCounter  # noqa: E402
from rag.config import ChunkingConfig  # noqa: E402
from rag.schemas import (  # noqa: E402
    DocumentMetadata,
    Paragraph,
    ParsedDocument,
    ParsedPage,
    ParsedSection,
    page_id_for,
    section_id_for,
)


class WordTokenCounter(TokenCounter):
    """Deterministic counter: one token per whitespace word.

    Chunking assertions need exact token budgets, which a real subword
    tokenizer would make brittle and which would also require the model on disk.
    """

    def __init__(self) -> None:
        super().__init__(tokenizer_path=None)
        self._load_attempted = True

    def count(self, text: str) -> int:
        return max(1, len(text.split()))


@pytest.fixture
def counter() -> WordTokenCounter:
    return WordTokenCounter()


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Small budgets so the fixtures stay readable."""
    return ChunkingConfig(
        target_tokens=100,
        overlap_tokens=20,
        min_tokens=30,
        max_tokens=140,
    )


@pytest.fixture
def metadata(tmp_path: Path) -> DocumentMetadata:
    pdf_path = tmp_path / "student_book.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 test")
    return DocumentMetadata(
        document_id="grade_03:science:unit_01_matter:student:student_book",
        document_title="Matter - Student Book",
        local_pdf_path=pdf_path,
        relative_pdf_path="raw/ncert/science/grade_03/our_wondrous_world/student/ceev101.pdf",
        filename="student_book.pdf",
        grade=3,
        subject="science",
        grade_id="grade_03",
        subject_id="grade_03:science",
        unit_id="grade_03:science:unit_01_matter",
        unit_slug="unit_01_matter",
        unit_title="Matter",
        unit_number=1,
        resource_type="student_book",
        audience="student",
        source_id="ncert_textbook",
        publisher="National Council of Educational Research and Training",
        source_role="primary",
        licence="See source notices",
        content_partition="student_evidence",
        extract_images=False,
    )


def words(word: str, count: int, page: int) -> Paragraph:
    """A paragraph of ``count`` identical words, so token counts are exact."""
    return Paragraph(text=" ".join([word] * count), page_number=page)


def make_section(
    document_id: str,
    index: int,
    title: str,
    paragraphs: list[Paragraph],
) -> ParsedSection:
    pages = sorted({p.page_number for p in paragraphs})
    return ParsedSection(
        section_id=section_id_for(document_id, index),
        title=title,
        section_index=index,
        page_start=pages[0],
        page_end=pages[-1],
        text="\n\n".join(p.text for p in paragraphs),
        paragraphs=paragraphs,
        page_numbers=pages,
    )


def make_page(document_id: str, page_number: int, text: str) -> ParsedPage:
    return ParsedPage(
        page_id=page_id_for(document_id, page_number),
        page_number=page_number,
        page_index=page_number - 1,
        text=text,
        char_count=len(text),
        width=612.0,
        height=792.0,
        has_extractable_text=bool(text.strip()),
    )


@pytest.fixture
def parsed_document(metadata: DocumentMetadata) -> ParsedDocument:
    """Two sections: one short and coherent, one long enough to be split."""
    document_id = metadata.document_id
    short = make_section(
        document_id,
        0,
        "What Is Matter?",
        [words("solid", 40, 1), words("liquid", 30, 1)],
    )
    long = make_section(
        document_id,
        1,
        "States of Matter",
        [
            words("gas", 60, 2),
            words("melting", 60, 2),
            words("freezing", 60, 3),
            words("boiling", 60, 3),
        ],
    )
    return ParsedDocument(
        metadata=metadata,
        content_hash="deadbeefcafe",
        page_count=3,
        pages=[
            make_page(document_id, 1, "solid liquid"),
            make_page(document_id, 2, "gas melting"),
            make_page(document_id, 3, "freezing boiling"),
        ],
        sections=[short, long],
    )
