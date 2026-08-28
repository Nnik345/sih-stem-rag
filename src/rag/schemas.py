"""Dataclasses shared by parsing, ingestion, retrieval and generation.

Identifier scheme
-----------------
All identifiers are deterministic and derived from the corpus layout, so
re-running ingestion upserts the same nodes instead of duplicating them::

    grade_id    grade_01
    subject_id  grade_01:science
    unit_id     grade_01:science:unit_02_plant_and_animal_survival
    document_id <unit_id>:student:student_book
    page_id     <document_id>:p0007
    section_id  <document_id>:s0003
    chunk_id    <section_id>:c0001
    image_id    <page_id>:img01
    concept_id  concept:plant_life_cycle

The document content hash is stored alongside the document so a changed PDF can
be detected and re-ingested without wiping the database.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterator
from typing import Any

# --------------------------------------------------------------------------- #
# Identifier helpers
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Lowercase ASCII slug with single underscores; stable across runs."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return _SLUG_RE.sub("_", ascii_only.lower()).strip("_")


def grade_id_for(grade: int) -> str:
    return f"grade_{grade:02d}"


def subject_id_for(grade: int, subject: str) -> str:
    return f"{grade_id_for(grade)}:{slugify(subject)}"


def unit_id_for(grade: int, subject: str, unit_slug: str) -> str:
    return f"{subject_id_for(grade, subject)}:{unit_slug}"


def page_id_for(document_id: str, page_number: int) -> str:
    return f"{document_id}:p{page_number:04d}"


def section_id_for(document_id: str, section_index: int) -> str:
    return f"{document_id}:s{section_index:04d}"


def chunk_id_for(section_id: str, chunk_index: int) -> str:
    return f"{section_id}:c{chunk_index:04d}"


def image_id_for(page_id: str, image_index: int) -> str:
    return f"{page_id}:img{image_index:02d}"


def concept_id_for(normalized_name: str) -> str:
    return f"concept:{slugify(normalized_name)}"


# --------------------------------------------------------------------------- #
# Corpus metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DocumentMetadata:
    """Everything known about a source PDF before it is parsed.

    Fields come from `core_knowledge_stem/manifest.json` where available and are
    otherwise derived from the on-disk directory layout.
    """

    document_id: str
    document_title: str
    local_pdf_path: Path
    relative_pdf_path: str
    filename: str

    grade: int
    subject: str
    grade_id: str
    subject_id: str

    unit_id: str
    unit_slug: str
    unit_title: str
    unit_number: int | None

    # e.g. "student_book", "teacher_guide", "online_resources" (from manifest).
    resource_type: str
    # "student" | "teacher" | "other" (from the directory layout).
    audience: str

    # Provenance from the manifest; never fabricated.
    unit_page_url: str | None = None
    pdf_url: str | None = None
    from_manifest: bool = False

    def as_chunk_properties(self) -> dict[str, Any]:
        """Metadata copied onto every Chunk node for first-class filtering."""
        return {
            "grade": self.grade,
            "subject": self.subject,
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "resource_type": self.resource_type,
            "audience": self.audience,
            "local_pdf_path": str(self.local_pdf_path),
        }


# --------------------------------------------------------------------------- #
# Parsed PDF structures
# --------------------------------------------------------------------------- #


@dataclass
class ParsedImage:
    image_id: str
    local_path: Path
    source_pdf: Path
    page_number: int
    page_id: str
    grade: int
    subject: str
    unit_id: str
    document_id: str
    width: int
    height: int
    image_format: str
    # PDF xref of the embedded image, useful for tracing back into the PDF.
    xref: int | None = None

    def to_properties(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "local_path": str(self.local_path),
            "source_pdf": str(self.source_pdf),
            "page_number": self.page_number,
            "grade": self.grade,
            "subject": self.subject,
            "unit_id": self.unit_id,
            "document_id": self.document_id,
            "width": self.width,
            "height": self.height,
            "format": self.image_format,
            "xref": self.xref,
        }


@dataclass
class TextBlock:
    """One text line from the PDF, with the typography that produced it.

    ``block_index`` is the index of the enclosing PyMuPDF block, which is what
    lets consecutive lines be reassembled into paragraphs.
    """

    text: str
    page_number: int
    order: int
    block_index: int
    max_font_size: float
    is_heading: bool = False
    is_bold: bool = False


@dataclass(frozen=True)
class Paragraph:
    """A paragraph of body text together with the page it was found on."""

    text: str
    page_number: int


@dataclass
class ParsedPage:
    page_id: str
    page_number: int  # 1-based, matches the PDF page label position
    page_index: int  # 0-based order within the document
    text: str
    char_count: int
    width: float
    height: float
    has_extractable_text: bool
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ParsedImage] = field(default_factory=list)
    # True when the page has no usable text layer (likely a scan or full-page art).
    image_only: bool = False

    def to_properties(self, document: DocumentMetadata) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "char_count": self.char_count,
            "width": self.width,
            "height": self.height,
            "has_extractable_text": self.has_extractable_text,
            "image_only": self.image_only,
            "document_id": document.document_id,
            "grade": document.grade,
            "subject": document.subject,
            "unit_id": document.unit_id,
            # Retained so the page can be rendered and handed to Qwen3-VL later.
            "source_pdf": str(document.local_pdf_path),
        }


@dataclass
class ParsedSection:
    section_id: str
    title: str
    section_index: int
    page_start: int
    page_end: int
    text: str
    # Body paragraphs in reading order, each tagged with its page number.
    paragraphs: list[Paragraph] = field(default_factory=list)
    # Page numbers this section spans, in order.
    page_numbers: list[int] = field(default_factory=list)
    # True when the title was discovered from PDF structure rather than invented.
    title_from_document: bool = True

    def to_properties(self, document: DocumentMetadata) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "section_index": self.section_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_count": len(self.text),
            "title_from_document": self.title_from_document,
            "document_id": document.document_id,
            "grade": document.grade,
            "subject": document.subject,
            "unit_id": document.unit_id,
        }


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    content_hash: str
    page_count: int
    pages: list[ParsedPage] = field(default_factory=list)
    sections: list[ParsedSection] = field(default_factory=list)
    pdf_title: str | None = None
    pdf_toc: list[tuple[int, str, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def images(self) -> list[ParsedImage]:
        return [image for page in self.pages for image in page.images]

    @property
    def pages_without_text(self) -> int:
        return sum(1 for page in self.pages if not page.has_extractable_text)


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """A retrievable unit of curriculum text with full hierarchical lineage."""

    chunk_id: str
    text: str
    token_count: int
    chunk_index: int  # position within the document

    # Parent lineage: Chunk -> Section -> Page -> Document -> Unit -> Subject -> Grade
    section_id: str
    section_title: str
    section_index: int
    page_start: int
    page_end: int
    page_ids: list[str]

    document_id: str
    document_title: str
    unit_id: str
    unit_title: str
    subject_id: str
    grade_id: str

    grade: int
    subject: str
    resource_type: str
    audience: str
    local_pdf_path: str

    concepts: list[str] = field(default_factory=list)

    def to_properties(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
            "chunk_index": self.chunk_index,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "section_index": self.section_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "subject_id": self.subject_id,
            "grade_id": self.grade_id,
            "grade": self.grade,
            "subject": self.subject,
            "resource_type": self.resource_type,
            "audience": self.audience,
            "local_pdf_path": self.local_pdf_path,
        }


@dataclass(frozen=True)
class ConceptMention:
    """Evidence-backed link from a chunk to a concept.

    ``evidence`` records why the link exists (exact phrase match, section title,
    unit title). Nothing here comes from an LLM.
    """

    concept_id: str
    name: str
    normalized_name: str
    source: str  # "unit_title" | "section_title" | "curriculum_term"
    occurrences: int
    evidence: str


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

# Retrieval channel names, used as dict keys and in result metadata.
CHANNEL_DENSE = "dense"
CHANNEL_FULLTEXT = "fulltext"
CHANNEL_GRAPH = "graph"
ALL_CHANNELS = (CHANNEL_DENSE, CHANNEL_FULLTEXT, CHANNEL_GRAPH)


@dataclass(frozen=True)
class RetrievalFilter:
    """Metadata scope for a query.

    ``None`` means "do not restrict". The pipeline never infers grade or subject
    from the question text; the caller supplies whatever it already knows.
    """

    grade: int | None = None
    subject: str | None = None
    unit_id: str | None = None
    unit_title_contains: str | None = None
    resource_type: str | None = None
    audience: str | None = None
    document_id: str | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "grade",
                "subject",
                "unit_id",
                "unit_title_contains",
                "resource_type",
                "audience",
                "document_id",
            )
        )

    def describe(self) -> str:
        active = {
            name: getattr(self, name)
            for name in (
                "grade",
                "subject",
                "unit_id",
                "unit_title_contains",
                "resource_type",
                "audience",
                "document_id",
            )
            if getattr(self, name) is not None
        }
        return "corpus-wide (no filter)" if not active else str(active)


@dataclass
class RetrievedChunk:
    """A candidate chunk plus every intermediate retrieval signal.

    Nothing is discarded between stages: per-channel ranks and scores, the RRF
    score and the reranker score all survive to the end so retrieval
    architectures can be compared experimentally.
    """

    chunk_id: str
    text: str

    grade: int | None = None
    subject: str | None = None
    unit_id: str | None = None
    unit_title: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    resource_type: str | None = None
    audience: str | None = None
    local_pdf_path: str | None = None

    # Which channels produced this candidate, in discovery order.
    retrieval_sources: list[str] = field(default_factory=list)

    dense_rank: int | None = None
    dense_score: float | None = None
    fulltext_rank: int | None = None
    fulltext_score: float | None = None
    graph_rank: int | None = None
    graph_score: float | None = None
    # How graph expansion reached this chunk, e.g. "SAME_SECTION via <chunk_id>".
    graph_expansion_path: str | None = None
    graph_seed_chunk_id: str | None = None

    rrf_score: float | None = None
    rrf_rank: int | None = None
    # Per-channel RRF contribution, so a fused score can be attributed back to
    # the channels that produced it.
    rrf_contributions: dict[str, float] = field(default_factory=dict)
    rerank_score: float | None = None
    rerank_rank: int | None = None

    concepts: list[str] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def retrieval_source(self) -> str:
        """Comma-joined channel list, for compact display."""
        return "+".join(self.retrieval_sources) if self.retrieval_sources else "unknown"

    @property
    def retrieval_score(self) -> float:
        """Best available score: reranker, else RRF, else the channel score."""
        for value in (self.rerank_score, self.rrf_score):
            if value is not None:
                return value
        for value in (self.dense_score, self.fulltext_score, self.graph_score):
            if value is not None:
                return value
        return 0.0

    @property
    def page_range(self) -> str:
        if self.page_start is None:
            return "?"
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    def add_source(self, channel: str) -> None:
        if channel not in self.retrieval_sources:
            self.retrieval_sources.append(channel)

    def preview(self, width: int = 160) -> str:
        collapsed = " ".join(self.text.split())
        return collapsed if len(collapsed) <= width else collapsed[: width - 3] + "..."

    def provenance(self) -> dict[str, Any]:
        """Source metadata kept for citation and traceability."""
        return {
            "chunk_id": self.chunk_id,
            "grade": self.grade,
            "subject": self.subject,
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_title": self.section_title,
            "pages": self.page_range,
            "local_pdf_path": self.local_pdf_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "grade": self.grade,
            "subject": self.subject,
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "resource_type": self.resource_type,
            "audience": self.audience,
            "local_pdf_path": self.local_pdf_path,
            "retrieval_sources": list(self.retrieval_sources),
            "retrieval_source": self.retrieval_source,
            "retrieval_score": self.retrieval_score,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "fulltext_rank": self.fulltext_rank,
            "fulltext_score": self.fulltext_score,
            "graph_rank": self.graph_rank,
            "graph_score": self.graph_score,
            "graph_expansion_path": self.graph_expansion_path,
            "graph_seed_chunk_id": self.graph_seed_chunk_id,
            "rrf_score": self.rrf_score,
            "rrf_rank": self.rrf_rank,
            "rrf_contributions": dict(self.rrf_contributions),
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "concepts": list(self.concepts),
            "images": list(self.images),
        }


@dataclass
class RetrievalDiagnostics:
    """Every intermediate stage of one retrieval, preserved for inspection."""

    query: str
    scope: RetrievalFilter
    dense: list[RetrievedChunk] = field(default_factory=list)
    fulltext: list[RetrievedChunk] = field(default_factory=list)
    graph: list[RetrievedChunk] = field(default_factory=list)
    fused: list[RetrievedChunk] = field(default_factory=list)
    reranked: list[RetrievedChunk] = field(default_factory=list)
    graph_seeds: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def channel_counts(self) -> dict[str, int]:
        return {
            "dense": len(self.dense),
            "fulltext": len(self.fulltext),
            "graph": len(self.graph),
            "fused": len(self.fused),
            "reranked": len(self.reranked),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope.describe(),
            "counts": self.channel_counts(),
            "graph_seeds": list(self.graph_seeds),
            "timings_ms": dict(self.timings_ms),
            "notes": list(self.notes),
            "dense": [c.to_dict() for c in self.dense],
            "fulltext": [c.to_dict() for c in self.fulltext],
            "graph": [c.to_dict() for c in self.graph],
            "fused": [c.to_dict() for c in self.fused],
            "reranked": [c.to_dict() for c in self.reranked],
        }


@dataclass
class RetrievalResponse:
    """Final retrieval output: selected evidence plus full diagnostics."""

    query: str
    scope: RetrievalFilter
    results: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[RetrievedChunk]:
        return iter(self.results)


# --------------------------------------------------------------------------- #
# Evidence gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvidenceCheck:
    name: str
    passed: bool
    detail: str
    value: Any = None


@dataclass
class EvidenceDecision:
    """Outcome of the evidence sufficiency gate.

    ``sufficient=False`` means the tutoring layer must NOT ask the generator to
    answer from its own parametric knowledge.
    """

    sufficient: bool
    checks: list[EvidenceCheck] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    kept_chunks: list[RetrievedChunk] = field(default_factory=list)
    # Coarse confidence label derived from the checks; not a calibrated score.
    confidence: str = "unknown"

    @property
    def failed_checks(self) -> list[EvidenceCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "value": c.value,
                }
                for c in self.checks
            ],
            "kept_chunk_ids": [c.chunk_id for c in self.kept_chunks],
        }
