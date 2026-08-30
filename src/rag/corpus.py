"""Discovery of the CBSE/NCERT STEM corpus from the committed source catalog.

Raw files live under ``curriculum/raw/ncert/`` and are never modified. Metadata
comes from ``rag.curriculum_catalog`` plus ``curriculum/manifests/sources.yaml``
hashes written by ``scripts/download_curriculum.py``.

Each catalog row is one textbook. Ingest creates **one Document per chapter PDF**
under that book's ``student/`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .curriculum_catalog import (
    ALIGNMENT_STATUS_NATIVE,
    SourceFile,
    chapter_number_from_filename,
    ingestible_files,
    is_answers_member,
    is_chapter_pdf,
)
from .logging_utils import get_logger
from .partitions import EVALUATION_ONLY, partition_from_filename
from .schemas import DocumentMetadata, grade_id_for, subject_id_for, unit_id_for

LOGGER = get_logger(__name__)


class CorpusError(RuntimeError):
    """Raised when the corpus is missing or unusable."""


@dataclass(frozen=True)
class CorpusStats:
    pdf_count: int
    unit_count: int
    manifest_records: int
    manifest_matched: int
    unmatched_pdfs: int


def _chapter_documents(record: SourceFile, corpus_root: Path) -> list[DocumentMetadata]:
    chapter_dir = corpus_root / record.local_path
    if not chapter_dir.is_dir():
        LOGGER.warning("Catalog book directory missing on disk: %s", record.local_path)
        return []

    documents: list[DocumentMetadata] = []
    unit_id = unit_id_for(record.grade, record.subject, record.unit_slug)
    for path in sorted(chapter_dir.glob("*.pdf")):
        if is_answers_member(path.name):
            continue
        if not is_chapter_pdf(path.name, record.ncert_code):
            continue
        relative = path.relative_to(corpus_root).as_posix()
        stem = path.stem
        document_id = f"{unit_id}:{record.audience}:{stem}"
        chapter_no = chapter_number_from_filename(path.name)
        if chapter_no is not None:
            title = f"{record.unit_title} - Chapter {chapter_no}"
        else:
            title = f"{record.unit_title} - {stem}"
        partition = partition_from_filename(relative)
        if record.allowed_partitions and partition not in record.allowed_partitions:
            if partition != EVALUATION_ONLY:
                partition = record.allowed_partitions[0]
        documents.append(
            DocumentMetadata(
                document_id=document_id,
                document_title=title,
                local_pdf_path=path,
                relative_pdf_path=relative,
                filename=path.name,
                grade=record.grade,
                subject=record.subject,
                grade_id=grade_id_for(record.grade),
                subject_id=subject_id_for(record.grade, record.subject),
                unit_id=unit_id,
                unit_slug=record.unit_slug,
                unit_title=record.unit_title,
                unit_number=record.unit_number,
                resource_type=record.resource_type,
                audience=record.audience,
                unit_page_url=record.official_page_url,
                pdf_url=record.direct_download_url,
                from_manifest=True,
                source_id=record.source_id,
                publisher=record.publisher,
                source_role=record.source_role,
                licence=record.licence,
                licence_url=record.licence_url,
                source_url=record.official_page_url,
                content_partition=partition,
                cisce_outcome_ids=(),
                alignment_status=ALIGNMENT_STATUS_NATIVE,
                file_format="pdf",
                extract_images=record.extract_images,
                file_id=f"{record.file_id}:{stem}",
            )
        )
    return documents


def discover_documents(
    corpus_root: Path,
    *,
    grades: tuple[int, ...] | None = None,
    subjects: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[list[DocumentMetadata], CorpusStats]:
    """Return ingestible chapter PDFs listed under catalog book directories."""
    if not corpus_root.is_dir():
        raise CorpusError(
            f"Corpus directory not found: {corpus_root}. Download the curriculum "
            f"first (python scripts/download_curriculum.py) or set CORPUS_PATH."
        )

    documents: list[DocumentMetadata] = []
    unmatched = 0
    catalog = ingestible_files()
    for record in catalog:
        if grades is not None and record.grade not in grades:
            continue
        if subjects is not None and record.subject not in subjects:
            continue
        chapters = _chapter_documents(record, corpus_root)
        if not chapters:
            unmatched += 1
            continue
        documents.extend(chapters)

    documents.sort(
        key=lambda d: (d.grade, d.subject, d.source_role, d.unit_slug, d.filename)
    )
    if limit is not None:
        documents = documents[:limit]

    if not documents:
        raise CorpusError(
            f"No ingestible curriculum files found under {corpus_root}. "
            "Run: python scripts/download_curriculum.py"
        )

    stats = CorpusStats(
        pdf_count=len(documents),
        unit_count=len({d.unit_id for d in documents}),
        manifest_records=len(catalog),
        manifest_matched=len(documents),
        unmatched_pdfs=unmatched,
    )
    LOGGER.info(
        "Discovered %d chapter PDFs across %d books (%d catalog records, "
        "%d books missing on disk)",
        stats.pdf_count,
        stats.unit_count,
        stats.manifest_records,
        stats.unmatched_pdfs,
    )
    return documents, stats
