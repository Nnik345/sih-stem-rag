"""Discovery of the Grade 3–5 STEM corpus from the committed source catalog.

Raw files live under ``curriculum/raw/`` and are never modified. Metadata comes
from ``rag.curriculum_catalog`` plus ``curriculum/manifests/sources.yaml``
hashes written by ``scripts/download_curriculum.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .alignment import outcome_ids_for
from .curriculum_catalog import SourceFile, ingestible_files
from .logging_utils import get_logger
from .partitions import partition_from_filename
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


def _metadata_from_record(record: SourceFile, corpus_root: Path) -> DocumentMetadata:
    path = corpus_root / record.local_path
    unit_id = unit_id_for(record.grade, record.subject, record.unit_slug)
    stem = Path(record.local_path).stem
    document_id = f"{unit_id}:{record.audience}:{stem}"
    outcomes, _granularity, status = outcome_ids_for(
        grade=record.grade, subject=record.subject, unit_slug=record.unit_slug
    )
    partition = partition_from_filename(record.local_path)
    if record.allowed_partitions and partition not in record.allowed_partitions:
        partition = record.allowed_partitions[0]
    return DocumentMetadata(
        document_id=document_id,
        document_title=f"{record.unit_title} - {record.resource_type.replace('_', ' ').title()}",
        local_pdf_path=path,
        relative_pdf_path=record.local_path,
        filename=Path(record.local_path).name,
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
        cisce_outcome_ids=tuple(outcomes),
        alignment_status=status,
        file_format=record.file_format,
        extract_images=record.extract_images,
        file_id=record.file_id,
    )


def discover_documents(
    corpus_root: Path,
    *,
    grades: tuple[int, ...] | None = None,
    subjects: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[list[DocumentMetadata], CorpusStats]:
    """Return ingestible documents listed in the catalog that exist on disk."""
    if not corpus_root.is_dir():
        raise CorpusError(
            f"Corpus directory not found: {corpus_root}. Download the curriculum "
            f"first (python scripts/download_curriculum.py) or set CORPUS_PATH."
        )

    documents: list[DocumentMetadata] = []
    unmatched = 0
    catalog = ingestible_files()
    for record in catalog:
        metadata = _metadata_from_record(record, corpus_root)
        if grades is not None and metadata.grade not in grades:
            continue
        if subjects is not None and metadata.subject not in subjects:
            continue
        if not metadata.local_pdf_path.is_file():
            LOGGER.warning("Catalog file missing on disk: %s", metadata.relative_pdf_path)
            unmatched += 1
            continue
        documents.append(metadata)

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
        "Discovered %d curriculum files across %d units (%d catalog records, "
        "%d missing on disk)",
        stats.pdf_count,
        stats.unit_count,
        stats.manifest_records,
        stats.unmatched_pdfs,
    )
    return documents, stats
