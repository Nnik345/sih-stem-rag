"""Discovery of the Core Knowledge corpus and derivation of document metadata.

The raw corpus is read-only. Metadata comes from
`core_knowledge_stem/manifest.json` when a PDF is listed there, and is otherwise
derived from the directory layout::

    core_knowledge_stem/grade_02/science/unit_01_properties_of_matter/student/student_book.pdf
                        ^grade    ^subject ^unit                      ^audience ^filename
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .schemas import (
    DocumentMetadata,
    grade_id_for,
    slugify,
    subject_id_for,
    unit_id_for,
)

LOGGER = get_logger(__name__)

_GRADE_DIR_RE = re.compile(r"^grade_(\d+)$")
_UNIT_PREFIX_RE = re.compile(r"^unit_(\d+)_(.*)$")

KNOWN_AUDIENCES = ("student", "teacher", "other")


class CorpusError(RuntimeError):
    """Raised when the corpus is missing or unusable."""


@dataclass(frozen=True)
class CorpusStats:
    pdf_count: int
    unit_count: int
    manifest_records: int
    manifest_matched: int
    unmatched_pdfs: int


def _readable_resource_type(resource_type: str) -> str:
    return resource_type.replace("_", " ").strip().title()


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Index manifest records by their corpus-relative PDF path.

    A missing or unreadable manifest is not fatal: metadata then comes purely
    from the directory layout, which is logged as a warning.
    """
    if not manifest_path.is_file():
        LOGGER.warning(
            "No manifest at %s; metadata will be derived from the directory layout only",
            manifest_path,
        )
        return {}

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            "Could not read manifest %s (%s); falling back to directory layout",
            manifest_path,
            exc,
        )
        return {}

    records = raw if isinstance(raw, list) else raw.get("resources", [])
    if not isinstance(records, list):
        LOGGER.warning("Manifest %s has an unexpected shape; ignoring it", manifest_path)
        return {}

    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        local_path = record.get("local_path")
        if not isinstance(local_path, str) or not local_path:
            LOGGER.debug("Skipping manifest record without local_path: %r", record)
            continue
        indexed[local_path.replace("\\", "/").lstrip("./")] = record

    LOGGER.info("Loaded %d manifest records from %s", len(indexed), manifest_path)
    return indexed


def _parse_layout(relative_path: str) -> dict[str, Any] | None:
    """Extract grade/subject/unit/audience from the corpus-relative path."""
    parts = relative_path.split("/")
    if len(parts) < 4:
        return None

    grade_match = _GRADE_DIR_RE.match(parts[0])
    if not grade_match:
        return None

    subject = slugify(parts[1])
    unit_dir = parts[2]
    unit_match = _UNIT_PREFIX_RE.match(unit_dir)
    unit_number = int(unit_match.group(1)) if unit_match else None

    audience = parts[3] if len(parts) >= 5 and parts[3] in KNOWN_AUDIENCES else "other"

    return {
        "grade": int(grade_match.group(1)),
        "subject": subject,
        "unit_slug": unit_dir,
        "unit_number": unit_number,
        "audience": audience,
        "filename": parts[-1],
    }


def _unit_title_from_slug(unit_slug: str) -> str:
    match = _UNIT_PREFIX_RE.match(unit_slug)
    body = match.group(2) if match else unit_slug
    return body.replace("_", " ").strip().title()


def build_document_metadata(
    pdf_path: Path,
    corpus_root: Path,
    manifest: dict[str, dict[str, Any]],
) -> DocumentMetadata:
    """Build metadata for one PDF, preferring manifest values over derived ones."""
    relative_path = pdf_path.relative_to(corpus_root).as_posix()
    layout = _parse_layout(relative_path)
    if layout is None:
        raise CorpusError(
            f"PDF path does not match the expected corpus layout "
            f"grade_XX/<subject>/unit_XX_.../<audience>/<file>.pdf: {relative_path}"
        )

    record = manifest.get(relative_path, {})
    from_manifest = bool(record)

    grade = record.get("grade") or layout["grade"]
    try:
        grade = int(grade)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Malformed grade %r in manifest for %s; using directory value %s",
            record.get("grade"),
            relative_path,
            layout["grade"],
        )
        grade = layout["grade"]

    subject = slugify(str(record.get("subject") or layout["subject"]))
    unit_slug = layout["unit_slug"]
    unit_title = str(record.get("unit_title") or "").strip() or _unit_title_from_slug(
        unit_slug
    )
    unit_number = record.get("unit_number")
    if unit_number is None:
        unit_number = layout["unit_number"]
    resource_type = str(
        record.get("resource_type") or Path(layout["filename"]).stem
    ).strip()

    unit_id = unit_id_for(grade, subject, unit_slug)
    document_id = f"{unit_id}:{layout['audience']}:{Path(layout['filename']).stem}"
    document_title = f"{unit_title} - {_readable_resource_type(resource_type)}"

    return DocumentMetadata(
        document_id=document_id,
        document_title=document_title,
        local_pdf_path=pdf_path,
        relative_pdf_path=relative_path,
        filename=layout["filename"],
        grade=grade,
        subject=subject,
        grade_id=grade_id_for(grade),
        subject_id=subject_id_for(grade, subject),
        unit_id=unit_id,
        unit_slug=unit_slug,
        unit_title=unit_title,
        unit_number=unit_number,
        resource_type=resource_type,
        audience=layout["audience"],
        unit_page_url=record.get("unit_page"),
        pdf_url=record.get("pdf_url"),
        from_manifest=from_manifest,
    )


def discover_documents(
    corpus_root: Path,
    *,
    grades: tuple[int, ...] | None = None,
    subjects: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> tuple[list[DocumentMetadata], CorpusStats]:
    """Find every PDF under ``corpus_root`` and attach its metadata.

    Results are sorted by (grade, subject, unit, audience, filename) so ingestion
    order is deterministic and resumable.
    """
    if not corpus_root.is_dir():
        raise CorpusError(
            f"Corpus directory not found: {corpus_root}. Download the curriculum "
            f"first (python scripts/download_core_knowledge_stem.py) or set "
            f"CORPUS_PATH."
        )

    manifest = load_manifest(corpus_root / "manifest.json")
    pdf_paths = sorted(p for p in corpus_root.rglob("*.pdf") if p.is_file())
    if not pdf_paths:
        raise CorpusError(f"No PDF files found under {corpus_root}")

    documents: list[DocumentMetadata] = []
    unmatched = 0
    matched = 0
    for pdf_path in pdf_paths:
        try:
            metadata = build_document_metadata(pdf_path, corpus_root, manifest)
        except CorpusError as exc:
            LOGGER.warning("Skipping %s: %s", pdf_path, exc)
            unmatched += 1
            continue

        if grades is not None and metadata.grade not in grades:
            continue
        if subjects is not None and metadata.subject not in subjects:
            continue

        matched += int(metadata.from_manifest)
        documents.append(metadata)

    documents.sort(
        key=lambda d: (d.grade, d.subject, d.unit_slug, d.audience, d.filename)
    )
    if limit is not None:
        documents = documents[:limit]

    stats = CorpusStats(
        pdf_count=len(documents),
        unit_count=len({d.unit_id for d in documents}),
        manifest_records=len(manifest),
        manifest_matched=matched,
        unmatched_pdfs=unmatched,
    )
    LOGGER.info(
        "Discovered %d PDFs across %d units (%d matched to manifest records, "
        "%d paths skipped)",
        stats.pdf_count,
        stats.unit_count,
        stats.manifest_matched,
        stats.unmatched_pdfs,
    )
    return documents, stats
