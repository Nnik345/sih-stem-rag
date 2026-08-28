"""Structured PDF parsing with PyMuPDF.

Produces pages, sections and extracted images while preserving every piece of
provenance needed later (source path, page number, page order, section title).

Design notes
------------
* The Core Knowledge PDFs are digitally generated, so the text layer is used
  directly. OCR is opt-in and only ever attempted for a page with no usable text.
* Almost none of these PDFs have a usable table of contents (the few TOC entries
  that exist point at cover artwork), so section boundaries are detected from
  typography. Body font size is measured per document rather than assumed: a
  Grade 1 student reader sets body text at 19 pt while a teacher guide uses
  12 pt, so a fixed threshold would be wrong for one of them.
* Vector diagrams cannot be extracted as raster images. That is expected and
  never fails ingestion -- the Page node keeps the source PDF path and page
  number so the page can be rendered for Qwen3-VL on demand.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .config import IngestConfig
from .logging_utils import get_logger
from .schemas import (
    DocumentMetadata,
    Paragraph,
    ParsedDocument,
    ParsedImage,
    ParsedPage,
    ParsedSection,
    TextBlock,
    image_id_for,
    page_id_for,
    section_id_for,
)

LOGGER = get_logger(__name__)


class PdfParseError(RuntimeError):
    """Raised when a PDF cannot be opened or contains no usable content."""


# --- Heading detection thresholds ------------------------------------------ #
# A heading must be visually distinct *and* short. These are typography
# heuristics, not semantics: nothing is inferred by a language model.
# Bold alone is deliberately not sufficient: these documents use bold title-case
# lead-ins inside body paragraphs, which produced ~3x too many sections.
_HEADING_SIZE_RATIO = 1.12
_HEADING_MAX_CHARS = 95
_HEADING_MAX_WORDS = 14
_BOLD_FONT_MARKERS = ("bold", "black", "heavy", "semibold", "-bd")
# Headings do not read as sentences; a trailing full stop signals body text.
_SENTENCE_END = (".", "!", "?", ",", ";")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?[ivxlcdm\d]{1,6}\s*$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")

# --- Junk-text filters ------------------------------------------------------ #
# Table-of-contents dot leaders ("Introduction . . . . . . 11") and decorative
# rules carry no retrievable meaning, so punctuation-dominated text is dropped.
_MIN_ALNUM_RATIO = 0.55
_MIN_PARAGRAPH_CHARS = 3
# A run of dot leaders means the line is a contents entry, e.g.
# "Chapter 4 Saving Earth . . . . . . . 32". The replacement character appears
# because some of these PDFs use a non-standard glyph for the leader dot.
_DOT_LEADER_RE = re.compile(r"(?:[.\u00b7\u2022\u2026\ufffd]\s*){4,}")
# Sections that exist purely for navigation.
_NAVIGATION_TITLES = frozenset(
    {"table of contents", "table of content", "contents", "index"}
)

# Images that repeat across most pages are page furniture (logos, rules,
# decorative borders) rather than curriculum diagrams.
_TEMPLATE_IMAGE_PAGE_RATIO = 0.4
_TEMPLATE_IMAGE_MIN_PAGES = 6

_IMAGE_EXTENSIONS = {"png", "jpeg", "jpg", "jpx", "gif", "bmp", "tiff", "webp"}


def compute_content_hash(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """SHA-256 of the file, used to detect a changed PDF between ingest runs."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_line(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


@dataclass
class _Line:
    text: str
    max_size: float
    is_bold: bool
    page_number: int
    order: int
    block_index: int


def _extract_lines(page: pymupdf.Page, page_number: int) -> list[_Line]:
    """Text lines in reading order, carrying the typography we need."""
    lines: list[_Line] = []
    try:
        page_dict = page.get_text("dict", sort=True)
    except Exception as exc:  # a single malformed page must not kill the document
        LOGGER.warning("Could not read text layout on page %d: %s", page_number, exc)
        return lines

    order = 0
    for block_index, block in enumerate(page_dict.get("blocks", [])):
        if block.get("type") != 0:  # 0 == text block
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = _clean_line("".join(span.get("text", "") for span in spans))
            if not text:
                continue
            max_size = max(float(span.get("size", 0.0)) for span in spans)
            fonts = " ".join(str(span.get("font", "")).lower() for span in spans)
            is_bold = any(marker in fonts for marker in _BOLD_FONT_MARKERS)
            lines.append(
                _Line(
                    text=text,
                    max_size=max_size,
                    is_bold=is_bold,
                    page_number=page_number,
                    order=order,
                    block_index=block_index,
                )
            )
            order += 1
    return lines


def _body_font_size(all_lines: list[_Line]) -> float:
    """Most common font size weighted by characters -- the document's body text."""
    weighted: Counter[float] = Counter()
    for line in all_lines:
        weighted[round(line.max_size, 1)] += len(line.text)
    if not weighted:
        return 0.0
    return weighted.most_common(1)[0][0]


def _looks_like_heading(line: _Line, body_size: float) -> bool:
    text = line.text
    if len(text) < 3 or len(text) > _HEADING_MAX_CHARS:
        return False
    if len(text.split()) > _HEADING_MAX_WORDS:
        return False
    if _PAGE_NUMBER_RE.match(text):
        return False
    if text.endswith(_SENTENCE_END):
        return False
    # Require at least one letter, so "1.2" or "----" is not a heading.
    if not any(ch.isalpha() for ch in text):
        return False

    larger = body_size > 0 and line.max_size >= body_size * _HEADING_SIZE_RATIO
    letters = [ch for ch in text if ch.isalpha()]
    all_caps = len(letters) >= 2 and all(ch.isupper() for ch in letters)
    return bool(larger or all_caps)


def _is_meaningful_text(text: str) -> bool:
    """Reject dot leaders, rules and other punctuation-dominated fragments."""
    stripped = text.strip()
    if len(stripped) < _MIN_PARAGRAPH_CHARS:
        return False
    if _DOT_LEADER_RE.search(stripped):
        return False
    dense = [ch for ch in stripped if not ch.isspace()]
    if not dense:
        return False
    alnum = sum(1 for ch in dense if ch.isalnum())
    return (alnum / len(dense)) >= _MIN_ALNUM_RATIO


def _collect_page_images(
    doc: pymupdf.Document,
    page_index: int,
) -> list[tuple[int, int, int]]:
    """(xref, width, height) for every embedded raster image on a page."""
    try:
        raw = doc[page_index].get_images(full=True)
    except Exception as exc:
        LOGGER.debug("get_images failed on page %d: %s", page_index + 1, exc)
        return []
    out = []
    for entry in raw:
        xref = int(entry[0])
        width = int(entry[2])
        height = int(entry[3])
        out.append((xref, width, height))
    return out


def _template_xrefs(
    xrefs_per_page: list[list[tuple[int, int, int]]],
    page_count: int,
) -> set[int]:
    """Identify repeated page furniture that should not become Image nodes."""
    pages_per_xref: Counter[int] = Counter()
    for entries in xrefs_per_page:
        for xref in {x for x, _, _ in entries}:
            pages_per_xref[xref] += 1

    template = set()
    for xref, pages in pages_per_xref.items():
        if pages >= _TEMPLATE_IMAGE_MIN_PAGES and pages >= (
            page_count * _TEMPLATE_IMAGE_PAGE_RATIO
        ):
            template.add(xref)
    return template


def _image_output_path(
    images_root: Path,
    metadata: DocumentMetadata,
    page_number: int,
    xref: int,
    extension: str,
) -> Path:
    """Deterministic, traceable location under data/processed/images/."""
    directory = (
        images_root
        / metadata.grade_id
        / metadata.subject
        / metadata.unit_slug
        / metadata.audience
    )
    stem = Path(metadata.filename).stem
    return directory / f"{stem}_p{page_number:04d}_x{xref}.{extension}"


def _save_pixmap_as_png(
    doc: pymupdf.Document, xref: int, destination: Path
) -> bool:
    """Re-render an embedded image through a pixmap and write it as PNG.

    Used for images ``extract_image`` refuses to hand back, most commonly ones
    carrying an alpha channel whose stored encoding is JPEG. PNG keeps the alpha
    channel, so nothing is lost.
    """
    pixmap = pymupdf.Pixmap(doc, xref)
    try:
        # PNG has no CMYK representation, so convert those to RGB first.
        if pixmap.colorspace is not None and pixmap.colorspace.n == 4:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            pixmap.save(destination)
        return True
    finally:
        # Pixmaps hold decoded bitmaps; release them rather than waiting for GC.
        pixmap = None


def _save_embedded_image(
    doc: pymupdf.Document,
    xref: int,
    page_number: int,
    metadata: DocumentMetadata,
    images_root: Path,
    warnings: list[str],
) -> tuple[Path, str] | None:
    """Write one embedded image to disk, returning its path and format."""
    try:
        extracted = doc.extract_image(xref)
    except Exception as extract_error:
        png_path = _image_output_path(images_root, metadata, page_number, xref, "png")
        try:
            if _save_pixmap_as_png(doc, xref, png_path):
                return png_path, "png"
        except Exception as render_error:
            warnings.append(
                f"image xref {xref} on page {page_number}: {extract_error}; "
                f"PNG fallback also failed: {render_error}"
            )
            return None
        warnings.append(f"image xref {xref} on page {page_number}: {extract_error}")
        return None

    payload = extracted.get("image")
    image_format = str(extracted.get("ext", "png")).lower()
    if not payload or image_format not in _IMAGE_EXTENSIONS:
        warnings.append(
            f"image xref {xref} on page {page_number}: "
            f"unsupported format {image_format!r}"
        )
        return None

    local_path = _image_output_path(
        images_root, metadata, page_number, xref, image_format
    )
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.is_file():
            local_path.write_bytes(payload)
    except OSError as exc:
        warnings.append(f"could not write image {local_path}: {exc}")
        return None
    return local_path, image_format


def _extract_images_for_page(
    doc: pymupdf.Document,
    page_index: int,
    page_number: int,
    page_id: str,
    entries: list[tuple[int, int, int]],
    template: set[int],
    metadata: DocumentMetadata,
    images_root: Path,
    config: IngestConfig,
    saved_cache: dict[int, Path],
    warnings: list[str],
) -> list[ParsedImage]:
    images: list[ParsedImage] = []
    seen_on_page: set[int] = set()

    for index, (xref, width, height) in enumerate(entries, start=1):
        if xref in template or xref in seen_on_page:
            continue
        if width * height < config.min_image_pixels:
            continue
        seen_on_page.add(xref)

        cached = saved_cache.get(xref)
        if cached is not None:
            local_path, image_format = cached, cached.suffix.lstrip(".")
        else:
            saved = _save_embedded_image(
                doc, xref, page_number, metadata, images_root, warnings
            )
            if saved is None:
                continue
            local_path, image_format = saved
            saved_cache[xref] = local_path

        images.append(
            ParsedImage(
                image_id=image_id_for(page_id, index),
                local_path=local_path,
                source_pdf=metadata.local_pdf_path,
                page_number=page_number,
                page_id=page_id,
                grade=metadata.grade,
                subject=metadata.subject,
                unit_id=metadata.unit_id,
                document_id=metadata.document_id,
                width=width,
                height=height,
                image_format=image_format,
                xref=xref,
            )
        )
    return images


def _ocr_page_text(page: pymupdf.Page) -> str:
    """Best-effort OCR for a page with no text layer. Requires Tesseract."""
    try:
        textpage = page.get_textpage_ocr(flags=0, full=True)
        return page.get_text("text", textpage=textpage)
    except Exception as exc:
        raise PdfParseError(
            f"OCR requested but unavailable ({exc}). Install Tesseract and set "
            f"TESSDATA_PREFIX, or leave INGEST_ENABLE_OCR=false."
        ) from exc


@dataclass
class _RawSection:
    title: str | None
    from_document: bool
    paragraphs: list[Paragraph]

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.paragraphs)


def _collect_raw_sections(pages: list[ParsedPage]) -> list[_RawSection]:
    """Split the document at heading boundaries, rebuilding paragraphs.

    Lines belonging to the same PyMuPDF block are rejoined into one paragraph, so
    the chunker can respect real paragraph boundaries instead of line wraps.
    """
    raw: list[_RawSection] = []
    current = _RawSection(title=None, from_document=False, paragraphs=[])
    pending_lines: list[str] = []
    pending_key: tuple[int, int] | None = None

    def flush_paragraph() -> None:
        nonlocal pending_lines, pending_key
        if pending_lines and pending_key is not None:
            text = _clean_line(" ".join(pending_lines))
            if _is_meaningful_text(text):
                current.paragraphs.append(Paragraph(text=text, page_number=pending_key[0]))
        pending_lines, pending_key = [], None

    for page in pages:
        for block in page.blocks:
            if block.is_heading:
                flush_paragraph()
                if current.paragraphs or current.title:
                    raw.append(current)
                current = _RawSection(
                    title=block.text, from_document=True, paragraphs=[]
                )
                continue
            key = (page.page_number, block.block_index)
            if key != pending_key:
                flush_paragraph()
                pending_key = key
            pending_lines.append(block.text)

    flush_paragraph()
    if current.paragraphs or current.title:
        raw.append(current)
    return [section for section in raw if section.paragraphs]


def _merge_short_sections(
    raw: list[_RawSection], min_section_chars: int
) -> list[_RawSection]:
    """Fold undersized sections forward so chunks can reach the target size.

    These PDFs use many short sub-headings; taken literally they yield sections
    of a paragraph or two and therefore chunks far below the configured target.
    A too-short section absorbs the following one, and the absorbed heading is
    kept inline as a paragraph so its wording stays searchable.
    """
    if min_section_chars <= 0:
        return raw

    merged: list[_RawSection] = []
    for section in raw:
        if merged and merged[-1].char_count < min_section_chars:
            previous = merged[-1]
            if section.title:
                page = (
                    section.paragraphs[0].page_number
                    if section.paragraphs
                    else previous.paragraphs[-1].page_number
                )
                previous.paragraphs.append(
                    Paragraph(text=section.title, page_number=page)
                )
            previous.paragraphs.extend(section.paragraphs)
            continue
        merged.append(section)
    return merged


def _build_sections(
    metadata: DocumentMetadata,
    pages: list[ParsedPage],
    min_section_chars: int,
) -> list[ParsedSection]:
    """Detect sections, merge undersized ones, then assign stable identifiers."""
    raw = _merge_short_sections(_collect_raw_sections(pages), min_section_chars)

    sections: list[ParsedSection] = []
    for section in raw:
        paragraphs = section.paragraphs
        if not paragraphs:
            continue
        title = section.title or f"{metadata.document_title} (front matter)"
        if title.strip().lower() in _NAVIGATION_TITLES:
            continue
        index = len(sections)
        page_numbers = sorted({p.page_number for p in paragraphs})
        sections.append(
            ParsedSection(
                section_id=section_id_for(metadata.document_id, index),
                title=title,
                section_index=index,
                page_start=page_numbers[0],
                page_end=page_numbers[-1],
                text="\n\n".join(p.text for p in paragraphs),
                paragraphs=paragraphs,
                page_numbers=page_numbers,
                title_from_document=section.from_document,
            )
        )

    if not sections:
        LOGGER.debug("No sections built for %s", metadata.document_id)
    return sections


def parse_document(
    metadata: DocumentMetadata,
    *,
    images_root: Path,
    config: IngestConfig,
) -> ParsedDocument:
    """Parse one PDF into pages, sections and images.

    Raises :class:`PdfParseError` only when the file cannot be opened at all; a
    PDF with no extractable text returns a document with warnings so ingestion
    can record and skip it explicitly rather than failing silently.
    """
    path = metadata.local_pdf_path
    if not path.is_file():
        raise PdfParseError(f"PDF not found: {path}")

    content_hash = compute_content_hash(path)

    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise PdfParseError(f"Could not open PDF {path}: {exc}") from exc

    warnings: list[str] = []
    pages: list[ParsedPage] = []

    try:
        if doc.is_encrypted and not doc.authenticate(""):
            raise PdfParseError(f"PDF is encrypted and cannot be read: {path}")

        page_count = doc.page_count
        if page_count == 0:
            raise PdfParseError(f"PDF has no pages: {path}")

        images_per_page = (
            [_collect_page_images(doc, i) for i in range(page_count)]
            if config.extract_images
            else [[] for _ in range(page_count)]
        )
        template = _template_xrefs(images_per_page, page_count)
        if template:
            LOGGER.debug(
                "%s: ignoring %d repeated template image(s)",
                metadata.document_id,
                len(template),
            )

        all_lines: list[list[_Line]] = []
        for page_index in range(page_count):
            page = doc[page_index]
            all_lines.append(_extract_lines(page, page_index + 1))

        body_size = _body_font_size([ln for lines in all_lines for ln in lines])

        saved_images: dict[int, Path] = {}
        for page_index in range(page_count):
            page = doc[page_index]
            page_number = page_index + 1
            page_id = page_id_for(metadata.document_id, page_number)
            lines = all_lines[page_index]
            page_text = "\n".join(line.text for line in lines).strip()

            has_text = len(page_text) >= config.min_page_chars
            if not has_text and config.enable_ocr:
                ocr_text = _ocr_page_text(page).strip()
                if len(ocr_text) >= config.min_page_chars:
                    page_text = ocr_text
                    has_text = True
                    warnings.append(f"page {page_number}: text recovered via OCR")

            blocks = [
                TextBlock(
                    text=line.text,
                    page_number=page_number,
                    order=line.order,
                    block_index=line.block_index,
                    max_font_size=line.max_size,
                    is_heading=_looks_like_heading(line, body_size),
                    is_bold=line.is_bold,
                )
                for line in lines
            ]

            rect = page.rect
            parsed_page = ParsedPage(
                page_id=page_id,
                page_number=page_number,
                page_index=page_index,
                text=page_text,
                char_count=len(page_text),
                width=float(rect.width),
                height=float(rect.height),
                has_extractable_text=has_text,
                blocks=blocks,
                image_only=not has_text,
            )

            if config.extract_images:
                parsed_page.images = _extract_images_for_page(
                    doc,
                    page_index,
                    page_number,
                    page_id,
                    images_per_page[page_index],
                    template,
                    metadata,
                    images_root,
                    config,
                    saved_images,
                    warnings,
                )

            pages.append(parsed_page)

        sections = _build_sections(metadata, pages, config.min_section_chars)
        pdf_title = (doc.metadata or {}).get("title") or None
        if pdf_title and pdf_title.strip().lower() in {"", "untitled"}:
            pdf_title = None
        try:
            toc = [tuple(entry[:3]) for entry in doc.get_toc()]
        except Exception:
            toc = []

    finally:
        doc.close()

    parsed = ParsedDocument(
        metadata=metadata,
        content_hash=content_hash,
        page_count=len(pages),
        pages=pages,
        sections=sections,
        pdf_title=pdf_title,
        pdf_toc=toc,  # type: ignore[arg-type]
        warnings=warnings,
    )

    if not any(page.has_extractable_text for page in pages):
        parsed.warnings.append(
            "no extractable text on any page; document is image-only "
            "(enable INGEST_ENABLE_OCR to attempt OCR)"
        )

    LOGGER.info(
        "Parsed %s: %d pages (%d without text), %d sections, %d images",
        metadata.relative_pdf_path,
        parsed.page_count,
        parsed.pages_without_text,
        len(parsed.sections),
        len(parsed.images),
    )
    return parsed
