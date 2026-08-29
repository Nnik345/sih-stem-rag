"""Parse learner ePUBs into the same page/section structures used for PDFs."""

from __future__ import annotations

import html
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import IngestConfig
from .logging_utils import get_logger
from .pdf_parser import compute_content_hash
from .schemas import (
    DocumentMetadata,
    Paragraph,
    ParsedDocument,
    ParsedImage,
    ParsedPage,
    ParsedSection,
    image_id_for,
    page_id_for,
    section_id_for,
)

LOGGER = get_logger(__name__)

_CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
_DC = "{http://purl.org/dc/elements/1.1/}"
_OPF = "{http://www.idpf.org/2007/opf}"
_IMG_SRC_RE = re.compile(r"<img[^>]+src\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_LICENCE_EXCEPTION_RE = re.compile(
    r"all rights reserved|not licensed|permission required|except where noted",
    re.IGNORECASE,
)
_SKIP_ASSET_NAME_RE = re.compile(
    r"(logo|icon|button|spacer|pixel|badge|favicon|ornament)",
    re.IGNORECASE,
)


class EpubParseError(RuntimeError):
    """Raised when an ePUB cannot be opened or has no readable spine."""


class _XHTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._current_tag = "p"
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "caption", "figcaption"}:
            self._flush()
            self._current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "caption", "figcaption"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._buffer.append(data)

    def _flush(self) -> None:
        text = html.unescape(" ".join("".join(self._buffer).split())).strip()
        self._buffer = []
        if text:
            self.blocks.append((self._current_tag, text))
        self._current_tag = "p"


def _opf_path(archive: zipfile.ZipFile) -> str:
    container = ET.fromstring(archive.read("META-INF/container.xml"))
    rootfile = container.find(".//c:rootfile", _CONTAINER_NS)
    if rootfile is None:
        rootfile = container.find(".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise EpubParseError("ePUB container.xml has no OPF rootfile")
    return rootfile.get("full-path") or ""


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _package_cc_by_covers_assets(rights: str) -> bool:
    """True only when package rights clearly grant CC-BY without an exception."""
    text = (rights or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if _LICENCE_EXCEPTION_RE.search(text):
        return False
    has_cc = "creative commons" in lowered or "cc by" in lowered or "cc-by" in lowered
    if not has_cc:
        return False
    if "noncommercial" in lowered or "non-commercial" in lowered or "cc by-nc" in lowered:
        return False
    return True


def _extract_epub_images_for_page(
    archive: zipfile.ZipFile,
    xhtml: str,
    xhtml_member: str,
    *,
    metadata: DocumentMetadata,
    page_id: str,
    page_number: int,
    images_root: Path,
    creator: str,
    rights: str,
    licence_url: str,
    warnings: list[str],
) -> tuple[list[ParsedImage], int]:
    """Extract CC-BY package images referenced by this spine item."""
    extracted: list[ParsedImage] = []
    skipped = 0
    xhtml_dir = posixpath.dirname(xhtml_member)
    seen: set[str] = set()
    for index, src in enumerate(_IMG_SRC_RE.findall(xhtml), start=1):
        href = posixpath.normpath(posixpath.join(xhtml_dir, src.split("#", 1)[0]))
        if href in seen:
            continue
        seen.add(href)
        name = posixpath.basename(href).lower()
        if _SKIP_ASSET_NAME_RE.search(name):
            skipped += 1
            warnings.append(f"skipped decorative asset {href}")
            continue
        if _LICENCE_EXCEPTION_RE.search(xhtml):
            skipped += 1
            warnings.append(f"skipped {href}: nearby rights exception in chapter")
            continue
        try:
            payload = archive.read(href)
        except KeyError:
            skipped += 1
            warnings.append(f"missing image member {href}")
            continue
        if len(payload) < 64:
            skipped += 1
            continue
        ext = Path(name).suffix.lstrip(".") or "png"
        relative = (
            Path(str(metadata.grade))
            / metadata.subject
            / metadata.unit_slug
            / metadata.audience
            / f"{Path(metadata.filename).stem}_p{page_number:04d}_{index}.{ext}"
        )
        destination = images_root / relative
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                destination.write_bytes(payload)
        except OSError as exc:
            skipped += 1
            warnings.append(f"could not write {destination}: {exc}")
            continue
        attribution = (
            f"{creator + ', ' if creator else ''}licensed under {rights.strip() or 'CC BY'}"
        )
        extracted.append(
            ParsedImage(
                image_id=image_id_for(page_id, index),
                local_path=destination,
                source_pdf=metadata.local_pdf_path,
                page_number=page_number,
                page_id=page_id,
                grade=metadata.grade,
                subject=metadata.subject,
                unit_id=metadata.unit_id,
                document_id=metadata.document_id,
                width=0,
                height=0,
                image_format=ext,
                creator=creator,
                licence=rights.strip() or metadata.licence,
                licence_url=licence_url or metadata.licence_url,
                attribution=attribution,
                source_href=href,
            )
        )
    return extracted, skipped


def parse_epub_document(
    metadata: DocumentMetadata,
    *,
    images_root: Path,
    config: IngestConfig,
) -> ParsedDocument:
    """Turn one learner ePUB into pages and heading-delimited sections."""
    path = metadata.local_pdf_path
    if not zipfile.is_zipfile(path):
        raise EpubParseError(f"Not an ePUB/zip: {path}")

    warnings: list[str] = []
    content_hash = compute_content_hash(path)
    pages: list[ParsedPage] = []
    sections: list[ParsedSection] = []
    extracted_images = 0
    skipped_images = 0

    with zipfile.ZipFile(path) as archive:
        opf_name = _opf_path(archive)
        opf_dir = posixpath.dirname(opf_name)
        opf = ET.fromstring(archive.read(opf_name))
        rights = _text(opf.find(f".//{_DC}rights")) or _text(
            opf.find(f".//{_OPF}meta[@name='rights']")
        )
        creator = _text(opf.find(f".//{_DC}creator"))
        if rights:
            warnings.append(f"package rights: {rights[:240]}")
        cover_assets = _package_cc_by_covers_assets(rights)
        want_images = bool(
            config.extract_images and metadata.extract_images and cover_assets
        )
        if metadata.extract_images and not cover_assets:
            warnings.append(
                "ePUB images skipped: package rights do not clearly cover assets "
                "under CC-BY without exception"
            )

        manifest = {
            item.get("id"): item
            for item in opf.findall(f".//{_OPF}item")
            if item.get("id")
        }
        spine = [
            item.get("idref")
            for item in opf.findall(f".//{_OPF}itemref")
            if item.get("idref")
        ]
        if not spine:
            raise EpubParseError(f"{path} has an empty spine")

        section_index = 0
        for spine_index, idref in enumerate(spine, start=1):
            item = manifest.get(idref or "")
            if item is None:
                continue
            href = item.get("href") or ""
            media = (item.get("media-type") or "").lower()
            if "html" not in media and not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            member = posixpath.normpath(posixpath.join(opf_dir, href))
            try:
                raw = archive.read(member)
            except KeyError:
                warnings.append(f"missing spine item {member}")
                continue
            decoded = raw.decode("utf-8", "ignore")
            parser = _XHTMLText()
            try:
                parser.feed(decoded)
                parser.close()
            except Exception as exc:
                warnings.append(f"could not parse {member}: {exc}")
                continue
            if not parser.blocks:
                continue

            page_id = page_id_for(metadata.document_id, spine_index)
            page_text = "\n\n".join(text for _, text in parser.blocks)
            page_images: list[ParsedImage] = []
            if want_images:
                page_images, skipped = _extract_epub_images_for_page(
                    archive,
                    decoded,
                    member,
                    metadata=metadata,
                    page_id=page_id,
                    page_number=spine_index,
                    images_root=images_root,
                    creator=creator,
                    rights=rights,
                    licence_url=metadata.licence_url,
                    warnings=warnings,
                )
                extracted_images += len(page_images)
                skipped_images += skipped
            pages.append(
                ParsedPage(
                    page_id=page_id,
                    page_number=spine_index,
                    page_index=spine_index - 1,
                    text=page_text,
                    char_count=len(page_text),
                    width=0.0,
                    height=0.0,
                    has_extractable_text=True,
                    image_only=False,
                    images=page_images,
                )
            )

            current_title = Path(href).stem.replace("_", " ").replace("-", " ")
            current_paras: list[Paragraph] = []
            current_from_heading = False

            def flush() -> None:
                nonlocal section_index, current_paras, current_title, current_from_heading
                if not current_paras:
                    return
                text = "\n\n".join(p.text for p in current_paras)
                sections.append(
                    ParsedSection(
                        section_id=section_id_for(metadata.document_id, section_index),
                        title=current_title,
                        section_index=section_index,
                        page_start=spine_index,
                        page_end=spine_index,
                        text=text,
                        paragraphs=list(current_paras),
                        page_numbers=[spine_index],
                        title_from_document=current_from_heading,
                    )
                )
                section_index += 1
                current_paras = []

            for tag, text in parser.blocks:
                if tag in {"h1", "h2", "h3"}:
                    flush()
                    current_title = text
                    current_from_heading = True
                    continue
                current_paras.append(Paragraph(text=text, page_number=spine_index))
            flush()

    if not pages:
        raise EpubParseError(f"No readable XHTML spine items in {path}")
    warnings.append(
        f"media audit: extracted {extracted_images} image(s), skipped {skipped_images}"
    )

    parsed = ParsedDocument(
        metadata=metadata,
        content_hash=content_hash,
        page_count=len(pages),
        pages=pages,
        sections=sections,
        warnings=warnings,
    )
    LOGGER.info(
        "Parsed ePUB %s: %d spine pages, %d sections",
        metadata.relative_pdf_path,
        len(pages),
        len(sections),
    )
    return parsed
