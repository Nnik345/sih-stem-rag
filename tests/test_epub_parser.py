"""ePUB spine items become pages/sections with chapter provenance."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

from rag.config import IngestConfig
from rag.epub_parser import parse_epub_document
from rag.schemas import DocumentMetadata


def _write_min_epub(path: Path, *, with_image: bool = False) -> None:
    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    image_item = (
        '<item id="img1" href="images/leaf.png" media-type="image/png"/>'
        if with_image
        else ""
    )
    opf = f"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="bookid" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Natural Sciences</dc:title>
    <dc:creator>Siyavula</dc:creator>
    <dc:rights>CC BY 4.0</dc:rights>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    {image_item}
  </manifest>
  <spine>
    <itemref idref="ch1"/>
  </spine>
</package>
"""
    img_tag = '<img src="images/leaf.png" alt="a leaf"/>' if with_image else ""
    chapter = f"""<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body>
<h1>Living and non-living things</h1>
<p>Plants need water, air and sunlight to grow.</p>
{img_tag}
</body>
</html>
"""
    png = (
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 80
        if with_image
        else b""
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/ch1.xhtml", chapter)
        if with_image:
            archive.writestr("OEBPS/images/leaf.png", png)


def test_epub_sections_keep_chapter_title(tmp_path: Path, metadata: DocumentMetadata):
    epub = tmp_path / "book.epub"
    _write_min_epub(epub)
    meta = replace(
        metadata,
        local_pdf_path=epub,
        relative_pdf_path="raw/siyavula/science/grade_04/book.epub",
        file_format="epub",
        extract_images=False,
    )
    parsed = parse_epub_document(meta, images_root=tmp_path, config=IngestConfig())
    assert parsed.page_count == 1
    assert parsed.sections
    assert parsed.sections[0].title == "Living and non-living things"
    assert parsed.sections[0].page_start == 1
    assert "sunlight" in parsed.sections[0].text
    assert parsed.images == []


def test_cc_by_epub_extracts_referenced_images(tmp_path: Path, metadata: DocumentMetadata):
    epub = tmp_path / "book.epub"
    _write_min_epub(epub, with_image=True)
    meta = replace(
        metadata,
        local_pdf_path=epub,
        relative_pdf_path="raw/siyavula/science/grade_04/book.epub",
        file_format="epub",
        extract_images=True,
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
    )
    parsed = parse_epub_document(meta, images_root=tmp_path / "images", config=IngestConfig())
    assert parsed.images
    image = parsed.images[0]
    assert image.licence.startswith("CC BY")
    assert image.local_path.is_file()
    assert "leaf.png" in image.source_href or image.local_path.suffix == ".png"
