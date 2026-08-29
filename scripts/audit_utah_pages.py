#!/usr/bin/env python3
"""Page-level extraction audit for Utah Grade 3 Science OER."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.chunker import TokenCounter, chunk_document  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.corpus import discover_documents  # noqa: E402
from rag.pdf_parser import parse_document  # noqa: E402


def main() -> int:
    config = load_config()
    documents, _ = discover_documents(
        config.paths.corpus_path, grades=(3,), subjects=("science",)
    )
    utah = [
        item
        for item in documents
        if item.source_id == "utah_science_oer" and item.grade == 3
    ]
    if not utah:
        print("No Utah Grade 3 document found under the corpus path.")
        return 1
    metadata = utah[0]
    parsed = parse_document(
        metadata,
        images_root=config.paths.images_dir,
        config=config.ingest,
    )
    chunks = chunk_document(parsed, TokenCounter(config.models.embedding_model_path), config.chunking)
    per_page: dict[int, int] = {}
    for chunk in chunks:
        for page in range(chunk.page_start, chunk.page_end + 1):
            per_page[page] = per_page.get(page, 0) + 1

    out_dir = config.paths.processed_data_path / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "utah_grade3_page_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "page",
                "char_count",
                "text_blocks",
                "chunk_count",
                "image_only",
                "page_role",
                "skip_reason",
            ]
        )
        for page in parsed.pages:
            writer.writerow(
                [
                    page.page_number,
                    page.char_count,
                    len(page.blocks),
                    per_page.get(page.page_number, 0),
                    page.image_only,
                    page.page_role,
                    page.skip_reason,
                ]
            )

    content_pages = [p for p in parsed.pages if p.page_role == "curriculum"]
    missing = [
        p.page_number
        for p in content_pages
        if per_page.get(p.page_number, 0) == 0 and p.char_count >= 40
    ]
    summary = {
        "document": metadata.relative_pdf_path,
        "pages": parsed.page_count,
        "chunks": len(chunks),
        "content_pages": len(content_pages),
        "image_only_pages": sum(1 for p in parsed.pages if p.image_only),
        "front_matter_or_credits": sum(
            1 for p in parsed.pages if p.page_role in {"front_matter", "credits"}
        ),
        "content_pages_with_no_chunk": missing,
        "csv": str(csv_path),
    }
    (out_dir / "utah_grade3_page_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if missing:
        print("Content-bearing pages without chunks:", missing)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
