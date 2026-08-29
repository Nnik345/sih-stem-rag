#!/usr/bin/env python3
"""Classify every parsed section and write a partition audit (no graph writes)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import load_config  # noqa: E402
from rag.corpus import discover_documents  # noqa: E402
from rag.epub_parser import parse_epub_document  # noqa: E402
from rag.partitions import classify_section  # noqa: E402
from rag.pdf_parser import parse_document  # noqa: E402


def main() -> int:
    config = load_config()
    documents, _ = discover_documents(config.paths.corpus_path)
    out_dir = config.paths.processed_data_path / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "partition_audit.csv"
    counts: dict[str, dict[str, int]] = {}
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_id",
                "document",
                "section_title",
                "page_start",
                "proposed_partition",
                "reason",
                "markers",
            ]
        )
        for metadata in documents:
            try:
                if metadata.file_format == "epub":
                    parsed = parse_epub_document(
                        metadata,
                        images_root=config.paths.images_dir,
                        config=config.ingest,
                    )
                else:
                    parsed = parse_document(
                        metadata,
                        images_root=config.paths.images_dir,
                        config=config.ingest,
                    )
            except Exception as exc:
                print(f"skip {metadata.relative_pdf_path}: {exc}")
                continue
            previous = ""
            source_counts = counts.setdefault(metadata.source_id, {})
            for section in parsed.sections:
                decision = classify_section(
                    section.title, section.text, previous_title=previous
                )
                previous = section.title
                source_counts[decision.partition] = source_counts.get(decision.partition, 0) + 1
                markers = []
                blob = f"{section.title}\n{section.text[:800]}".lower()
                for token in (
                    "answer key",
                    "sample response",
                    "exit ticket",
                    "homework",
                    "creative commons",
                    "isbn",
                ):
                    if token in blob:
                        markers.append(token)
                writer.writerow(
                    [
                        metadata.source_id,
                        metadata.relative_pdf_path,
                        section.title[:120],
                        section.page_start,
                        decision.partition,
                        decision.reason,
                        ";".join(markers),
                    ]
                )
    (out_dir / "partition_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
