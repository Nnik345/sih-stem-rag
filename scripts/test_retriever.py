#!/usr/bin/env python3
"""Inspect hybrid retrieval channel by channel. Never calls the generator.

Shows DENSE, FULL-TEXT, GRAPH, FUSED and RERANKED results side by side with each
candidate's per-channel ranks, RRF score and reranker score, so retrieval
behaviour can be understood and compared between configurations.

    python scripts/test_retriever.py -q "what are the components of food" -g 6 -s science
    python scripts/test_retriever.py -q "how does light reflect" -g 10 -s science --no-rerank
    python scripts/test_retriever.py -q "what is electrostatics" -g 12 -s science --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402
from rag.pipeline import HybridRetriever  # noqa: E402
from rag.schemas import RetrievalResponse, RetrievedChunk  # noqa: E402

PREVIEW_WIDTH = 118


def _fmt(value: float | int | None, spec: str = "7.4f") -> str:
    return "  --   " if value is None else format(value, spec)


def _rank(value: int | None) -> str:
    return " -" if value is None else f"{value:2d}"


def print_channel(title: str, results: list[RetrievedChunk], *, limit: int) -> None:
    print()
    print("=" * 132)
    print(f"{title}   ({len(results)} results)")
    print("=" * 132)
    if not results:
        print("  (no results)")
        return

    for index, chunk in enumerate(results[:limit], start=1):
        contributions = (
            f"dense#{_rank(chunk.dense_rank)} "
            f"ft#{_rank(chunk.fulltext_rank)} "
            f"graph#{_rank(chunk.graph_rank)}"
        )
        print(
            f"\n  [{index:2d}] G{chunk.grade} {chunk.subject:12s} | "
            f"unit: {(chunk.unit_title or '?')[:52]}"
        )
        print(
            f"       doc: {(chunk.document_title or '?')[:60]} | "
            f"pages {chunk.page_range}"
        )
        print(f"       section: {(chunk.section_title or '?')[:70]}")
        print(f"       chunk_id: {chunk.chunk_id}")
        print(
            f"       sources: {chunk.retrieval_source:22s} {contributions}   "
            f"dense={_fmt(chunk.dense_score)} ft={_fmt(chunk.fulltext_score, '7.3f')} "
            f"graph={_fmt(chunk.graph_score)}"
        )
        print(
            f"       rrf={_fmt(chunk.rrf_score, '9.6f')} "
            f"rrf_rank={_rank(chunk.rrf_rank)}   "
            f"rerank={_fmt(chunk.rerank_score, '8.4f')} "
            f"rerank_rank={_rank(chunk.rerank_rank)}"
        )
        print(
            f"       source={chunk.source_id or '?'} "
            f"role={chunk.source_role or '?'} "
            f"licence={(chunk.licence or '?')[:40]} "
            f"partition={chunk.content_partition or '?'}"
        )
        outcomes = ", ".join(chunk.cisce_outcome_ids) or "(none)"
        print(
            f"       cisce={outcomes} "
            f"alignment={chunk.alignment_status or '?'}"
        )
        if chunk.graph_expansion_path:
            print(f"       graph path: {chunk.graph_expansion_path[:PREVIEW_WIDTH]}")
        print(f"       text: {chunk.preview(PREVIEW_WIDTH)}")


def print_report(response: RetrievalResponse, *, limit: int) -> None:
    diagnostics = response.diagnostics

    print()
    print("#" * 132)
    print(f"QUERY : {response.query}")
    print(f"SCOPE : {response.scope.describe()}")
    print("#" * 132)

    print_channel("DENSE RESULTS (BGE-M3 + Neo4j vector index)", diagnostics.dense, limit=limit)
    print_channel("FULL-TEXT RESULTS (Neo4j Lucene index)", diagnostics.fulltext, limit=limit)
    print_channel("GRAPH RESULTS (bounded expansion from seeds)", diagnostics.graph, limit=limit)
    print_channel("FUSED RESULTS (weighted reciprocal rank fusion)", diagnostics.fused, limit=limit)
    print_channel("RERANKED FINAL RESULTS (bge-reranker-v2-m3)", diagnostics.reranked, limit=limit)

    print()
    print("=" * 132)
    print("DIAGNOSTICS")
    print("=" * 132)
    print(f"  candidate counts : {diagnostics.channel_counts()}")
    print(
        "  stage timings ms : "
        + ", ".join(f"{k}={v:.1f}" for k, v in diagnostics.timings_ms.items())
    )
    if diagnostics.graph_seeds:
        print(f"  graph seeds      : {len(diagnostics.graph_seeds)}")
        for seed in diagnostics.graph_seeds[:5]:
            print(f"    - {seed}")
    for note in diagnostics.notes:
        print(f"  note             : {note}")

    if diagnostics.reranked:
        print()
        print("  Reranker effect (fused rank -> reranked rank)")
        fused_rank = {c.chunk_id: c.rrf_rank for c in diagnostics.fused}
        for chunk in diagnostics.reranked:
            before = fused_rank.get(chunk.chunk_id)
            arrow = "unchanged" if before == chunk.rerank_rank else "MOVED"
            print(
                f"    fused #{_rank(before)} -> reranked #{_rank(chunk.rerank_rank)} "
                f"({arrow:9s}) score={_fmt(chunk.rerank_score, '8.4f')} "
                f"{chunk.chunk_id}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--query", help="Student question. Omit to be prompted.")
    parser.add_argument("-g", "--grade", type=int, help="Grade filter, e.g. 3")
    parser.add_argument("-s", "--subject", help="Subject filter: science | mathematics")
    parser.add_argument("-u", "--unit", help="Unit id filter")
    parser.add_argument("--resource-type", help="Resource type filter")
    parser.add_argument("--audience", help="student | teacher | other")
    parser.add_argument(
        "--limit", type=int, default=8, help="Rows shown per channel (default 8)"
    )
    parser.add_argument(
        "--final-top-k", type=int, help="Override the final reranked result count"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip the reranker stage"
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="Attach image metadata from the graph to final results",
    )
    parser.add_argument("--json", help="Also write full diagnostics to this JSON file")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    query = args.query
    grade = args.grade
    subject = args.subject
    if not query:
        try:
            query = input("Question: ").strip()
            if grade is None:
                raw_grade = input("Grade (blank for any): ").strip()
                grade = int(raw_grade) if raw_grade else None
            if subject is None:
                raw_subject = input("Subject (science/mathematics, blank for any): ").strip()
                subject = raw_subject or None
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
    if not query:
        print("No query given.", file=sys.stderr)
        return 2

    try:
        store = Neo4jStore(config.require_neo4j())
        store.connect()
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    retriever: HybridRetriever | None = None
    try:
        retriever = HybridRetriever(config, store)
        response = retriever.retrieve(
            query,
            grade=grade,
            subject=subject,
            unit=args.unit,
            resource_type=args.resource_type,
            audience=args.audience,
            rerank=not args.no_rerank,
            final_top_k=args.final_top_k,
            include_images=True if args.images else None,
        )
        print_report(response, limit=args.limit)

        if args.json:
            path = Path(args.json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(response.diagnostics.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nFull diagnostics written to {path}")

        if args.images:
            print()
            print("=" * 132)
            print("IMAGES ATTACHED TO FINAL RESULTS (graph metadata only, no visual embeddings)")
            print("=" * 132)
            for chunk in response.results:
                if not chunk.images:
                    continue
                print(f"  {chunk.chunk_id}")
                for image in chunk.images:
                    print(
                        f"    page {image['page_number']:>4} "
                        f"{image['width']}x{image['height']} {image['format']:5s} "
                        f"{image['local_path']}"
                    )
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        if retriever is not None:
            retriever.release_models()
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
