#!/usr/bin/env python3
"""Inspect the curriculum graph: counts, coverage and local neighbourhoods.

Read-only. Nothing is created, modified or deleted.

    python scripts/inspect_graph.py                       # counts + coverage
    python scripts/inspect_graph.py --units               # list every unit
    python scripts/inspect_graph.py --concepts 30         # top concepts
    python scripts/inspect_graph.py --unit grade_02:science:unit_01_properties_of_matter
    python scripts/inspect_graph.py --chunk <chunk_id>    # chunk neighbourhood
    python scripts/inspect_graph.py --images 10           # image -> page -> unit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.graph_schema import NODE_LABELS, RELATIONSHIP_TYPES  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402

RULE = "-" * 96


def show_counts(store: Neo4jStore) -> None:
    print("\nNODE COUNTS")
    print(RULE)
    node_counts = store.node_counts(NODE_LABELS)
    for label, count in node_counts.items():
        print(f"  {label:12s} {count:>10,}")
    print(f"  {'TOTAL':12s} {sum(node_counts.values()):>10,}")

    print("\nRELATIONSHIP COUNTS")
    print(RULE)
    rel_counts = store.relationship_counts(RELATIONSHIP_TYPES)
    for rel_type, count in rel_counts.items():
        print(f"  {rel_type:16s} {count:>10,}")
    print(f"  {'TOTAL':16s} {sum(rel_counts.values()):>10,}")


def show_coverage(store: Neo4jStore, embedding_version: str) -> None:
    print("\nCURRICULUM COVERAGE")
    print(RULE)
    rows = store.read(
        """
        MATCH (g:Grade)-[:HAS_SUBJECT]->(s:Subject)-[:HAS_UNIT]->(u:Unit)
              -[:HAS_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH g, s, count(DISTINCT u) AS units, count(DISTINCT d) AS documents,
             count(DISTINCT p) AS pages
        MATCH (c:Chunk {grade: g.grade, subject: s.subject})
        RETURN g.grade AS grade, s.subject AS subject, units, documents, pages,
               count(c) AS chunks
        ORDER BY grade, subject
        """
    )
    if not rows:
        print("  (graph is empty -- run: python scripts/ingest_corpus.py)")
        return
    header = f"  {'grade':>5} {'subject':14} {'units':>6} {'docs':>6} {'pages':>7} {'chunks':>8}"
    print(header)
    for row in rows:
        print(
            f"  {row['grade']:>5} {row['subject']:14} {row['units']:>6} "
            f"{row['documents']:>6} {row['pages']:>7,} {row['chunks']:>8,}"
        )

    print("\nEMBEDDING COVERAGE")
    print(RULE)
    embed = store.read(
        """
        MATCH (c:Chunk)
        RETURN count(c) AS chunks,
               count(c.embedding) AS embedded,
               count(CASE WHEN c.embedding_version = $version THEN 1 END) AS current
        """,
        {"version": embedding_version},
    )
    if embed:
        row = embed[0]
        print(f"  chunks total            : {row['chunks']:,}")
        print(f"  with an embedding       : {row['embedded']:,}")
        print(f"  at version {embedding_version:14s}: {row['current']:,}")
        missing = row["chunks"] - row["current"]
        if missing:
            print(
                f"  MISSING current vectors : {missing:,} "
                f"(run: python scripts/ingest_corpus.py)"
            )

    print("\nDOCUMENTS WITHOUT CHUNKS (no retrievable text)")
    print(RULE)
    empty = store.read(
        """
        MATCH (d:Document)
        WHERE coalesce(d.chunk_count, 0) = 0
        RETURN d.relative_pdf_path AS path, d.page_count AS pages
        ORDER BY path
        """
    )
    if not empty:
        print("  (none)")
    for row in empty:
        print(f"  {row['path']} ({row['pages']} pages)")


def show_units(store: Neo4jStore) -> None:
    print("\nUNITS")
    print(RULE)
    rows = store.read(
        """
        MATCH (u:Unit)
        OPTIONAL MATCH (u)-[:HAS_DOCUMENT]->(d:Document)
        OPTIONAL MATCH (c:Chunk {unit_id: u.unit_id})
        RETURN u.unit_id AS unit_id, u.title AS title, u.grade AS grade,
               u.subject AS subject, count(DISTINCT d) AS documents,
               count(DISTINCT c) AS chunks
        ORDER BY grade, subject, unit_id
        """
    )
    for row in rows:
        print(
            f"  G{row['grade']} {row['subject']:12s} docs={row['documents']:>2} "
            f"chunks={row['chunks']:>5} {row['unit_id']}"
        )
        print(f"      {row['title']}")


def show_concepts(store: Neo4jStore, limit: int) -> None:
    print(f"\nTOP {limit} CONCEPTS BY MENTIONS")
    print(RULE)
    rows = store.read(
        """
        MATCH (co:Concept)
        OPTIONAL MATCH (co)<-[m:MENTIONS]-(c:Chunk)
        RETURN co.name AS name, co.origin AS origin,
               count(m) AS mentions,
               count(DISTINCT c.unit_id) AS units,
               count(DISTINCT c.grade) AS grades
        ORDER BY mentions DESC, name
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    if not rows:
        print("  (no concepts)")
        return
    print(f"  {'mentions':>8} {'units':>6} {'grades':>7}  {'origin':15} name")
    for row in rows:
        print(
            f"  {row['mentions']:>8} {row['units']:>6} {row['grades']:>7}  "
            f"{(row['origin'] or '?'):15} {row['name']}"
        )


def show_unit_neighbourhood(store: Neo4jStore, unit_id: str) -> None:
    print(f"\nUNIT NEIGHBOURHOOD: {unit_id}")
    print(RULE)
    rows = store.read(
        """
        MATCH (g:Grade)-[:HAS_SUBJECT]->(s:Subject)-[:HAS_UNIT]->(u:Unit {unit_id: $unit_id})
        RETURN g.grade AS grade, s.subject AS subject, u.title AS title,
               u.unit_number AS unit_number
        """,
        {"unit_id": unit_id},
    )
    if not rows:
        print("  Unit not found. List units with --units")
        return
    row = rows[0]
    print(f"  Grade {row['grade']} / {row['subject']} / unit {row['unit_number']}")
    print(f"  Title: {row['title']}")

    print("\n  Documents")
    for doc in store.read(
        """
        MATCH (u:Unit {unit_id: $unit_id})-[:HAS_DOCUMENT]->(d:Document)
        RETURN d.document_id AS document_id, d.title AS title,
               d.resource_type AS resource_type, d.audience AS audience,
               d.page_count AS pages, coalesce(d.chunk_count, 0) AS chunks,
               coalesce(d.image_count, 0) AS images, d.local_pdf_path AS pdf
        ORDER BY audience, document_id
        """,
        {"unit_id": unit_id},
    ):
        print(
            f"    {doc['audience']:8s} {doc['resource_type']:18s} "
            f"pages={doc['pages']:>4} chunks={doc['chunks']:>4} "
            f"images={doc['images']:>4}"
        )
        print(f"      {doc['pdf']}")

    print("\n  Sections (first 15)")
    for section in store.read(
        """
        MATCH (:Unit {unit_id: $unit_id})-[:HAS_DOCUMENT]->(:Document)
              -[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(sec:Section)
        RETURN sec.title AS title, sec.page_start AS page_start,
               sec.page_end AS page_end, sec.section_id AS section_id
        ORDER BY sec.section_id
        LIMIT 15
        """,
        {"unit_id": unit_id},
    ):
        print(
            f"    p{section['page_start']:>4}-{section['page_end']:<4} "
            f"{section['title'][:70]}"
        )

    print("\n  Concepts mentioned in this unit (top 15)")
    for concept in store.read(
        """
        MATCH (c:Chunk {unit_id: $unit_id})-[m:MENTIONS]->(co:Concept)
        RETURN co.name AS name, count(m) AS mentions
        ORDER BY mentions DESC, name
        LIMIT 15
        """,
        {"unit_id": unit_id},
    ):
        print(f"    {concept['mentions']:>4}x {concept['name']}")


def show_chunk_neighbourhood(store: Neo4jStore, chunk_id: str) -> None:
    print(f"\nCHUNK NEIGHBOURHOOD: {chunk_id}")
    print(RULE)
    rows = store.read(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        OPTIONAL MATCH (g:Grade)-[:HAS_SUBJECT]->(s:Subject)-[:HAS_UNIT]->(u:Unit)
                       -[:HAS_DOCUMENT]->(d:Document)-[:HAS_PAGE]->(p:Page)
                       -[:HAS_SECTION]->(sec:Section)-[:HAS_CHUNK]->(c)
        RETURN c.text AS text, c.token_count AS tokens, c.page_start AS page_start,
               c.page_end AS page_end, c.local_pdf_path AS pdf,
               g.grade AS grade, s.subject AS subject, u.title AS unit_title,
               d.title AS document_title, sec.title AS section_title,
               p.page_id AS parent_page
        """,
        {"chunk_id": chunk_id},
    )
    if not rows:
        print("  Chunk not found.")
        return
    row = rows[0]
    print("  Full lineage (Chunk -> Section -> Page -> Document -> Unit -> Subject -> Grade)")
    print(f"    grade    : {row['grade']}")
    print(f"    subject  : {row['subject']}")
    print(f"    unit     : {row['unit_title']}")
    print(f"    document : {row['document_title']}")
    print(f"    section  : {row['section_title']}")
    print(f"    pages    : {row['page_start']}-{row['page_end']}")
    print(f"    pdf      : {row['pdf']}")
    print(f"    tokens   : {row['tokens']}")
    print(f"\n  Text\n    {' '.join((row['text'] or '').split())[:600]}")

    print("\n  Reading-order neighbours")
    for rel in store.read(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        OPTIONAL MATCH (c)-[:PREVIOUS]->(prev:Chunk)
        OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
        RETURN prev.chunk_id AS previous, next.chunk_id AS next
        """,
        {"chunk_id": chunk_id},
    ):
        print(f"    PREVIOUS: {rel['previous']}")
        print(f"    NEXT    : {rel['next']}")

    print("\n  Concepts")
    for concept in store.read(
        """
        MATCH (:Chunk {chunk_id: $chunk_id})-[m:MENTIONS]->(co:Concept)
        RETURN co.name AS name, m.source AS source, m.occurrences AS occurrences,
               coalesce(co.mention_count, 0) AS corpus_mentions
        ORDER BY occurrences DESC, name
        """,
        {"chunk_id": chunk_id},
    ):
        print(
            f"    {concept['name']:40s} source={concept['source']:15s} "
            f"here={concept['occurrences']} corpus={concept['corpus_mentions']}"
        )

    print("\n  Images on the same page(s)")
    images = store.read(
        """
        MATCH (:Chunk {chunk_id: $chunk_id})-[:ON_PAGE]->(p:Page)-[:HAS_IMAGE]->(i:Image)
        RETURN i.page_number AS page, i.width AS width, i.height AS height,
               i.format AS format, i.local_path AS path
        ORDER BY page, width * height DESC
        LIMIT 10
        """,
        {"chunk_id": chunk_id},
    )
    if not images:
        print("    (none)")
    for image in images:
        print(
            f"    p{image['page']:>4} {image['width']}x{image['height']} "
            f"{image['format']:5s} {image['path']}"
        )

    print("\n  Chunks reachable in one graph hop (same section / adjacent / same page)")
    for neighbour in store.read(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})
        CALL {
            WITH c
            MATCH (c)<-[:HAS_CHUNK]-(:Section)-[:HAS_CHUNK]->(n:Chunk)
            RETURN n, 'SAME_SECTION' AS relation
            UNION
            WITH c
            MATCH (c)-[:NEXT|PREVIOUS]-(n:Chunk)
            RETURN n, 'ADJACENT' AS relation
            UNION
            WITH c
            MATCH (c)-[:ON_PAGE]->(:Page)<-[:ON_PAGE]-(n:Chunk)
            RETURN n, 'SAME_PAGE' AS relation
        }
        WITH n, relation WHERE n.chunk_id <> $chunk_id
        RETURN DISTINCT relation, n.chunk_id AS chunk_id,
               n.section_title AS section_title
        ORDER BY relation, chunk_id
        LIMIT 15
        """,
        {"chunk_id": chunk_id},
    ):
        print(
            f"    {neighbour['relation']:14s} {neighbour['chunk_id']}\n"
            f"                   section: {(neighbour['section_title'] or '?')[:60]}"
        )


def show_images(store: Neo4jStore, limit: int) -> None:
    print(f"\nIMAGE TRACEABILITY (first {limit})")
    print(RULE)
    rows = store.read(
        """
        MATCH (g:Grade)-[:HAS_SUBJECT]->(s:Subject)-[:HAS_UNIT]->(u:Unit)
              -[:HAS_DOCUMENT]->(d:Document)-[:HAS_PAGE]->(p:Page)-[:HAS_IMAGE]->(i:Image)
        OPTIONAL MATCH (i)-[:ILLUSTRATES]->(co:Concept)
        RETURN i.image_id AS image_id, i.local_path AS path, i.page_number AS page,
               i.width AS width, i.height AS height, i.format AS format,
               i.source_pdf AS pdf, g.grade AS grade, s.subject AS subject,
               u.title AS unit_title, collect(DISTINCT co.name) AS concepts
        ORDER BY grade, subject, image_id
        LIMIT $limit
        """,
        {"limit": int(limit)},
    )
    if not rows:
        print("  (no images)")
        return
    for row in rows:
        print(
            f"  G{row['grade']} {row['subject']:12s} page {row['page']:>4} "
            f"{row['width']}x{row['height']} {row['format']}"
        )
        print(f"    unit   : {row['unit_title'][:70]}")
        print(f"    pdf    : {row['pdf']}")
        print(f"    file   : {row['path']}")
        concepts = [c for c in row["concepts"] if c]
        print(f"    concept: {', '.join(concepts) if concepts else '(none)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", action="store_true", help="List all units.")
    parser.add_argument(
        "--concepts", type=int, nargs="?", const=25, help="Show top N concepts."
    )
    parser.add_argument("--unit", help="Inspect one unit and its neighbourhood.")
    parser.add_argument("--chunk", help="Inspect one chunk and its neighbourhood.")
    parser.add_argument(
        "--images", type=int, nargs="?", const=10, help="Show N image records."
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        store = Neo4jStore(config.require_neo4j())
        store.connect()
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    try:
        print(f"Neo4j: {store.server_version()} at {config.neo4j.describe()}")  # type: ignore[union-attr]

        targeted = any([args.units, args.concepts, args.unit, args.chunk, args.images])
        if not targeted:
            show_counts(store)
            show_coverage(store, config.embedding_version)
        if args.units:
            show_units(store)
        if args.concepts:
            show_concepts(store, args.concepts)
        if args.unit:
            show_unit_neighbourhood(store, args.unit)
        if args.chunk:
            show_chunk_neighbourhood(store, args.chunk)
        if args.images:
            show_images(store, args.images)
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
