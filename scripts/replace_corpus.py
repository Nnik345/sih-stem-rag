#!/usr/bin/env python3
"""Explicit, source-aware curriculum replacement.

Ordinary ingestion never deletes the graph. This command is the only supported
way to remove the retired Core Knowledge corpus (or, with a separate flag, the
entire curriculum graph).

    python scripts/replace_corpus.py --purge-core-knowledge
    python scripts/replace_corpus.py --purge-core-knowledge --yes
    python scripts/replace_corpus.py --purge-all-curriculum --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.graph_schema import NODE_LABELS, RELATIONSHIP_TYPES, reset_graph  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402

_CK_MATCH = """
MATCH (d:Document)
WHERE toLower(coalesce(d.local_pdf_path, '')) CONTAINS 'core_knowledge'
   OR toLower(coalesce(d.relative_pdf_path, '')) CONTAINS 'core_knowledge'
   OR toLower(coalesce(d.publisher, '')) CONTAINS 'core knowledge'
   OR toLower(coalesce(d.source_id, '')) CONTAINS 'core_knowledge'
   OR toLower(coalesce(d.title, '')) CONTAINS 'core knowledge'
"""


def _count(store: Neo4jStore, cypher: str, **params: object) -> int:
    rows = store.read(cypher, params or None)
    return int(rows[0]["n"]) if rows else 0


def inspect_core_knowledge(store: Neo4jStore) -> dict[str, int]:
    docs = _count(store, _CK_MATCH + " RETURN count(d) AS n")
    chunks = _count(
        store,
        _CK_MATCH
        + """
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
        RETURN count(c) AS n
        """,
    )
    images = _count(
        store,
        _CK_MATCH
        + """
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_IMAGE]->(i:Image)
        RETURN count(i) AS n
        """,
    )
    pages = _count(
        store,
        _CK_MATCH + " OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page) RETURN count(p) AS n",
    )
    rels = _count(
        store,
        _CK_MATCH
        + """
        OPTIONAL MATCH (d)-[r]-()
        RETURN count(r) AS n
        """,
    )
    return {
        "documents": docs,
        "pages": pages,
        "chunks": chunks,
        "images": images,
        "document_relationships": rels,
    }


def purge_core_knowledge(store: Neo4jStore) -> dict[str, int]:
    """Delete matching documents and any hierarchy nodes they leave empty."""
    before = inspect_core_knowledge(store)
    store.execute_write(
        _CK_MATCH
        + """
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        OPTIONAL MATCH (p)-[:HAS_SECTION]->(sec:Section)
        OPTIONAL MATCH (sec)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:Image)
        WITH collect(DISTINCT d) + collect(DISTINCT p) + collect(DISTINCT sec)
             + collect(DISTINCT c) + collect(DISTINCT i) AS nodes
        UNWIND nodes AS node
        DETACH DELETE node
        """
    )
    # Orphaned units/subjects/grades and concepts with no remaining mentions.
    store.execute_write(
        """
        MATCH (u:Unit) WHERE NOT (u)-[:HAS_DOCUMENT]->() DETACH DELETE u
        """
    )
    store.execute_write(
        """
        MATCH (s:Subject) WHERE NOT (s)-[:HAS_UNIT]->() DETACH DELETE s
        """
    )
    store.execute_write(
        """
        MATCH (g:Grade) WHERE NOT (g)-[:HAS_SUBJECT]->() DETACH DELETE g
        """
    )
    store.execute_write(
        """
        MATCH (co:Concept) WHERE NOT ()-[:MENTIONS]->(co) AND NOT ()-[:ILLUSTRATES]->(co)
        DETACH DELETE co
        """
    )
    remaining = inspect_core_knowledge(store)
    return {"before": before, "remaining": remaining}


def verify_no_core_knowledge(store: Neo4jStore) -> int:
    leftover = store.read(
        """
        MATCH (n)
        WHERE toLower(coalesce(n.local_pdf_path, '')) CONTAINS 'core_knowledge'
           OR toLower(coalesce(n.relative_pdf_path, '')) CONTAINS 'core_knowledge'
           OR toLower(coalesce(n.publisher, '')) CONTAINS 'core knowledge'
           OR toLower(coalesce(n.source_id, '')) CONTAINS 'core_knowledge'
        RETURN count(n) AS n
        """
    )[0]["n"]
    return int(leftover)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--purge-core-knowledge",
        action="store_true",
        help="Delete nodes identified as Core Knowledge curriculum.",
    )
    parser.add_argument(
        "--purge-all-curriculum",
        action="store_true",
        help="DESTRUCTIVE: delete every node in the curriculum database.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    if not args.purge_core_knowledge and not args.purge_all_curriculum:
        print("Specify --purge-core-knowledge and/or --purge-all-curriculum.", file=sys.stderr)
        return 2

    try:
        config = load_config()
        store = Neo4jStore(config.require_neo4j())
        store.connect()
    except (ConfigError, Neo4jUnavailableError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    try:
        print("Current graph")
        for label, count in store.node_counts(NODE_LABELS).items():
            print(f"  {label:12s} {count:>10,}")
        print()
        ck = inspect_core_knowledge(store)
        print("Core Knowledge matches (before)")
        for key, value in ck.items():
            print(f"  {key:24s} {value:>10,}")
        print()

        if args.purge_core_knowledge:
            if not args.yes:
                if input("Type 'purge-core-knowledge' to confirm: ").strip() != "purge-core-knowledge":
                    print("Aborted; nothing was deleted.")
                    return 1
            result = purge_core_knowledge(store)
            print("Purged Core Knowledge matches:")
            print(f"  before : {result['before']}")
            print(f"  remaining matches: {result['remaining']}")
            leftover = verify_no_core_knowledge(store)
            print(f"  leftover Core Knowledge nodes: {leftover}")
            if leftover:
                print("ERROR: Core Knowledge nodes remain.", file=sys.stderr)
                return 5

        if args.purge_all_curriculum:
            if not args.yes:
                print(f"--purge-all-curriculum deletes ALL nodes in {config.neo4j.database!r}.")
                if input("Type 'purge-all' to confirm: ").strip() != "purge-all":
                    print("Aborted; nothing further was deleted.")
                    return 1
            summary = reset_graph(store)
            print(f"Full curriculum reset: {summary['nodes_deleted']} nodes deleted.")

        leftover = verify_no_core_knowledge(store)
        print(f"Verification: {leftover} Core Knowledge nodes remain.")
        print("\nGraph node counts after purge")
        for label, count in store.node_counts(NODE_LABELS).items():
            print(f"  {label:12s} {count:>10,}")
        print("\nRelationship counts after purge")
        for rel, count in store.relationship_counts(RELATIONSHIP_TYPES).items():
            print(f"  {rel:16s} {count:>10,}")
        return 0 if leftover == 0 else 5
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
