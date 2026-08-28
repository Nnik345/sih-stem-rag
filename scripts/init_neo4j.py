#!/usr/bin/env python3
"""Create the curriculum graph schema: constraints, vector and full-text indexes.

Safe and idempotent. An existing database is never deleted unless --reset is
passed explicitly.

    python scripts/init_neo4j.py
    python scripts/init_neo4j.py --show
    python scripts/init_neo4j.py --reset            # DESTRUCTIVE, asks first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.embeddings import EmbeddingError, read_hidden_size  # noqa: E402
from rag.graph_schema import (  # noqa: E402
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    initialize_schema,
    reset_graph,
)
from rag.logging_utils import get_logger, setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402

LOGGER = get_logger("init_neo4j")

_CONNECTION_HINT = """
Neo4j is not reachable. Start the project-local instance:

    scripts/neo4j_local.sh start
    scripts/neo4j_local.sh status

If .neo4j-local/ is missing, provision it first: see README.md -> 'Neo4j setup'.

Requirements: Neo4j 5.18 or newer (native vector indexes and
vector.similarity.cosine), running on Java 17 or 21.
""".strip()


def _connection_help() -> str:
    return _CONNECTION_HINT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Only report the current schema and node counts; create nothing.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DESTRUCTIVE: delete every node before creating the schema.",
    )
    parser.add_argument(
        "--drop-indexes",
        action="store_true",
        help="With --reset, also drop the vector/full-text/property indexes.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation for --reset.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    neo4j_config = config.require_neo4j()
    print(f"Neo4j target: {neo4j_config.describe()}\n")

    try:
        store = Neo4jStore(neo4j_config)
        store.connect()
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}\n", file=sys.stderr)
        print(_connection_help(), file=sys.stderr)
        return 3

    try:
        print(f"Server: {store.server_version()}\n")

        if args.reset:
            if not args.yes:
                node_total = 0 if store.is_empty() else -1
                warning = (
                    "The database appears empty."
                    if node_total == 0
                    else "The database contains data."
                )
                print(f"{warning} --reset deletes ALL nodes in database "
                      f"{neo4j_config.database!r}.")
                answer = input("Type 'reset' to confirm: ").strip()
                if answer != "reset":
                    print("Aborted; nothing was deleted.")
                    return 1
            summary = reset_graph(store, drop_indexes=args.drop_indexes)
            print(
                f"Reset complete: {summary['nodes_deleted']} nodes deleted, "
                f"{summary['indexes_dropped']} indexes dropped.\n"
            )

        if not args.show:
            try:
                dimension = read_hidden_size(config.models.embedding_model_path)
            except EmbeddingError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 4
            print(
                f"Embedding dimension read from "
                f"{config.models.embedding_model_path.name}/config.json: {dimension}\n"
            )
            report = initialize_schema(store, dimension)
            print("Schema created/verified")
            print(report.describe())
            print()

        print("Indexes")
        print("-" * 88)
        for index in store.list_indexes():
            labels = ",".join(index.get("labelsOrTypes") or [])
            props = ",".join(index.get("properties") or [])
            print(
                f"  {index['name']:32s} {index['type']:10s} {labels:10s} "
                f"{props:34s} {index['state']}"
            )

        print("\nConstraints")
        print("-" * 88)
        for constraint in store.list_constraints():
            labels = ",".join(constraint.get("labelsOrTypes") or [])
            props = ",".join(constraint.get("properties") or [])
            print(f"  {constraint['name']:32s} {constraint['type']:24s} {labels}.{props}")

        print("\nNode counts")
        print("-" * 88)
        for label, count in store.node_counts(NODE_LABELS).items():
            print(f"  {label:12s} {count:>10,}")

        print("\nRelationship counts")
        print("-" * 88)
        for rel_type, count in store.relationship_counts(RELATIONSHIP_TYPES).items():
            print(f"  {rel_type:16s} {count:>10,}")

    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
