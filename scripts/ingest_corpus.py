#!/usr/bin/env python3
"""Ingest the Core Knowledge corpus into the Neo4j curriculum graph.

Steps: discover PDFs (using manifest.json where available) -> parse structured
text, pages, sections and images with PyMuPDF -> hierarchical chunks -> graph
nodes and relationships -> conservative concept links -> BGE-M3 dense embeddings
-> vector and full-text indexes.

Default behaviour is safe and idempotent: unchanged, fully embedded documents are
skipped, so an interrupted run resumes simply by re-running the same command.
Nothing is deleted without --reset.

    python scripts/ingest_corpus.py
    python scripts/ingest_corpus.py --grade 1 --subject science
    python scripts/ingest_corpus.py --limit 5
    python scripts/ingest_corpus.py --force              # re-parse and re-embed
    python scripts/ingest_corpus.py --skip-embeddings    # structure only
    python scripts/ingest_corpus.py --reset --yes        # DESTRUCTIVE rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.corpus import CorpusError  # noqa: E402
from rag.embeddings import EmbeddingError  # noqa: E402
from rag.graph_schema import NODE_LABELS, RELATIONSHIP_TYPES, reset_graph  # noqa: E402
from rag.ingest import CorpusIngestor  # noqa: E402
from rag.logging_utils import setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grade",
        type=int,
        action="append",
        help="Restrict ingestion to a grade (repeatable).",
    )
    parser.add_argument(
        "--subject",
        action="append",
        help="Restrict ingestion to a subject, e.g. science (repeatable).",
    )
    parser.add_argument("--limit", type=int, help="Ingest at most N PDFs.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse and re-embed documents even if they are already current.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build the graph structure without computing embeddings.",
    )
    parser.add_argument(
        "--keep-embedder",
        action="store_true",
        help="Leave BGE-M3 loaded on exit instead of releasing memory.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DESTRUCTIVE: delete every node before ingesting.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Skip the --reset confirmation prompt."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if not config.paths.corpus_path.is_dir():
        print(
            f"Corpus not found at {config.paths.corpus_path}. Download it first:\n"
            f"  python scripts/download_core_knowledge_stem.py",
            file=sys.stderr,
        )
        return 2

    if not args.skip_embeddings and not config.models.embedding_model_path.is_dir():
        print(
            f"Embedding model missing at {config.models.embedding_model_path}. Run:\n"
            f"  python scripts/download_retrieval_models.py",
            file=sys.stderr,
        )
        return 2

    print(config.summary())
    print()

    try:
        store = Neo4jStore(config.require_neo4j())
        store.connect()
    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    try:
        if args.reset:
            if not args.yes:
                print(
                    f"--reset deletes ALL nodes in database "
                    f"{config.neo4j.database!r}."  # type: ignore[union-attr]
                )
                if input("Type 'reset' to confirm: ").strip() != "reset":
                    print("Aborted; nothing was deleted.")
                    return 1
            summary = reset_graph(store)
            print(f"Reset complete: {summary['nodes_deleted']} nodes deleted.\n")

        ingestor = CorpusIngestor(config, store)
        try:
            stats = ingestor.run(
                grades=tuple(args.grade) if args.grade else None,
                subjects=(
                    tuple(s.lower() for s in args.subject) if args.subject else None
                ),
                limit=args.limit,
                skip_embeddings=args.skip_embeddings,
                force=args.force,
            )
        except KeyboardInterrupt:
            print(
                "\nInterrupted. Re-run the same command to resume; completed "
                "documents will be skipped.",
                file=sys.stderr,
            )
            return 130
        except (CorpusError, EmbeddingError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 4
        finally:
            if not args.keep_embedder:
                ingestor.release_embedder()

        print()
        print(stats.describe())

        print("\nGraph node counts")
        print("-" * 60)
        for label, count in store.node_counts(NODE_LABELS).items():
            print(f"  {label:12s} {count:>10,}")

        print("\nGraph relationship counts")
        print("-" * 60)
        total_relationships = 0
        for rel_type, count in store.relationship_counts(RELATIONSHIP_TYPES).items():
            total_relationships += count
            print(f"  {rel_type:16s} {count:>10,}")
        print(f"  {'TOTAL':16s} {total_relationships:>10,}")

        return 0 if not stats.failures else 5

    except Neo4jUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
