#!/usr/bin/env python3
"""Embed existing Neo4j :Image nodes with SigLIP. Does not re-parse PDFs.

Skips images whose embedding_version already matches the current SigLIP
generation. Student uploads are never written here.

    python scripts/embed_images.py
    python scripts/embed_images.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import ConfigError, load_config  # noqa: E402
from rag.image_index import embed_curriculum_images  # noqa: E402
from rag.logging_utils import get_logger, setup_logging  # noqa: E402
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError  # noqa: E402

LOGGER = get_logger("embed_images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every on-disk Image even if embedding_version matches.",
    )
    parser.add_argument("--log-level", default="INFO")
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
        stats = embed_curriculum_images(config, store, force=args.force)
    finally:
        store.close()

    print("Image embedding job")
    print("-" * 40)
    for key, value in stats.items():
        print(f"  {key:16s}: {value}")
    return 1 if int(stats.get("errors") or 0) and not stats.get("embedded") else 0


if __name__ == "__main__":
    raise SystemExit(main())
