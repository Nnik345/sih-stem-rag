#!/usr/bin/env python3
"""Download both Qwen3-VL checkpoints used by this project.

Fetches the 8B Socratic tutor and the 2B query rewriter. Already-complete
directories are skipped. Retrieval embedder/reranker stay in
``scripts/download_retrieval_models.py`` (which also fetches the 2B rewriter
and mentions SigLIP for textbook figure matching).

    python scripts/download_qwen_models.py
    python scripts/download_qwen_models.py --force
    python scripts/download_qwen_models.py --model qwen3-vl-2b-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_retrieval_models import (  # noqa: E402
    ModelSpec,
    download,
    LOGGER,
)
from rag.logging_utils import setup_logging  # noqa: E402

QWEN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen3-vl-2b-instruct",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        local_dir=PROJECT_ROOT / "models" / "qwen3-vl-2b-instruct",
        required_files=("config.json", "tokenizer.json"),
        ignore_patterns=("*.png", "*.jpg", "*.webp", "*.DS_Store"),
    ),
    ModelSpec(
        key="qwen3-vl-8b-instruct",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir=PROJECT_ROOT / "models" / "qwen3-vl-8b-instruct",
        required_files=("config.json", "tokenizer.json"),
        ignore_patterns=("*.png", "*.jpg", "*.webp", "*.DS_Store"),
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[m.key for m in QWEN_MODELS],
        action="append",
        help="Download only the named checkpoint (repeatable). Default: both.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model already looks complete.",
    )
    args = parser.parse_args()

    setup_logging()
    selected = [m for m in QWEN_MODELS if not args.model or m.key in args.model]
    results = {spec.repo_id: download(spec, args.force) for spec in selected}

    print("\nQwen model status")
    print("-" * 60)
    for spec in selected:
        state = "OK" if results[spec.repo_id] else "FAILED"
        print(f"  {state:7s} {spec.repo_id:32s} -> {spec.local_dir}")

    if not all(results.values()):
        LOGGER.error("One or more Qwen downloads failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
