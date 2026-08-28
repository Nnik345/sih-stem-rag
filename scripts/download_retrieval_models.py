#!/usr/bin/env python3
"""Download the retrieval models (BGE-M3 embedder, BGE reranker) into models/.

Only downloads what is missing. The generator model (Qwen3-VL-8B-Instruct) is
never touched by this script.

    python scripts/download_retrieval_models.py
    python scripts/download_retrieval_models.py --force
    python scripts/download_retrieval_models.py --model bge-m3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.logging_utils import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_dir: Path
    # Files that must exist for the download to count as complete.
    required_files: tuple[str, ...]
    # Large artefacts we do not use are skipped: the ONNX export duplicates the
    # PyTorch weights (~2.3 GB) and the README images are irrelevant.
    ignore_patterns: tuple[str, ...] = field(default=())


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="bge-m3",
        repo_id="BAAI/bge-m3",
        local_dir=PROJECT_ROOT / "models" / "bge-m3",
        required_files=("config.json", "tokenizer.json", "pytorch_model.bin"),
        ignore_patterns=("onnx/*", "imgs/*", "*.jpg", "*.webp", "*.DS_Store"),
    ),
    ModelSpec(
        key="bge-reranker-v2-m3",
        repo_id="BAAI/bge-reranker-v2-m3",
        local_dir=PROJECT_ROOT / "models" / "bge-reranker-v2-m3",
        required_files=("config.json", "tokenizer.json", "model.safetensors"),
        ignore_patterns=("assets/*", "*.png", "*.DS_Store"),
    ),
)


def is_present(spec: ModelSpec) -> bool:
    return spec.local_dir.is_dir() and all(
        (spec.local_dir / name).is_file() for name in spec.required_files
    )


def directory_size_mb(path: Path) -> float:
    if not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e6


def download(spec: ModelSpec, force: bool) -> bool:
    """Return True if the model is available locally after this call."""
    if is_present(spec) and not force:
        LOGGER.info(
            "%s already present at %s (%.0f MB) - skipping download",
            spec.repo_id,
            spec.local_dir,
            directory_size_mb(spec.local_dir),
        )
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        LOGGER.error("huggingface-hub is not installed: %s", exc)
        return False

    LOGGER.info("Downloading %s -> %s", spec.repo_id, spec.local_dir)
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(spec.local_dir),
            ignore_patterns=list(spec.ignore_patterns) or None,
        )
    except Exception as exc:
        LOGGER.error("Download of %s failed: %s", spec.repo_id, exc)
        return False

    missing = [n for n in spec.required_files if not (spec.local_dir / n).is_file()]
    if missing:
        LOGGER.error(
            "%s downloaded but required files are missing: %s",
            spec.repo_id,
            ", ".join(missing),
        )
        return False

    LOGGER.info(
        "%s ready at %s (%.0f MB)",
        spec.repo_id,
        spec.local_dir,
        directory_size_mb(spec.local_dir),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[m.key for m in MODELS],
        action="append",
        help="Download only the named model (repeatable). Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model already looks complete.",
    )
    args = parser.parse_args()

    setup_logging()
    selected = [m for m in MODELS if not args.model or m.key in args.model]

    results = {spec.repo_id: download(spec, args.force) for spec in selected}

    print("\nRetrieval model status")
    print("-" * 60)
    for spec in selected:
        state = "OK" if results[spec.repo_id] else "FAILED"
        print(f"  {state:7s} {spec.repo_id:26s} -> {spec.local_dir}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
