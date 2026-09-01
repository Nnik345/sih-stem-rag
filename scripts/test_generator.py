#!/usr/bin/env python3
"""Sanity test for text-only generation with the local Qwen3-VL-8B-Instruct model.

Loads through :class:`rag.generator.QwenGenerator` so weight placement matches
the tutor path (GPU/RAM split from live device memory).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import load_config  # noqa: E402
from rag.generator import QwenGenerator  # noqa: E402

USER_PROMPT = (
    "Explain why plants need sunlight in simple words for a Grade 3 student."
)


def main() -> None:
    config = load_config(require_neo4j=False)
    model_path = config.models.generator_model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    print("Loading generator (weights follow this machine's VRAM and RAM)...")
    generator = QwenGenerator.from_config(config.models)
    try:
        generator.load()
        print("Model loaded successfully.")
        print("Generating response...")
        print("\n---\n")
        print("## TEST QUERY\n")
        print(USER_PROMPT)
        print("\n---\n")
        print("## MODEL RESPONSE\n")
        generator.generate(
            [{"role": "user", "content": USER_PROMPT}],
            on_token=lambda piece: print(piece, end="", flush=True),
        )
        print()
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
    finally:
        generator.unload()


if __name__ == "__main__":
    main()
