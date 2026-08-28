#!/usr/bin/env python3
"""Sanity test for text-only generation with the local Qwen3-VL-8B-Instruct model."""

from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, TextStreamer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-vl-8b-instruct"

USER_PROMPT = (
    "Explain why plants need sunlight in simple words for a Grade 3 student."
)
MAX_NEW_TOKENS = 150


def main() -> None:
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Local model directory not found: {MODEL_PATH}"
        )

    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        print("Loading model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
        )
        model.eval()
        print("Model loaded successfully.")

        messages = [{"role": "user", "content": USER_PROMPT}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Offloaded layers make generation slow, so stream tokens to show progress.
        streamer = TextStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        print("Generating response...")
        print("\n---\n")
        print("## TEST QUERY\n")
        print(USER_PROMPT)
        print("\n---\n")
        print("## MODEL RESPONSE\n")

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                streamer=streamer,
            )
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
