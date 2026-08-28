"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

The reranker scores (query, chunk_text) pairs jointly, which is far more
accurate than comparing independent embeddings, but too slow to run over the
whole corpus. It therefore sits after rank fusion: ~20 fused candidates in,
~5 ranked chunks out.

Scores are raw logits: unbounded, with 0 corresponding to sigmoid 0.5. They are
comparable within one query but not across queries, so the evidence gate treats
its threshold as a tunable heuristic rather than a calibrated probability.
Loading is lazy and :meth:`unload` frees memory before the generator is loaded.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ModelConfig
from .logging_utils import get_logger
from .schemas import RetrievedChunk

LOGGER = get_logger(__name__)


class RerankerError(RuntimeError):
    """Raised when the reranker model is missing or scoring fails."""


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


def _resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEReranker:
    """Lazy-loading cross-encoder reranker that can be released to free memory."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "BGEReranker":
        return cls(
            config.reranker_model_path,
            device=config.reranker_device,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise RerankerError(
                f"Reranker model directory not found: {self.model_path}. Run "
                f"python scripts/download_retrieval_models.py"
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RerankerError(f"torch/transformers unavailable: {exc}") from exc

        device = _resolve_device(self.requested_device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        LOGGER.info(
            "Loading BGE reranker from %s (device=%s, dtype=%s)",
            self.model_path,
            device,
            dtype,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), use_fast=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), dtype=dtype
            )
            model.eval()
            model.to(device)
        except Exception as exc:
            raise RerankerError(
                f"Could not load reranker from {self.model_path}: {exc}"
            ) from exc

        self._model = model
        self._device = device
        LOGGER.info("BGE reranker ready")

    def unload(self) -> None:
        if self._model is None:
            return
        LOGGER.info("Releasing BGE reranker from %s", self._device)
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        self._device = None

    def __enter__(self) -> "BGEReranker":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevance logits for each (query, text) pair, in input order."""
        if not texts:
            return []
        self.load()
        import torch

        assert self._model is not None and self._tokenizer is not None
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            pairs = [[query, text if text.strip() else " "] for text in batch]
            try:
                encoded = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                with torch.inference_mode():
                    logits = self._model(**encoded).logits
                scores.extend(logits.view(-1).float().cpu().tolist())
            except Exception as exc:
                raise RerankerError(
                    f"Reranking failed for batch at offset {start} "
                    f"(size {len(batch)}): {exc}"
                ) from exc
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Score and reorder candidates, annotating each with its rerank score.

        The candidate objects are mutated in place so that every earlier signal
        (dense/fulltext/graph ranks, RRF score) is preserved for later analysis.
        """
        if not candidates:
            return []

        scores = self.score_pairs(query, [c.text for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ordered = sorted(
            candidates,
            key=lambda c: (
                c.rerank_score if c.rerank_score is not None else float("-inf")
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(ordered, start=1):
            candidate.rerank_rank = rank

        LOGGER.info(
            "Reranked %d candidates (top score %.4f, bottom %.4f)",
            len(ordered),
            ordered[0].rerank_score or 0.0,
            ordered[-1].rerank_score or 0.0,
        )
        return ordered[:top_k] if top_k else ordered
