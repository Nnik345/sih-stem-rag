"""BGE-M3 embeddings for retrieval.

Dense embeddings only are used: the CLS hidden state, L2-normalised so
that cosine similarity and dot product coincide, which is what the Neo4j vector
index expects.

Why Transformers instead of FlagEmbedding
----------------------------------------
The generator depends on a git checkout of Transformers (Qwen3-VL support), and
installing `FlagEmbedding` would drag in sentence-transformers / peft / datasets
with their own Transformers pin, risking a downgrade that breaks the generator.
BGE-M3's dense head is just "CLS state, then normalise", so it is reproduced
here directly against the local checkpoint. Nothing about the weights changes.

Extending to sparse / multi-vector later
----------------------------------------
:class:`EmbeddingOutput` already carries ``sparse`` and ``colbert`` slots, and
:meth:`BGEM3Embedder.encode` takes the requested modes as an argument. The
checkpoint's ``sparse_linear.pt`` and ``colbert_linear.pt`` heads are downloaded
alongside the model, so adding those modes means implementing the two extra
projections and a matching index -- not rewriting the retrieval pipeline.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import ModelConfig
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

MODE_DENSE = "dense"
MODE_SPARSE = "sparse"
MODE_COLBERT = "colbert"
IMPLEMENTED_MODES = (MODE_DENSE,)


class EmbeddingError(RuntimeError):
    """Raised when the embedding model is missing or cannot encode."""


@dataclass
class EmbeddingOutput:
    """Container for the embedding representations of a batch of texts.

    Only ``dense`` is populated today; the other fields are reserved so that
    downstream code can start consuming them without a signature change.
    """

    dense: np.ndarray | None = None
    sparse: list[dict[int, float]] | None = None
    colbert: list[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return 0 if self.dense is None else int(self.dense.shape[0])


def _resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_hidden_size(model_path: Path) -> int:
    """Dense dimensionality straight from the checkpoint's config.

    Read from the model directory rather than assumed, so the Neo4j vector index
    can be created before the weights are loaded.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise EmbeddingError(
            f"Embedding model config not found: {config_path}. Run "
            f"python scripts/download_retrieval_models.py"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingError(f"Could not read {config_path}: {exc}") from exc

    hidden_size = config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise EmbeddingError(
            f"{config_path} does not declare a usable hidden_size: {hidden_size!r}"
        )
    return hidden_size


class BGEM3Embedder:
    """Lazy-loading BGE-M3 encoder that can be released between pipeline stages.

    The embedder is not held open longer than needed. Use it as a context manager
    or call :meth:`unload` explicitly.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1024,
        modes: Sequence[str] = (MODE_DENSE,),
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.batch_size = batch_size
        self.max_length = max_length

        unsupported = [m for m in modes if m not in IMPLEMENTED_MODES]
        if unsupported:
            raise EmbeddingError(
                f"Embedding mode(s) {unsupported} are not implemented. "
                f"Available: {list(IMPLEMENTED_MODES)}. Sparse and ColBERT modes "
                f"are planned future work."
            )
        self.modes = tuple(modes)

        self._model = None
        self._tokenizer = None
        self._device: str | None = None
        self._dimension: int | None = None

    # -- lifecycle --------------------------------------------------------- #

    @classmethod
    def from_config(cls, config: ModelConfig) -> "BGEM3Embedder":
        return cls(
            config.embedding_model_path,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
            max_length=config.embedding_max_length,
            modes=(config.embedding_mode,),
        )

    @property
    def dimension(self) -> int:
        """Dense embedding size, from the loaded model when available."""
        if self._dimension is None:
            if self._model is not None:
                self._dimension = int(self._model.config.hidden_size)
            else:
                self._dimension = read_hidden_size(self.model_path)
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise EmbeddingError(
                f"Embedding model directory not found: {self.model_path}. Run "
                f"python scripts/download_retrieval_models.py"
            )

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise EmbeddingError(f"torch/transformers unavailable: {exc}") from exc

        device = _resolve_device(self.requested_device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        LOGGER.info(
            "Loading BGE-M3 from %s (device=%s, dtype=%s, batch=%d, max_len=%d)",
            self.model_path,
            device,
            dtype,
            self.batch_size,
            self.max_length,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), use_fast=True
            )
            model = AutoModel.from_pretrained(str(self.model_path), dtype=dtype)
            model.eval()
            model.to(device)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model from {self.model_path}: {exc}"
            ) from exc

        self._model = model
        self._device = device
        self._dimension = int(model.config.hidden_size)
        LOGGER.info("BGE-M3 ready (dense dimension=%d)", self._dimension)

    def unload(self) -> None:
        """Release the model and free memory before another model is loaded."""
        if self._model is None:
            return
        LOGGER.info("Releasing BGE-M3 from %s", self._device)
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

    def __enter__(self) -> "BGEM3Embedder":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    # -- encoding ---------------------------------------------------------- #

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch

        assert self._model is not None and self._tokenizer is not None
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**encoded)
            # BGE-M3's dense representation is the CLS hidden state.
            cls = outputs.last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(cls.float(), p=2, dim=-1)
        return normalized.cpu().numpy().astype(np.float32)

    def encode(
        self,
        texts: Iterable[str],
        *,
        modes: Sequence[str] | None = None,
        progress_every: int = 0,
    ) -> EmbeddingOutput:
        """Encode texts into normalized dense vectors, in batches."""
        requested = tuple(modes or self.modes)
        unsupported = [m for m in requested if m not in IMPLEMENTED_MODES]
        if unsupported:
            raise EmbeddingError(
                f"Embedding mode(s) {unsupported} are not implemented"
            )

        items = [text if text.strip() else " " for text in texts]
        if not items:
            return EmbeddingOutput(
                dense=np.zeros((0, self.dimension), dtype=np.float32),
                metadata={"count": 0, "modes": list(requested)},
            )

        self.load()
        vectors: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            try:
                vectors.append(self._encode_batch(batch))
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding failed for batch at offset {start} "
                    f"(size {len(batch)}): {exc}"
                ) from exc
            if progress_every and (start // self.batch_size) % progress_every == 0:
                LOGGER.debug("Embedded %d/%d texts", min(start + len(batch), len(items)), len(items))

        dense = np.vstack(vectors)
        return EmbeddingOutput(
            dense=dense,
            metadata={
                "count": int(dense.shape[0]),
                "dimension": int(dense.shape[1]),
                "modes": list(requested),
                "device": self._device,
            },
        )

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Dense vectors for corpus chunks."""
        output = self.encode(texts)
        assert output.dense is not None
        return output.dense

    def encode_query(self, query: str) -> np.ndarray:
        """Dense vector for a single query.

        BGE-M3 needs no instruction prefix, so queries and documents go through
        exactly the same encoder.
        """
        output = self.encode([query])
        assert output.dense is not None
        return output.dense[0]
