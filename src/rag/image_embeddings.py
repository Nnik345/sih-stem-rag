"""SigLIP image embeddings for textbook figure matching.

Vectors are L2-normalised so cosine similarity and the Neo4j vector index
agree. Dimensionality is read from the checkpoint config, never hardcoded.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ModelConfig
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


class ImageEmbeddingError(RuntimeError):
    """Raised when the vision encoder is missing or cannot encode."""


def _resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_image_hidden_size(model_path: Path) -> int:
    """Projection / vision hidden size from the SigLIP checkpoint config."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ImageEmbeddingError(
            f"Image embedding model config not found: {config_path}. Run "
            f"python scripts/download_retrieval_models.py --model siglip-base-patch16-224"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImageEmbeddingError(f"Could not read {config_path}: {exc}") from exc

    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    for candidate in (
        config.get("projection_dim"),
        vision.get("projection_dim"),
        vision.get("hidden_size"),
        config.get("hidden_size"),
    ):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    raise ImageEmbeddingError(
        f"{config_path} does not declare a usable image embedding size"
    )


class SiglipImageEmbedder:
    """Lazy-loading SigLIP encoder that can be released between pipeline stages."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._processor = None
        self._device: str | None = None
        self._dimension: int | None = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "SiglipImageEmbedder":
        return cls(
            config.image_embedding_model_path,
            device=config.embedding_device,
            batch_size=config.embedding_batch_size,
        )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = read_image_hidden_size(self.model_path)
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise ImageEmbeddingError(
                f"Image embedding model directory not found: {self.model_path}. "
                "Download it with: python scripts/download_retrieval_models.py "
                "--model siglip-base-patch16-224"
            )
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImageEmbeddingError(
                f"Transformers/torch is required for SigLIP ({exc})"
            ) from exc

        device = _resolve_device(self.requested_device)
        LOGGER.info("Loading SigLIP from %s on %s", self.model_path, device)
        try:
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            model = AutoModel.from_pretrained(str(self.model_path))
            model.eval()
            model.to(device)
        except Exception as exc:
            raise ImageEmbeddingError(
                f"Could not load SigLIP from {self.model_path}: {exc}"
            ) from exc

        self._model = model
        self._device = device
        if hasattr(model, "config"):
            projection = getattr(model.config, "projection_dim", None)
            if isinstance(projection, int) and projection > 0:
                self._dimension = projection
        if self._dimension is None:
            self._dimension = read_image_hidden_size(self.model_path)
        LOGGER.info("SigLIP ready (dimension=%d)", self._dimension)

    def unload(self) -> None:
        if self._model is None and self._processor is None:
            return
        LOGGER.info("Releasing SigLIP from %s", self._device)
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        self._device = None

    def __enter__(self) -> "SiglipImageEmbedder":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    def _encode_batch(self, paths: Sequence[Path]) -> np.ndarray:
        import torch
        from PIL import Image

        assert self._model is not None and self._processor is not None
        images = []
        for path in paths:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_image_features(**inputs)
            normalized = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        return normalized.cpu().numpy().astype(np.float32)

    def encode_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Encode local image files into L2-normalised dense vectors."""
        items = [Path(path) for path in paths]
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)
        missing = [str(path) for path in items if not path.is_file()]
        if missing:
            raise ImageEmbeddingError(
                f"Image file(s) not found: {', '.join(missing[:5])}"
            )
        self.load()
        vectors: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            try:
                vectors.append(self._encode_batch(batch))
            except Exception as exc:
                raise ImageEmbeddingError(
                    f"Image embedding failed at offset {start}: {exc}"
                ) from exc
        return np.vstack(vectors)

    def encode_image(self, path: str | Path) -> np.ndarray:
        """Single-image query vector."""
        matrix = self.encode_paths([path])
        return matrix[0]
