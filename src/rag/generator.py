"""Qwen3-VL-8B-Instruct generator wrapper.

This is the single place the generator is loaded. It uses the original
non-quantized checkpoint in ``models/qwen3-vl-8b-instruct`` with
``dtype="auto"`` and ``device_map="auto"``. Weight placement follows free VRAM
and physical RAM: leftover layers are offloaded to CPU, so a 32 GiB GPU can run
the 8B tutor on-device while an 8 GiB GPU keeps most of it in RAM.

``scripts/test_generator.py`` calls into this module rather than reimplementing
model loading.

Multimodal note: the processor accepts images, and :meth:`stream` will pass them
through when given. Retrieved image paths travel as metadata only by default.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import ModelConfig
from .image_paths import select_vision_image_paths
from .logging_utils import get_logger
from .model_memory import (
    adaptive_max_image_pixels,
    cuda_max_memory_map,
    cuda_memory_gib,
    empty_cuda_cache,
    tighter_max_memory_map,
)

LOGGER = get_logger(__name__)


class GeneratorError(RuntimeError):
    """Raised when the generator model is missing or generation fails."""


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 640
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        if not self.do_sample:
            return {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


class QwenGenerator:
    """Lazy-loading wrapper around the local Qwen3-VL-8B-Instruct checkpoint."""

    def __init__(
        self,
        model_path: Path,
        *,
        settings: GenerationSettings | None = None,
        vram_reserve_gib: float = 5.5,
        vram_headroom_fraction: float = 0.25,
        cpu_ram_fraction: float = 0.80,
        max_image_pixels: int = 0,
    ) -> None:
        self.model_path = Path(model_path)
        self.settings = settings or GenerationSettings()
        self.vram_reserve_gib = vram_reserve_gib
        self.vram_headroom_fraction = vram_headroom_fraction
        self.cpu_ram_fraction = cpu_ram_fraction
        self.max_image_pixels = max_image_pixels
        self._model = None
        self._processor = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> "QwenGenerator":
        return cls(
            config.generator_model_path,
            settings=GenerationSettings(
                max_new_tokens=config.generator_max_new_tokens,
                temperature=config.generator_temperature,
            ),
            vram_reserve_gib=config.generator_vram_reserve_gib,
            vram_headroom_fraction=config.generator_vram_headroom_fraction,
            cpu_ram_fraction=config.generator_cpu_ram_fraction,
            max_image_pixels=config.generator_max_image_pixels,
        )

    # -- lifecycle --------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise GeneratorError(
                f"Generator model directory not found: {self.model_path}. Download "
                f"it with: python scripts/download_qwen_models.py"
            )
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise GeneratorError(
                f"Transformers does not expose Qwen3VLForConditionalGeneration "
                f"({exc}). Qwen3-VL needs the git build: "
                f"pip install git+https://github.com/huggingface/transformers"
            ) from exc

        LOGGER.info("Loading Qwen3-VL processor from %s", self.model_path)
        try:
            empty_cuda_cache()
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            self._limit_processor_pixels()
            LOGGER.info(
                "Loading Qwen3-VL weights (device_map=auto; layers that do not fit "
                "in device memory are offloaded to CPU RAM, so generation is slow "
                "on small GPUs)"
            )
            max_memory = self._max_memory_map()
            load_kwargs: dict[str, Any] = {
                "dtype": "auto",
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            except Exception as capped_error:
                if max_memory is None:
                    raise
                # Never retry without a cap: uncapped device_map fills the GPU
                # with weights and OOMs on vision prefill. Offload more instead.
                LOGGER.warning(
                    "Capped load failed (%s); retrying with more CPU offload",
                    capped_error,
                )
                load_kwargs["max_memory"] = tighter_max_memory_map(max_memory)
                empty_cuda_cache()
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            model.eval()
        except Exception as exc:
            self._processor = None
            raise GeneratorError(
                f"Could not load the generator from {self.model_path}: {exc}"
            ) from exc

        self._model = model
        LOGGER.info("Qwen3-VL ready (device=%s)", getattr(model, "device", "unknown"))

    def _max_memory_map(self) -> dict[Any, str] | None:
        """Per-device weight budgets from this machine's free VRAM and RAM."""
        return cuda_max_memory_map(
            vram_reserve_gib=self.vram_reserve_gib,
            cpu_ram_fraction=self.cpu_ram_fraction,
            headroom_fraction=self.vram_headroom_fraction,
        )

    def _limit_processor_pixels(self) -> None:
        stats = cuda_memory_gib()
        total = stats[1] if stats else None
        max_pixels = adaptive_max_image_pixels(total, self.max_image_pixels)
        if max_pixels is None or self._processor is None:
            return
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels
            LOGGER.info(
                "Capping tutor vision at max_pixels=%d for this GPU", max_pixels
            )

    def unload(self) -> None:
        if self._model is None:
            return
        LOGGER.info("Releasing Qwen3-VL")
        self._model = None
        self._processor = None
        empty_cuda_cache()

    def __enter__(self) -> "QwenGenerator":
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    # -- generation -------------------------------------------------------- #

    @staticmethod
    def _to_chat_messages(
        messages: Sequence[dict[str, str]],
        image_paths: Sequence[str] = (),
        *,
        images_dir: Path | None = None,
        extra_allowed: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Convert plain {role, content} messages to Qwen3-VL content lists."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": message["content"]}
            ]
            converted.append({"role": message["role"], "content": content})
        allowed = list(image_paths)
        if images_dir is not None:
            allowed = select_vision_image_paths(
                image_paths, images_dir=images_dir, extra_allowed=extra_allowed
            )
        if allowed and converted:
            last = converted[-1]
            last["content"] = [
                {"type": "image", "image": str(path)} for path in allowed
            ] + last["content"]
        return converted

    def _prepare_inputs(
        self,
        messages: Sequence[dict[str, str]],
        image_paths: Sequence[str] = (),
    ) -> Any:
        assert self._processor is not None and self._model is not None
        chat = self._to_chat_messages(messages, image_paths)
        inputs = self._processor.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self._activation_device())

    def _activation_device(self) -> Any:
        """First device Accelerate placed, so inputs match a split model."""
        import torch

        mapping = getattr(self._model, "hf_device_map", None)
        if mapping:
            for loc in mapping.values():
                if loc in ("cpu", "disk"):
                    continue
                if isinstance(loc, int):
                    return torch.device(f"cuda:{loc}")
                if isinstance(loc, str) and loc.startswith("cuda"):
                    return torch.device(loc)
            return torch.device("cpu")
        device = getattr(self._model, "device", None)
        return device if device is not None else torch.device("cpu")

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        settings: GenerationSettings | None = None,
        image_paths: Sequence[str] = (),
    ) -> Iterator[str]:
        """Yield generated text incrementally as the model produces it."""
        self.load()
        import torch
        from transformers import TextIteratorStreamer

        assert self._processor is not None and self._model is not None
        active = settings or self.settings
        inputs = self._prepare_inputs(messages, image_paths)
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        error: list[BaseException] = []

        def _generate() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(  # type: ignore[union-attr]
                        **inputs,
                        streamer=streamer,
                        **active.to_kwargs(),
                    )
            except BaseException as exc:  # surfaced to the consumer below
                error.append(exc)
            finally:
                streamer.end()

        worker = threading.Thread(target=_generate, name="qwen-generate", daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            worker.join()

        if error:
            raise GeneratorError(f"Generation failed: {error[0]}") from error[0]

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        settings: GenerationSettings | None = None,
        image_paths: Sequence[str] = (),
        on_token: Any = None,
    ) -> str:
        """Generate a full response, optionally forwarding each piece live.

        Streaming is always used internally so callers can show progress; pass
        ``on_token=print`` (or any callable) to display tokens as they arrive.
        """
        pieces: list[str] = []
        for piece in self.stream(
            messages, settings=settings, image_paths=image_paths
        ):
            pieces.append(piece)
            if on_token is not None:
                on_token(piece)
        return "".join(pieces)
