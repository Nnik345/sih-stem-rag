"""Qwen3-VL-8B-Instruct generator wrapper.

This is the single place the generator is loaded. It uses the original
non-quantized checkpoint in ``models/qwen3-vl-8b-instruct`` with
``dtype="auto"`` and ``device_map="auto"``, plus token streaming.

``scripts/test_generator.py`` calls into this module rather than reimplementing
model loading.

Multimodal note: the processor accepts images, and :meth:`stream` will pass them
through when given. Retrieved image paths travel as metadata only by default.
"""

from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import ModelConfig
from .logging_utils import get_logger

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
        vram_reserve_gib: float = 3.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.settings = settings or GenerationSettings()
        self.vram_reserve_gib = vram_reserve_gib
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
                f"it with: huggingface-cli download Qwen/Qwen3-VL-8B-Instruct "
                f"--local-dir {self.model_path}"
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
            self._processor = AutoProcessor.from_pretrained(str(self.model_path))
            LOGGER.info(
                "Loading Qwen3-VL weights (device_map=auto; layers that do not fit "
                "in device memory are offloaded to CPU, so generation is slow)"
            )
            max_memory = self._max_memory_map()
            load_kwargs: dict[str, Any] = {"dtype": "auto", "device_map": "auto"}
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
                LOGGER.info(
                    "Capping generator weights at %s so ~%.1f GiB of device "
                    "memory stays free for the prefill logits and KV cache; the "
                    "remaining layers are offloaded to CPU",
                    max_memory[0],
                    self.vram_reserve_gib,
                )
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            except Exception as capped_error:
                if max_memory is None:
                    raise
                # Fall back to the plain, already-validated load rather than
                # failing outright because of the memory cap.
                LOGGER.warning(
                    "Capped load failed (%s); retrying with plain device_map=auto",
                    capped_error,
                )
                load_kwargs.pop("max_memory")
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(self.model_path), **load_kwargs
                )
            model.eval()
        except Exception as exc:
            raise GeneratorError(
                f"Could not load the generator from {self.model_path}: {exc}"
            ) from exc

        self._model = model
        LOGGER.info("Qwen3-VL ready (device=%s)", getattr(model, "device", "unknown"))

    def _max_memory_map(self) -> dict[Any, str] | None:
        """Per-device weight budgets, holding back memory for generation.

        Returns ``None`` when there is no CUDA device or the reserve leaves too
        little to be worth capping, in which case plain ``device_map="auto"``
        is used. Accelerate only considers devices present in this map, so the
        CPU entry is required for the offloaded layers.
        """
        if self.vram_reserve_gib <= 0:
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            # Free, not total: other processes may already hold memory that the
            # model can never use.
            free_vram = torch.cuda.mem_get_info(0)[0] / 1024**3
        except Exception as exc:
            LOGGER.debug("Could not query device memory (%s); not capping", exc)
            return None

        gpu_budget = free_vram - self.vram_reserve_gib
        if gpu_budget < 2.0:
            LOGGER.warning(
                "Only %.1f GiB of device memory is free and %.1f GiB is reserved for "
                "generation, leaving %.1f GiB for weights; not capping",
                free_vram,
                self.vram_reserve_gib,
                gpu_budget,
            )
            return None

        # Offloaded layers live in RAM. Total (not currently-free) physical RAM
        # is the right basis: the page cache is reclaimable.
        total_ram = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )
        return {0: f"{gpu_budget:.1f}GiB", "cpu": f"{total_ram * 0.6:.0f}GiB"}

    def unload(self) -> None:
        if self._model is None:
            return
        LOGGER.info("Releasing Qwen3-VL")
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

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
    ) -> list[dict[str, Any]]:
        """Convert plain {role, content} messages to Qwen3-VL content lists."""
        converted: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": message["content"]}
            ]
            converted.append({"role": message["role"], "content": content})
        if image_paths and converted:
            last = converted[-1]
            last["content"] = [
                {"type": "image", "image": str(path)} for path in image_paths
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
        return inputs.to(self._model.device)

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
