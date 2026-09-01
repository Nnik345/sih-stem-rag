"""Place Hugging Face weights from live GPU and RAM, not a fixed card size.

``device_map="auto"`` without a cap fills the accelerator with layers and then
OOMs during vision prefill. This module always leaves generation headroom on
the GPU and sends leftover layers to system RAM.

The split follows the machine:

* ~32 GiB VRAM: an 8B bf16 tutor (~16 GiB) plus headroom still fits, so weights
  stay on the GPU.
* ~12 GiB VRAM: part of the tutor stays on the GPU; the rest is CPU-offloaded.
* ~8 GiB VRAM: most layers sit in RAM; only a thin GPU slice remains.
"""

from __future__ import annotations

import gc
from typing import Any

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# Extra VRAM held back beyond the configured floor, as a fraction of *total*
# device memory. A large card therefore reserves more absolute headroom (KV
# cache, vision) while still leaving enough for an 8B model to sit fully on GPU.
DEFAULT_HEADROOM_FRACTION = 0.25
DEFAULT_HEADROOM_CAP_GIB = 10.0
DEFAULT_CPU_RAM_FRACTION = 0.80
MIN_GPU_WEIGHT_GIB = 1.5
MIN_HEADROOM_GIB = 2.0

# Qwen-VL processors count visual tokens in 28x28 patches. ``None`` leaves the
# processor default (full resolution) for large cards.
PIXELS_HIGH: int | None = None
PIXELS_MEDIUM = 768 * 28 * 28
PIXELS_LOW = 512 * 28 * 28
PIXELS_MIN = 256 * 28 * 28


def physical_ram_gib() -> float:
    import os

    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def cuda_memory_gib(device: int = 0) -> tuple[float, float] | None:
    """Return ``(free_gib, total_gib)`` for ``device``, or ``None`` without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info(device)
        return free_b / 1024**3, total_b / 1024**3
    except Exception as exc:
        LOGGER.debug("Could not query device memory (%s)", exc)
        return None


def empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generation_headroom_gib(
    total_vram_gib: float,
    *,
    reserve_floor_gib: float,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    headroom_cap_gib: float = DEFAULT_HEADROOM_CAP_GIB,
) -> float:
    """VRAM kept free for prefill, vision activations, and the KV cache.

    Scales with the card: 32 GiB reserves more bytes than 8 GiB, but a smaller
    *fraction* of a large card so an 8B checkpoint can still run fully on GPU.
    """
    scaled = min(headroom_cap_gib, total_vram_gib * headroom_fraction)
    return max(reserve_floor_gib, scaled, MIN_HEADROOM_GIB)


def gpu_weight_budget_gib(
    free_vram_gib: float,
    headroom_gib: float,
    *,
    min_gpu_weight_gib: float = MIN_GPU_WEIGHT_GIB,
    min_headroom_gib: float = MIN_HEADROOM_GIB,
) -> float:
    """GiB of weights allowed on the GPU after holding back generation headroom."""
    budget = free_vram_gib - headroom_gib
    if budget >= min_gpu_weight_gib:
        return budget
    # Tight card, or another process is using it. Still cap: returning no map
    # lets device_map pack the GPU full of weights, which OOMs on generate().
    squeezed = free_vram_gib - min_headroom_gib
    if squeezed > 0:
        return max(0.5, min(min_gpu_weight_gib, squeezed))
    return max(0.5, free_vram_gib * 0.25)


def cuda_max_memory_map(
    *,
    vram_reserve_gib: float,
    cpu_ram_fraction: float = DEFAULT_CPU_RAM_FRACTION,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    headroom_cap_gib: float = DEFAULT_HEADROOM_CAP_GIB,
    device: int = 0,
    cuda_stats: tuple[float, float] | None = None,
    total_ram_gib: float | None = None,
) -> dict[Any, str] | None:
    """Accelerate ``max_memory`` from this machine's free VRAM and physical RAM.

    Always includes a ``cpu`` budget so leftover layers go to system RAM.
    Returns ``None`` only when there is no CUDA device or capping is disabled
    (``vram_reserve_gib <= 0``).
    """
    if vram_reserve_gib <= 0:
        return None
    stats = cuda_stats if cuda_stats is not None else cuda_memory_gib(device)
    if stats is None:
        return None
    free_vram, total_vram = stats
    headroom = generation_headroom_gib(
        total_vram,
        reserve_floor_gib=vram_reserve_gib,
        headroom_fraction=headroom_fraction,
        headroom_cap_gib=headroom_cap_gib,
    )
    gpu_budget = gpu_weight_budget_gib(free_vram, headroom)
    ram = physical_ram_gib() if total_ram_gib is None else total_ram_gib
    cpu_budget = max(4.0, ram * cpu_ram_fraction)
    LOGGER.info(
        "Weight placement: %.1f GiB GPU (%.1f GiB free of %.1f GiB, %.1f GiB held "
        "back for generation) + %.0f GiB system RAM",
        gpu_budget,
        free_vram,
        total_vram,
        headroom,
        cpu_budget,
    )
    return {device: f"{gpu_budget:.1f}GiB", "cpu": f"{cpu_budget:.0f}GiB"}


def tighter_max_memory_map(max_memory: dict[Any, str], *, gpu_gib: float = 1.2) -> dict[Any, str]:
    """Retry map: same RAM budget, less GPU, after a capped load fails."""
    tighter = dict(max_memory)
    gpu_keys = [key for key in tighter if key != "cpu"]
    if gpu_keys:
        tighter[gpu_keys[0]] = f"{gpu_gib:.1f}GiB"
    return tighter


def adaptive_max_image_pixels(
    total_vram_gib: float | None,
    override: int = 0,
) -> int | None:
    """Cap Qwen-VL image tokens on small GPUs.

    ``override > 0`` wins. ``None`` means leave the processor default.
    """
    if override > 0:
        return override
    if total_vram_gib is None:
        return PIXELS_LOW
    if total_vram_gib >= 24:
        return PIXELS_HIGH
    if total_vram_gib >= 16:
        return PIXELS_MEDIUM
    if total_vram_gib >= 10:
        return PIXELS_LOW
    return PIXELS_MIN


def generator_should_yield_gpu(free_gib: float, total_gib: float) -> bool:
    """Whether the resident 8B tutor must be unloaded so retrieval can use the GPU.

    A 32 GiB card can keep the tutor loaded beside the 2B rewriter. An 8–12 GiB
    card cannot.
    """
    return total_gib < 20.0 or free_gib < 8.0
