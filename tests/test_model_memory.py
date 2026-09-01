"""Hardware-aware GPU/RAM weight placement. No live model load."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.config import load_config
from rag.model_memory import (
    PIXELS_LOW,
    PIXELS_MEDIUM,
    PIXELS_MIN,
    adaptive_max_image_pixels,
    cuda_max_memory_map,
    generator_should_yield_gpu,
    tighter_max_memory_map,
)
from rag.pipeline import SocraticRagPipeline


def _gib(value: str) -> float:
    assert value.endswith("GiB"), value
    return float(value[:-3])


def test_32gb_gpu_leaves_room_for_8b_on_device():
    mapping = cuda_max_memory_map(
        vram_reserve_gib=5.5,
        cuda_stats=(30.0, 32.0),
        total_ram_gib=32.0,
    )
    assert mapping is not None
    assert "cpu" in mapping
    # 8B bf16 is ~16 GiB; 32 GiB minus scaled headroom still fits it on GPU.
    assert _gib(mapping[0]) >= 16.0
    assert _gib(mapping["cpu"]) >= 24.0


def test_8gb_gpu_offloads_more_than_12gb():
    map12 = cuda_max_memory_map(
        vram_reserve_gib=5.5,
        cuda_stats=(11.0, 11.6),
        total_ram_gib=32.0,
    )
    map8 = cuda_max_memory_map(
        vram_reserve_gib=5.5,
        cuda_stats=(7.5, 8.0),
        total_ram_gib=32.0,
    )
    assert map12 is not None and map8 is not None
    assert _gib(map8[0]) < _gib(map12[0])
    assert _gib(map8[0]) >= 0.5
    assert "cpu" in map8
    assert _gib(map8["cpu"]) >= 24.0


def test_tight_free_vram_still_caps_instead_of_packing_the_gpu():
    mapping = cuda_max_memory_map(
        vram_reserve_gib=5.5,
        cuda_stats=(3.0, 11.6),
        total_ram_gib=32.0,
    )
    assert mapping is not None
    assert _gib(mapping[0]) < 3.0
    assert "cpu" in mapping


def test_disabled_reserve_skips_the_cap():
    assert (
        cuda_max_memory_map(
            vram_reserve_gib=0.0,
            cuda_stats=(11.0, 12.0),
            total_ram_gib=32.0,
        )
        is None
    )


def test_tighter_retry_keeps_cpu_budget():
    original = {0: "6.0GiB", "cpu": "26GiB"}
    tighter = tighter_max_memory_map(original)
    assert tighter["cpu"] == "26GiB"
    assert _gib(tighter[0]) < _gib(original[0])


def test_image_pixels_scale_with_vram():
    assert adaptive_max_image_pixels(32.0) is None
    assert adaptive_max_image_pixels(16.0) == PIXELS_MEDIUM
    assert adaptive_max_image_pixels(12.0) == PIXELS_LOW
    assert adaptive_max_image_pixels(8.0) == PIXELS_MIN
    assert adaptive_max_image_pixels(8.0, override=12345) == 12345


def test_large_gpu_keeps_tutor_loaded_for_next_retrieval():
    assert generator_should_yield_gpu(16.0, 32.0) is False
    assert generator_should_yield_gpu(4.0, 11.6) is True
    assert generator_should_yield_gpu(7.0, 8.0) is True


def test_stream_answer_releases_retrieval_models_before_generate():
    config = load_config(require_neo4j=False)
    released: list[str] = []
    pipeline = SocraticRagPipeline.__new__(SocraticRagPipeline)
    pipeline.config = config
    pipeline.retriever = MagicMock()
    pipeline.retriever.release_models = lambda: released.append("release")
    fake_gen = MagicMock()
    fake_gen.stream = MagicMock(return_value=iter(["ok"]))
    pipeline._generator = fake_gen

    result = MagicMock()
    result.turn.messages = [{"role": "user", "content": "hi"}]
    result.image_paths = []

    assert list(pipeline.stream_answer(result)) == ["ok"]
    assert released == ["release"]
    fake_gen.stream.assert_called_once()
