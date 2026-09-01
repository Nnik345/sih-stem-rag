"""Single source of truth for every tunable value in the RAG pipeline.

No module outside this file should contain a magic retrieval number. Values are
read from the environment (optionally via a `.env` file) and otherwise fall back
to the documented defaults.

Usage::

    from rag.config import load_config
    config = load_config()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bumping this string invalidates cached embeddings and lets future experiments
# keep several embedding generations side by side in the same graph.
EMBEDDING_VERSION = "bge-m3-dense-v1"
IMAGE_EMBEDDING_VERSION = "siglip-base-patch16-224-v1"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or self-contradictory."""


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def _env_path(name: str, default: str) -> Path:
    raw = _env_str(name, default)
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        password = os.environ.get("NEO4J_PASSWORD", "").strip()
        if not password:
            raise ConfigError(
                "NEO4J_PASSWORD is not set. Copy .env.example to .env and set the "
                "Neo4j credentials (see README.md -> 'Neo4j environment variables')."
            )
        if password == "change-me":
            raise ConfigError(
                "NEO4J_PASSWORD is still the .env.example placeholder 'change-me'. "
                "Set the real local Neo4j password in .env."
            )
        return cls(
            uri=_env_str("NEO4J_URI", "bolt://localhost:7687"),
            user=_env_str("NEO4J_USER", "neo4j"),
            password=password,
            database=_env_str("NEO4J_DATABASE", "neo4j"),
        )

    def describe(self) -> str:
        """Connection summary that never contains the password."""
        return f"{self.uri} (user={self.user}, database={self.database})"


@dataclass(frozen=True)
class ChunkingConfig:
    """Hierarchical, token-aware chunking parameters."""

    target_tokens: int = 600
    overlap_tokens: int = 100
    # A section shorter than this is emitted as a single chunk instead of being
    # split, and is merged forward if it is smaller than min_tokens on its own.
    min_tokens: int = 80
    # Hard ceiling; a single paragraph longer than this is split by sentence.
    max_tokens: int = 900

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        cfg = cls(
            target_tokens=_env_int("CHUNK_TARGET_TOKENS", 600),
            overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 100),
            min_tokens=_env_int("CHUNK_MIN_TOKENS", 80),
            max_tokens=_env_int("CHUNK_MAX_TOKENS", 900),
        )
        if cfg.overlap_tokens >= cfg.target_tokens:
            raise ConfigError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS "
                f"({cfg.overlap_tokens} >= {cfg.target_tokens})"
            )
        if cfg.max_tokens < cfg.target_tokens:
            raise ConfigError(
                "CHUNK_MAX_TOKENS must be >= CHUNK_TARGET_TOKENS "
                f"({cfg.max_tokens} < {cfg.target_tokens})"
            )
        return cfg


@dataclass(frozen=True)
class RetrievalConfig:
    """Top-K values, graph traversal limits and rank-fusion parameters."""

    dense_top_k: int = 20
    fulltext_top_k: int = 20
    graph_seed_top_k: int = 10
    graph_max_depth: int = 2
    # Upper bound on graph-expansion candidates, so traversal cannot flood fusion.
    graph_top_k: int = 20
    fusion_top_k: int = 20
    final_top_k: int = 5

    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_fulltext: float = 1.0
    # Graph expansion is a secondary signal.
    weight_graph: float = 0.5
    # SigLIP kNN over textbook Image nodes (student photos are queries only).
    image_top_k: int = 8
    min_image_score: float = 0.25
    weight_image: float = 1.0
    max_attached_figures: int = 2

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        cfg = cls(
            dense_top_k=_env_int("DENSE_TOP_K", 20),
            fulltext_top_k=_env_int("FULLTEXT_TOP_K", 20),
            graph_seed_top_k=_env_int("GRAPH_SEED_TOP_K", 10),
            graph_max_depth=_env_int("GRAPH_MAX_DEPTH", 2),
            graph_top_k=_env_int("GRAPH_TOP_K", 20),
            fusion_top_k=_env_int("FUSION_TOP_K", 20),
            final_top_k=_env_int("FINAL_TOP_K", 5),
            rrf_k=_env_int("RRF_K", 60),
            weight_dense=_env_float("WEIGHT_DENSE", 1.0),
            weight_fulltext=_env_float("WEIGHT_FULLTEXT", 1.0),
            weight_graph=_env_float("WEIGHT_GRAPH", 0.5),
            image_top_k=_env_int("IMAGE_TOP_K", 8),
            min_image_score=_env_float("IMAGE_MIN_SCORE", 0.25),
            weight_image=_env_float("WEIGHT_IMAGE", 1.0),
            max_attached_figures=_env_int("MAX_ATTACHED_FIGURES", 2),
        )
        if not 1 <= cfg.graph_max_depth <= 3:
            raise ConfigError(
                "GRAPH_MAX_DEPTH must be between 1 and 3 to keep traversal bounded, "
                f"got {cfg.graph_max_depth}"
            )
        if cfg.rrf_k <= 0:
            raise ConfigError(f"RRF_K must be positive, got {cfg.rrf_k}")
        for name, k in (
            ("DENSE_TOP_K", cfg.dense_top_k),
            ("FULLTEXT_TOP_K", cfg.fulltext_top_k),
            ("FUSION_TOP_K", cfg.fusion_top_k),
            ("FINAL_TOP_K", cfg.final_top_k),
            ("IMAGE_TOP_K", cfg.image_top_k),
            ("MAX_ATTACHED_FIGURES", cfg.max_attached_figures),
        ):
            if k <= 0:
                raise ConfigError(f"{name} must be positive, got {k}")
        return cfg


@dataclass(frozen=True)
class EvidenceConfig:
    """Evidence sufficiency gate thresholds.

    These are heuristics chosen for a first implementation. They are NOT
    validated thresholds; treat them as a configurable starting point and expect
    to replace this component with a stronger grounding check later.
    """

    # Raw bge-reranker-v2-m3 logit. 0.0 is sigmoid(0)=0.5, not "no relevance".
    # Keep this floor unless a reviewed calibration set shows a better cut-point.
    min_rerank_score: float = 0.0
    min_chunks: int = 1
    # How many chunks must clear min_rerank_score.
    min_strong_chunks: int = 1
    # Reject evidence coming from outside an explicitly requested grade/subject.
    require_scope_match: bool = True
    # Fraction of the query's content words that must appear in the evidence.
    min_query_term_overlap: float = 0.15
    # Prior-grade (lookback) chunks must clear this reranker logit. Higher than
    # min_rerank_score so weak older-chapter near-misses stay out.
    min_prior_grade_rerank_score: float = 1.0

    @classmethod
    def from_env(cls) -> "EvidenceConfig":
        return cls(
            min_rerank_score=_env_float("EVIDENCE_MIN_RERANK_SCORE", 0.0),
            min_chunks=_env_int("EVIDENCE_MIN_CHUNKS", 1),
            min_strong_chunks=_env_int("EVIDENCE_MIN_STRONG_CHUNKS", 1),
            require_scope_match=_env_bool("EVIDENCE_REQUIRE_SCOPE_MATCH", True),
            min_query_term_overlap=_env_float("EVIDENCE_MIN_QUERY_TERM_OVERLAP", 0.15),
            min_prior_grade_rerank_score=_env_float(
                "EVIDENCE_MIN_PRIOR_GRADE_RERANK_SCORE", 1.0
            ),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Local model paths and per-model runtime settings.

    The embedder and reranker are loaded lazily and can be released between
    stages so the generator can run without keeping every model resident.
    """

    embedding_model_path: Path
    reranker_model_path: Path
    rewriter_model_path: Path
    generator_model_path: Path
    image_embedding_model_path: Path

    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    embedding_max_length: int = 1024
    # Dense vectors only today; sparse and ColBERT modes are reserved for later.
    embedding_mode: str = "dense"

    reranker_device: str = "auto"
    reranker_batch_size: int = 4
    reranker_max_length: int = 1024

    generator_max_new_tokens: int = 640
    generator_temperature: float = 0.7
    # Floor on VRAM held back from generator weights for vision prefill and the
    # KV cache. Actual headroom also scales with the card (see model_memory).
    # Set to 0 to disable the cap (device_map packs the GPU; generation may OOM).
    generator_vram_reserve_gib: float = 5.5
    # Extra headroom as a fraction of *total* device memory. On a 32 GiB card
    # this leaves the 8B tutor on GPU; on 8 GiB more layers go to system RAM.
    generator_vram_headroom_fraction: float = 0.25
    # Fraction of physical RAM Accelerate may use for CPU-offloaded layers.
    generator_cpu_ram_fraction: float = 0.80
    # 0 = choose Qwen-VL max_pixels from VRAM size. >0 forces that pixel budget.
    generator_max_image_pixels: int = 0
    # Per-chunk character budget for the evidence block. Transformers computes
    # fp32 logits over every prompt position during prefill, so prompt length
    # costs memory quadratically in practice; trimming here is the cheapest lever.
    generator_max_evidence_chars: int = 1000

    @classmethod
    def from_env(cls) -> "ModelConfig":
        reserve = _env_float("GENERATOR_VRAM_RESERVE_GIB", 5.5)
        headroom = _env_float("GENERATOR_VRAM_HEADROOM_FRACTION", 0.25)
        cpu_frac = _env_float("GENERATOR_CPU_RAM_FRACTION", 0.80)
        if reserve < 0:
            raise ConfigError(
                f"GENERATOR_VRAM_RESERVE_GIB must be >= 0, got {reserve}"
            )
        if not 0 < headroom < 0.9:
            raise ConfigError(
                "GENERATOR_VRAM_HEADROOM_FRACTION must be between 0 and 0.9, "
                f"got {headroom}"
            )
        if not 0.1 <= cpu_frac <= 0.95:
            raise ConfigError(
                "GENERATOR_CPU_RAM_FRACTION must be between 0.1 and 0.95, "
                f"got {cpu_frac}"
            )
        return cls(
            embedding_model_path=_env_path("EMBEDDING_MODEL_PATH", "models/bge-m3"),
            reranker_model_path=_env_path(
                "RERANKER_MODEL_PATH", "models/bge-reranker-v2-m3"
            ),
            rewriter_model_path=_env_path(
                "REWRITER_MODEL_PATH", "models/qwen3-vl-2b-instruct"
            ),
            generator_model_path=_env_path(
                "GENERATOR_MODEL_PATH", "models/qwen3-vl-8b-instruct"
            ),
            image_embedding_model_path=_env_path(
                "IMAGE_EMBEDDING_MODEL_PATH", "models/siglip-base-patch16-224"
            ),
            embedding_device=_env_str("EMBEDDING_DEVICE", "auto"),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 8),
            embedding_max_length=_env_int("EMBEDDING_MAX_LENGTH", 1024),
            embedding_mode=_env_str("EMBEDDING_MODE", "dense"),
            reranker_device=_env_str("RERANKER_DEVICE", "auto"),
            reranker_batch_size=_env_int("RERANKER_BATCH_SIZE", 4),
            reranker_max_length=_env_int("RERANKER_MAX_LENGTH", 1024),
            generator_max_new_tokens=_env_int("GENERATOR_MAX_NEW_TOKENS", 640),
            generator_temperature=_env_float("GENERATOR_TEMPERATURE", 0.7),
            generator_vram_reserve_gib=reserve,
            generator_vram_headroom_fraction=headroom,
            generator_cpu_ram_fraction=cpu_frac,
            generator_max_image_pixels=_env_int("GENERATOR_MAX_IMAGE_PIXELS", 0),
            generator_max_evidence_chars=_env_int(
                "GENERATOR_MAX_EVIDENCE_CHARS", 1000
            ),
        )


@dataclass(frozen=True)
class PathConfig:
    corpus_path: Path
    processed_data_path: Path

    @classmethod
    def from_env(cls) -> "PathConfig":
        return cls(
            corpus_path=_env_path("CORPUS_PATH", "curriculum"),
            processed_data_path=_env_path("PROCESSED_DATA_PATH", "curriculum/processed"),
        )

    @property
    def manifest_path(self) -> Path:
        return self.corpus_path / "manifests" / "sources.yaml"

    @property
    def text_dir(self) -> Path:
        return self.processed_data_path / "text"

    @property
    def images_dir(self) -> Path:
        return self.processed_data_path / "images"

    @property
    def manifests_dir(self) -> Path:
        return self.processed_data_path / "manifests"

    @property
    def cache_dir(self) -> Path:
        return self.processed_data_path / "cache"

    @property
    def uploads_dir(self) -> Path:
        """Temporary student photos. Never ingested as Image nodes."""
        return self.processed_data_path / "uploads"

    @property
    def evaluation_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "evaluation"

    def ensure_processed_dirs(self) -> None:
        """Create the processed-data tree. The raw corpus is never modified."""
        for directory in (
            self.text_dir,
            self.images_dir,
            self.manifests_dir,
            self.cache_dir,
            self.uploads_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class IngestConfig:
    """Corpus-ingestion behaviour."""

    # Minimum characters of extracted text for a page to count as machine-readable.
    min_page_chars: int = 40
    # Heading-delimited sections shorter than this are merged into the preceding
    # section. These PDFs use many short sub-headings, and taking every one as a
    # section boundary yields chunks far below the configured token target.
    min_section_chars: int = 400
    # Embedded images below this pixel area are decorative (rules, bullets, logos).
    min_image_pixels: int = 100 * 100
    extract_images: bool = True
    # OCR is off by default: the approved STEM PDFs/ePUBs have a text layer.
    # Pages with no extractable text are recorded as image-only rather than OCRed.
    enable_ocr: bool = False
    neo4j_batch_size: int = 200

    @classmethod
    def from_env(cls) -> "IngestConfig":
        return cls(
            min_page_chars=_env_int("INGEST_MIN_PAGE_CHARS", 40),
            min_section_chars=_env_int("INGEST_MIN_SECTION_CHARS", 400),
            min_image_pixels=_env_int("INGEST_MIN_IMAGE_PIXELS", 10_000),
            extract_images=_env_bool("INGEST_EXTRACT_IMAGES", True),
            enable_ocr=_env_bool("INGEST_ENABLE_OCR", False),
            neo4j_batch_size=_env_int("INGEST_NEO4J_BATCH_SIZE", 200),
        )


@dataclass(frozen=True)
class RagConfig:
    """Aggregate configuration passed through the whole pipeline."""

    paths: PathConfig
    neo4j: Neo4jConfig | None
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    evidence: EvidenceConfig
    models: ModelConfig
    ingest: IngestConfig

    embedding_version: str = EMBEDDING_VERSION
    image_embedding_version: str = IMAGE_EMBEDDING_VERSION
    # Student photos are input-only. Textbook figures are not attached to answers.
    multimodal_enabled: bool = True

    def require_neo4j(self) -> Neo4jConfig:
        if self.neo4j is None:
            raise ConfigError(
                "Neo4j configuration is unavailable. Set NEO4J_URI / NEO4J_USER / "
                "NEO4J_PASSWORD / NEO4J_DATABASE in .env (see .env.example)."
            )
        return self.neo4j

    def with_overrides(self, **kwargs: object) -> "RagConfig":
        """Return a copy with top-level fields replaced (used by CLI flags)."""
        return replace(self, **kwargs)  # type: ignore[arg-type]

    def summary(self) -> str:
        neo4j = self.neo4j.describe() if self.neo4j else "<not configured>"
        return "\n".join(
            [
                "RAG configuration",
                f"  corpus              : {self.paths.corpus_path}",
                f"  processed data      : {self.paths.processed_data_path}",
                f"  embedding model     : {self.models.embedding_model_path}",
                f"  image embedding     : {self.models.image_embedding_model_path}",
                f"  reranker model      : {self.models.reranker_model_path}",
                f"  rewriter model      : {self.models.rewriter_model_path}",
                f"  generator model     : {self.models.generator_model_path}",
                f"  embedding version   : {self.embedding_version}",
                f"  image embed version : {self.image_embedding_version}",
                f"  neo4j               : {neo4j}",
                f"  chunk target/overlap: {self.chunking.target_tokens}"
                f"/{self.chunking.overlap_tokens} tokens",
                f"  top-k dense/ft/graph: {self.retrieval.dense_top_k}"
                f"/{self.retrieval.fulltext_top_k}/{self.retrieval.graph_top_k}",
                f"  fusion/final top-k  : {self.retrieval.fusion_top_k}"
                f"/{self.retrieval.final_top_k}",
                f"  rrf k / weights     : {self.retrieval.rrf_k} / "
                f"dense={self.retrieval.weight_dense} "
                f"fulltext={self.retrieval.weight_fulltext} "
                f"graph={self.retrieval.weight_graph} "
                f"image={self.retrieval.weight_image}",
                f"  graph max depth     : {self.retrieval.graph_max_depth}",
                f"  multimodal retrieval: "
                f"{'enabled' if self.multimodal_enabled else 'disabled'}",
            ]
        )


_ENV_LOADED = False


def load_dotenv_once() -> None:
    """Load `.env` from the project root if python-dotenv is installed.

    Real environment variables always win over `.env` values.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        LOGGER.debug("No .env file at %s; relying on the process environment", env_path)
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        LOGGER.warning(
            "%s exists but python-dotenv is not installed; export the Neo4j "
            "variables manually or run: pip install python-dotenv",
            env_path,
        )
        return
    load_dotenv(dotenv_path=env_path, override=False)
    LOGGER.debug("Loaded environment from %s", env_path)


def load_config(*, require_neo4j: bool = True) -> RagConfig:
    """Build the configuration from the environment.

    Set ``require_neo4j=False`` for tools that must work without a database
    (chunking experiments, unit tests, model downloads).
    """
    load_dotenv_once()

    try:
        neo4j = Neo4jConfig.from_env()
    except ConfigError:
        if require_neo4j:
            raise
        neo4j = None

    return RagConfig(
        paths=PathConfig.from_env(),
        neo4j=neo4j,
        chunking=ChunkingConfig.from_env(),
        retrieval=RetrievalConfig.from_env(),
        evidence=EvidenceConfig.from_env(),
        models=ModelConfig.from_env(),
        ingest=IngestConfig.from_env(),
        multimodal_enabled=_env_bool("MULTIMODAL_ENABLED", True),
        image_embedding_version=_env_str(
            "IMAGE_EMBEDDING_VERSION", IMAGE_EMBEDDING_VERSION
        ),
    )
