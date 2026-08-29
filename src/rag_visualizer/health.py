"""Health checks that never load large models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import PROJECT_ROOT, ConfigError, load_config
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError


def frontend_dist_dir() -> Path:
    return PROJECT_ROOT / "frontend" / "dist"


def check_health() -> dict[str, Any]:
    """Path and connectivity probes only — models are not loaded."""
    try:
        config = load_config(require_neo4j=False)
    except ConfigError:
        config = None

    neo4j_status = "unavailable"
    if config is not None and config.neo4j is not None:
        try:
            store = Neo4jStore(config.neo4j)
            store.read("RETURN 1 AS ok")
            store.close()
            neo4j_status = "ok"
        except (Neo4jUnavailableError, ConfigError, Exception):
            neo4j_status = "unavailable"

    models = config.models if config is not None else None
    paths = config.paths if config is not None else None
    return {
        "status": "ok",
        "api": "ok",
        "neo4j": neo4j_status,
        "corpus_path_present": bool(paths and paths.corpus_path.exists()) if paths else False,
        "embedding_model_path_present": bool(
            models and models.embedding_model_path.is_dir()
        )
        if models
        else False,
        "reranker_model_path_present": bool(
            models and models.reranker_model_path.is_dir()
        )
        if models
        else False,
        "generator_model_path_present": bool(
            models and models.generator_model_path.is_dir()
        )
        if models
        else False,
        "frontend_build_present": (frontend_dist_dir() / "index.html").is_file(),
    }
