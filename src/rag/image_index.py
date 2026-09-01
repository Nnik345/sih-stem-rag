"""One-off job: embed existing :Image nodes with SigLIP. No PDF re-ingest."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import IMAGE_EMBEDDING_VERSION, RagConfig
from .graph_schema import create_image_vector_index, wait_for_indexes
from .image_embeddings import ImageEmbeddingError, SiglipImageEmbedder, read_image_hidden_size
from .image_paths import resolve_curriculum_image
from .logging_utils import get_logger
from .neo4j_store import Neo4jStore

LOGGER = get_logger(__name__)

_LIST_IMAGES = """
MATCH (i:Image)
WHERE i.local_path IS NOT NULL
RETURN i.image_id AS image_id,
       i.local_path AS local_path,
       i.embedding_version AS embedding_version
ORDER BY i.image_id
"""

_SET_IMAGE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (i:Image {image_id: row.image_id})
CALL db.create.setNodeVectorProperty(i, 'embedding', row.embedding)
SET i.embedding_version = $embedding_version,
    i.embedding_model = $embedding_model,
    i.embedding_dim = $embedding_dim
"""


def embed_curriculum_images(
    config: RagConfig,
    store: Neo4jStore,
    *,
    force: bool = False,
    embedder: SiglipImageEmbedder | None = None,
) -> dict[str, Any]:
    """Update Image.embedding for files that exist under ``images_dir``.

    Nodes whose ``embedding_version`` already matches are skipped unless
    ``force`` is true. Missing files are counted and left unchanged.
    """
    stats = {
        "scanned": 0,
        "embedded": 0,
        "skipped": 0,
        "missing_files": 0,
        "errors": 0,
    }
    try:
        dimension = read_image_hidden_size(config.models.image_embedding_model_path)
    except ImageEmbeddingError as exc:
        LOGGER.warning("Image embed job skipped: %s", exc)
        stats["errors"] = 1
        stats["reason"] = str(exc)
        return stats

    create_image_vector_index(store, dimension)
    wait_for_indexes(store, timeout_seconds=60)

    records = store.read(_LIST_IMAGES)
    stats["scanned"] = len(records)
    if not records:
        LOGGER.info("No Image nodes with a local_path; nothing to embed")
        return stats

    version = config.image_embedding_version
    pending: list[dict[str, Any]] = []
    for record in records:
        image_id = str(record.get("image_id") or "")
        current = record.get("embedding_version")
        if not force and current == version:
            stats["skipped"] += 1
            continue
        resolved = resolve_curriculum_image(
            record.get("local_path"), config.paths.images_dir
        )
        if resolved is None:
            stats["missing_files"] += 1
            continue
        pending.append({"image_id": image_id, "path": resolved})

    if not pending:
        LOGGER.info(
            "Image embeddings already current (%s): %d skipped, %d missing files",
            version,
            stats["skipped"],
            stats["missing_files"],
        )
        return stats

    owner = embedder is None
    encoder = embedder or SiglipImageEmbedder.from_config(config.models)
    try:
        encoder.load()
        batch_size = encoder.batch_size
        model_name = Path(encoder.model_path).name
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                vectors = encoder.encode_paths([row["path"] for row in batch])
            except ImageEmbeddingError as exc:
                LOGGER.warning("Image batch at offset %d failed: %s", start, exc)
                stats["errors"] += len(batch)
                continue
            rows = [
                {
                    "image_id": row["image_id"],
                    "embedding": vectors[index].tolist(),
                }
                for index, row in enumerate(batch)
            ]
            store.execute_write_batches(
                _SET_IMAGE_EMBEDDINGS,
                rows,
                extra_parameters={
                    "embedding_version": version,
                    "embedding_model": model_name,
                    "embedding_dim": int(encoder.dimension),
                },
            )
            stats["embedded"] += len(rows)
            LOGGER.info(
                "Embedded images %d–%d / %d",
                start + 1,
                start + len(rows),
                len(pending),
            )
    finally:
        if owner:
            encoder.unload()

    LOGGER.info(
        "Image embed job: scanned=%d embedded=%d skipped=%d missing=%d errors=%d",
        stats["scanned"],
        stats["embedded"],
        stats["skipped"],
        stats["missing_files"],
        stats["errors"],
    )
    return stats
