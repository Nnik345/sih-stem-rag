"""SigLIP kNN over :Image nodes, then page chunks for fusion.

Student photos are embedded only as queries. Hits are Neo4j Image rows, filtered
to the same grade/subject lineage as text retrieval (never a higher class, never
across maths/science).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import RetrievalConfig
from .curriculum_catalog import in_lineage_scope
from .graph_schema import IMAGE_VECTOR_INDEX
from .image_embeddings import ImageEmbeddingError, SiglipImageEmbedder
from .logging_utils import Timer, get_logger
from .neo4j_store import Neo4jStore
from .retrieval_base import CHUNK_PROJECTION, chunk_from_record
from .schemas import CHANNEL_IMAGE, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)

DEFAULT_OVERSAMPLE = 8
MAX_CANDIDATE_K = 400


@dataclass
class ImageHit:
    """One textbook figure from the image vector index."""

    image_id: str
    local_path: str
    score: float
    page_number: int | None = None
    grade: int | None = None
    subject: str | None = None
    document_id: str | None = None
    unit_id: str | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "local_path": self.local_path,
            "score": self.score,
            "page_number": self.page_number,
            "grade": self.grade,
            "subject": self.subject,
            "document_id": self.document_id,
            "unit_id": self.unit_id,
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
        }


def filter_image_hits(
    records: Sequence[dict[str, Any]],
    *,
    grade: int | None,
    subject: str | None,
    allow_prior_grades: bool = True,
) -> list[dict[str, Any]]:
    """Keep image rows in the caller's class or an allowed earlier class."""
    kept: list[dict[str, Any]] = []
    for record in records:
        if in_lineage_scope(
            chunk_grade=record.get("grade"),
            chunk_subject=record.get("subject"),
            current_grade=grade,
            current_subject=subject,
            allow_prior_grades=allow_prior_grades,
        ):
            kept.append(record)
    return kept


class ImageRetriever:
    """Image-vector retrieval channel. Page chunks feed text fusion."""

    def __init__(
        self,
        store: Neo4jStore,
        embedder: SiglipImageEmbedder,
        config: RetrievalConfig,
        *,
        embedding_version: str,
        oversample: int = DEFAULT_OVERSAMPLE,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.embedding_version = embedding_version
        self.oversample = max(1, int(oversample))
        self.last_timing_ms: float = 0.0
        self.last_hits: list[ImageHit] = []
        self.last_note: str = ""

    def retrieve(
        self,
        image_path: str | Path,
        *,
        scope: RetrievalFilter | None = None,
    ) -> tuple[list[ImageHit], list[RetrievedChunk]]:
        """Embed the student image and return lineage-filtered figure hits."""
        timer = Timer()
        self.last_hits = []
        self.last_note = ""
        path = Path(image_path)
        if not path.is_file():
            self.last_note = "student image file missing"
            self.last_timing_ms = timer.stop() * 1000
            return [], []

        try:
            vector = self.embedder.encode_image(path)
        except ImageEmbeddingError as exc:
            LOGGER.warning("Image kNN skipped: %s", exc)
            self.last_note = str(exc)
            self.last_timing_ms = timer.stop() * 1000
            return [], []

        top_k = max(1, int(self.config.image_top_k))
        candidate_k = min(MAX_CANDIDATE_K, top_k * self.oversample)
        if scope is not None and scope.allow_prior_grades:
            candidate_k = min(MAX_CANDIDATE_K, candidate_k * 2)

        params: dict[str, Any] = {
            "index_name": IMAGE_VECTOR_INDEX,
            "candidate_k": int(candidate_k),
            "vector": vector.tolist(),
            "embedding_version": self.embedding_version,
            "top_k": int(top_k),
        }
        where_parts = ["i.embedding_version = $embedding_version"]
        if scope is not None and scope.grade is not None:
            if scope.allow_prior_grades:
                where_parts.append("i.grade <= $flt_grade")
            else:
                where_parts.append("i.grade = $flt_grade")
            params["flt_grade"] = int(scope.grade)
        if scope is not None and scope.subject is not None:
            from .curriculum_catalog import lineage_subjects

            if scope.allow_prior_grades:
                where_parts.append("i.subject IN $flt_subjects")
                params["flt_subjects"] = list(lineage_subjects(str(scope.subject)))
            else:
                where_parts.append("toLower(i.subject) = $flt_subject")
                params["flt_subject"] = str(scope.subject).lower()

        where = " AND ".join(where_parts)
        try:
            records = self.store.read(
                f"""
                CALL db.index.vector.queryNodes($index_name, $candidate_k, $vector)
                YIELD node AS i, score
                WHERE {where}
                RETURN i.image_id AS image_id,
                       i.local_path AS local_path,
                       i.page_number AS page_number,
                       i.grade AS grade,
                       i.subject AS subject,
                       i.document_id AS document_id,
                       i.unit_id AS unit_id,
                       score AS score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                params,
            )
        except Exception as exc:
            LOGGER.warning("Image vector query failed: %s", exc)
            self.last_note = f"image vector query failed: {exc}"
            self.last_timing_ms = timer.stop() * 1000
            return [], []

        raw = [dict(record) for record in records]
        if scope is not None:
            raw = filter_image_hits(
                raw,
                grade=scope.grade,
                subject=scope.subject,
                allow_prior_grades=scope.allow_prior_grades,
            )

        hits: list[ImageHit] = []
        for rank, row in enumerate(raw, start=1):
            image_id = str(row.get("image_id") or "")
            if not image_id:
                continue
            hits.append(
                ImageHit(
                    image_id=image_id,
                    local_path=str(row.get("local_path") or ""),
                    score=float(row.get("score") or 0.0),
                    page_number=row.get("page_number"),
                    grade=row.get("grade"),
                    subject=row.get("subject"),
                    document_id=row.get("document_id"),
                    unit_id=row.get("unit_id"),
                )
            )
            if rank >= top_k:
                break

        chunks = self._chunks_for_hits(hits)
        self.last_hits = hits
        self.last_timing_ms = timer.stop() * 1000
        LOGGER.info(
            "Image kNN: %d hits, %d page chunks in %.1f ms",
            len(hits),
            len(chunks),
            self.last_timing_ms,
        )
        return hits, chunks

    def _chunks_for_hits(self, hits: Sequence[ImageHit]) -> list[RetrievedChunk]:
        if not hits:
            return []
        by_id = {hit.image_id: hit for hit in hits}
        records = self.store.read(
            f"""
            UNWIND $image_ids AS image_id
            MATCH (i:Image {{image_id: image_id}})<-[:HAS_IMAGE]-(p:Page)
                  <-[:ON_PAGE]-(c:Chunk)
            RETURN image_id,
                   {CHUNK_PROJECTION}
            """,
            {"image_ids": [hit.image_id for hit in hits]},
        )
        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rank, record in enumerate(records, start=1):
            image_id = str(record.get("image_id") or "")
            hit = by_id.get(image_id)
            chunk = chunk_from_record(record)
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunk.add_source(CHANNEL_IMAGE)
            chunk.image_rank = rank
            chunk.image_score = hit.score if hit is not None else None
            chunk.matched_image_id = image_id
            if hit is not None:
                hit.chunks.append(chunk)
            chunks.append(chunk)
        return chunks
