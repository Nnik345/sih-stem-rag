"""Dense semantic retrieval over the Neo4j vector index.

    query -> BGE-M3 dense embedding -> Neo4j vector index -> metadata filter -> top-K

Metadata filtering and approximate vector search
------------------------------------------------
Neo4j's vector index cannot pre-filter by property, so a narrow scope (say
Grade 3 Science only) risks the approximate search returning ``k`` neighbours
that are then almost all filtered away. Two strategies are combined:

* **indexed** -- ask the index for ``top_k * oversample`` neighbours and apply
  the metadata predicate inside the same Cypher statement, then take ``top_k``.
* **exact** -- when the indexed path still yields fewer than ``top_k`` in-scope
  results, fall back to ``vector.similarity.cosine`` over the filtered chunks.
  Slower, but the filter is applied first, so recall inside the scope is exact.

Either way the filter is enforced in the database, never by discarding unrelated
results in Python afterwards. Each result records which strategy produced it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import RetrievalConfig
from .embeddings import BGEM3Embedder
from .graph_schema import CHUNK_VECTOR_INDEX
from .logging_utils import Timer, get_logger
from .neo4j_store import Neo4jStore
from .retrieval_base import CHUNK_PROJECTION, build_filter_clause, chunk_from_record, where_clause
from .schemas import CHANNEL_DENSE, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)

# How many extra neighbours to request from the approximate index when a
# metadata filter is active.
DEFAULT_OVERSAMPLE = 8
# Absolute ceiling on the neighbours requested from the index.
MAX_CANDIDATE_K = 1000

STRATEGY_INDEXED = "vector_index"
STRATEGY_EXACT = "exact_filtered_scan"


class DenseRetriever:
    """Vector-index retrieval channel."""

    def __init__(
        self,
        store: Neo4jStore,
        embedder: BGEM3Embedder,
        config: RetrievalConfig,
        *,
        embedding_version: str,
        oversample: int = DEFAULT_OVERSAMPLE,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config
        self.embedding_version = embedding_version
        self.oversample = max(1, oversample)
        self.last_strategy: str | None = None
        self.last_timing_ms: float = 0.0
        self.last_query_vector_norm: float | None = None
        self.last_vector_preview: list[float] = []
        self.last_embedding_dim: int | None = None

    # -- Cypher ------------------------------------------------------------ #

    @staticmethod
    def _indexed_query(filter_clause: str) -> str:
        version_clause = "c.embedding_version = $embedding_version"
        return f"""
        CALL db.index.vector.queryNodes($index_name, $candidate_k, $vector)
        YIELD node AS c, score
        {where_clause(version_clause, filter_clause)}
        RETURN
            {CHUNK_PROJECTION},
            score             AS score
        ORDER BY score DESC
        LIMIT $top_k
        """

    @staticmethod
    def _exact_query(filter_clause: str) -> str:
        clauses = where_clause(
            "c.embedding IS NOT NULL",
            "c.embedding_version = $embedding_version",
            filter_clause,
        )
        return f"""
        MATCH (c:Chunk)
        {clauses}
        WITH c, vector.similarity.cosine(c.embedding, $vector) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN
            {CHUNK_PROJECTION},
            score             AS score
        """

    # -- retrieval --------------------------------------------------------- #

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode_query(query)

    def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalFilter | None = None,
        top_k: int | None = None,
        query_vector: Sequence[float] | np.ndarray | None = None,
    ) -> list[RetrievedChunk]:
        """Top-K semantically similar chunks inside ``scope``."""
        scope = scope or RetrievalFilter()
        limit = top_k or self.config.dense_top_k
        timer = Timer()

        vector = (
            self.embed_query(query) if query_vector is None else np.asarray(query_vector)
        )
        vector_list = [float(value) for value in np.asarray(vector).ravel()]
        self.last_embedding_dim = len(vector_list)
        self.last_query_vector_norm = float(np.linalg.norm(vector_list)) if vector_list else 0.0
        self.last_vector_preview = vector_list[:8]

        filter_clause, filter_params = build_filter_clause(scope, "c")
        candidate_k = min(
            MAX_CANDIDATE_K,
            limit * self.oversample if filter_clause else limit,
        )

        params: dict[str, Any] = {
            "index_name": CHUNK_VECTOR_INDEX,
            "candidate_k": int(candidate_k),
            "top_k": int(limit),
            "vector": vector_list,
            "embedding_version": self.embedding_version,
            **filter_params,
        }

        records = self.store.read(self._indexed_query(filter_clause), params)
        strategy = STRATEGY_INDEXED

        if filter_clause and len(records) < limit:
            LOGGER.debug(
                "Vector index returned %d/%d in-scope results at candidate_k=%d; "
                "falling back to an exact filtered scan",
                len(records),
                limit,
                candidate_k,
            )
            records = self.store.read(self._exact_query(filter_clause), params)
            strategy = STRATEGY_EXACT

        results: list[RetrievedChunk] = []
        for rank, record in enumerate(records, start=1):
            chunk = chunk_from_record(record)
            chunk.dense_rank = rank
            chunk.dense_score = float(record["score"])
            chunk.add_source(CHANNEL_DENSE)
            results.append(chunk)

        self.last_strategy = strategy
        self.last_timing_ms = timer.stop() * 1000
        LOGGER.info(
            "Dense retrieval: %d results via %s in %.1f ms (scope=%s)",
            len(results),
            strategy,
            self.last_timing_ms,
            scope.describe(),
        )
        return results
