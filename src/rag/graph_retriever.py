"""Graph-based expansion channel.

Expansion starts from the strongest dense/full-text chunks (the *seeds*) rather
than traversing the graph blindly. Traversal is a fixed set of named, bounded
patterns instead of a variable-length path, so it cannot run away:

    hop 1  SAME_SECTION   sibling chunks under the seed's Section
    hop 1  ADJACENT       seed -[:NEXT|PREVIOUS]- chunk (reading order)
    hop 1  SAME_PAGE      chunks sharing a Page with the seed
    hop 2  SHARED_CONCEPT chunks mentioning a Concept the seed mentions
    hop 2  SAME_LESSON    chunks in the same Unit under an identically titled
                          Section (links student and teacher material)

``graph_max_depth`` selects which patterns are allowed, each pattern is capped
per seed, and hub concepts above ``max_concept_mentions`` are excluded so a very
common term cannot drag in hundreds of loosely related chunks. Seeds themselves
are excluded from the output, so this channel only ever contributes candidates
the other two channels missed.

Every candidate keeps the relation and seed it came from, so graph-derived
evidence stays attributable to the curriculum structure that produced it.
"""

from __future__ import annotations

from typing import Any, Sequence

from .config import RetrievalConfig
from .graph_trace import (
    DEFAULT_TRACE_MAX_NODES,
    DEFAULT_TRACE_PER_RELATION,
    classify_graph_trace,
    collect_diagnostic_rows,
    primary_records_to_rows,
)
from .logging_utils import Timer, get_logger
from .neo4j_store import Neo4jStore
from .retrieval_base import (
    build_filter_clause,
    chunk_from_record,
    chunk_projection,
    where_clause,
)
from .schemas import CHANNEL_GRAPH, RetrievalFilter, RetrievedChunk
from .trace import TraceObserver, emit

LOGGER = get_logger(__name__)

# Relation name -> hop count. A relation is used only when its hop count is
# within the configured graph_max_depth.
RELATION_DEPTH: dict[str, int] = {
    "SAME_SECTION": 1,
    "ADJACENT": 1,
    "SAME_PAGE": 1,
    "SHARED_CONCEPT": 2,
    "SAME_LESSON": 2,
}

# Relative usefulness of each relation, used to score graph candidates.
RELATION_WEIGHT: dict[str, float] = {
    "SAME_SECTION": 1.0,
    "ADJACENT": 0.9,
    "SAME_PAGE": 0.7,
    "SHARED_CONCEPT": 0.6,
    "SAME_LESSON": 0.5,
}

# Per-seed, per-relation cap on neighbours.
DEFAULT_PER_RELATION_LIMIT = 8
# Concepts mentioned by more chunks than this are hubs and are not traversed.
DEFAULT_MAX_CONCEPT_MENTIONS = 400

_BRANCHES: dict[str, str] = {
    "SAME_SECTION": """
        MATCH (seed)<-[:HAS_CHUNK]-(:Section)-[:HAS_CHUNK]->(n:Chunk)
        RETURN n AS n, 'SAME_SECTION' AS relation, null AS via
        LIMIT $per_relation_limit
    """,
    "ADJACENT": """
        MATCH (seed)-[:NEXT|PREVIOUS]-(n:Chunk)
        RETURN n AS n, 'ADJACENT' AS relation, null AS via
        LIMIT $per_relation_limit
    """,
    "SAME_PAGE": """
        MATCH (seed)-[:ON_PAGE]->(p:Page)<-[:ON_PAGE]-(n:Chunk)
        RETURN n AS n, 'SAME_PAGE' AS relation, p.page_id AS via
        LIMIT $per_relation_limit
    """,
    "SHARED_CONCEPT": """
        MATCH (seed)-[:MENTIONS]->(co:Concept)<-[:MENTIONS]-(n:Chunk)
        WHERE coalesce(co.mention_count, 0) <= $max_concept_mentions
        RETURN n AS n, 'SHARED_CONCEPT' AS relation, co.name AS via
        LIMIT $per_relation_limit
    """,
    "SAME_LESSON": """
        MATCH (n:Chunk)
        WHERE n.unit_id = seed.unit_id
          AND n.section_title = seed.section_title
          AND n.document_id <> seed.document_id
        RETURN n AS n, 'SAME_LESSON' AS relation, seed.section_title AS via
        LIMIT $per_relation_limit
    """,
}


def _build_query(relations: Sequence[str], filter_clause: str) -> str:
    branches = " UNION ".join(_BRANCHES[relation] for relation in relations)
    predicate = where_clause("n.chunk_id <> seed.chunk_id", filter_clause)
    # CALL (seed) { ... } is the variable-scope form; importing with `WITH seed`
    # inside the subquery is deprecated as of Neo4j 5.23.
    return f"""
    /* graph-primary */
    UNWIND $seeds AS seed_row
    MATCH (seed:Chunk {{chunk_id: seed_row.chunk_id}})
    CALL (seed) {{
        {branches}
    }}
    WITH seed, seed_row, n, relation, via
    {predicate}
    RETURN
        {chunk_projection("n")},
        relation          AS relation,
        via               AS via,
        seed.chunk_id     AS seed_chunk_id,
        seed_row.weight   AS seed_weight
    """


class GraphRetriever:
    """Bounded graph expansion channel."""

    def __init__(
        self,
        store: Neo4jStore,
        config: RetrievalConfig,
        *,
        per_relation_limit: int = DEFAULT_PER_RELATION_LIMIT,
        max_concept_mentions: int = DEFAULT_MAX_CONCEPT_MENTIONS,
    ) -> None:
        self.store = store
        self.config = config
        self.per_relation_limit = per_relation_limit
        self.max_concept_mentions = max_concept_mentions
        self.last_timing_ms: float = 0.0
        self.last_seeds: list[str] = []
        self.last_trace: dict[str, Any] | None = None
        self.trace_per_relation_limit = DEFAULT_TRACE_PER_RELATION
        self.trace_max_nodes = DEFAULT_TRACE_MAX_NODES

    def active_relations(self, max_depth: int | None = None) -> list[str]:
        depth = max_depth if max_depth is not None else self.config.graph_max_depth
        return [name for name, hops in RELATION_DEPTH.items() if hops <= depth]

    @staticmethod
    def select_seeds(
        channels: Sequence[Sequence[RetrievedChunk]],
        seed_top_k: int,
    ) -> list[dict[str, Any]]:
        """Pick seed chunks from the highest-ranked results of other channels.

        A seed's weight is ``1 / rank`` in its own channel, so expansion around a
        strong hit counts for more than expansion around a marginal one.
        """
        weights: dict[str, float] = {}
        for results in channels:
            for rank, chunk in enumerate(results, start=1):
                weight = 1.0 / rank
                if weight > weights.get(chunk.chunk_id, 0.0):
                    weights[chunk.chunk_id] = weight

        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [
            {"chunk_id": chunk_id, "weight": weight}
            for chunk_id, weight in ordered[:seed_top_k]
        ]

    def expand(
        self,
        seeds: Sequence[dict[str, Any]],
        *,
        scope: RetrievalFilter | None = None,
        top_k: int | None = None,
        max_depth: int | None = None,
        observer: TraceObserver | None = None,
    ) -> list[RetrievedChunk]:
        """Expand from ``seeds`` and return the best new candidates."""
        scope = scope or RetrievalFilter()
        limit = top_k or self.config.graph_top_k
        timer = Timer()
        self.last_seeds = [seed["chunk_id"] for seed in seeds]
        self.last_trace = None

        if not seeds:
            LOGGER.info("Graph expansion skipped: no seed chunks")
            self.last_timing_ms = timer.stop() * 1000
            self._emit_graph_trace(
                observer,
                seeds=[],
                scope=scope,
                selected=[],
                primary_records=[],
                enabled=[],
                disabled=list(RELATION_DEPTH),
            )
            return []

        relations = self.active_relations(max_depth)
        disabled = [name for name in RELATION_DEPTH if name not in relations]
        if not relations:
            LOGGER.warning(
                "No graph relations enabled at depth %s", max_depth
            )
            self.last_timing_ms = timer.stop() * 1000
            self._emit_graph_trace(
                observer,
                seeds=list(seeds),
                scope=scope,
                selected=[],
                primary_records=[],
                enabled=[],
                disabled=disabled,
            )
            return []

        filter_clause, filter_params = build_filter_clause(scope, "n")
        params: dict[str, Any] = {
            "seeds": list(seeds),
            "per_relation_limit": int(self.per_relation_limit),
            "max_concept_mentions": int(self.max_concept_mentions),
            **filter_params,
        }
        records = self.store.read(_build_query(relations, filter_clause), params)

        # Aggregate: a chunk reached by several seeds or relations scores higher.
        aggregated: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}
        paths: dict[str, list[str]] = {}
        seed_ids = set(self.last_seeds)

        for record in records:
            chunk_id = record["chunk_id"]
            if chunk_id in seed_ids:
                continue
            relation = record["relation"]
            weight = float(record.get("seed_weight") or 0.0)
            contribution = weight * RELATION_WEIGHT.get(relation, 0.5)

            if chunk_id not in aggregated:
                chunk = chunk_from_record(record)
                chunk.add_source(CHANNEL_GRAPH)
                chunk.graph_seed_chunk_id = record["seed_chunk_id"]
                aggregated[chunk_id] = chunk
                scores[chunk_id] = 0.0
                paths[chunk_id] = []

            scores[chunk_id] += contribution
            via = record.get("via")
            descriptor = (
                f"{relation} via {via} from {record['seed_chunk_id']}"
                if via
                else f"{relation} from {record['seed_chunk_id']}"
            )
            if descriptor not in paths[chunk_id]:
                paths[chunk_id].append(descriptor)

        ordered = sorted(
            aggregated.values(), key=lambda c: scores[c.chunk_id], reverse=True
        )[:limit]
        for rank, chunk in enumerate(ordered, start=1):
            chunk.graph_rank = rank
            chunk.graph_score = round(scores[chunk.chunk_id], 6)
            chunk.graph_expansion_path = "; ".join(paths[chunk.chunk_id][:3])

        self.last_timing_ms = timer.stop() * 1000
        LOGGER.info(
            "Graph expansion: %d candidates from %d seeds via %s in %.1f ms",
            len(ordered),
            len(seeds),
            ", ".join(relations),
            self.last_timing_ms,
        )
        self._emit_graph_trace(
            observer,
            seeds=list(seeds),
            scope=scope,
            selected=ordered,
            primary_records=records,
            enabled=relations,
            disabled=disabled,
            graph_top_k=limit,
        )
        return ordered

    def retrieve(
        self,
        channels: Sequence[Sequence[RetrievedChunk]],
        *,
        scope: RetrievalFilter | None = None,
        top_k: int | None = None,
        seed_top_k: int | None = None,
        max_depth: int | None = None,
        observer: TraceObserver | None = None,
    ) -> list[RetrievedChunk]:
        """Select seeds from other channels, then expand."""
        seeds = self.select_seeds(
            channels, seed_top_k or self.config.graph_seed_top_k
        )
        return self.expand(
            seeds,
            scope=scope,
            top_k=top_k,
            max_depth=max_depth,
            observer=observer,
        )

    def _emit_graph_trace(
        self,
        observer: TraceObserver | None,
        *,
        seeds: Sequence[dict[str, Any]],
        scope: RetrievalFilter,
        selected: Sequence[RetrievedChunk],
        primary_records: Sequence[dict[str, Any]],
        enabled: Sequence[str],
        disabled: Sequence[str],
        graph_top_k: int | None = None,
    ) -> None:
        if observer is None:
            self.last_trace = None
            return
        diagnostic_rows, hubs, caps = collect_diagnostic_rows(
            self.store,
            seeds=seeds,
            enabled_relations=enabled,
            max_concept_mentions=self.max_concept_mentions,
            trace_per_relation=self.trace_per_relation_limit,
        )
        trace = classify_graph_trace(
            seeds=seeds,
            scope=scope,
            selected=selected,
            primary_rows=primary_records_to_rows(primary_records, seeds),
            diagnostic_rows=diagnostic_rows,
            hub_concepts=hubs,
            enabled_relations=list(enabled),
            disabled_relations=list(disabled),
            per_relation_limit=self.per_relation_limit,
            graph_top_k=graph_top_k or self.config.graph_top_k,
            max_concept_mentions=self.max_concept_mentions,
            max_paths_per_seed_relation=self.trace_per_relation_limit,
            max_nodes=self.trace_max_nodes,
        )
        if caps:
            trace.truncated = True
            for cap in caps:
                if cap not in trace.truncation_caps:
                    trace.truncation_caps.append(cap)
            trace.truncation_warning = (
                "Graph trace was truncated by a safety cap "
                f"({', '.join(trace.truncation_caps)}). Neighbours beyond the cap "
                "were not examined and are not shown as ignored."
            )
        if len(trace.nodes) >= self.trace_max_nodes:
            trace.truncated = True
            if "max_nodes" not in trace.truncation_caps:
                trace.truncation_caps.append("max_nodes")
        self.last_trace = trace.to_dict()
        emit(
            observer,
            "graph_completed",
            graph=trace,
            elapsed_ms=self.last_timing_ms,
            summary=(
                f"{len(selected)} selected, "
                f"{trace.counters.get('ignored_candidates', 0)} ignored"
            ),
        )

    # -- neighbourhood inspection ------------------------------------------ #

    def images_for_chunks(
        self, chunk_ids: Sequence[str], *, limit_per_chunk: int = 3
    ) -> dict[str, list[dict[str, Any]]]:
        """Image metadata for the pages a chunk covers.

        Used when multimodal retrieval is enabled; the graph already knows where
        every image belongs, so no visual embedding is required to find them.
        """
        if not chunk_ids:
            return {}
        records = self.store.read(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})-[:ON_PAGE]->(p:Page)
                  -[:HAS_IMAGE]->(i:Image)
            WITH chunk_id, i, p
            ORDER BY i.width * i.height DESC
            RETURN chunk_id,
                   collect({
                     image_id: i.image_id,
                     local_path: i.local_path,
                     page_number: i.page_number,
                     source_pdf: i.source_pdf,
                     width: i.width,
                     height: i.height,
                     format: i.format,
                     licence: i.licence,
                     licence_url: i.licence_url,
                     attribution: i.attribution,
                     creator: i.creator
                   })[0..$limit_per_chunk] AS images
            """,
            {"chunk_ids": list(chunk_ids), "limit_per_chunk": int(limit_per_chunk)},
        )
        return {record["chunk_id"]: record["images"] for record in records}
