"""Weighted Reciprocal Rank Fusion across the three retrieval channels.

Raw channel scores are never added together: a Neo4j cosine similarity (0..1), a
Lucene BM25-style score (unbounded) and a graph-expansion weight live on
different scales, and summing them would silently let one channel dominate. RRF
uses only *ranks*, which is scale-free:

    score(chunk) = sum over channels of  weight_channel / (k + rank_channel)

``k`` (default 60, the conventional value) damps the influence of top ranks; the
per-channel weights let dense and full-text act as primary signals while graph
expansion stays secondary.

Fusion also deduplicates: a chunk found by several channels becomes one
candidate whose ``retrieval_sources`` lists every channel that found it, and all
per-channel ranks and scores are carried through for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import RetrievalConfig
from .logging_utils import get_logger
from .schemas import (
    CHANNEL_DENSE,
    CHANNEL_FULLTEXT,
    CHANNEL_GRAPH,
    RetrievedChunk,
)

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ChannelResults:
    """One ranked channel entering fusion."""

    name: str
    results: Sequence[RetrievedChunk]
    weight: float


def rrf_score(rank: int, k: int, weight: float = 1.0) -> float:
    """Weighted reciprocal-rank contribution of a single ranked hit."""
    if rank < 1:
        raise ValueError(f"Ranks are 1-based, got {rank}")
    return weight / (k + rank)


def _merge_into(target: RetrievedChunk, source: RetrievedChunk) -> None:
    """Copy channel-specific signals from a duplicate into the kept candidate."""
    if not target.text and source.text:
        target.text = source.text
    for field_name in (
        "grade",
        "subject",
        "unit_id",
        "unit_title",
        "document_id",
        "document_title",
        "section_id",
        "section_title",
        "page_start",
        "page_end",
        "resource_type",
        "audience",
        "local_pdf_path",
    ):
        if getattr(target, field_name) is None:
            setattr(target, field_name, getattr(source, field_name))

    for field_name in (
        "dense_rank",
        "dense_score",
        "fulltext_rank",
        "fulltext_score",
        "graph_rank",
        "graph_score",
        "graph_expansion_path",
        "graph_seed_chunk_id",
    ):
        value = getattr(source, field_name)
        if value is not None and getattr(target, field_name) is None:
            setattr(target, field_name, value)

    for channel in source.retrieval_sources:
        target.add_source(channel)


def fuse(
    channels: Iterable[ChannelResults],
    *,
    k: int,
    top_k: int,
) -> list[RetrievedChunk]:
    """Fuse ranked channels into one deduplicated ranking."""
    # Materialised because the channels are traversed again for the summary log.
    channel_list = list(channels)
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}

    for channel in channel_list:
        for rank, chunk in enumerate(channel.results, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id in fused:
                _merge_into(fused[chunk_id], chunk)
            else:
                fused[chunk_id] = chunk
                scores[chunk_id] = 0.0
                contributions[chunk_id] = {}

            contribution = rrf_score(rank, k, channel.weight)
            scores[chunk_id] += contribution
            contributions[chunk_id][channel.name] = round(contribution, 8)

    ordered = sorted(
        fused.values(),
        key=lambda c: (scores[c.chunk_id], -(c.dense_rank or 10**6)),
        reverse=True,
    )

    for rank, chunk in enumerate(ordered, start=1):
        chunk.rrf_score = round(scores[chunk.chunk_id], 8)
        chunk.rrf_rank = rank
        chunk.rrf_contributions = contributions[chunk.chunk_id]

    selected = ordered[:top_k]
    LOGGER.info(
        "RRF fusion (k=%d): %d unique candidates from %d channel hits, "
        "keeping top %d",
        k,
        len(fused),
        sum(len(channel.results) for channel in channel_list),
        len(selected),
    )
    return selected


def fuse_standard_channels(
    dense: Sequence[RetrievedChunk],
    fulltext: Sequence[RetrievedChunk],
    graph: Sequence[RetrievedChunk],
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    """Fuse the three retrieval channels using the configured weights."""
    return fuse(
        [
            ChannelResults(CHANNEL_DENSE, dense, config.weight_dense),
            ChannelResults(CHANNEL_FULLTEXT, fulltext, config.weight_fulltext),
            ChannelResults(CHANNEL_GRAPH, graph, config.weight_graph),
        ],
        k=config.rrf_k,
        top_k=config.fusion_top_k,
    )
