"""Weighted Reciprocal Rank Fusion.

Fusion is pure arithmetic over ranks, so it is tested directly with synthetic
channel results: no Neo4j, no models.
"""

from __future__ import annotations

import pytest

from rag.config import RetrievalConfig
from rag.fusion import ChannelResults, fuse, fuse_standard_channels, rrf_score
from rag.schemas import (
    CHANNEL_DENSE,
    CHANNEL_FULLTEXT,
    CHANNEL_GRAPH,
    RetrievedChunk,
)


def dense_hit(chunk_id: str, rank: int, score: float = 0.9) -> RetrievedChunk:
    chunk = RetrievedChunk(chunk_id=chunk_id, text=f"text of {chunk_id}", grade=1)
    chunk.dense_rank = rank
    chunk.dense_score = score
    chunk.add_source(CHANNEL_DENSE)
    return chunk


def fulltext_hit(chunk_id: str, rank: int, score: float = 4.2) -> RetrievedChunk:
    chunk = RetrievedChunk(chunk_id=chunk_id, text=f"text of {chunk_id}")
    chunk.fulltext_rank = rank
    chunk.fulltext_score = score
    chunk.add_source(CHANNEL_FULLTEXT)
    return chunk


def graph_hit(chunk_id: str, rank: int, seed: str = "seed-1") -> RetrievedChunk:
    chunk = RetrievedChunk(chunk_id=chunk_id, text=f"text of {chunk_id}")
    chunk.graph_rank = rank
    chunk.graph_score = 0.5
    chunk.graph_seed_chunk_id = seed
    chunk.graph_expansion_path = f"SAME_SECTION via {seed}"
    chunk.add_source(CHANNEL_GRAPH)
    return chunk


class TestRrfScore:
    def test_matches_the_formula(self):
        assert rrf_score(1, 60) == pytest.approx(1 / 61)
        assert rrf_score(3, 60, 0.5) == pytest.approx(0.5 / 63)

    def test_decreases_with_rank(self):
        scores = [rrf_score(rank, 60) for rank in range(1, 6)]
        assert scores == sorted(scores, reverse=True)

    def test_k_damps_the_top_rank_advantage(self):
        small_k = rrf_score(1, 1) / rrf_score(2, 1)
        large_k = rrf_score(1, 60) / rrf_score(2, 60)
        assert small_k > large_k

    def test_zero_and_negative_ranks_rejected(self):
        """Ranks are 1-based; a 0 would silently inflate a candidate."""
        with pytest.raises(ValueError):
            rrf_score(0, 60)
        with pytest.raises(ValueError):
            rrf_score(-1, 60)


class TestFusion:
    def test_deduplicates_across_channels(self):
        fused = fuse(
            [
                ChannelResults(CHANNEL_DENSE, [dense_hit("a", 1), dense_hit("b", 2)], 1.0),
                ChannelResults(
                    CHANNEL_FULLTEXT, [fulltext_hit("b", 1), fulltext_hit("c", 2)], 1.0
                ),
            ],
            k=60,
            top_k=10,
        )
        assert [c.chunk_id for c in fused].count("b") == 1
        assert len(fused) == 3

    def test_multi_channel_agreement_wins(self):
        """A chunk found by two channels outranks one found by a single channel."""
        fused = fuse(
            [
                ChannelResults(CHANNEL_DENSE, [dense_hit("a", 1), dense_hit("b", 2)], 1.0),
                ChannelResults(CHANNEL_FULLTEXT, [fulltext_hit("b", 1)], 1.0),
            ],
            k=60,
            top_k=10,
        )
        assert fused[0].chunk_id == "b"
        assert set(fused[0].retrieval_sources) == {CHANNEL_DENSE, CHANNEL_FULLTEXT}

    def test_records_contributing_channels(self):
        fused = fuse(
            [
                ChannelResults(CHANNEL_DENSE, [dense_hit("a", 1)], 1.0),
                ChannelResults(CHANNEL_FULLTEXT, [fulltext_hit("a", 2)], 1.0),
                ChannelResults(CHANNEL_GRAPH, [graph_hit("a", 1)], 0.5),
            ],
            k=60,
            top_k=10,
        )
        candidate = fused[0]
        assert candidate.retrieval_sources == [
            CHANNEL_DENSE,
            CHANNEL_FULLTEXT,
            CHANNEL_GRAPH,
        ]
        assert candidate.retrieval_source == "dense+fulltext+graph"
        assert set(candidate.rrf_contributions) == {"dense", "fulltext", "graph"}
        # Both sides are rounded to 8 decimals independently, hence the tolerance.
        assert candidate.rrf_score == pytest.approx(
            sum(candidate.rrf_contributions.values()), abs=1e-7
        )

    def test_weights_change_the_ordering(self):
        channels = lambda graph_weight: [  # noqa: E731
            ChannelResults(CHANNEL_DENSE, [dense_hit("a", 1)], 1.0),
            ChannelResults(CHANNEL_GRAPH, [graph_hit("g", 1)], graph_weight),
        ]
        secondary = fuse(channels(0.4), k=60, top_k=10)
        assert secondary[0].chunk_id == "a"

        dominant = fuse(channels(5.0), k=60, top_k=10)
        assert dominant[0].chunk_id == "g"

    def test_assigns_dense_rrf_ranks_from_one(self):
        fused = fuse(
            [
                ChannelResults(
                    CHANNEL_DENSE,
                    [dense_hit("a", 1), dense_hit("b", 2), dense_hit("c", 3)],
                    1.0,
                ),
            ],
            k=60,
            top_k=10,
        )
        assert [c.rrf_rank for c in fused] == [1, 2, 3]
        assert [c.chunk_id for c in fused] == ["a", "b", "c"]

    def test_ranks_are_assigned_before_truncation(self):
        """top_k must not renumber the candidates that survive."""
        hits = [dense_hit(f"c{i}", i) for i in range(1, 6)]
        fused = fuse([ChannelResults(CHANNEL_DENSE, hits, 1.0)], k=60, top_k=2)
        assert len(fused) == 2
        assert [c.rrf_rank for c in fused] == [1, 2]

    def test_scores_are_monotonically_non_increasing(self):
        hits_dense = [dense_hit(f"d{i}", i) for i in range(1, 5)]
        hits_lexical = [fulltext_hit(f"d{i}", 5 - i) for i in range(1, 5)]
        fused = fuse(
            [
                ChannelResults(CHANNEL_DENSE, hits_dense, 1.0),
                ChannelResults(CHANNEL_FULLTEXT, hits_lexical, 1.0),
            ],
            k=60,
            top_k=10,
        )
        scores = [c.rrf_score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_per_channel_signals_for_later_analysis(self):
        """Nothing about how a candidate was found may be discarded."""
        fused = fuse(
            [
                ChannelResults(CHANNEL_DENSE, [dense_hit("a", 2, score=0.77)], 1.0),
                ChannelResults(CHANNEL_FULLTEXT, [fulltext_hit("a", 5, score=3.1)], 1.0),
                ChannelResults(CHANNEL_GRAPH, [graph_hit("a", 1, seed="seed-9")], 0.4),
            ],
            k=60,
            top_k=10,
        )
        candidate = fused[0]
        assert candidate.dense_rank == 2
        assert candidate.dense_score == pytest.approx(0.77)
        assert candidate.fulltext_rank == 5
        assert candidate.fulltext_score == pytest.approx(3.1)
        assert candidate.graph_rank == 1
        assert candidate.graph_seed_chunk_id == "seed-9"
        assert "SAME_SECTION" in candidate.graph_expansion_path

    def test_metadata_is_filled_in_from_whichever_channel_has_it(self):
        """Graph hits can arrive without full metadata; merging must repair it."""
        bare = graph_hit("a", 1)
        bare.grade = None
        fused = fuse(
            [
                ChannelResults(CHANNEL_GRAPH, [bare], 0.4),
                ChannelResults(CHANNEL_DENSE, [dense_hit("a", 1)], 1.0),
            ],
            k=60,
            top_k=10,
        )
        assert fused[0].grade == 1

    def test_empty_channels_produce_empty_result(self):
        assert fuse([], k=60, top_k=5) == []
        assert (
            fuse([ChannelResults(CHANNEL_DENSE, [], 1.0)], k=60, top_k=5) == []
        )

    def test_accepts_a_generator_of_channels(self):
        """The summary log re-reads the channels, so they must be materialised."""
        channels = (
            ChannelResults(name, [dense_hit("a", 1)], 1.0)
            for name in (CHANNEL_DENSE, CHANNEL_FULLTEXT)
        )
        assert len(fuse(channels, k=60, top_k=5)) == 1


class TestStandardChannels:
    def test_uses_configured_weights_and_limits(self):
        config = RetrievalConfig(
            rrf_k=60,
            fusion_top_k=2,
            weight_dense=1.0,
            weight_fulltext=1.0,
            weight_graph=0.4,
        )
        fused = fuse_standard_channels(
            [dense_hit("a", 1), dense_hit("b", 2)],
            [fulltext_hit("c", 1)],
            [graph_hit("d", 1)],
            config,
        )
        assert len(fused) == config.fusion_top_k
        assert fused[0].chunk_id in {"a", "c"}

    def test_graph_is_secondary_by_default(self):
        config = RetrievalConfig()
        assert config.weight_graph < config.weight_dense
        assert config.weight_graph < config.weight_fulltext
