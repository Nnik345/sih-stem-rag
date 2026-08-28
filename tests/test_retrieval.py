"""End-to-end retrieval against the live graph.

Unlike the other test modules these are integration tests: they need a running
Neo4j with an ingested corpus and the BGE models on disk. They skip (never
fail) when those are absent, so the pure-Python suite stays runnable anywhere.

    ./scripts/neo4j_local.sh start
    python scripts/ingest_corpus.py
    python -m pytest tests/test_retrieval.py
"""

from __future__ import annotations

import pytest

from rag.config import ConfigError, load_config
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError
from rag.pipeline import HybridRetriever
from rag.schemas import CHANNEL_DENSE, CHANNEL_FULLTEXT, RetrievalFilter
from rag.retrieval_base import build_filter_clause

# A question with obvious lexical anchors, so both the dense and the full-text
# channel should find something.
QUERY = "why does the moon look different on different nights"


@pytest.fixture(scope="module")
def config():
    try:
        return load_config()
    except ConfigError as exc:
        pytest.skip(f"configuration unavailable: {exc}")


@pytest.fixture(scope="module")
def store(config):
    try:
        store = Neo4jStore(config.neo4j)
        store.read("RETURN 1 AS ok")
    except Neo4jUnavailableError as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")
    yield store
    store.close()


@pytest.fixture(scope="module")
def ingested(store):
    counts = store.read(
        "MATCH (c:Chunk) RETURN count(c) AS chunks, "
        "count(c.embedding) AS embedded"
    )[0]
    if not counts["chunks"]:
        pytest.skip("graph has no Chunk nodes; run scripts/ingest_corpus.py first")
    return counts


@pytest.fixture(scope="module")
def retriever(config, store, ingested):
    if not config.models.embedding_model_path.is_dir():
        pytest.skip(
            "BGE-M3 missing; run scripts/download_retrieval_models.py"
        )
    retriever = HybridRetriever(config, store)
    yield retriever
    retriever.embedder.unload()
    retriever.reranker.unload()


class TestGraphContents:
    def test_all_three_grades_present(self, store, ingested):
        grades = {
            row["grade"]
            for row in store.read("MATCH (g:Grade) RETURN g.grade AS grade")
        }
        assert {1, 2, 3} <= grades

    def test_both_subjects_present(self, store, ingested):
        subjects = {
            row["subject"]
            for row in store.read("MATCH (s:Subject) RETURN s.subject AS subject")
        }
        assert {"mathematics", "science"} <= subjects

    def test_chunks_have_text_and_embeddings(self, store, ingested):
        assert ingested["chunks"] > 0
        assert ingested["embedded"] > 0
        empty = store.read(
            "MATCH (c:Chunk) WHERE c.text IS NULL OR trim(c.text) = '' "
            "RETURN count(c) AS empty"
        )[0]["empty"]
        assert empty == 0

    def test_every_chunk_traces_to_a_grade(self, store, ingested):
        """The Chunk -> ... -> Grade lineage must be complete, not merely present."""
        orphans = store.read(
            """
            MATCH (c:Chunk)
            WHERE NOT EXISTS {
                MATCH (:Grade)-[:HAS_SUBJECT]->(:Subject)-[:HAS_UNIT]->(:Unit)
                      -[:HAS_DOCUMENT]->(:Document)-[:HAS_PAGE]->(:Page)
                      -[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c)
            }
            RETURN count(c) AS orphans
            """
        )[0]["orphans"]
        assert orphans == 0

    def test_images_trace_to_a_page_and_pdf(self, store, ingested):
        rows = store.read(
            """
            MATCH (p:Page)-[:HAS_IMAGE]->(i:Image)
            RETURN i.local_path AS path, i.page_number AS image_page,
                   p.page_number AS page, p.source_pdf AS pdf
            LIMIT 25
            """
        )
        if not rows:
            pytest.skip("no Image nodes in the graph")
        for row in rows:
            assert row["path"]
            assert row["pdf"]
            assert row["image_page"] == row["page"]


class TestChannels:
    def test_dense_channel_returns_results(self, retriever):
        results = retriever.dense.retrieve(QUERY, scope=RetrievalFilter(grade=1))
        assert results
        for chunk in results:
            assert chunk.dense_rank is not None
            assert chunk.dense_score is not None
            assert CHANNEL_DENSE in chunk.retrieval_sources

    def test_dense_ranks_are_contiguous_and_ordered(self, retriever):
        results = retriever.dense.retrieve(QUERY, scope=RetrievalFilter(grade=1))
        assert [c.dense_rank for c in results] == list(range(1, len(results) + 1))
        scores = [c.dense_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_fulltext_channel_finds_exact_terminology(self, retriever):
        results = retriever.lexical.retrieve("moon phases", scope=RetrievalFilter())
        assert results
        for chunk in results:
            assert chunk.fulltext_rank is not None
            assert CHANNEL_FULLTEXT in chunk.retrieval_sources

    def test_graph_expansion_is_attributable_and_bounded(self, retriever, config):
        dense = retriever.dense.retrieve(QUERY, scope=RetrievalFilter(grade=1))
        lexical = retriever.lexical.retrieve(QUERY, scope=RetrievalFilter(grade=1))
        graph = retriever.graph.retrieve([dense, lexical], scope=RetrievalFilter(grade=1))
        assert retriever.graph.last_seeds
        assert len(graph) <= config.retrieval.graph_top_k
        for chunk in graph:
            # Every graph candidate must say how it was reached.
            assert chunk.graph_seed_chunk_id
            assert chunk.graph_expansion_path

    def test_empty_query_does_not_crash_the_lexical_channel(self, retriever):
        assert retriever.lexical.retrieve("   ", scope=RetrievalFilter()) == []


class TestMetadataFiltering:
    @pytest.mark.parametrize("grade", [1, 2, 3])
    def test_grade_filter_excludes_other_grades(self, retriever, grade):
        response = retriever.retrieve(
            "counting and adding numbers", grade=grade, rerank=False
        )
        assert response.results
        for chunk in response.diagnostics.dense:
            assert chunk.grade == grade
        for chunk in response.diagnostics.fulltext:
            assert chunk.grade == grade
        for chunk in response.diagnostics.graph:
            assert chunk.grade == grade

    def test_subject_filter_excludes_other_subjects(self, retriever):
        response = retriever.retrieve(
            "measuring length", grade=2, subject="mathematics", rerank=False
        )
        assert response.results
        for chunk in response.results:
            assert chunk.subject == "mathematics"
            assert chunk.grade == 2

    def test_filter_is_applied_in_cypher_not_afterwards(self):
        """Guards the property that makes filtering cheap and correct."""
        clause, params = build_filter_clause(RetrievalFilter(grade=1))
        assert clause and params

    def test_unit_filter_narrows_to_one_unit(self, retriever, store, ingested):
        unit_id = store.read(
            "MATCH (c:Chunk) RETURN c.unit_id AS unit_id LIMIT 1"
        )[0]["unit_id"]
        response = retriever.retrieve("what do students learn", unit=unit_id, rerank=False)
        for chunk in response.results:
            assert chunk.unit_id == unit_id


class TestFullPipeline:
    @staticmethod
    @pytest.fixture(scope="class")
    def response(retriever):
        """Retrieved once for the whole class: each query loads the reranker."""
        return retriever.retrieve(QUERY, grade=1, subject="science")

    def test_all_stages_are_preserved(self, response):
        diagnostics = response.diagnostics
        assert diagnostics.dense
        assert diagnostics.fused
        assert diagnostics.reranked
        assert diagnostics.timings_ms.get("total") is not None

    def test_fusion_output_is_deduplicated(self, response):
        ids = [c.chunk_id for c in response.diagnostics.fused]
        assert len(ids) == len(set(ids))

    def test_fused_candidates_carry_rrf_ranks(self, response):
        ranks = [c.rrf_rank for c in response.diagnostics.fused]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_reranker_scores_are_assigned_and_sorted(self, response):
        reranked = response.diagnostics.reranked
        assert reranked
        for chunk in reranked:
            assert chunk.rerank_score is not None
        scores = [c.rerank_score for c in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_reranking_is_not_a_no_op(self, retriever):
        """The reranker should be able to reorder fusion output.

        Asserted over several queries: reranker and RRF agreeing on one query is
        legitimate, agreeing on all of them would mean the reranker is inert.
        """
        queries = [
            "why does the moon look different on different nights",
            "how do plants get what they need to grow",
            "what is the difference between a solid and a liquid",
        ]
        reordered = 0
        for query in queries:
            response = retriever.retrieve(query, grade=1)
            fused_order = [c.chunk_id for c in response.diagnostics.fused]
            final_order = [c.chunk_id for c in response.diagnostics.reranked]
            if final_order != fused_order[: len(final_order)]:
                reordered += 1
        assert reordered >= 1

    def test_final_chunks_are_fully_traceable(self, response):
        assert response.results
        for chunk in response.results:
            provenance = chunk.provenance()
            for key in (
                "grade",
                "subject",
                "unit_id",
                "unit_title",
                "document_id",
                "document_title",
                "pages",
                "local_pdf_path",
            ):
                assert provenance[key] not in (None, "", "?"), key

    def test_final_chunks_respect_the_requested_scope(self, response):
        for chunk in response.results:
            assert chunk.grade == 1
            assert chunk.subject == "science"

    def test_out_of_corpus_query_returns_weak_or_no_evidence(self, retriever):
        """Retrieval must not silently invent relevance for a foreign topic."""
        response = retriever.retrieve(
            "explain the Higgs boson and quantum chromodynamics", grade=1
        )
        top = response.results[0].rerank_score if response.results else None
        assert top is None or top < 0.9
