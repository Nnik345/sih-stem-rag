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
from rag.curriculum_catalog import in_lineage_scope
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError
from rag.pipeline import HybridRetriever
from rag.query_rewrite import QueryRewriteResult
from rag.schemas import CHANNEL_DENSE, CHANNEL_FULLTEXT, RetrievalFilter
from rag.retrieval_base import build_filter_clause

# A question with obvious lexical anchors in NCERT maths (ingested first).
QUERY = "counting numbers and shapes"


def _assert_lineage(chunk, grade: int, subject: str) -> None:
    assert in_lineage_scope(
        chunk_grade=chunk.grade,
        chunk_subject=chunk.subject,
        current_grade=grade,
        current_subject=subject,
        allow_prior_grades=True,
    )


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


class _PassthroughRewriter:
    """Avoid loading the 2B VL rewriter on every retrieval integration test."""

    def rewrite(self, query: str, *, grade=None, subject=None, **kwargs) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            retrieval_query=query,
            intent="other",
            fallback=True,
            reason="test passthrough",
        )

    def unload(self) -> None:
        return None


@pytest.fixture(scope="module")
def retriever(config, store, ingested):
    if not config.models.embedding_model_path.is_dir():
        pytest.skip(
            "BGE-M3 missing; run scripts/download_retrieval_models.py"
        )
    retriever = HybridRetriever(config, store, rewriter=_PassthroughRewriter())
    yield retriever
    retriever.release_models()


@pytest.fixture(scope="module")
def any_grade(store, ingested):
    rows = store.read("MATCH (g:Grade) RETURN g.grade AS grade ORDER BY g.grade LIMIT 1")
    return int(rows[0]["grade"])


class TestGraphContents:
    def test_grades_are_within_one_to_twelve(self, store, ingested):
        grades = {
            row["grade"]
            for row in store.read("MATCH (g:Grade) RETURN g.grade AS grade")
        }
        assert grades
        assert grades <= set(range(1, 13))

    def test_only_ncert_source_ids(self, store, ingested):
        leftover = store.read(
            """
            MATCH (c:Chunk)
            WHERE coalesce(c.source_id, '') <> 'ncert_textbook'
            RETURN count(c) AS n
            """
        )[0]["n"]
        assert leftover == 0

    def test_no_core_knowledge_nodes(self, store, ingested):
        leftover = store.read(
            """
            MATCH (n)
            WHERE toLower(coalesce(n.local_pdf_path, '')) CONTAINS 'core_knowledge'
               OR toLower(coalesce(n.publisher, '')) CONTAINS 'core knowledge'
            RETURN count(n) AS n
            """
        )[0]["n"]
        assert leftover == 0

    def test_no_evaluation_only_chunks(self, store, ingested):
        count = store.read(
            "MATCH (c:Chunk {content_partition: 'evaluation_only'}) RETURN count(c) AS n"
        )[0]["n"]
        assert count == 0

    def test_both_subjects_present(self, store, ingested):
        subjects = {
            row["subject"]
            for row in store.read("MATCH (s:Subject) RETURN s.subject AS subject")
        }
        if {"mathematics", "science"} - subjects:
            pytest.skip("both subjects not ingested yet")
        assert {"mathematics", "science"} <= subjects

    def test_class_11_12_pcb_subjects_when_ingested(self, store, ingested):
        subjects = {
            (row["grade"], row["subject"])
            for row in store.read(
                "MATCH (g:Grade)-[:HAS_SUBJECT]->(s:Subject) "
                "RETURN g.grade AS grade, s.subject AS subject"
            )
        }
        pcb = {"physics", "chemistry", "biology"}
        senior = {subject for grade, subject in subjects if grade in (11, 12)}
        if not senior:
            pytest.skip("classes 11–12 not ingested yet")
        if not (pcb & senior):
            pytest.skip("PCB subjects not ingested yet")
        assert pcb <= senior
        assert "science" not in senior

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
    def test_dense_channel_returns_results(self, retriever, any_grade):
        results = retriever.dense.retrieve(QUERY, scope=RetrievalFilter(grade=any_grade))
        if not results:
            pytest.skip(f"no dense hits for grade {any_grade} yet")
        for chunk in results:
            assert chunk.dense_rank is not None
            assert chunk.dense_score is not None
            assert CHANNEL_DENSE in chunk.retrieval_sources

    def test_dense_ranks_are_contiguous_and_ordered(self, retriever, any_grade):
        results = retriever.dense.retrieve(QUERY, scope=RetrievalFilter(grade=any_grade))
        if not results:
            pytest.skip(f"no dense hits for grade {any_grade} yet")
        assert [c.dense_rank for c in results] == list(range(1, len(results) + 1))
        scores = [c.dense_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_fulltext_channel_finds_exact_terminology(self, retriever):
        results = retriever.lexical.retrieve("numbers", scope=RetrievalFilter())
        if not results:
            pytest.skip("no lexical hits yet")
        for chunk in results:
            assert chunk.fulltext_rank is not None
            assert CHANNEL_FULLTEXT in chunk.retrieval_sources

    def test_graph_expansion_is_attributable_and_bounded(self, retriever, config, any_grade):
        scope = RetrievalFilter(grade=any_grade)
        dense = retriever.dense.retrieve(QUERY, scope=scope)
        lexical = retriever.lexical.retrieve(QUERY, scope=scope)
        if not dense and not lexical:
            pytest.skip(f"no seeds for grade {any_grade} yet")
        graph = retriever.graph.retrieve([dense, lexical], scope=scope)
        assert retriever.graph.last_seeds
        assert len(graph) <= config.retrieval.graph_top_k
        for chunk in graph:
            assert chunk.graph_seed_chunk_id
            assert chunk.graph_expansion_path

    def test_empty_query_does_not_crash_the_lexical_channel(self, retriever):
        assert retriever.lexical.retrieve("   ", scope=RetrievalFilter()) == []


class TestMetadataFiltering:
    @pytest.mark.parametrize("grade", [6, 10, 12])
    def test_grade_filter_excludes_other_grades(self, retriever, store, grade):
        present = {
            row["grade"]
            for row in store.read("MATCH (g:Grade) RETURN g.grade AS grade")
        }
        if grade not in present:
            pytest.skip(f"grade {grade} not ingested yet")
        response = retriever.retrieve(
            "numbers and measurement", grade=grade, rerank=False
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
            "how do I measure length", grade=6, subject="mathematics", rerank=False
        )
        if not response.results:
            pytest.skip("class 6 mathematics not ingested yet")
        for chunk in response.results:
            _assert_lineage(chunk, 6, "mathematics")

    def test_unsupported_grades_return_no_results(self, retriever):
        response = retriever.retrieve("numbers", grade=13, rerank=False)
        assert not response.results
        assert not response.diagnostics.dense
        assert not response.diagnostics.fulltext

    def test_filter_is_applied_in_cypher_not_afterwards(self):
        """Guards the property that makes filtering cheap and correct."""
        clause, params = build_filter_clause(RetrievalFilter(grade=3))
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
        result = retriever.retrieve("what are the components of food", grade=6, subject="science")
        if not result.results and not result.diagnostics.dense:
            pytest.skip("class 6 science not ingested yet")
        return result

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
            "what are the components of food",
            "how does light reflect from a plane mirror",
            "what is electrostatics",
        ]
        reordered = 0
        ran = 0
        for query in queries:
            response = retriever.retrieve(query, grade=6)
            if not response.diagnostics.fused:
                continue
            ran += 1
            fused_order = [c.chunk_id for c in response.diagnostics.fused]
            final_order = [c.chunk_id for c in response.diagnostics.reranked]
            if final_order != fused_order[: len(final_order)]:
                reordered += 1
        if ran == 0:
            pytest.skip("class 6 not ingested yet")
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
                "source_id",
                "source_role",
                "licence",
                "content_partition",
            ):
                assert provenance[key] not in (None, "", "?"), key
            assert chunk.alignment_status
            assert chunk.content_partition in {
                "student_evidence",
                "teacher_strategy",
            }

    def test_final_chunks_respect_the_requested_scope(self, response):
        for chunk in response.results:
            _assert_lineage(chunk, 6, "science")

    def test_ncert_maths_retrieves_primary_textbook(self, retriever):
        response = retriever.retrieve(
            "what is a fraction",
            grade=6,
            subject="mathematics",
            rerank=False,
        )
        if not response.results:
            pytest.skip("class 6 mathematics not ingested yet")
        for chunk in response.results:
            assert chunk.source_id == "ncert_textbook"
            assert chunk.source_role == "primary"
            assert chunk.subject == "mathematics"
            assert chunk.licence
            assert chunk.content_partition != "evaluation_only"
            assert not chunk.cisce_outcome_ids

    def test_ncert_science_retrieves_primary_textbook(self, retriever):
        response = retriever.retrieve(
            "what are the components of food",
            grade=6,
            subject="science",
            rerank=False,
        )
        if not response.results:
            pytest.skip("class 6 science not ingested yet")
        for chunk in response.results:
            assert chunk.source_id == "ncert_textbook"
            assert chunk.source_role == "primary"
            assert chunk.licence
            assert chunk.content_partition != "evaluation_only"
            assert not chunk.cisce_outcome_ids

    def test_class10_light_retrieves_ncert_science(self, retriever):
        response = retriever.retrieve(
            "how does light reflect from a plane mirror",
            grade=10,
            subject="science",
            rerank=False,
        )
        if not response.results:
            pytest.skip("class 10 science not ingested yet")
        for chunk in response.results:
            assert chunk.source_id == "ncert_textbook"
            _assert_lineage(chunk, 10, "science")

    def test_class12_electrostatics_retrieves_ncert_physics(self, retriever):
        response = retriever.retrieve(
            "what is electrostatics and electric charge",
            grade=12,
            subject="physics",
            rerank=False,
        )
        if not response.results:
            pytest.skip("class 12 physics not ingested yet")
        for chunk in response.results:
            assert chunk.source_id == "ncert_textbook"
            _assert_lineage(chunk, 12, "physics")

    def test_production_never_returns_evaluation_only(self, retriever):
        response = retriever.retrieve(
            "answers to the exercises",
            grade=6,
            subject="mathematics",
            rerank=False,
        )
        for chunk in (
            list(response.diagnostics.dense)
            + list(response.diagnostics.fulltext)
            + list(response.diagnostics.graph)
            + list(response.results)
        ):
            assert chunk.content_partition != "evaluation_only"

    def test_homework_and_answer_key_queries_stay_on_safe_partitions(self, retriever):
        for query in (
            "answers to exercises",
            "answer key",
            "solutions to the exercises",
        ):
            response = retriever.retrieve(
                query, grade=6, subject="mathematics", rerank=False
            )
            for chunk in (
                list(response.diagnostics.dense)
                + list(response.diagnostics.fulltext)
                + list(response.results)
            ):
                assert chunk.content_partition in {
                    None,
                    "student_evidence",
                    "teacher_strategy",
                }

    def test_licence_boilerplate_is_not_returned(self, retriever):
        response = retriever.retrieve(
            "ncert copyright all rights reserved isbn",
            grade=6,
            rerank=False,
        )
        for chunk in response.results:
            text = (chunk.text or "").lower()
            assert "this work is licensed under a creative commons" not in text

    def test_class6_food_retrieves_curriculum_text(self, retriever):
        response = retriever.retrieve(
            "what are the components of food",
            grade=6,
            subject="science",
            rerank=False,
        )
        if not response.results:
            pytest.skip("class 6 science not ingested yet")
        blob = " ".join(c.text.lower() for c in response.results)
        assert "food" in blob or "nutrient" in blob or "carbohydrate" in blob or "protein" in blob

    def test_out_of_corpus_query_returns_weak_or_no_evidence(self, retriever):
        """Retrieval must not silently invent relevance for a foreign topic."""
        response = retriever.retrieve(
            "explain the Higgs boson and quantum chromodynamics", grade=6
        )
        top = response.results[0].rerank_score if response.results else None
        assert top is None or top < 0.9


class TestLiveQueryRewrite:
    """Uses the real 2B rewriter. Skips when the checkpoint is not on disk."""

    def test_class12_maths_colloquial_rewrite_retrieves_derivatives(
        self, config, store, ingested
    ):
        if not config.models.rewriter_model_path.is_dir():
            pytest.skip("Qwen3-VL-2B rewriter missing; run scripts/download_qwen_models.py")
        retriever = HybridRetriever(config, store)
        try:
            response = retriever.retrieve(
                "what is differentiation of x squared",
                grade=12,
                subject="mathematics",
                rerank=True,
            )
        finally:
            retriever.release_models()
        assert response.diagnostics.retrieval_query
        if response.diagnostics.rewrite_fallback:
            pytest.skip(
                f"rewriter fell back: {response.diagnostics.notes}"
            )
        rewritten = response.diagnostics.retrieval_query.lower()
        assert rewritten != "what is differentiation of x squared" or "derivative" in rewritten
        if not response.results:
            pytest.skip("class 12 mathematics not ingested yet")
        for chunk in response.results:
            _assert_lineage(chunk, 12, "mathematics")
        blob = " ".join(c.text.lower() for c in response.results)
        assert any(
            token in blob
            for token in ("derivative", "differenti", "limit", "function")
        )

