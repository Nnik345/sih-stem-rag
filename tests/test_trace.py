"""Trace model serialization and pipeline observer behaviour."""

from __future__ import annotations

import json

import pytest

from rag.config import EvidenceConfig, RetrievalConfig
from rag.evidence import EvidenceGate
from rag.fusion import fuse_standard_channels
from rag.pipeline import (
    _build_evidence_trace,
    _build_fusion_trace,
    _build_prompt_trace,
    _build_reranker_trace,
)
from rag.schemas import (
    CHANNEL_DENSE,
    CHANNEL_FULLTEXT,
    CHANNEL_GRAPH,
    RetrievalDiagnostics,
    RetrievalFilter,
    RetrievedChunk,
)
from rag.socratic import SocraticController, TutorState
from rag.trace import (
    DenseTrace,
    RecordingObserver,
    RunTrace,
    RunTraceCollector,
    candidate_from_chunk,
    emit,
    payload_contains_secrets,
    safe_error_message,
)


def _chunk(chunk_id: str, text: str = "Weather can change from day to day", **kwargs) -> RetrievedChunk:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        grade=3,
        subject="science",
        unit_id="grade_03:science:unit_01_science_oer",
        unit_title="Utah Science OER Textbook",
        document_id="doc-1",
        document_title="Student Book",
        section_title="Weather",
        page_start=4,
        page_end=5,
        local_pdf_path="/tmp/book.pdf",
    )
    for key, value in kwargs.items():
        setattr(chunk, key, value)
    return chunk


class TestTraceModels:
    def test_run_trace_serializes(self):
        trace = RunTrace(query="how does weather change", retrieval_only=True)
        payload = trace.to_dict()
        assert payload["query"] == "how does weather change"
        assert payload["status"] == "queued"
        assert set(payload["stages"]) == {
            "query",
            "filters",
            "rewrite",
            "image",
            "dense",
            "lexical",
            "graph",
            "fusion",
            "reranker",
            "evidence",
            "prompt",
            "generator",
        }
        json.dumps(payload)

    def test_diagnostics_keeps_existing_fields(self):
        diagnostics = RetrievalDiagnostics(
            query="q",
            scope=RetrievalFilter(grade=1),
            dense=[_chunk("c1", dense_rank=1, dense_score=0.9)],
        )
        payload = diagnostics.to_dict()
        for key in (
            "query",
            "scope",
            "counts",
            "graph_seeds",
            "timings_ms",
            "notes",
            "dense",
            "fulltext",
            "graph",
            "fused",
            "reranked",
        ):
            assert key in payload
        assert payload["graph_trace"] is None
        assert payload["dense"][0]["chunk_id"] == "c1"

    def test_no_secrets_in_serialized_trace(self):
        collector = RunTraceCollector()
        collector.on_event(
            "run_failed",
            {"error": safe_error_message(RuntimeError("NEO4J_PASSWORD=super-secret"))},
        )
        blob = json.dumps(collector.snapshot().to_dict())
        assert "NEO4J_PASSWORD" not in blob
        assert "super-secret" not in blob
        assert not payload_contains_secrets(collector.snapshot().to_dict())


class TestObserver:
    def test_disabled_observer_emits_nothing(self):
        observer = RecordingObserver()
        emit(None, "dense_started")
        emit(None, "run_failed", error="nope")
        assert observer.events == []

    def test_enabled_observer_records_events(self):
        observer = RecordingObserver()
        emit(observer, "run_started", query="q")
        emit(observer, "dense_started")
        emit(observer, "dense_completed", dense=DenseTrace(strategy="vector_index"))
        assert observer.names == ["run_started", "dense_started", "dense_completed"]

    def test_event_order_matches_pipeline_order(self):
        collector = RunTraceCollector()
        order = [
            "run_started",
            "filters_applied",
            "dense_started",
            "dense_completed",
            "lexical_started",
            "lexical_completed",
            "graph_started",
            "graph_completed",
            "fusion_completed",
            "reranker_started",
            "reranker_completed",
            "evidence_completed",
            "prompt_built",
            "generation_started",
            "generation_token",
            "generation_completed",
            "run_completed",
        ]
        for event in order:
            collector.on_event(event, {"token": "x"} if event == "generation_token" else {"query": "q"})
        assert collector.trace.events == order
        assert collector.trace.status.value == "completed"

    def test_run_completed_replaces_graph_with_annotated_trace(self):
        collector = RunTraceCollector()
        collector.on_event(
            "graph_completed",
            {
                "graph": {
                    "nodes": [
                        {
                            "node_id": "seed-1",
                            "node_kind": "seed",
                            "status": "seed",
                            "final_evidence": False,
                            "metadata": {},
                        }
                    ]
                }
            },
        )
        collector.on_event(
            "run_completed",
            {
                "diagnostics": {
                    "graph_trace": {
                        "nodes": [
                            {
                                "node_id": "seed-1",
                                "node_kind": "seed",
                                "status": "evidence",
                                "final_evidence": True,
                                "metadata": {"visual": "evidence"},
                            }
                        ]
                    }
                }
            },
        )
        node = collector.trace.graph["nodes"][0]
        assert node["final_evidence"] is True
        assert node["status"] == "evidence"
        assert node["metadata"]["visual"] == "evidence"

    def test_attach_graph_trace_writes_collector_snapshot(self):
        from rag.trace import attach_graph_trace

        collector = RunTraceCollector()
        attach_graph_trace(collector, {"nodes": [{"node_id": "c1", "final_evidence": True}]})
        assert collector.trace.graph["nodes"][0]["final_evidence"] is True

    def test_annotate_graph_marks_seed_as_final_evidence(self):
        from rag.pipeline import _annotate_graph_later_status
        from rag.schemas import RetrievalDiagnostics, RetrievalResponse

        scope = RetrievalFilter(grade=3, subject="science")
        graph = {
            "nodes": [
                {"node_id": "seed-1", "node_kind": "seed", "status": "seed", "metadata": {}},
                {"node_id": "n2", "node_kind": "candidate", "status": "accepted", "metadata": {}},
            ]
        }
        retrieval = RetrievalResponse(
            query="q",
            scope=scope,
            results=[],
            diagnostics=RetrievalDiagnostics(query="q", scope=scope, graph_trace=graph),
        )
        _annotate_graph_later_status(retrieval, {"seed-1", "n2"}, {"seed-1"})
        by_id = {node["node_id"]: node for node in graph["nodes"]}
        assert by_id["seed-1"]["final_evidence"] is True
        assert by_id["seed-1"]["status"] == "evidence"
        assert by_id["seed-1"]["metadata"]["visual"] == "evidence"
        assert by_id["n2"]["final_evidence"] is False
        assert by_id["n2"]["status"] == "accepted"

    def test_stage_error_produces_run_failed(self):
        collector = RunTraceCollector()
        collector.on_event("run_started", {"query": "q"})
        collector.on_event("dense_started", {})
        collector.on_event("run_failed", {"error": "boom", "stage": "dense"})
        assert collector.trace.status.value == "failed"
        assert collector.trace.stages["dense"].status.value == "failed"
        assert collector.trace.error == "boom"


class TestChannelTraces:
    def test_dense_candidates_have_ranks_and_scores(self):
        chunk = _chunk("c1", dense_rank=1, dense_score=0.87)
        chunk.add_source(CHANNEL_DENSE)
        traced = candidate_from_chunk(chunk, rank=1, score=0.87)
        payload = traced.to_dict()
        assert payload["rank"] == 1
        assert payload["score"] == pytest.approx(0.87)
        assert payload["provenance"]["chunk_id"] == "c1"

    def test_fusion_contributions_sum_to_rrf_score(self):
        dense = [_chunk("a", dense_rank=1, dense_score=0.9)]
        dense[0].add_source(CHANNEL_DENSE)
        lexical = [_chunk("a", fulltext_rank=1, fulltext_score=3.0)]
        lexical[0].add_source(CHANNEL_FULLTEXT)
        graph = [_chunk("a", graph_rank=1, graph_score=0.4)]
        graph[0].add_source(CHANNEL_GRAPH)
        fused = fuse_standard_channels(dense, lexical, graph, RetrievalConfig())
        trace = _build_fusion_trace(fused, RetrievalConfig())
        candidate = trace.candidates[0]
        assert candidate.rrf_score == pytest.approx(
            candidate.dense_contribution
            + candidate.lexical_contribution
            + candidate.graph_contribution,
            abs=1e-7,
        )

    def test_reranker_rank_movement(self):
        first = _chunk("a", rrf_rank=2, rerank_rank=1, rerank_score=1.2)
        second = _chunk("b", rrf_rank=1, rerank_rank=2, rerank_score=0.1)
        trace = _build_reranker_trace([first, second], {"a"})
        by_id = {c.chunk_id: c for c in trace.candidates}
        assert by_id["a"].rank_movement == 1
        assert by_id["b"].rank_movement == -1
        assert by_id["a"].survived_final_top_k is True
        assert by_id["b"].survived_final_top_k is False

    def test_evidence_trace_matches_decision(self):
        gate = EvidenceGate(EvidenceConfig(min_rerank_score=-10.0, min_query_term_overlap=0.0))
        chunk = _chunk("c1", rerank_score=2.0, rerank_rank=1)
        decision = gate.evaluate("how does weather change", [chunk], scope=RetrievalFilter(grade=3, subject="science"))
        trace = _build_evidence_trace(decision, gate.config)
        assert trace.sufficient is decision.sufficient
        assert trace.confidence == decision.confidence
        assert [c.name for c in trace.checks] == [c.name for c in decision.checks]
        assert [c.passed for c in trace.checks] == [c.passed for c in decision.checks]
        assert trace.kept_chunk_ids == [c.chunk_id for c in decision.kept_chunks]

    def test_prompt_trace_matches_generator_messages(self):
        gate = EvidenceGate(EvidenceConfig(min_rerank_score=-10.0, min_query_term_overlap=0.0))
        chunk = _chunk("c1", rerank_score=2.0)
        decision = gate.evaluate("how does weather change", [chunk], scope=RetrievalFilter(grade=3, subject="science"))
        controller = SocraticController(max_evidence_chars=200)
        turn = controller.build_turn(
            "how does weather change",
            decision,
            scope=RetrievalFilter(grade=3, subject="science"),
        )
        from rag.config import ModelConfig
        from pathlib import Path

        models = ModelConfig(
            embedding_model_path=Path("models/bge-m3"),
            reranker_model_path=Path("models/bge-reranker-v2-m3"),
            rewriter_model_path=Path("models/qwen3-vl-2b-instruct"),
            generator_model_path=Path("models/qwen3-vl-8b-instruct"),
            image_embedding_model_path=Path("models/siglip-base-patch16-224"),
        )
        trace = _build_prompt_trace(turn, models)
        assert trace.system_prompt == turn.system_prompt
        assert trace.user_prompt == turn.user_prompt
        assert turn.messages[0]["content"] == trace.system_prompt
        assert turn.messages[1]["content"] == trace.user_prompt
        assert trace.tutor_state == TutorState.GIVE_HINT.value
        assert trace.evidence_blocks[0]["chunk_id"] == "c1"
