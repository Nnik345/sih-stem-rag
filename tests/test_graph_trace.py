"""Graph diagnostic classification. Uses fixtures, not a live database."""

from __future__ import annotations

from rag.config import RetrievalConfig
from rag.graph_retriever import GraphRetriever
from rag.graph_trace import (
    DUPLICATE_PATH,
    GRAPH_TOP_K_CUTOFF,
    HUB_CONCEPT_EXCLUDED,
    METADATA_FILTER_MISMATCH,
    PER_RELATION_LIMIT,
    RELATION_DISABLED_BY_DEPTH,
    SEED_NODE,
    SELECTED_GRAPH_CANDIDATE,
    TRACE_SAFETY_LIMIT,
    DiagnosticRow,
    HubConceptRow,
    classify_graph_trace,
)
from rag.schemas import RetrievalFilter, RetrievedChunk
from rag.trace import RecordingObserver


def _row(
    seed: str,
    relation: str,
    candidate: str,
    position: int,
    *,
    grade: int = 1,
    subject: str = "science",
    via: str | None = None,
    via_id: str | None = None,
    via_type: str | None = None,
    seed_weight: float = 1.0,
    truncated_group: bool = False,
    text: str = "chunk text",
) -> DiagnosticRow:
    return DiagnosticRow(
        seed_chunk_id=seed,
        seed_weight=seed_weight,
        relation=relation,
        candidate_chunk_id=candidate,
        position=position,
        text=text,
        via=via,
        via_id=via_id,
        via_type=via_type,
        via_label=via,
        grade=grade,
        subject=subject,
        unit_id="u1",
        section_title="Moon",
        truncated_group=truncated_group,
    )


def _selected(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    chunk = RetrievedChunk(chunk_id=chunk_id, text="accepted", grade=1, subject="science")
    chunk.graph_score = score
    chunk.graph_rank = 1
    return chunk


def _classify(**kwargs):
    defaults = dict(
        seeds=[{"chunk_id": "seed-1", "weight": 1.0}],
        scope=RetrievalFilter(grade=1, subject="science"),
        selected=[],
        primary_rows=[],
        enabled_relations=["SAME_SECTION", "ADJACENT"],
        disabled_relations=["SHARED_CONCEPT", "SAME_LESSON"],
        per_relation_limit=2,
        graph_top_k=1,
        max_paths_per_seed_relation=50,
        max_nodes=1000,
    )
    defaults.update(kwargs)
    return classify_graph_trace(**defaults)


class TestReasonCodes:
    def test_seed_node_exclusion(self):
        trace = _classify(
            primary_rows=[_row("seed-1", "ADJACENT", "seed-1", 1)],
        )
        paths = [p for p in trace.paths if p.candidate_chunk_id == "seed-1"]
        assert paths
        assert all(p.reason_code == SEED_NODE for p in paths)
        seed_node = next(n for n in trace.nodes if n.node_id == "seed-1")
        assert seed_node.node_kind == "seed"
        assert SEED_NODE in seed_node.reason_codes

    def test_metadata_mismatch(self):
        row = _row("seed-1", "SAME_SECTION", "other-grade", 1, grade=3)
        trace = _classify(diagnostic_rows=[row], enabled_relations=["SAME_SECTION"])
        path = next(p for p in trace.paths if p.candidate_chunk_id == "other-grade")
        assert path.reason_code == METADATA_FILTER_MISMATCH
        assert path.accepted is False
        node = next(n for n in trace.nodes if n.node_id == "other-grade")
        assert node.metadata.get("visual") == "metadata_mismatch"

    def test_hub_concept_exclusion_does_not_expand_chunks(self):
        trace = _classify(
            hub_concepts=[
                HubConceptRow(
                    seed_chunk_id="seed-1",
                    concept_id="concept:water",
                    name="water",
                    mention_count=900,
                )
            ],
            enabled_relations=["SHARED_CONCEPT"],
        )
        hub = next(n for n in trace.nodes if n.node_id == "concept:water")
        assert hub.label == "Concept"
        assert HUB_CONCEPT_EXCLUDED in hub.reason_codes
        assert hub.metadata["mention_count"] == 900
        chunk_nodes = [n for n in trace.nodes if n.label == "Chunk" and n.node_kind == "candidate"]
        assert chunk_nodes == []

    def test_duplicate_path_keeps_all_paths_and_node_accepted(self):
        selected = [_selected("cand-1")]
        rows = [
            _row("seed-1", "SAME_SECTION", "cand-1", 1, via="sec-a", via_id="sec-a", via_type="Section"),
            _row("seed-1", "ADJACENT", "cand-1", 1),
        ]
        trace = _classify(
            selected=selected,
            primary_rows=rows,
            enabled_relations=["SAME_SECTION", "ADJACENT"],
        )
        cand_paths = [p for p in trace.paths if p.candidate_chunk_id == "cand-1"]
        assert len(cand_paths) == 2
        assert cand_paths[0].reason_code == SELECTED_GRAPH_CANDIDATE
        assert cand_paths[1].reason_code == DUPLICATE_PATH
        node = next(n for n in trace.nodes if n.node_id == "cand-1")
        assert node.status == "accepted"

    def test_per_relation_limit(self):
        row = _row("seed-1", "SAME_SECTION", "too-far", 3)
        trace = _classify(
            diagnostic_rows=[row],
            per_relation_limit=2,
            enabled_relations=["SAME_SECTION"],
        )
        path = next(p for p in trace.paths if p.candidate_chunk_id == "too-far")
        assert path.reason_code == PER_RELATION_LIMIT

    def test_graph_top_k_cutoff(self):
        selected = [_selected("keep")]
        rows = [
            _row("seed-1", "SAME_SECTION", "keep", 1),
            _row("seed-1", "SAME_SECTION", "cut", 2),
        ]
        trace = _classify(
            selected=selected,
            primary_rows=rows,
            graph_top_k=1,
            per_relation_limit=8,
            enabled_relations=["SAME_SECTION"],
        )
        cut = next(p for p in trace.paths if p.candidate_chunk_id == "cut")
        assert cut.reason_code == GRAPH_TOP_K_CUTOFF
        assert trace.selected_chunk_ids == ["keep"]

    def test_disabled_relation_is_relation_level(self):
        trace = _classify(
            enabled_relations=["SAME_SECTION"],
            disabled_relations=["SHARED_CONCEPT", "SAME_LESSON"],
        )
        disabled = {d.relation: d for d in trace.disabled_relations}
        assert "SHARED_CONCEPT" in disabled
        assert disabled["SHARED_CONCEPT"].reason_code == RELATION_DISABLED_BY_DEPTH
        assert not any(p.relation == "SHARED_CONCEPT" for p in trace.paths)

    def test_trace_safety_limit(self):
        row = _row("seed-1", "SAME_SECTION", "overflow", 51, truncated_group=True)
        trace = _classify(
            diagnostic_rows=[row],
            max_paths_per_seed_relation=50,
            enabled_relations=["SAME_SECTION"],
        )
        assert trace.truncated is True
        assert "per_seed_relation" in trace.truncation_caps
        path = next(p for p in trace.paths if p.candidate_chunk_id == "overflow")
        assert path.reason_code == TRACE_SAFETY_LIMIT
        assert "not examined" in trace.truncation_warning.lower() or "truncated" in trace.truncation_warning.lower()


class TestGraphTraceShape:
    def test_context_nodes_and_logical_edges(self):
        selected = [_selected("lesson-chunk")]
        rows = [
            _row(
                "seed-1",
                "SAME_LESSON",
                "lesson-chunk",
                1,
                via="Moon",
                via_id="lesson:Moon",
                via_type="Section",
            )
        ]
        trace = _classify(
            selected=selected,
            primary_rows=rows,
            enabled_relations=["SAME_LESSON"],
            disabled_relations=[],
        )
        logical = [e for e in trace.edges if not e.physical]
        assert logical
        assert all("(logical)" in e.to_dict()["label"] or not e.physical for e in logical)
        context = [n for n in trace.nodes if n.node_kind == "context"]
        assert context
        assert all(n.status == "context" for n in context)

    def test_selected_ids_match_retriever_output(self):
        selected = [_selected("g1"), _selected("g2")]
        # graph_top_k=1 would cut g2 from "would have been selected" but we pass selected explicitly
        trace = _classify(
            selected=selected,
            primary_rows=[
                _row("seed-1", "ADJACENT", "g1", 1),
                _row("seed-1", "ADJACENT", "g2", 2),
            ],
            graph_top_k=20,
            enabled_relations=["ADJACENT"],
        )
        assert trace.selected_chunk_ids == ["g1", "g2"]
        accepted = {
            n.node_id for n in trace.nodes if n.status == "accepted" and n.node_kind == "candidate"
        }
        assert accepted == {"g1", "g2"}

    def test_physical_section_context(self):
        trace = _classify(
            selected=[_selected("sib")],
            primary_rows=[
                _row(
                    "seed-1",
                    "SAME_SECTION",
                    "sib",
                    1,
                    via="Intro",
                    via_id="sec-1",
                    via_type="Section",
                )
            ],
            enabled_relations=["SAME_SECTION"],
        )
        section = next(n for n in trace.nodes if n.node_id == "sec-1")
        assert section.node_kind == "context"
        assert section.label == "Section"
        physical = [e for e in trace.edges if e.physical]
        assert physical


class FakeStore:
    def __init__(self, primary, diagnostic=None, hubs=None):
        self.primary = primary
        self.diagnostic = diagnostic or []
        self.hubs = hubs or []
        self.queries: list[str] = []

    def read(self, query, parameters=None):
        self.queries.append(query)
        if "graph-primary" in query:
            return list(self.primary)
        if "graph-trace:HUB" in query:
            return list(self.hubs)
        if "graph-trace:" in query:
            return list(self.diagnostic)
        return []


class TestGraphRetrieverObserver:
    def _chunk_record(self, chunk_id, seed="seed-1", relation="ADJACENT"):
        return {
            "chunk_id": chunk_id,
            "text": f"text {chunk_id}",
            "grade": 1,
            "subject": "science",
            "unit_id": "u1",
            "unit_title": "U",
            "document_id": "d1",
            "document_title": "Doc",
            "section_id": "s1",
            "section_title": "Sec",
            "page_start": 1,
            "page_end": 1,
            "resource_type": "student_book",
            "audience": "student",
            "local_pdf_path": "/tmp/a.pdf",
            "relation": relation,
            "via": None,
            "seed_chunk_id": seed,
            "seed_weight": 1.0,
        }

    def test_trace_disabled_skips_diagnostic_queries(self):
        store = FakeStore([self._chunk_record("cand-1")])
        retriever = GraphRetriever(store, RetrievalConfig(graph_top_k=5, graph_max_depth=1))
        seeds = [{"chunk_id": "seed-1", "weight": 1.0}]
        retriever.expand(seeds, scope=RetrievalFilter(grade=1), observer=None)
        assert all("graph-trace:" not in q for q in store.queries)
        assert any("graph-primary" in q for q in store.queries)

    def test_trace_enabled_and_disabled_return_identical_selected(self):
        primary = [self._chunk_record("cand-1"), self._chunk_record("cand-2")]
        extra = [
            {
                **self._chunk_record("ignored-grade"),
                "grade": 3,
                "chunk_id": "ignored-grade",
            }
        ]
        store_off = FakeStore(primary)
        store_on = FakeStore(primary, diagnostic=extra)
        config = RetrievalConfig(graph_top_k=5, graph_max_depth=1)
        off = GraphRetriever(store_off, config).expand(
            [{"chunk_id": "seed-1", "weight": 1.0}],
            scope=RetrievalFilter(grade=1),
            observer=None,
        )
        observer = RecordingObserver()
        retriever_on = GraphRetriever(store_on, config)
        on = retriever_on.expand(
            [{"chunk_id": "seed-1", "weight": 1.0}],
            scope=RetrievalFilter(grade=1),
            observer=observer,
        )
        assert [c.chunk_id for c in off] == [c.chunk_id for c in on]
        assert [c.graph_score for c in off] == [c.graph_score for c in on]
        assert any("graph-trace:" in q for q in store_on.queries)
        assert "graph_completed" in observer.names
        assert retriever_on.last_trace is not None
        assert retriever_on.last_trace["selected_chunk_ids"] == [c.chunk_id for c in on]

    def test_selected_trace_ids_equal_returned_candidates(self):
        primary = [self._chunk_record("cand-1"), self._chunk_record("cand-2")]
        retriever = GraphRetriever(
            FakeStore(primary, diagnostic=[]),
            RetrievalConfig(graph_top_k=5, graph_max_depth=1),
        )
        observer = RecordingObserver()
        selected = retriever.expand(
            [{"chunk_id": "seed-1", "weight": 1.0}],
            scope=RetrievalFilter(grade=1),
            observer=observer,
        )
        assert retriever.last_trace is not None
        assert retriever.last_trace["selected_chunk_ids"] == [c.chunk_id for c in selected]
