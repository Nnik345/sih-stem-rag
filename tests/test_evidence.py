"""Evidence-gate partition filters and reranker score semantics."""

from __future__ import annotations

from rag.config import EvidenceConfig
from rag.evidence import EvidenceGate
from rag.reranker import _sigmoid
from rag.schemas import RetrievalFilter, RetrievedChunk


def _chunk(chunk_id: str, text: str, **kwargs) -> RetrievedChunk:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        grade=3,
        subject="science",
        content_partition="student_evidence",
        rerank_score=1.5,
        raw_rerank_score=1.5,
        rerank_probability=_sigmoid(1.5),
    )
    for key, value in kwargs.items():
        setattr(chunk, key, value)
    return chunk


def test_sigmoid_zero_is_half():
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(10.0) > 0.99
    assert _sigmoid(-10.0) < 0.01


def test_gate_rejects_boilerplate_even_with_high_score():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=-5.0, min_query_term_overlap=0.0))
    chunk = _chunk(
        "cc",
        "This work is licensed under a Creative Commons Attribution 4.0 License. https://creativecommons.org/licenses/by/4.0/",
        rerank_score=8.0,
    )
    decision = gate.evaluate(
        "how does weather change from day to day",
        [chunk],
        scope=RetrievalFilter(grade=3, subject="science"),
    )
    assert not decision.sufficient
    assert any(check.name == "safe_partition" and not check.passed for check in decision.checks)


def test_gate_rejects_evaluation_partition():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=-5.0, min_query_term_overlap=0.0))
    chunk = _chunk(
        "key",
        "The correct answer is 12. Sample response: students should have written 12.",
        content_partition="evaluation_only",
        rerank_score=6.0,
    )
    decision = gate.evaluate("what is 3 times 4", [chunk], scope=RetrievalFilter(grade=3))
    assert not decision.sufficient


def test_gate_accepts_supported_weather_passage():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "wx",
        "Weather can change from day to day. A rainy day may be followed by a sunny day. "
        "Temperature, wind and clouds help describe how weather changes.",
        rerank_score=0.4,
    )
    decision = gate.evaluate(
        "how does weather change from day to day",
        [chunk],
        scope=RetrievalFilter(grade=3, subject="science"),
    )
    assert decision.sufficient
    assert decision.kept_chunks[0].chunk_id == "wx"


def test_gate_declines_unsupported_topic():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "wx",
        "Weather can change from day to day. Clouds and rain are part of weather.",
        rerank_score=-2.5,
    )
    decision = gate.evaluate(
        "how do black holes evaporate via hawking radiation",
        [chunk],
        scope=RetrievalFilter(grade=3, subject="science"),
    )
    assert not decision.sufficient


def test_similar_boilerplate_chunk_is_rejected_not_kept():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=-10.0, min_query_term_overlap=0.0))
    relevant = _chunk(
        "ok",
        "A rain gauge measures how much rain falls. Weather changes from day to day.",
        rerank_score=1.0,
    )
    junk = _chunk(
        "cc",
        "This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike license.",
        rerank_score=9.0,
    )
    decision = gate.evaluate(
        "how does weather change from day to day",
        [junk, relevant],
        scope=RetrievalFilter(grade=3, subject="science"),
    )
    assert decision.sufficient
    assert [c.chunk_id for c in decision.kept_chunks] == ["ok"]
