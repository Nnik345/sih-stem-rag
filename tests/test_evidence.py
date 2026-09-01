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


def test_maths_overlap_ignores_algebraic_instance_tokens():
    from rag.evidence import concept_terms

    query = "derivative of the polynomial x squared plus 3x"
    terms = concept_terms(query, mathematics=True)
    assert "derivative" in terms
    assert "polynomial" in terms
    assert "plus" not in terms
    assert "squared" not in terms
    assert "3x" not in terms
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "power",
        "The derivative of a polynomial is found using the power rule and the sum rule. "
        "The derivative of x to the power n is n times x to the power n minus one.",
        grade=11,
        subject="mathematics",
        rerank_score=1.2,
    )
    decision = gate.evaluate(
        query,
        [chunk],
        scope=RetrievalFilter(grade=11, subject="mathematics"),
    )
    assert decision.sufficient


def test_maths_photo_equation_does_not_need_the_same_polynomial_in_the_book():
    from rag.evidence import concept_terms

    query = "d/dx (3x^2 - 4x + 3) = 6x - 4"
    assert "3x" not in concept_terms(query, mathematics=True)
    assert concept_terms(query, mathematics=True) == set()
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "power",
        "The derivative of a polynomial is found using the power rule and the sum rule. "
        "The derivative of x to the power n is n times x to the power n minus one.",
        grade=11,
        subject="mathematics",
        rerank_score=1.2,
    )
    decision = gate.evaluate(
        query,
        [chunk],
        scope=RetrievalFilter(grade=11, subject="mathematics"),
    )
    assert decision.sufficient
    overlap = next(c for c in decision.checks if c.name == "query_term_overlap")
    assert overlap.passed
    assert "skipped" in overlap.detail


def test_science_overlap_still_rejects_unrelated_topic():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "wx",
        "Weather can change from day to day. Clouds and rain are part of weather.",
        rerank_score=0.4,
    )
    decision = gate.evaluate(
        "how do black holes evaporate via hawking radiation",
        [chunk],
        scope=RetrievalFilter(grade=3, subject="science"),
    )
    assert not decision.sufficient


def test_negative_rerank_still_keeps_best_in_scope_maths_chunk():
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    chunk = _chunk(
        "c11",
        "Now, let us tackle derivatives of some standard functions. "
        "The derivative of x to the power n is n times x to the power n minus one.",
        grade=11,
        subject="mathematics",
        rerank_score=-0.625,
    )
    decision = gate.evaluate(
        "d/dx (3x^2 - 4x + 3) = 6x - 4",
        [chunk],
        scope=RetrievalFilter(grade=12, subject="mathematics", allow_prior_grades=True),
    )
    assert decision.sufficient
    assert [c.chunk_id for c in decision.kept_chunks] == ["c11"]
    assert any("promoted" in c.detail for c in decision.checks)


def test_class12_keeps_modest_class11_chunk_when_this_class_has_no_hit():
    """Derivatives live in Class 11. A Class 12 student must still use that page."""
    gate = EvidenceGate(EvidenceConfig(min_rerank_score=0.0, min_query_term_overlap=0.15))
    prior = _chunk(
        "c11",
        "The derivative of x to the power n is n times x to the power n minus one.",
        grade=11,
        subject="mathematics",
        rerank_score=0.075,
    )
    decision = gate.evaluate(
        "d/dx (3x^2 - 4x + 3) = 6x - 4",
        [prior],
        scope=RetrievalFilter(grade=12, subject="mathematics", allow_prior_grades=True),
    )
    assert decision.sufficient
    assert [c.chunk_id for c in decision.kept_chunks] == ["c11"]


def test_high_scoring_prior_grade_maths_chunk_is_kept():
    gate = EvidenceGate(
        EvidenceConfig(
            min_rerank_score=0.0,
            min_query_term_overlap=0.0,
            min_prior_grade_rerank_score=1.0,
        )
    )
    prior = _chunk(
        "c11",
        "The derivative of x to the power n is n times x to the power n minus one.",
        grade=11,
        subject="mathematics",
        rerank_score=2.4,
    )
    current = _chunk(
        "c12",
        "The chain rule is used to differentiate a composite function.",
        grade=12,
        subject="mathematics",
        rerank_score=0.2,
    )
    decision = gate.evaluate(
        "derivative power rule sum rule",
        [prior, current],
        scope=RetrievalFilter(grade=12, subject="mathematics", allow_prior_grades=True),
    )
    assert decision.sufficient
    assert {c.chunk_id for c in decision.kept_chunks} == {"c11", "c12"}


def test_weak_prior_grade_chunk_is_dropped():
    gate = EvidenceGate(
        EvidenceConfig(
            min_rerank_score=0.0,
            min_query_term_overlap=0.0,
            min_prior_grade_rerank_score=1.0,
        )
    )
    weak = _chunk(
        "c9",
        "Polynomials can be added by combining like terms.",
        grade=9,
        subject="mathematics",
        rerank_score=0.3,
    )
    current = _chunk(
        "c12",
        "The chain rule is used to differentiate a composite function.",
        grade=12,
        subject="mathematics",
        rerank_score=0.4,
    )
    decision = gate.evaluate(
        "chain rule composite function",
        [weak, current],
        scope=RetrievalFilter(grade=12, subject="mathematics", allow_prior_grades=True),
    )
    assert decision.sufficient
    assert [c.chunk_id for c in decision.kept_chunks] == ["c12"]


def test_physics_may_keep_high_scoring_science_not_maths_or_chemistry():
    gate = EvidenceGate(
        EvidenceConfig(
            min_rerank_score=0.0,
            min_query_term_overlap=0.0,
            min_prior_grade_rerank_score=1.0,
        )
    )
    science = _chunk(
        "sci",
        "Force is a push or a pull. Motion changes when an unbalanced force acts.",
        grade=10,
        subject="science",
        rerank_score=2.1,
    )
    maths = _chunk(
        "math",
        "Force is a push or a pull. Motion changes when an unbalanced force acts.",
        grade=10,
        subject="mathematics",
        rerank_score=3.0,
    )
    chemistry = _chunk(
        "chem",
        "Force is a push or a pull. Motion changes when an unbalanced force acts.",
        grade=11,
        subject="chemistry",
        rerank_score=3.0,
    )
    physics = _chunk(
        "phy",
        "Newton's laws of motion describe how a force changes the motion of a body.",
        grade=12,
        subject="physics",
        rerank_score=0.5,
    )
    decision = gate.evaluate(
        "force and motion newton",
        [science, maths, chemistry, physics],
        scope=RetrievalFilter(grade=12, subject="physics", allow_prior_grades=True),
    )
    assert decision.sufficient
    kept = {c.chunk_id for c in decision.kept_chunks}
    assert "sci" in kept
    assert "phy" in kept
    assert "math" not in kept
    assert "chem" not in kept
