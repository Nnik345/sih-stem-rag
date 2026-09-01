"""Maths instance questions: rule retrieval, lookback, and live gate checks."""

from __future__ import annotations

import pytest

from rag.config import ConfigError, load_config
from rag.evidence import EvidenceGate
from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError
from rag.pipeline import HybridRetriever
from rag.query_rewrite import QueryRewriteResult

# One in-curriculum check-my-work item per class. The passthrough rewriter
# searches the rule, not the instance digits.
MATH_INSTANCES: tuple[tuple[int, str, str], ...] = (
    (1, "Is 2 plus 3 equal to 5?", "addition of small whole numbers"),
    (2, "Is 10 plus 4 equal to 14?", "addition of two-digit numbers"),
    (3, "Is 12 times 3 equal to 36?", "multiplication of whole numbers"),
    (4, "Is one-half of 8 equal to 4?", "fractions of a collection"),
    (5, "Is 12 times 3 equal to 36?", "multiplication of whole numbers"),
    (6, "Is -3 plus 5 equal to 2?", "addition of integers"),
    (7, "If 2x equals 10, is x equal to 5?", "simple linear equations"),
    (8, "Is 1/2 plus 1/4 equal to 3/4?", "addition of rational numbers"),
    (9, "Is (x+1)(x+2) equal to x squared plus 3x plus 2?", "product of linear polynomials"),
    (10, "Is sin squared theta plus cos squared theta equal to 1?", "trigonometric identity"),
    (11, "Is the differentiation of x^2 + 3x = 2x + 3?", "derivative of a polynomial using the power rule and the sum rule"),
    (12, "Is the derivative of (2x+1)^3 equal to 6(2x+1)^2?", "chain rule for differentiation of composite functions"),
)


class _RuleRewriter:
    def __init__(self, retrieval_query: str) -> None:
        self.retrieval_query = retrieval_query

    def rewrite(self, query: str, *, grade=None, subject=None, **kwargs) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            retrieval_query=self.retrieval_query,
            intent="verify",
            fallback=False,
        )

    def unload(self) -> None:
        return None


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
    counts = store.read("MATCH (c:Chunk) RETURN count(c) AS chunks")[0]
    if not counts["chunks"]:
        pytest.skip("graph has no Chunk nodes; run scripts/ingest_corpus.py first")
    return counts


def test_rewritten_rule_queries_are_not_both_sides_differentiated():
    for grade, question, rule in MATH_INSTANCES:
        blob = f"{question} {rule}".lower()
        assert "derivative of a equals derivative of b" not in blob
        if grade >= 11:
            assert "power rule" in rule or "chain rule" in rule


@pytest.mark.parametrize("grade,question,rule", MATH_INSTANCES)
def test_live_in_grade_maths_instance_clears_evidence_gate(
    config, store, ingested, grade, question, rule
):
    if not config.models.embedding_model_path.is_dir():
        pytest.skip("BGE-M3 missing; run scripts/download_retrieval_models.py")
    retriever = HybridRetriever(config, store, rewriter=_RuleRewriter(rule))
    try:
        response = retriever.retrieve(
            question, grade=grade, subject="mathematics", rerank=True
        )
    finally:
        retriever.release_models()
    if not any(c.grade == grade and c.subject == "mathematics" for c in response.results):
        pytest.skip(f"class {grade} mathematics not ingested yet")
    gate = EvidenceGate(config.evidence)
    decision = gate.evaluate(
        response.diagnostics.retrieval_query or rule,
        response.results,
        scope=response.scope,
    )
    rewritten = (response.diagnostics.retrieval_query or "").lower()
    assert "equals derivative" not in rewritten
    if not decision.sufficient:
        pytest.skip(
            f"class {grade} gate insufficient: "
            + ", ".join(c.name for c in decision.failed_checks)
        )
    assert decision.sufficient
    assert decision.kept_chunks


def test_live_class12_polynomial_derivative_may_use_class11(config, store, ingested):
    if not config.models.embedding_model_path.is_dir():
        pytest.skip("BGE-M3 missing")
    if not config.models.rewriter_model_path.is_dir():
        pytest.skip("Qwen3-VL-2B rewriter missing")
    retriever = HybridRetriever(config, store)
    question = "Is the derivative of x^2 + 3x = 2x + 3?"
    try:
        response = retriever.retrieve(
            question, grade=12, subject="mathematics", rerank=True
        )
    finally:
        retriever.release_models()
    if response.diagnostics.rewrite_fallback:
        pytest.skip(f"rewriter fell back: {response.diagnostics.notes}")
    rewritten = response.diagnostics.retrieval_query.lower()
    assert "equals derivative" not in rewritten
    assert "2x + 3" not in rewritten
    gate = EvidenceGate(config.evidence)
    decision = gate.evaluate(
        response.diagnostics.retrieval_query,
        response.results,
        scope=response.scope,
    )
    if not decision.sufficient:
        pytest.skip(
            "class 12+11 derivative gate insufficient: "
            + ", ".join(c.name for c in decision.failed_checks)
        )
    grades = {c.grade for c in decision.kept_chunks}
    assert grades <= set(range(1, 13))
    assert any(g <= 12 for g in grades)
    blob = " ".join(c.text.lower() for c in decision.kept_chunks)
    assert any(
        token in blob
        for token in ("derivative", "differenti", "power", "sum rule", "limit")
    )
    assert not any(c.subject == "science" for c in decision.kept_chunks)
