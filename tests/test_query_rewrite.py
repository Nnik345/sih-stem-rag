"""Query rewriter prompt constraints, JSON parse, and pipeline unload order."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.config import load_config
from rag.pipeline import HybridRetriever
from rag.query_rewrite import (
    REWRITE_SYSTEM_PROMPT,
    QueryRewriteResult,
    build_rewrite_user_prompt,
    parse_rewrite_output,
    question_needs_textbook_figure,
    specialize_maths_retrieval_query,
)


def test_rewrite_prompt_forbids_answers_and_inferred_scope():
    blob = REWRITE_SYSTEM_PROMPT.lower()
    assert "do not solve" in blob
    assert "do not guess" in blob or "do not infer" in blob
    assert "json" in blob
    user = build_rewrite_user_prompt(
        "differentiation of x^2", grade=12, subject="mathematics"
    )
    assert "Class (given by the application, do not infer): 12" in user
    assert "mathematics" in user
    assert "differentiation of x^2" in user


def test_rewrite_prompt_defines_verify_check_my_work():
    blob = REWRITE_SYSTEM_PROMPT.lower()
    assert "verify" in blob
    assert "expr is the problem" in blob
    assert "do not apply the operation to result" in blob
    assert "is the differentiation of x^2 + 3x = 2x + 3?" in blob
    assert '"intent": "verify"' in blob
    assert "proposed answer" in blob
    assert "power rule" in blob
    assert "chain rule" in blob
    assert "do not put the instance expression" in blob
    assert "derivative of a polynomial using the power rule and the sum rule" in blob


def test_parse_rewrite_output_accepts_json():
    result = parse_rewrite_output(
        '{"retrieval_query": "derivative of x squared", "intent": "explain"}',
        "differentiation of x^2",
    )
    assert result.fallback is False
    assert result.retrieval_query == "derivative of x squared"
    assert result.intent == "explain"
    assert result.input_kind == "other"


def test_parse_rewrite_output_strips_fences_and_falls_back():
    fenced = parse_rewrite_output(
        '```json\n{"retrieval_query": "electric charge", "intent": "explain"}\n```',
        "what is charge",
    )
    assert fenced.retrieval_query == "electric charge"
    bad = parse_rewrite_output("not json at all", "original question")
    assert bad.fallback is True
    assert bad.retrieval_query == "original question"
    empty = parse_rewrite_output('{"retrieval_query": "", "intent": "explain"}', "keep me")
    assert empty.fallback is True
    assert empty.retrieval_query == "keep me"


def test_rewriter_unloads_before_dense_uses_rewritten_query():
    config = load_config(require_neo4j=False)
    order: list[object] = []

    class FakeRewriter:
        def rewrite(self, query, *, grade=None, subject=None, **kwargs):
            order.append("rewrite")
            assert grade == 12
            assert subject == "mathematics"
            return QueryRewriteResult(
                original_query=query,
                retrieval_query="derivative of x squared",
                intent="explain",
                fallback=False,
            )

        def unload(self):
            order.append("unload")

    retriever = HybridRetriever(config, MagicMock(), rewriter=FakeRewriter())
    retriever.dense.retrieve = lambda query, scope=None: (
        order.append(("dense", query)) or []
    )
    retriever.lexical.retrieve = lambda query, scope=None: (
        order.append(("lexical", query)) or []
    )
    retriever.graph.retrieve = lambda *args, **kwargs: order.append("graph") or []

    response = retriever.retrieve(
        "differentiation of x^2", grade=12, subject="mathematics"
    )
    assert order[0] == "rewrite"
    assert order[1] == "unload"
    ncert_query = "algebra of derivative of functions derivative of x to the power n"
    assert order[2] == ("dense", ncert_query)
    assert order[3] == ("lexical", ncert_query)
    assert response.diagnostics.retrieval_query == ncert_query
    assert response.diagnostics.rewrite_fallback is False
    assert response.query == "differentiation of x^2"
    assert response.scope.allow_prior_grades is True


def test_retrieve_without_subject_does_not_look_back():
    config = load_config(require_neo4j=False)
    captured: dict[str, object] = {}

    class FakeRewriter:
        def rewrite(self, query, *, grade=None, subject=None, **kwargs):
            return QueryRewriteResult(
                original_query=query,
                retrieval_query=query,
                intent="other",
                fallback=True,
                reason="test",
            )

        def unload(self):
            return None

    retriever = HybridRetriever(config, MagicMock(), rewriter=FakeRewriter())

    def capture_dense(query, scope=None):
        captured["scope"] = scope
        return []

    retriever.dense.retrieve = capture_dense
    retriever.lexical.retrieve = lambda query, scope=None: []
    retriever.graph.retrieve = lambda *args, **kwargs: []
    retriever.retrieve("counting numbers", grade=3)
    assert captured["scope"].allow_prior_grades is False


def test_rewrite_prompt_does_not_request_figures_for_ordinary_explanations():
    blob = REWRITE_SYSTEM_PROMPT.lower()
    assert "diagram" in blob
    assert "only when the student asked to see one" in blob


def test_specialize_maths_retrieval_query_replaces_photo_equation():
    equation = "d/dx (3x^2 - 4x + 3) = 6x - 4"
    out = specialize_maths_retrieval_query(equation, transcribed=equation)
    assert "algebra of derivative" in out
    assert "power n" in out
    assert "3x" not in out
    assert "power rule" not in out
    # Differentiation cues always map to NCERT wording, even if the rewriter
    # already produced a "rule" phrase with no instance tokens.
    rewriter = "derivative of a quadratic polynomial using the power rule"
    assert specialize_maths_retrieval_query(rewriter, original=equation) == out


def test_ddx_solution_request_is_not_quadratic_roots():
    query = "provide a solution for d/dx 3x^2 - 4x + 3"
    out = specialize_maths_retrieval_query(query, original=query)
    assert "algebra of derivative" in out
    assert "power n" in out
    blob = out.lower()
    assert "root" not in blob
    assert "zero" not in blob
    assert "quadratic equation" not in blob
    assert "factoris" not in blob


def test_original_ddx_overrides_quadratic_rewriter_query():
    query = "provide a solution for d/dx 3x^2 - 4x + 3"
    out = specialize_maths_retrieval_query(
        "quadratic equations and roots",
        original=query,
    )
    assert out == "algebra of derivative of functions derivative of x to the power n"
    assert "quadratic" not in out.lower()
    assert "root" not in out.lower()


def test_maths_instance_retrieval_query_is_specialised_before_dense():
    config = load_config(require_neo4j=False)
    order: list[object] = []
    equation = "d/dx (3x^2 - 4x + 3) = 6x - 4"

    class FakeRewriter:
        def rewrite(self, query, *, grade=None, subject=None, **kwargs):
            return QueryRewriteResult(
                original_query=query,
                retrieval_query=equation,
                intent="verify",
                fallback=False,
                transcribed_question=equation,
                input_kind="math_problem",
            )

        def unload(self):
            return None

    retriever = HybridRetriever(config, MagicMock(), rewriter=FakeRewriter())
    retriever.dense.retrieve = lambda query, scope=None: (
        order.append(("dense", query)) or []
    )
    retriever.lexical.retrieve = lambda query, scope=None: []
    retriever.graph.retrieve = lambda *args, **kwargs: []
    response = retriever.retrieve(equation, grade=11, subject="mathematics")
    expected = "algebra of derivative of functions derivative of x to the power n"
    assert order[0] == ("dense", expected)
    assert response.diagnostics.retrieval_query == expected


def test_question_needs_textbook_figure_only_for_visuals():
    assert question_needs_textbook_figure(query="diagram of a plant cell")
    assert question_needs_textbook_figure(query="labelled diagram of a neuron")
    assert question_needs_textbook_figure(query="what does a plant cell look like")
    assert question_needs_textbook_figure(
        query="what is this", input_kind="diagram"
    )
    assert not question_needs_textbook_figure(query="what is photosynthesis")
    assert not question_needs_textbook_figure(
        query="is the differentiation of x^2 + 3x = 2x + 3?"
    )
    assert not question_needs_textbook_figure(query="how many significant figures")
    assert not question_needs_textbook_figure(
        query="check my work", input_kind="math_problem"
    )
