"""Buffered structured generation for all tutor states. Mocks only."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from rag.pipeline import SocraticRagPipeline
from rag.schemas import RetrievalFilter, RetrievedChunk
from rag.socratic import TutorState, TutorTurn
from rag.structured_tutor import FALLBACKS, generate_structured_reply
from rag.tutor_json import settings_for

HINT_JSON = json.dumps(
    {
        "hint": "Look at how the power of x changes.",
        "guiding_question": "What happens to the exponent when you differentiate?",
    }
)
EXPLAIN_MATHS_JSON = json.dumps(
    {
        "explanation": "A power of x is differentiated by bringing the exponent down.",
        "formula_or_rule": "The derivative of x to the power n is n times x to the power n minus one.",
        "worked_example": {
            "problem": "Find the derivative of 5x^2 - 2x + 1",
            "steps": [
                "Differentiate each term using the power rule.",
                "The constant term becomes zero.",
            ],
            "answer": "10x - 2",
        },
    }
)
EXPLAIN_SCIENCE_JSON = json.dumps(
    {
        "explanation": "Photosynthesis uses light to make food in green leaves.",
        "formula_or_rule": None,
        "worked_example": None,
    }
)
INSUFFICIENT_JSON = json.dumps(
    {
        "decline": "Verified curriculum material does not cover this question.",
        "nearby_coverage": None,
        "next_step": "Please rephrase or ask your teacher.",
    }
)
CONFIRM_JSON = json.dumps(
    {
        "verdict": "correct",
        "brief_reason": "The power rule and sum rule give a matching derivative.",
        "mistake_groups": [],
    }
)


class FakeGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.complete_calls: list[dict] = []
        self.stream_calls = 0

    def complete(self, messages, *, settings=None, image_paths=()):
        self.complete_calls.append(
            {"messages": messages, "settings": settings, "image_paths": image_paths}
        )
        if not self.outputs:
            raise AssertionError("complete called more times than outputs provided")
        return self.outputs.pop(0)

    def stream(self, messages, *, settings=None, image_paths=()):
        self.stream_calls += 1
        raise AssertionError("structured tutor must not stream")


def _turn(
    state: TutorState,
    question: str,
    *,
    subject: str = "mathematics",
    evidence: tuple[RetrievedChunk, ...] = (),
) -> TutorTurn:
    return TutorTurn(
        question=question,
        state=state,
        system_prompt="sys",
        user_prompt=question,
        scope=RetrievalFilter(grade=12, subject=subject),
        evidence=list(evidence),
    )


def _pipeline(generator) -> SocraticRagPipeline:
    pipeline = SocraticRagPipeline.__new__(SocraticRagPipeline)
    pipeline.config = MagicMock()
    pipeline.retriever = MagicMock()
    pipeline.retriever.release_models = MagicMock()
    pipeline._generator = generator
    return pipeline


class _Observer:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_event(self, event, payload):
        self.events.append(event)


def test_no_state_calls_stream_and_each_is_greedy():
    payloads = {
        TutorState.GIVE_HINT: HINT_JSON,
        TutorState.EXPLAIN_CONCEPT: EXPLAIN_MATHS_JSON,
        TutorState.CONFIRM_ANSWER: CONFIRM_JSON,
        TutorState.INSUFFICIENT_EVIDENCE: INSUFFICIENT_JSON,
    }
    questions = {
        TutorState.GIVE_HINT: "how do I differentiate 2x^2?",
        TutorState.EXPLAIN_CONCEPT: "explain d/dx of 2x^2",
        TutorState.CONFIRM_ANSWER: "is the derivative of 3x^2 - 4x + 3 equal to 6x - 4?",
        TutorState.INSUFFICIENT_EVIDENCE: "what is a made-up particle?",
    }
    for state, payload in payloads.items():
        gen = FakeGenerator([payload])
        subject = "science" if state is TutorState.INSUFFICIENT_EVIDENCE else "mathematics"
        text = generate_structured_reply(
            gen, _turn(state, questions[state], subject=subject)
        )
        assert gen.stream_calls == 0
        assert len(gen.complete_calls) == 1
        assert gen.complete_calls[0]["settings"].do_sample is False
        assert gen.complete_calls[0]["settings"].max_new_tokens == settings_for(state).max_new_tokens
        assert "{" not in text
        assert "verdict" not in text or state is TutorState.CONFIRM_ANSWER and "Correct." in text


def test_stream_answer_yields_one_validated_response_without_tokens():
    gen = FakeGenerator([HINT_JSON])
    pipeline = _pipeline(gen)
    result = MagicMock()
    result.turn = _turn(TutorState.GIVE_HINT, "how do I differentiate 2x^2?")
    result.image_paths = []
    observer = _Observer()
    pieces = list(pipeline.stream_answer(result, observer=observer))
    assert len(pieces) == 1
    assert pieces[0] == result.response_text
    assert "Look at how the power" in pieces[0]
    assert "generation_token" not in observer.events
    assert gen.stream_calls == 0


def test_each_state_retries_once_then_falls_back():
    cases = [
        (TutorState.GIVE_HINT, HINT_JSON, "how do I start?"),
        (TutorState.EXPLAIN_CONCEPT, EXPLAIN_MATHS_JSON, "explain d/dx of 2x^2"),
        (
            TutorState.CONFIRM_ANSWER,
            CONFIRM_JSON,
            "is the derivative of 3x^2 - 4x + 3 equal to 6x - 4?",
        ),
        (TutorState.INSUFFICIENT_EVIDENCE, INSUFFICIENT_JSON, "unknown topic"),
    ]
    for state, good, question in cases:
        gen = FakeGenerator(["not json", good])
        subject = "science" if state is TutorState.INSUFFICIENT_EVIDENCE else "mathematics"
        text = generate_structured_reply(
            gen, _turn(state, question, subject=subject)
        )
        assert len(gen.complete_calls) == 2
        assert "JSON REPAIR" in gen.complete_calls[1]["messages"][-1]["content"]
        assert text != FALLBACKS[state]
        gen_fail = FakeGenerator(["nope", '{"verdict":"nope"}'])
        fallback = generate_structured_reply(
            gen_fail, _turn(state, question, subject=subject)
        )
        assert fallback == FALLBACKS[state]
        assert "{" not in fallback
        assert len(gen_fail.complete_calls) == 2


def test_word_limit_and_evidence_labels_are_rejected():
    too_long = json.dumps(
        {
            "hint": "word " * 100,
            "guiding_question": "What next?",
        }
    )
    leaked = json.dumps(
        {
            "hint": "See [E1] for the rule.",
            "guiding_question": "What is the next step?",
        }
    )
    for payload in (too_long, leaked):
        gen = FakeGenerator([payload, payload])
        text = generate_structured_reply(
            gen, _turn(TutorState.GIVE_HINT, "how do I start?")
        )
        assert text == FALLBACKS[TutorState.GIVE_HINT]


def test_give_hint_one_question_and_rejects_answer_leak():
    gen = FakeGenerator([HINT_JSON])
    text = generate_structured_reply(
        gen, _turn(TutorState.GIVE_HINT, "how do I differentiate 2x^2?")
    )
    assert text.count("?") == 1
    assert "Look at how the power" in text
    leaked = json.dumps(
        {
            "hint": "The answer is 4x.",
            "guiding_question": "Can you see why?",
        }
    )
    gen2 = FakeGenerator([leaked, leaked])
    assert generate_structured_reply(
        gen2, _turn(TutorState.GIVE_HINT, "how do I differentiate 2x^2?")
    ) == FALLBACKS[TutorState.GIVE_HINT]


def test_explain_maths_requires_different_worked_example():
    gen = FakeGenerator([EXPLAIN_MATHS_JSON])
    text = generate_structured_reply(
        gen, _turn(TutorState.EXPLAIN_CONCEPT, "explain d/dx of 2x^2")
    )
    assert "Worked example" in text
    assert "5x^2 - 2x + 1" in text
    same = json.dumps(
        {
            "explanation": "Use the power rule.",
            "formula_or_rule": "n x^{n-1}",
            "worked_example": {
                "problem": "explain d/dx of 2x^2",
                "steps": ["Differentiate."],
                "answer": "4x",
            },
        }
    )
    gen2 = FakeGenerator([same, same])
    assert generate_structured_reply(
        gen2, _turn(TutorState.EXPLAIN_CONCEPT, "explain d/dx of 2x^2")
    ) == FALLBACKS[TutorState.EXPLAIN_CONCEPT]


def test_explain_non_maths_allows_null_formula_and_example():
    gen = FakeGenerator([EXPLAIN_SCIENCE_JSON])
    text = generate_structured_reply(
        gen,
        _turn(
            TutorState.EXPLAIN_CONCEPT,
            "what is photosynthesis?",
            subject="science",
        ),
    )
    assert "Photosynthesis" in text
    assert "Worked example" not in text
    assert "Formula or rule" not in text


def test_insufficient_rejects_answering_the_question():
    answering = json.dumps(
        {
            "decline": "The answer is 42.",
            "nearby_coverage": None,
            "next_step": "Ask your teacher.",
        }
    )
    gen = FakeGenerator([answering, answering])
    text = generate_structured_reply(
        gen,
        _turn(
            TutorState.INSUFFICIENT_EVIDENCE,
            "what is a made-up particle?",
            subject="science",
        ),
    )
    assert text == FALLBACKS[TutorState.INSUFFICIENT_EVIDENCE]


def test_answer_on_token_skipped_for_confirm_only():
    gen = FakeGenerator([CONFIRM_JSON])
    pipeline = _pipeline(gen)
    result = MagicMock()
    result.turn = _turn(
        TutorState.CONFIRM_ANSWER,
        "is the derivative of 3x^2 - 4x + 3 equal to 6x - 4?",
    )
    result.image_paths = []
    result.answered = True
    result.notes = []
    seen: list[str] = []
    pipeline.prepare = MagicMock(return_value=result)
    out = pipeline.answer(
        result.turn.question,
        requested_state=TutorState.CONFIRM_ANSWER,
        on_token=seen.append,
    )
    assert out.response_text.startswith("Correct.")
    assert seen == []
