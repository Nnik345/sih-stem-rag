"""Structured CONFIRM_ANSWER evaluation. Mocks only; no live 8B load."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from rag.confirm_eval import (
    CONFIRM_GENERATION_SETTINGS,
    SAFE_FALLBACK,
    evaluate_confirm,
    format_confirm_response,
    parse_confirm_json,
    validate_confirm_assessment,
    verify_derivative_claim,
)
from rag.pipeline import SocraticRagPipeline
from rag.socratic import TutorState

REPORTED = "is the derivative of 3x^2 - 4x + 3 equal to 6x - 4?"

CORRECT_JSON = json.dumps(
    {
        "verdict": "correct",
        "brief_reason": "The power rule and sum rule give a matching derivative.",
        "mistake_groups": [],
    }
)
INCORRECT_JSON = json.dumps(
    {
        "verdict": "incorrect",
        "brief_reason": "The proposed derivative does not match the rule.",
        "mistake_groups": [
            {
                "mistake": "The variable term was not differentiated with the power rule.",
                "hint": "Apply the power rule to each power of x separately.",
            }
        ],
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
        raise AssertionError("CONFIRM_ANSWER must not stream evaluator output")


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "tutor"},
        {"role": "user", "content": REPORTED},
    ]


def test_reported_derivative_is_correct_and_private():
    gen = FakeGenerator([CORRECT_JSON])
    text = evaluate_confirm(
        gen, _messages(), question=REPORTED, subject="mathematics"
    )
    assert text.startswith("Correct.")
    assert "power rule" in text.lower() or "matching" in text.lower()
    assert "Not quite" not in text
    assert "incorrect" not in text.lower()
    assert "{" not in text
    assert "verdict" not in text
    assert "[E1]" not in text
    assert gen.stream_calls == 0
    assert gen.complete_calls[0]["settings"].do_sample is False


def test_equivalent_and_reordered_derivatives_are_accepted():
    assert (
        verify_derivative_claim(
            "is the derivative of 3x^2 - 4x + 3 equal to 6x - 4 + 0?",
            subject="mathematics",
        )
        is True
    )
    assert (
        verify_derivative_claim(
            "is the derivative of 3x^2 - 4x + 3 equal to -4 + 6x?",
            subject="mathematics",
        )
        is True
    )
    gen = FakeGenerator([CORRECT_JSON])
    text = evaluate_confirm(
        gen,
        _messages(),
        question="is the derivative of 3x^2 - 4x + 3 equal to 6x - 4 + 0?",
        subject="mathematics",
    )
    assert text.startswith("Correct.")


def test_incorrect_derivative_hints_without_revealing_answer():
    gen = FakeGenerator([INCORRECT_JSON])
    text = evaluate_confirm(
        gen,
        _messages(),
        question="is the derivative of 3x^2 - 4x + 3 equal to 5x - 4?",
        subject="mathematics",
    )
    assert text.startswith("Not quite.")
    assert "Hint:" in text
    assert "6x-4" not in text.replace(" ", "")
    assert "6*x - 4" not in text
    assert "the correct derivative is" not in text.lower()
    locked = gen.complete_calls[0]["messages"][-1]["content"]
    assert "LOCKED SYMBOLIC VERDICT: incorrect" in locked


def test_identical_mistakes_are_grouped():
    payload = {
        "verdict": "incorrect",
        "brief_reason": "The same power-rule error appears more than once.",
        "mistake_groups": [
            {"mistake": "Missed the power rule on x squared", "hint": "Use n x^{n-1}."},
            {"mistake": "Missed the power rule on x squared", "hint": "Duplicate hint."},
        ],
    }
    assessment = validate_confirm_assessment(payload, locked_verdict=False)
    assert assessment is not None
    assert len(assessment.mistake_groups) == 1
    text = format_confirm_response(assessment)
    assert text.count("- ") == 1
    assert "Duplicate hint" not in text


def test_distinct_mistakes_each_get_a_hint():
    payload = {
        "verdict": "incorrect",
        "brief_reason": "Two different differentiation errors.",
        "mistake_groups": [
            {"mistake": "Power rule was not applied to x squared", "hint": "Check the exponent."},
            {"mistake": "The linear term was left unchanged", "hint": "Differentiate every term."},
        ],
    }
    assessment = validate_confirm_assessment(payload, locked_verdict=False)
    assert assessment is not None
    text = format_confirm_response(assessment)
    assert text.startswith("Not quite.")
    assert text.count("Hint:") == 2
    assert "Power rule was not applied" in text
    assert "linear term" in text


def test_malformed_json_retries_once_then_succeeds():
    gen = FakeGenerator(["not json at all", CORRECT_JSON])
    text = evaluate_confirm(
        gen, _messages(), question=REPORTED, subject="mathematics"
    )
    assert text.startswith("Correct.")
    assert len(gen.complete_calls) == 2
    assert "JSON REPAIR" in gen.complete_calls[1]["messages"][-1]["content"]


def test_second_invalid_result_uses_safe_fallback():
    gen = FakeGenerator(["not json", '{"verdict": "nope"}'])
    text = evaluate_confirm(
        gen, _messages(), question=REPORTED, subject="mathematics"
    )
    assert text == SAFE_FALLBACK
    assert len(gen.complete_calls) == 2
    assert "{" not in text
    assert "verdict" not in text
    assert "Traceback" not in text


def test_locked_verdict_mismatch_is_invalid():
    payload = parse_confirm_json(INCORRECT_JSON)
    assert validate_confirm_assessment(payload, locked_verdict=True) is None
    payload_ok = parse_confirm_json(CORRECT_JSON)
    assert validate_confirm_assessment(payload_ok, locked_verdict=False) is None


def test_sympy_does_not_run_for_other_subjects():
    assert verify_derivative_claim(REPORTED, subject="science") is None
    assert verify_derivative_claim("what is a plant cell?", subject="mathematics") is None


def test_confirm_settings_are_greedy_and_restricted():
    assert CONFIRM_GENERATION_SETTINGS.do_sample is False
    assert CONFIRM_GENERATION_SETTINGS.max_new_tokens == 400


class _Observer:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.payloads: list[dict] = []

    def on_event(self, event, payload):
        self.events.append(event)
        self.payloads.append(payload)


def _pipeline_with_generator(generator) -> SocraticRagPipeline:
    pipeline = SocraticRagPipeline.__new__(SocraticRagPipeline)
    pipeline.config = MagicMock()
    pipeline.retriever = MagicMock()
    pipeline.retriever.release_models = MagicMock()
    pipeline._generator = generator
    return pipeline


def test_confirm_answer_does_not_stream_raw_evaluator_output():
    gen = FakeGenerator([CORRECT_JSON])
    pipeline = _pipeline_with_generator(gen)
    result = MagicMock()
    result.turn.state = TutorState.CONFIRM_ANSWER
    result.turn.messages = _messages()
    result.turn.question = REPORTED
    result.turn.scope.subject = "mathematics"
    result.image_paths = []
    observer = _Observer()

    pieces = list(pipeline.stream_answer(result, observer=observer))
    assert pieces == [result.response_text]
    assert result.response_text.startswith("Correct.")
    assert gen.stream_calls == 0
    assert len(gen.complete_calls) == 1
    assert gen.complete_calls[0]["settings"].do_sample is False
    assert "generation_token" not in observer.events
    assert "generation_started" in observer.events


def test_other_tutor_states_still_stream():
    gen = MagicMock()
    gen.stream = MagicMock(return_value=iter(["Hel", "lo"]))
    gen.complete = MagicMock(side_effect=AssertionError("other states must stream"))
    pipeline = _pipeline_with_generator(gen)
    result = MagicMock()
    result.turn.state = TutorState.GIVE_HINT
    result.turn.messages = [{"role": "user", "content": "hint please"}]
    result.image_paths = []
    observer = _Observer()

    assert list(pipeline.stream_answer(result, observer=observer)) == ["Hel", "lo"]
    gen.stream.assert_called_once()
    gen.complete.assert_not_called()
    assert observer.events.count("generation_token") == 2
    assert [p["token"] for p in observer.payloads if "token" in p] == ["Hel", "lo"]


def test_answer_does_not_forward_confirm_tokens_to_callback():
    gen = FakeGenerator([CORRECT_JSON])
    pipeline = _pipeline_with_generator(gen)
    result = MagicMock()
    result.turn.state = TutorState.CONFIRM_ANSWER
    result.turn.question = REPORTED
    result.turn.scope.subject = "mathematics"
    result.turn.messages = _messages()
    result.image_paths = []
    result.answered = True
    result.notes = []
    seen: list[str] = []
    pipeline.prepare = MagicMock(return_value=result)
    out = pipeline.answer(
        REPORTED,
        requested_state=TutorState.CONFIRM_ANSWER,
        on_token=seen.append,
    )
    assert out.response_text.startswith("Correct.")
    assert seen == []
    assert gen.stream_calls == 0
