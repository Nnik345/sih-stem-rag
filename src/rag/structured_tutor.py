"""Buffered structured generation for every tutoring state."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .confirm_eval import evaluate_confirm
from .logging_utils import get_logger
from .socratic import TutorState, TutorTurn, _is_mathematics
from .tutor_json import (
    complete_json,
    has_leak,
    parse_json_object,
    reveals_final_answer,
    settings_for,
    within_word_limit,
)

LOGGER = get_logger(__name__)

FALLBACKS: dict[TutorState, str] = {
    TutorState.GIVE_HINT: (
        "I could not form a reliable hint just now. "
        "Which part of this problem do you want to try first?"
    ),
    TutorState.EXPLAIN_CONCEPT: (
        "I could not put together a reliable explanation just now. "
        "Please ask again or check this topic with your teacher."
    ),
    TutorState.CONFIRM_ANSWER: (
        "I could not verify that answer reliably. Please try again or ask your teacher."
    ),
    TutorState.INSUFFICIENT_EVIDENCE: (
        "Verified curriculum material is not enough for this question. "
        "Please rephrase or ask your teacher."
    ),
}

_HINT_REPAIR = (
    "JSON REPAIR\nReply with one JSON object only: "
    '{"hint":"...","guiding_question":"...?"}. Both fields non-empty. '
    "Exactly one hint and exactly one guiding question ending with ?. "
    "Do not reveal the final answer or list solution steps."
)
_EXPLAIN_REPAIR = (
    "JSON REPAIR\nReply with one JSON object only with keys explanation, "
    "formula_or_rule, worked_example. explanation must be non-empty. "
    "formula_or_rule is a string or null. worked_example is null or "
    '{"problem":"...","steps":["..."],"answer":"..."}. For mathematics, '
    "worked_example is required and must differ from the student's problem."
)
_INSUFFICIENT_REPAIR = (
    "JSON REPAIR\nReply with one JSON object only: "
    '{"decline":"...","nearby_coverage":"... or null","next_step":"..."}. '
    "decline and next_step must be non-empty. Do not answer the original "
    "question from general knowledge."
)

_STEP_SEQUENCE_RE = re.compile(
    r"\bstep\s*1\b.+\bstep\s*2\b",
    re.IGNORECASE | re.DOTALL,
)
_ANSWERING_WHILE_DECLINING_RE = re.compile(
    r"\bthe\s+(?:correct\s+)?answer\s+is\b|"
    r"\btherefore\s+(?:the\s+)?(?:result|derivative|value)\b|"
    r"\bequals\s+[-+]?\d",
    re.IGNORECASE,
)


def generate_structured_reply(
    generator: Any,
    turn: TutorTurn,
    *,
    image_paths: Sequence[str] = (),
) -> str:
    """Buffered JSON generation, validation, and Python formatting."""
    state = turn.state
    if not isinstance(state, TutorState):
        try:
            state = TutorState(str(state))
        except ValueError:
            LOGGER.warning("Unknown tutor state %r; using GIVE_HINT fallback", state)
            return FALLBACKS[TutorState.GIVE_HINT]
    if state is TutorState.CONFIRM_ANSWER:
        return evaluate_confirm(
            generator,
            turn.messages,
            question=turn.question,
            subject=turn.scope.subject,
            image_paths=image_paths,
        )
    if state is TutorState.GIVE_HINT:
        validate = lambda raw: _validate_hint(raw)
        repair = _HINT_REPAIR
    elif state is TutorState.EXPLAIN_CONCEPT:
        validate = lambda raw: _validate_explain(
            raw, question=turn.question, subject=turn.scope.subject
        )
        repair = _EXPLAIN_REPAIR
    elif state is TutorState.INSUFFICIENT_EVIDENCE:
        evidence_text = " ".join(chunk.text for chunk in turn.evidence)
        validate = lambda raw: _validate_insufficient(
            raw, question=turn.question, evidence_text=evidence_text
        )
        repair = _INSUFFICIENT_REPAIR
    else:
        return FALLBACKS[TutorState.GIVE_HINT]
    formatted = complete_json(
        generator,
        turn.messages,
        settings=settings_for(state),
        image_paths=image_paths,
        validate=validate,
        repair_instruction=repair,
    )
    if formatted is None:
        LOGGER.warning("%s evaluation failed after retry; using fallback", state.value)
        return FALLBACKS[state]
    return formatted


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _reject_common(*parts: str) -> bool:
    blob = "\n".join(parts)
    return has_leak(blob) or reveals_final_answer(blob)


def _validate_hint(raw: str) -> str | None:
    payload = parse_json_object(raw)
    if payload is None:
        return None
    hint = _nonempty_str(payload.get("hint"))
    question = _nonempty_str(payload.get("guiding_question"))
    if hint is None or question is None:
        return None
    if "?" in hint:
        return None
    if not question.endswith("?") or question.count("?") != 1:
        return None
    if _STEP_SEQUENCE_RE.search(hint) or _STEP_SEQUENCE_RE.search(question):
        return None
    if _reject_common(hint, question):
        return None
    formatted = f"{hint}\n\n{question}"
    if not within_word_limit(formatted, TutorState.GIVE_HINT):
        return None
    return formatted


def _normalize_instance(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower().replace("**", "^"))


def _validate_explain(raw: str, *, question: str, subject: str | None) -> str | None:
    payload = parse_json_object(raw)
    if payload is None:
        return None
    explanation = _nonempty_str(payload.get("explanation"))
    if explanation is None:
        return None
    formula = payload.get("formula_or_rule", None)
    if formula is not None and not isinstance(formula, str):
        return None
    formula_text = formula.strip() if isinstance(formula, str) and formula.strip() else None
    example = payload.get("worked_example", None)
    maths = _is_mathematics(subject)
    if maths:
        if not isinstance(example, dict):
            return None
    elif example is not None and not isinstance(example, dict):
        return None
    example_block = None
    if isinstance(example, dict):
        problem = _nonempty_str(example.get("problem"))
        answer = _nonempty_str(example.get("answer"))
        steps_raw = example.get("steps")
        if problem is None or answer is None or not isinstance(steps_raw, list) or not steps_raw:
            return None
        steps: list[str] = []
        for step in steps_raw:
            text = _nonempty_str(step)
            if text is None:
                return None
            steps.append(text)
        student_norm = _normalize_instance(question)
        problem_norm = _normalize_instance(problem)
        if student_norm and problem_norm:
            if problem_norm == student_norm:
                return None
            if len(problem_norm) >= 12 and problem_norm in student_norm:
                return None
            if len(student_norm) >= 12 and student_norm in problem_norm:
                return None
        example_block = (problem, steps, answer)
    parts = [explanation]
    if formula_text:
        parts.extend(["Formula or rule", formula_text])
    if example_block is not None:
        problem, steps, answer = example_block
        parts.append("Worked example")
        parts.append(problem)
        for index, step in enumerate(steps, start=1):
            parts.append(f"{index}. {step}")
        parts.append(f"Answer: {answer}")
    if _reject_common(*parts):
        return None
    formatted = "\n\n".join(parts)
    if not within_word_limit(formatted, TutorState.EXPLAIN_CONCEPT):
        return None
    return formatted


def _token_overlap(left: str, right: str) -> int:
    def tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z]{4,}", text.lower())}

    return len(tokens(left) & tokens(right))


def _validate_insufficient(
    raw: str, *, question: str, evidence_text: str
) -> str | None:
    payload = parse_json_object(raw)
    if payload is None:
        return None
    decline = _nonempty_str(payload.get("decline"))
    next_step = _nonempty_str(payload.get("next_step"))
    if decline is None or next_step is None:
        return None
    nearby_raw = payload.get("nearby_coverage", None)
    if nearby_raw is not None and not isinstance(nearby_raw, str):
        return None
    nearby = nearby_raw.strip() if isinstance(nearby_raw, str) and nearby_raw.strip() else None
    if nearby:
        if not evidence_text.strip():
            return None
        if _token_overlap(nearby, evidence_text) < 1:
            return None
    blob = "\n".join(p for p in (decline, nearby or "", next_step) if p)
    if _reject_common(blob) or _ANSWERING_WHILE_DECLINING_RE.search(blob):
        return None
    if _ANSWERING_WHILE_DECLINING_RE.search(question) and nearby:
        # Nearby must not look like a solution to the asked question.
        if _normalize_instance(question) and _normalize_instance(question) in _normalize_instance(
            nearby
        ):
            return None
    formatted_parts = [decline]
    if nearby:
        formatted_parts.append(nearby)
    formatted_parts.append(next_step)
    formatted = "\n\n".join(formatted_parts)
    if not within_word_limit(formatted, TutorState.INSUFFICIENT_EVIDENCE):
        return None
    return formatted
