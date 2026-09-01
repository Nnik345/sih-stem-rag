"""Structured CONFIRM_ANSWER evaluation: greedy JSON call, then Python format.

The student never sees the evaluator JSON. SymPy may lock a derivative verdict
when the question is safely parseable mathematics; other subjects are unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .generator import GenerationSettings
from .logging_utils import get_logger
from .socratic import TutorState, _STATE_WORD_LIMITS

LOGGER = get_logger(__name__)

CONFIRM_MAX_NEW_TOKENS = 400
CONFIRM_GENERATION_SETTINGS = GenerationSettings(
    max_new_tokens=CONFIRM_MAX_NEW_TOKENS,
    do_sample=False,
)
CONFIRM_WORD_LIMIT = _STATE_WORD_LIMITS[TutorState.CONFIRM_ANSWER]
SAFE_FALLBACK = (
    "I could not verify that answer reliably. Please try again or ask your teacher."
)
JSON_REPAIR_INSTRUCTION = (
    "JSON REPAIR\nYour previous output was not valid. Reply with one JSON object "
    "only: no markdown, no student-facing prose, no Correct. or Not quite. "
    "prefixes in any field. Required keys: verdict (correct or incorrect), "
    "brief_reason (non-empty string), mistake_groups (list). If correct, "
    "mistake_groups must be []. If incorrect, each item needs non-empty mistake "
    "and hint strings, grouped by distinct mistake, and must not reveal the "
    "corrected final answer."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_DERIVATIVE_CLAIM_RE = re.compile(
    r"(?is)^\s*(?:is\s+)?(?:the\s+)?"
    r"(?:derivative|differentiation)\s+of\s+(?P<expr>.+?)\s+"
    r"(?:equal(?:s)?\s+to|=)\s+(?P<result>.+?)\s*\??\s*$"
)
_ALLOWED_EXPR = re.compile(r"^[0-9A-Za-z+\-*/^().,\s]+$")
_BANNED_EXPR = re.compile(
    r"(import|eval|exec|lambda|compile|open|__|os\.|sys\.)",
    re.IGNORECASE,
)
_VERDICT_PREFIX_RE = re.compile(
    r"^\s*(correct|not quite|incorrect)\b[.:!]?",
    re.IGNORECASE,
)
_LEAK_RE = re.compile(
    r"\[E\d+\]|\bE[123]\b|CURRICULUM EVIDENCE|according to the evidence",
    re.IGNORECASE,
)
_REVEALS_ANSWER_RE = re.compile(
    r"\b(?:the\s+(?:correct\s+)?(?:final\s+)?answer\s+is|"
    r"corrected\s+(?:final\s+)?answer|"
    r"complete(?:d)?\s+replacement\s+solution|"
    r"the\s+correct\s+derivative\s+is)\b",
    re.IGNORECASE,
)
_INCORRECT_IN_REASON_RE = re.compile(
    r"\bnot quite\b|\bincorrect\b|\bwrong\b",
    re.IGNORECASE,
)
_CORRECT_IN_REASON_RE = re.compile(
    r"\b(?:is|are)\s+correct\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MistakeGroup:
    mistake: str
    hint: str


@dataclass(frozen=True)
class ConfirmAssessment:
    verdict: str
    brief_reason: str
    mistake_groups: tuple[MistakeGroup, ...]


def verify_derivative_claim(
    question: str,
    *,
    subject: str | None = None,
) -> bool | None:
    """Return True/False when a derivative claim can be checked; else None."""
    verdict, _expected = _derivative_symbolic_check(question, subject=subject)
    return verdict


def _derivative_symbolic_check(
    question: str,
    *,
    subject: str | None = None,
) -> tuple[bool | None, Any]:
    if (subject or "").strip().lower() != "mathematics":
        return None, None
    text = (question or "").replace("\u2212", "-").replace("\u00d7", "*").strip()
    match = _DERIVATIVE_CLAIM_RE.match(text)
    if match is None:
        return None, None
    expr = _parse_math(match.group("expr"))
    proposed = _parse_math(match.group("result"))
    if expr is None or proposed is None:
        return None, None
    try:
        from sympy import diff, expand, simplify
    except ImportError:
        LOGGER.warning("SymPy is not installed; skipping symbolic confirmation")
        return None, None
    symbols = expr.free_symbols | proposed.free_symbols
    if len(symbols) > 1:
        return None, None
    try:
        if symbols:
            variable = next(iter(symbols))
        else:
            from sympy import Symbol

            variable = Symbol("x")
        expected = diff(expr, variable)
        delta = simplify(expand(expected - proposed))
    except (TypeError, ValueError):
        return None, None
    return bool(delta == 0), expected


def _parse_math(raw: str) -> Any | None:
    text = (raw or "").strip()
    if not text or len(text) > 120:
        return None
    if not _ALLOWED_EXPR.match(text) or _BANNED_EXPR.search(text):
        return None
    try:
        from sympy import Add, Float, Integer, Mul, Pow, Rational, Symbol, Tuple
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError:
        return None
    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )
    global_dict = {
        "Symbol": Symbol,
        "Integer": Integer,
        "Float": Float,
        "Rational": Rational,
        "Mul": Mul,
        "Add": Add,
        "Pow": Pow,
        "Tuple": Tuple,
        "__builtins__": {},
    }
    try:
        return parse_expr(
            text,
            transformations=transformations,
            evaluate=True,
            global_dict=global_dict,
        )
    except Exception:
        return None


def format_confirm_response(assessment: ConfirmAssessment) -> str:
    if assessment.verdict == "correct":
        return f"Correct. {assessment.brief_reason.strip()}"
    lines = ["Not quite.", ""]
    for group in assessment.mistake_groups:
        lines.append(f"- {group.mistake.strip()}")
        lines.append(f"  Hint: {group.hint.strip()}")
    return "\n".join(lines)


def parse_confirm_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def validate_confirm_assessment(
    payload: dict[str, Any] | None,
    *,
    locked_verdict: bool | None = None,
    expected_expr: Any = None,
) -> ConfirmAssessment | None:
    if not isinstance(payload, dict):
        return None
    verdict_raw = payload.get("verdict")
    if not isinstance(verdict_raw, str):
        return None
    verdict = verdict_raw.strip().lower()
    if verdict not in {"correct", "incorrect"}:
        return None
    if locked_verdict is True and verdict != "correct":
        return None
    if locked_verdict is False and verdict != "incorrect":
        return None
    reason = payload.get("brief_reason")
    if not isinstance(reason, str) or not reason.strip():
        return None
    reason = reason.strip()
    groups_raw = payload.get("mistake_groups")
    if not isinstance(groups_raw, list):
        return None
    groups: list[MistakeGroup] = []
    for item in groups_raw:
        if not isinstance(item, dict):
            return None
        mistake = item.get("mistake")
        hint = item.get("hint")
        if not isinstance(mistake, str) or not mistake.strip():
            return None
        if not isinstance(hint, str) or not hint.strip():
            return None
        groups.append(MistakeGroup(mistake=mistake.strip(), hint=hint.strip()))
    groups = _group_repeated_mistakes(groups)
    if verdict == "correct":
        if groups:
            return None
        if _INCORRECT_IN_REASON_RE.search(reason):
            return None
    else:
        if not groups:
            return None
        if _CORRECT_IN_REASON_RE.search(reason):
            return None
    fields = [reason, *(g.mistake for g in groups), *(g.hint for g in groups)]
    for field in fields:
        if _VERDICT_PREFIX_RE.match(field):
            return None
        if _LEAK_RE.search(field):
            return None
    if verdict == "incorrect":
        for field in fields:
            if _REVEALS_ANSWER_RE.search(field):
                return None
            if expected_expr is not None and _contains_expected(field, expected_expr):
                return None
    assessment = ConfirmAssessment(
        verdict=verdict,
        brief_reason=reason,
        mistake_groups=tuple(groups),
    )
    formatted = format_confirm_response(assessment)
    if len(formatted.split()) > CONFIRM_WORD_LIMIT:
        return None
    if verdict == "correct" and re.search(r"\bnot quite\b", formatted, re.I):
        return None
    if verdict == "incorrect" and re.match(r"^\s*Correct\.", formatted):
        return None
    return assessment


def _group_repeated_mistakes(groups: Sequence[MistakeGroup]) -> list[MistakeGroup]:
    seen: set[str] = set()
    out: list[MistakeGroup] = []
    for group in groups:
        key = re.sub(r"\s+", " ", group.mistake.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(group)
    return out


def _expr_fingerprint(value: Any) -> str:
    text = str(value).lower().replace("**", "^")
    text = re.sub(r"\s+", "", text)
    text = text.replace("*", "")
    return text


def _contains_expected(text: str, expected: Any) -> bool:
    needle = _expr_fingerprint(expected)
    if len(needle) < 2:
        return False
    return needle in _expr_fingerprint(text)


def _with_locked_verdict(
    messages: Sequence[dict[str, str]],
    locked: bool | None,
) -> list[dict[str, str]]:
    copied = [{"role": m["role"], "content": m["content"]} for m in messages]
    if locked is None or not copied:
        return copied
    if locked:
        extra = (
            "LOCKED SYMBOLIC VERDICT: correct\n"
            "A safe symbolic check found the student's proposed derivative is "
            "algebraically equivalent to the expected derivative. Your JSON "
            "verdict MUST be \"correct\". Do not mark it incorrect."
        )
    else:
        extra = (
            "LOCKED SYMBOLIC VERDICT: incorrect\n"
            "A safe symbolic check found the student's proposed derivative is "
            "not algebraically equivalent. Your JSON verdict MUST be "
            "\"incorrect\". Give mistake groups and hints. Do not state the "
            "correct derivative."
        )
    copied[-1]["content"] = copied[-1]["content"] + "\n\n" + extra
    return copied


def _with_repair(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    copied = [{"role": m["role"], "content": m["content"]} for m in messages]
    if copied:
        copied[-1]["content"] = copied[-1]["content"] + "\n\n" + JSON_REPAIR_INSTRUCTION
    return copied


def evaluate_confirm(
    generator: Any,
    messages: Sequence[dict[str, str]],
    *,
    question: str,
    subject: str | None = None,
    image_paths: Sequence[str] = (),
) -> str:
    """Run one or two greedy JSON evaluations and format the student reply."""
    locked, expected = _derivative_symbolic_check(question, subject=subject)
    eval_messages = _with_locked_verdict(messages, locked)

    def _attempt(prompt_messages: Sequence[dict[str, str]]) -> ConfirmAssessment | None:
        raw = generator.complete(
            prompt_messages,
            settings=CONFIRM_GENERATION_SETTINGS,
            image_paths=image_paths,
        )
        return validate_confirm_assessment(
            parse_confirm_json(raw if isinstance(raw, str) else ""),
            locked_verdict=locked,
            expected_expr=expected,
        )

    assessment = _attempt(eval_messages)
    if assessment is None:
        assessment = _attempt(_with_repair(eval_messages))
    if assessment is None:
        LOGGER.warning("CONFIRM_ANSWER evaluation failed after retry; using fallback")
        return SAFE_FALLBACK
    return format_confirm_response(assessment)
