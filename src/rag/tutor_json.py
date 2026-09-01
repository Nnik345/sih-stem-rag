"""Shared buffered JSON generation: parse, leak checks, retry, word limits."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from .generator import GenerationSettings
from .socratic import TutorState, _STATE_WORD_LIMITS

ParseValidate = Callable[[str], str | None]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_LEAK_RE = re.compile(
    r"\[E\d+\]|\bE[123]\b|CURRICULUM EVIDENCE|"
    r"according to the (?:curriculum|evidence)|"
    r"the retrieved material|"
    r"\bchunk[_ ]id\b|"
    r"\bGIVE_HINT\b|\bEXPLAIN_CONCEPT\b|\bCONFIRM_ANSWER\b|"
    r"\bINSUFFICIENT_EVIDENCE\b|"
    r"JSON REPAIR|LOCKED SYMBOLIC",
    re.IGNORECASE,
)
_REVEALS_ANSWER_RE = re.compile(
    r"\b(?:the\s+(?:correct\s+)?(?:final\s+)?answer\s+is|"
    r"corrected\s+(?:final\s+)?answer|"
    r"complete(?:d)?\s+replacement\s+solution|"
    r"the\s+correct\s+derivative\s+is)\b",
    re.IGNORECASE,
)

STATE_WORD_LIMITS = _STATE_WORD_LIMITS

STATE_MAX_NEW_TOKENS: dict[TutorState, int] = {
    TutorState.GIVE_HINT: 256,
    TutorState.EXPLAIN_CONCEPT: 640,
    TutorState.CONFIRM_ANSWER: 400,
    TutorState.INSUFFICIENT_EVIDENCE: 256,
}


def settings_for(state: TutorState) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=STATE_MAX_NEW_TOKENS[state],
        do_sample=False,
    )


def word_count(text: str) -> int:
    return len((text or "").split())


def within_word_limit(text: str, state: TutorState) -> bool:
    return word_count(text) <= STATE_WORD_LIMITS[state]


def has_leak(text: str) -> bool:
    return bool(_LEAK_RE.search(text or ""))


def reveals_final_answer(text: str) -> bool:
    return bool(_REVEALS_ANSWER_RE.search(text or ""))


def parse_json_object(raw: str) -> dict[str, Any] | None:
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


def copy_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def append_to_last_user(
    messages: Sequence[dict[str, str]], extra: str
) -> list[dict[str, str]]:
    copied = copy_messages(messages)
    if copied:
        copied[-1]["content"] = copied[-1]["content"] + "\n\n" + extra
    return copied


def complete_json(
    generator: Any,
    messages: Sequence[dict[str, str]],
    *,
    settings: GenerationSettings,
    image_paths: Sequence[str] = (),
    validate: ParseValidate,
    repair_instruction: str,
) -> str | None:
    """One greedy completion, then exactly one repair attempt if invalid."""
    raw = generator.complete(
        messages, settings=settings, image_paths=image_paths
    )
    result = validate(raw if isinstance(raw, str) else "")
    if result is not None:
        return result
    repaired = append_to_last_user(messages, repair_instruction)
    raw2 = generator.complete(
        repaired, settings=settings, image_paths=image_paths
    )
    return validate(raw2 if isinstance(raw2, str) else "")
