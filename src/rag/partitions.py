"""Classify curriculum text into retrieval partitions.

Production tutor retrieval may use only ``student_evidence`` and safe
``teacher_strategy``. Everything else is stored (when useful for audits) or
skipped, but is never returned on the normal tutor path.

Partitions
----------
* ``student_evidence`` -- conceptual exposition and definitions
* ``teacher_strategy`` -- pedagogy, misconceptions, questioning (no answers)
* ``practice_only`` -- unsolved exercises, homework, exit tickets, problem sets
* ``evaluation_only`` -- answer keys, solutions, marking guides, completed work
* ``excluded_boilerplate`` -- licences, credits, copyright, publication metadata
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .curriculum_catalog import EVALUATION_PARTITION, PRODUCTION_PARTITIONS

STUDENT_EVIDENCE = "student_evidence"
TEACHER_STRATEGY = "teacher_strategy"
PRACTICE_ONLY = "practice_only"
EVALUATION_ONLY = EVALUATION_PARTITION
EXCLUDED_BOILERPLATE = "excluded_boilerplate"

PRODUCTION_TUTOR_PARTITIONS = PRODUCTION_PARTITIONS
NON_PRODUCTION = (PRACTICE_ONLY, EVALUATION_ONLY, EXCLUDED_BOILERPLATE)

_EVAL_NAME_RE = re.compile(
    r"(answer[\s_-]*key|worked[\s_-]*solution|marking[\s_-]*guide|"
    r"scoring[\s_-]*rubric|end[\s_-]*of[\s_-]*module[\s_-]*assessment|"
    r"mid[\s_-]*module[\s_-]*assessment|exit[\s_-]*ticket[\s_-]*sample|"
    r"sprint[\s_-]*answers?|homework[\s_-]*answers?)",
    re.IGNORECASE,
)

_EVAL_HEADING_RE = re.compile(
    r"(answer\s*key|worked\s+solutions?|sample\s+responses?|acceptable\s+answers?|"
    r"mid[\s-]*module\s+assessment|end[\s-]*of[\s-]*module\s+assessment|"
    r"assessment\s+(task|summary)|evaluation\s+rubric|possible\s+solutions?|"
    r"exit\s+ticket\s+(sample|answers?)|homework\s+(answer|solution)|"
    r"marking\s+guide|scoring\s+rubric|\bsolutions?\b)",
    re.IGNORECASE,
)

_PRACTICE_HEADING_RE = re.compile(
    r"(exit\s+ticket|homework|problem\s+set|sprint\b|problem\s+set|"
    r"practice\s+set|fluency\s+sprints?|application\s+problem|"
    r"unsolved|independent\s+practice)",
    re.IGNORECASE,
)

_STRATEGY_HEADING_RE = re.compile(
    r"^(notes\s+on|scaffolding|concept\s+development|fluency\s+practice|"
    r"student\s+debrief|misconception|questioning|pedagogy|teacher\s+note|"
    r"notes\s+for\s+teachers)\b",
    re.IGNORECASE,
)

_BOILERPLATE_HEADING_RE = re.compile(
    r"(credits?\s+and\s+copyright|credits?$|copyright|creative\s+commons|"
    r"acknowledg|isbn|table\s+of\s+contents|^contents$|about\s+the\s+authors?|"
    r"publication\s+information|license\s+notice|licensing)",
    re.IGNORECASE,
)

_BOILERPLATE_BODY_RE = re.compile(
    r"(this work is licensed under a creative commons|"
    r"creative commons attribution|"
    r"unless otherwise noted, the contents of this book are licensed|"
    r"©\s*copyright|all rights reserved|"
    r"isbn[-\s]?\d|"
    r"great minds\.?\s*eureka-math|"
    r"nys common core|"
    r"https?://creativecommons\.org|"
    r"https?://www\.engageny\.org|"
    r"https?://greatminds\.org)",
    re.IGNORECASE,
)

_SOLUTION_BODY_RE = re.compile(
    r"(sample\s+response|acceptable\s+answer|answers?\s+will\s+vary|"
    r"the\s+correct\s+answer\s+is|answer\s*key|"
    r"students?\s+should\s+(have\s+)?(written|said|got|answered)|"
    r"expected\s+response)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_CC_FOOTER_RE = re.compile(
    r"this work is licensed under a creative commons", re.IGNORECASE
)


@dataclass(frozen=True)
class PartitionDecision:
    partition: str
    reason: str


def partition_from_filename(filename: str) -> str:
    if _EVAL_NAME_RE.search(filename or ""):
        return EVALUATION_ONLY
    return STUDENT_EVIDENCE


def partition_from_heading(title: str, *, default: str = STUDENT_EVIDENCE) -> str:
    return classify_section(title, "").partition


def is_boilerplate_text(text: str) -> bool:
    """True when the passage is licence/credits/publication metadata, not curriculum."""
    blob = (text or "").strip()
    if not blob:
        return False
    if _BOILERPLATE_HEADING_RE.search(blob[:120]):
        return True
    urls = len(_URL_RE.findall(blob))
    words = max(1, len(blob.split()))
    if urls >= 3 and urls / words > 0.04:
        return True
    if _CC_FOOTER_RE.search(blob) and len(blob) < 900:
        return True
    if _BOILERPLATE_BODY_RE.search(blob):
        # A long instructional section may mention the licence once in a footer.
        licence_hits = len(_BOILERPLATE_BODY_RE.findall(blob))
        if licence_hits >= 1 and len(blob) < 700:
            return True
        if licence_hits >= 2:
            return True
        # Mostly licence language by proportion.
        if licence_hits >= 1 and words < 80:
            return True
    return False


def classify_section(
    title: str,
    text: str = "",
    *,
    previous_title: str = "",
    default: str = STUDENT_EVIDENCE,
) -> PartitionDecision:
    """Use heading, body, and neighboring labels — not a single regex."""
    heading = (title or "").strip()
    body = (text or "").strip()
    combined = f"{heading}\n{body}"
    prev = (previous_title or "").strip()

    if is_boilerplate_text(heading) or is_boilerplate_text(body[:800]):
        if _BOILERPLATE_HEADING_RE.search(heading) or is_boilerplate_text(body[:500]):
            if len(body) < 1200 or _BOILERPLATE_HEADING_RE.search(heading):
                return PartitionDecision(EXCLUDED_BOILERPLATE, "licence/credits/publication text")

    if _EVAL_HEADING_RE.search(heading) or _EVAL_NAME_RE.search(heading):
        return PartitionDecision(EVALUATION_ONLY, "heading names an answer key, assessment, or solution")
    if _SOLUTION_BODY_RE.search(body[:1500]):
        return PartitionDecision(EVALUATION_ONLY, "body contains sample/acceptable answers or answer-key language")
    if _EVAL_HEADING_RE.search(prev) and _SOLUTION_BODY_RE.search(body[:800]):
        return PartitionDecision(EVALUATION_ONLY, "follows an answer-key heading and discloses answers")

    if _STRATEGY_HEADING_RE.search(heading):
        if _SOLUTION_BODY_RE.search(body[:1500]):
            return PartitionDecision(
                EVALUATION_ONLY, "teacher section discloses completed answers"
            )
        return PartitionDecision(TEACHER_STRATEGY, "pedagogical guidance heading")

    if _PRACTICE_HEADING_RE.search(heading):
        if _SOLUTION_BODY_RE.search(body[:1500]) or _EVAL_HEADING_RE.search(heading):
            return PartitionDecision(EVALUATION_ONLY, "practice heading with disclosed answers")
        return PartitionDecision(PRACTICE_ONLY, "unsolved exercise, homework, or exit ticket")

    if is_boilerplate_text(combined) and len(body) < 700:
        return PartitionDecision(EXCLUDED_BOILERPLATE, "licence or credits boilerplate")

    return PartitionDecision(default, "default conceptual exposition")


def is_production_partition(partition: str) -> bool:
    return partition in PRODUCTION_TUTOR_PARTITIONS
