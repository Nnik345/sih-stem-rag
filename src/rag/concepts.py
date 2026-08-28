"""Conservative concept extraction for the curriculum graph.

Concept extraction deliberately refuses to invent semantics. Concepts come only from text
that actually exists in the corpus:

* unit titles (from `manifest.json` or the directory layout)
* section titles / lesson headings discovered by the PDF parser
* segments of those titles split on punctuation and conjunctions, which are
  still verbatim phrases from the source

A ``(:Chunk)-[:MENTIONS]->(:Concept)`` edge is created only when the concept
phrase occurs *verbatim* (case-insensitive, whitespace-normalised, on word
boundaries) in the chunk text, or when the concept is the chunk's own section or
unit title. No LLM is involved, and no relationship type beyond MENTIONS /
ILLUSTRATES is produced -- CAUSES, PREREQUISITE_OF and friends would require
inference this system does not trust yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .logging_utils import get_logger
from .schemas import ConceptMention, concept_id_for

LOGGER = get_logger(__name__)

# Sources of concept evidence, stored on the MENTIONS edge.
SOURCE_UNIT_TITLE = "unit_title"
SOURCE_SECTION_TITLE = "section_title"
SOURCE_CURRICULUM_TERM = "curriculum_term"

_PUNCT_RE = re.compile(r"[^\w\s-]+")
_WS_RE = re.compile(r"\s+")
_SEGMENT_SPLIT_RE = re.compile(
    r"\s*(?:[,;:/|]|\u2013|\u2014|--|\band\b|\bor\b|\bwith\b)\s*", re.IGNORECASE
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
# Publisher/structural prefixes on Core Knowledge unit titles, e.g.
# "CKMath Grade 1 - Connecting Math to Our World: Math All Around Us".
_TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:ck(?:math|sci|la)\s*)?(?:grade\s*\d+\s*)?(?:unit\s*\d+\s*)?[-\u2013\u2014:]?\s*",
    re.IGNORECASE,
)
_STRUCTURAL_RE = re.compile(
    r"^(?:unit|lesson|chapter|activity|session|segment|day|week|page|part|step|"
    r"appendix|figure|table|grade|core\s+knowledge)\b[\s\d.:-]*$",
    re.IGNORECASE,
)
_HAS_DIGIT_ONLY_RE = re.compile(r"^[\W\d_]+$")

# Boilerplate headings that are structure, not curriculum content.
_BLOCKED_PHRASES = frozenset(
    {
        "overview",
        "at a glance",
        "unit opener",
        "table of contents",
        "contents",
        "online resources",
        "core knowledge",
        "core knowledge foundation",
        "introduction",
        "acknowledgments",
        "acknowledgements",
        "credits",
        "copyright",
        "license",
        "licensing",
        "creative commons",
        "glossary",
        "index",
        "answer key",
        "teacher guide",
        "teacher support",
        "student book",
        "student reader",
        "student workbook",
        "materials",
        "preparation",
        "advance preparation",
        "objectives",
        "learning objectives",
        "standards",
        "assessment",
        "vocabulary",
        "note",
        "notes",
        "tip",
        "tips",
        "warning",
        "caution",
        "activity page",
        "activity pages",
        "front matter",
        "back cover",
        "front cover",
        "big question",
        "what to do",
        "what to expect",
        "for teachers",
        "for students",
        "name",
        "date",
        "directions",
        "instructions",
        "review",
        "practice",
        "homework",
        "exit ticket",
        "warm up",
        "warm-up",
        "cool down",
        "wrap up",
        "wrap-up",
        "closing",
        "opening",
        "extension",
        "differentiation",
        "support",
        "challenge",
    }
)

# Generic words that carry no curriculum meaning on their own.
_BLOCKED_SINGLE_WORDS = frozenset(
    {
        "about",
        "again",
        "also",
        "answer",
        "answers",
        "book",
        "books",
        "class",
        "classroom",
        "each",
        "example",
        "examples",
        "goal",
        "goals",
        "here",
        "idea",
        "ideas",
        "item",
        "items",
        "kind",
        "kinds",
        "learn",
        "learning",
        "lesson",
        "lessons",
        "look",
        "make",
        "many",
        "more",
        "other",
        "others",
        "page",
        "pages",
        "people",
        "question",
        "questions",
        "read",
        "reading",
        "resource",
        "resources",
        "school",
        "student",
        "students",
        "teacher",
        "teachers",
        "thing",
        "things",
        "think",
        "together",
        "topic",
        "topics",
        "unit",
        "units",
        "use",
        "using",
        "way",
        "ways",
        "week",
        "word",
        "words",
        "work",
        "world",
        "write",
        "writing",
        "year",
    }
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

MIN_PHRASE_CHARS = 4
MAX_PHRASE_WORDS = 6
# Endings where a trailing "s" is part of the word, not a plural marker. Without
# "ys" and "ics", "Always" would normalise to "alway" and "Mathematics" to
# "mathematic".
_NO_DEPLURAL_SUFFIXES = ("ss", "us", "is", "os", "as", "ys", "ics", "ous", "ns")
# Standards codes such as "NOS4", "SEP1", "LS1" head many curriculum headings but
# are references, not concepts.
_CODE_TOKEN_RE = re.compile(r"^[a-z]{1,4}\d+[a-z]?$", re.IGNORECASE)


def _depluralize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(_NO_DEPLURAL_SUFFIXES):
        return word[:-1]
    return word


def normalize_concept(name: str) -> str:
    """Canonical key used to deduplicate concept spellings.

    Lowercases, strips punctuation and articles, collapses whitespace and folds a
    simple trailing plural, so "Plant Life Cycles", "plant life cycle" and
    "The Plant Life Cycle" all resolve to ``plant life cycle``.
    """
    text = _PUNCT_RE.sub(" ", name.replace("-", " "))
    text = _WS_RE.sub(" ", text).strip().lower()
    text = _LEADING_ARTICLE_RE.sub("", text).strip()
    if not text:
        return ""

    words = text.split()
    words[-1] = _depluralize(words[-1])
    return " ".join(words)


def is_valid_concept_phrase(phrase: str) -> bool:
    """Quality gate: reject structure, boilerplate and generic filler."""
    cleaned = _WS_RE.sub(" ", phrase).strip()
    if len(cleaned) < MIN_PHRASE_CHARS:
        return False
    if _HAS_DIGIT_ONLY_RE.match(cleaned):
        return False
    if _STRUCTURAL_RE.match(cleaned):
        return False

    normalized = normalize_concept(cleaned)
    if not normalized or len(normalized) < MIN_PHRASE_CHARS:
        return False
    if normalized in _BLOCKED_PHRASES or cleaned.strip().lower() in _BLOCKED_PHRASES:
        return False

    words = normalized.split()
    if len(words) > MAX_PHRASE_WORDS:
        return False
    if all(word in _STOPWORDS for word in words):
        return False
    if any(_CODE_TOKEN_RE.match(word) for word in words):
        return False
    # Reject worksheet references like "AP 1", "AP A 3" or "2 Read Together"
    # while keeping genuine titles such as "Numbers to 1000": a leading bare
    # number, or bare numbers/single letters making up half the phrase, means
    # the phrase is a reference rather than a concept.
    if words[0].isdigit() or len(words[0]) == 1:
        return False
    filler = sum(1 for word in words if word.isdigit() or len(word) == 1)
    if filler * 2 >= len(words):
        return False
    if len(words) == 1:
        word = words[0]
        if word in _BLOCKED_SINGLE_WORDS or word in _STOPWORDS:
            return False
        if len(word) < 5:
            return False
    return True


@dataclass(frozen=True)
class ConceptCandidate:
    """A concept phrase plus the verbatim text it was taken from."""

    name: str
    normalized_name: str
    concept_id: str
    source: str
    evidence: str

    @property
    def word_count(self) -> int:
        return len(self.normalized_name.split())


def _clean_title(title: str) -> str:
    without_prefix = _TITLE_PREFIX_RE.sub("", title, count=1)
    return _WS_RE.sub(" ", without_prefix).strip() or _WS_RE.sub(" ", title).strip()


def candidates_from_title(title: str, source: str) -> list[ConceptCandidate]:
    """Concept candidates from one title: the whole phrase and its segments.

    Splitting on commas and conjunctions keeps every candidate a verbatim
    substring of the title, so "Sun, Moon, and Stars" yields the full title plus
    "Sun", "Moon" and "Stars" without inventing anything.
    """
    cleaned = _clean_title(title)
    if not cleaned:
        return []

    pieces = [cleaned]
    # Titles like "Connecting Math to Our World: Math All Around Us" put the
    # topic after the colon; keep that tail as its own candidate.
    pieces.extend(part for part in _SEGMENT_SPLIT_RE.split(cleaned) if part)

    seen: set[str] = set()
    candidates: list[ConceptCandidate] = []
    for piece in pieces:
        phrase = piece.strip(" -\u2013\u2014:;,.")
        if not is_valid_concept_phrase(phrase):
            continue
        normalized = normalize_concept(phrase)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(
            ConceptCandidate(
                name=phrase,
                normalized_name=normalized,
                concept_id=concept_id_for(normalized),
                source=source,
                evidence=cleaned,
            )
        )
    return candidates


def build_vocabulary(
    unit_titles: Iterable[str],
    section_titles: Iterable[str] = (),
    *,
    min_support_single_word: int = 2,
) -> dict[str, ConceptCandidate]:
    """Merge title-derived candidates into a normalized concept vocabulary.

    Unit titles win on collision because they are the most reliable source.

    ``min_support_single_word`` requires a one-word concept to appear in at least
    that many distinct titles. Without it, incidental words from a single heading
    (story character names, for instance) become concepts and add noise to the
    graph; requiring corroboration keeps single-word concepts to recurring
    curriculum terms.
    """
    from collections import Counter

    support: Counter[str] = Counter()
    candidates: dict[str, ConceptCandidate] = {}

    def register(title: str, source: str, *, overwrite: bool) -> None:
        seen_in_title: set[str] = set()
        for candidate in candidates_from_title(title, source):
            if candidate.normalized_name not in seen_in_title:
                support[candidate.normalized_name] += 1
                seen_in_title.add(candidate.normalized_name)
            if overwrite or candidate.normalized_name not in candidates:
                candidates[candidate.normalized_name] = candidate

    for title in section_titles:
        register(title, SOURCE_SECTION_TITLE, overwrite=False)
    for title in unit_titles:
        register(title, SOURCE_UNIT_TITLE, overwrite=True)

    vocabulary = {
        normalized: candidate
        for normalized, candidate in candidates.items()
        if candidate.word_count > 1 or support[normalized] >= min_support_single_word
    }
    LOGGER.info(
        "Concept vocabulary contains %d normalized concepts "
        "(%d single-word candidates dropped for lack of support)",
        len(vocabulary),
        len(candidates) - len(vocabulary),
    )
    return vocabulary


class ConceptMatcher:
    """Finds verbatim occurrences of vocabulary phrases in chunk text.

    Two matching paths are available. :meth:`find` tests each phrase separately,
    which is clear but O(vocabulary) per chunk. :meth:`find_all` scans once with a
    single combined alternation and maps each hit back through
    :func:`normalize_concept`; with a few thousand phrases and a few thousand
    chunks that is the difference between minutes and seconds, so the linking
    pass uses it.
    """

    def __init__(self, vocabulary: dict[str, ConceptCandidate]) -> None:
        self.vocabulary = vocabulary
        self._patterns: dict[str, re.Pattern[str]] = {}
        for normalized, candidate in vocabulary.items():
            self._patterns[normalized] = self._build_pattern(candidate.name)
        self._combined = self._build_combined_pattern(vocabulary)

    @staticmethod
    def _build_combined_pattern(
        vocabulary: dict[str, ConceptCandidate],
    ) -> re.Pattern[str] | None:
        """One alternation of every phrase, longest first so it wins the match."""
        if not vocabulary:
            return None
        alternatives: list[str] = []
        for candidate in sorted(
            vocabulary.values(), key=lambda c: len(c.name), reverse=True
        ):
            words = [
                w
                for w in _PUNCT_RE.sub(" ", candidate.name.replace("-", " ")).split()
                if w
            ]
            if not words:
                continue
            parts = [re.escape(word) for word in words[:-1]]
            last = re.escape(words[-1])
            if not words[-1].lower().endswith(_NO_DEPLURAL_SUFFIXES):
                last = f"{last}s?"
            parts.append(last)
            alternatives.append(r"[\s\-]+".join(parts))
        if not alternatives:
            return None
        return re.compile(
            r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE
        )

    @staticmethod
    def _build_pattern(name: str) -> re.Pattern[str]:
        """Word-boundary pattern tolerant of whitespace and optional plural 's'.

        Only whitespace and a trailing plural are allowed to vary; the words
        themselves must match exactly, which keeps matching conservative.
        """
        words = [w for w in _PUNCT_RE.sub(" ", name.replace("-", " ")).split() if w]
        if not words:
            return re.compile(r"(?!x)x")
        parts = [re.escape(word) for word in words[:-1]]
        last = re.escape(words[-1])
        if not words[-1].lower().endswith(_NO_DEPLURAL_SUFFIXES):
            last = f"{last}s?"
        parts.append(last)
        return re.compile(r"\b" + r"[\s\-]+".join(parts) + r"\b", re.IGNORECASE)

    def find(self, text: str, *, limit: int | None = None) -> list[ConceptMention]:
        """Concept mentions supported by verbatim occurrences in ``text``."""
        mentions: list[ConceptMention] = []
        for normalized, pattern in self._patterns.items():
            matches = pattern.findall(text)
            if not matches:
                continue
            candidate = self.vocabulary[normalized]
            mentions.append(
                ConceptMention(
                    concept_id=candidate.concept_id,
                    name=candidate.name,
                    normalized_name=normalized,
                    source=SOURCE_CURRICULUM_TERM,
                    occurrences=len(matches),
                    evidence=f"verbatim phrase match ({len(matches)}x)",
                )
            )

        return self._ranked(mentions, limit)

    def find_all(self, text: str, *, limit: int | None = None) -> list[ConceptMention]:
        """Single-scan equivalent of :meth:`find`, for bulk linking."""
        if self._combined is None:
            return []
        counts: dict[str, int] = {}
        for match in self._combined.finditer(text):
            normalized = normalize_concept(match.group(0))
            if normalized in self.vocabulary:
                counts[normalized] = counts.get(normalized, 0) + 1

        mentions = [
            ConceptMention(
                concept_id=self.vocabulary[normalized].concept_id,
                name=self.vocabulary[normalized].name,
                normalized_name=normalized,
                source=SOURCE_CURRICULUM_TERM,
                occurrences=occurrences,
                evidence=f"verbatim phrase match ({occurrences}x)",
            )
            for normalized, occurrences in counts.items()
        ]
        return self._ranked(mentions, limit)

    @staticmethod
    def _ranked(
        mentions: list[ConceptMention], limit: int | None
    ) -> list[ConceptMention]:
        # Longer, more specific phrases first; then by frequency.
        mentions.sort(
            key=lambda m: (-len(m.normalized_name.split()), -m.occurrences, m.name)
        )
        return mentions[:limit] if limit else mentions

    def __len__(self) -> int:
        return len(self._patterns)


def mentions_from_own_titles(
    section_title: str,
    unit_title: str,
) -> list[ConceptMention]:
    """Structural concept links: a chunk mentions its own section/unit topics."""
    mentions: list[ConceptMention] = []
    seen: set[str] = set()
    for title, source in (
        (section_title, SOURCE_SECTION_TITLE),
        (unit_title, SOURCE_UNIT_TITLE),
    ):
        if not title:
            continue
        for candidate in candidates_from_title(title, source):
            if candidate.normalized_name in seen:
                continue
            seen.add(candidate.normalized_name)
            mentions.append(
                ConceptMention(
                    concept_id=candidate.concept_id,
                    name=candidate.name,
                    normalized_name=candidate.normalized_name,
                    source=source,
                    occurrences=1,
                    evidence=f"{source}: {candidate.evidence}",
                )
            )
    return mentions
