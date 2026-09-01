"""Evidence sufficiency gate.

Decides whether the retrieved curriculum actually supports answering the
question. When it does not, the pipeline must return a structured
insufficient-evidence result instead of asking the generator to fill the gap
from its own parametric memory.

Honest statement of what this is
--------------------------------
These are heuristics, not validated thresholds. The reranker emits an unbounded
logit whose scale shifts with query phrasing, so ``min_rerank_score`` is a
starting point to be tuned against the labelled evaluation set in
``data/evaluation/``, not a calibrated probability. The component is deliberately
isolated behind :class:`EvidenceGate` so it can be replaced by a stronger
grounding or hallucination-detection method without touching retrieval or
generation.

Checks performed today:

1. any candidates were retrieved at all
2. the best reranker score clears the configured floor
3. enough chunks independently clear that floor (mutual corroboration)
4. the evidence comes from the requested grade/subject, when one was requested
5. the evidence text actually shares content words with the question
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import EvidenceConfig
from .curriculum_catalog import in_lineage_scope
from .logging_utils import get_logger
from .partitions import is_boilerplate_text, is_production_partition
from .schemas import EvidenceCheck, EvidenceDecision, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)

_TOKEN_RE = re.compile(r"[\w'’]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "a", "about", "all", "an", "and", "any", "are", "as", "at", "be", "because",
        "been", "but", "by", "can", "could", "did", "do", "does", "each", "for",
        "from", "get", "give", "had", "has", "have", "he", "her", "him", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "just", "know", "like", "make",
        "many", "me", "more", "most", "my", "no", "not", "of", "on", "one", "or",
        "other", "our", "out", "over", "same", "she", "should", "so", "some", "such",
        "tell", "than", "that", "the", "their", "them", "then", "there", "these",
        "they", "this", "to", "under", "up", "us", "use", "very", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "why", "will", "with",
        "would", "you", "your",
    }
)
_ALGEBRA_OPERATOR_WORDS = frozenset(
    {"plus", "minus", "times", "equals", "equal", "squared", "cubed"}
)
_INSTANCE_TOKEN_RE = re.compile(r"^(?:\d+[a-z]+|[a-z]+\d+|\d+)$", re.IGNORECASE)


def content_terms(text: str) -> set[str]:
    """Lowercased content words, with stopwords and single letters removed."""
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def concept_terms(text: str, *, mathematics: bool = False) -> set[str]:
    """Content words, optionally dropping algebraic instance tokens for maths.

    When every leftover token is an instance (``3x``, ``6x``, ``plus``), return
    an empty set. Callers must not fall back to the instance tokens: that would
    demand the student's exact polynomial appear in NCERT.
    """
    terms = content_terms(text)
    if not mathematics:
        return terms
    return {
        token
        for token in terms
        if token not in _ALGEBRA_OPERATOR_WORDS
        and _INSTANCE_TOKEN_RE.match(token) is None
    }


def has_maths_instance_token(text: str) -> bool:
    """True when ``text`` contains digits-as-algebra (``3x``, ``6``, …)."""
    return any(_INSTANCE_TOKEN_RE.match(token) for token in _TOKEN_RE.findall(text.lower()))


def strip_maths_instance_text(text: str) -> str:
    """Keep rule words; drop numbers, ``3x``-style tokens, and operator words."""
    kept: list[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        low = token.lower()
        if len(low) <= 2 or low in _STOPWORDS or low in _ALGEBRA_OPERATOR_WORDS:
            continue
        if _INSTANCE_TOKEN_RE.match(low):
            continue
        kept.append(low)
    return " ".join(kept)


class EvidenceGate:
    """Applies the sufficiency checks to a reranked candidate list."""

    def __init__(self, config: EvidenceConfig) -> None:
        self.config = config

    def evaluate(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        scope: RetrievalFilter | None = None,
    ) -> EvidenceDecision:
        scope = scope or RetrievalFilter()
        checks: list[EvidenceCheck] = []
        reasons: list[str] = []

        # 1. anything retrieved at all
        has_candidates = len(candidates) > 0
        checks.append(
            EvidenceCheck(
                name="candidates_exist",
                passed=has_candidates,
                detail=f"{len(candidates)} candidate chunk(s) after reranking",
                value=len(candidates),
            )
        )
        if not has_candidates:
            reasons.append("Retrieval returned no curriculum chunks for this query.")
            return EvidenceDecision(
                sufficient=False, checks=checks, reasons=reasons, confidence="none"
            )

        scored = [c for c in candidates if c.rerank_score is not None]
        if not scored:
            checks.append(
                EvidenceCheck(
                    name="reranked",
                    passed=False,
                    detail="candidates carry no reranker score; cannot judge relevance",
                )
            )
            reasons.append("Candidates were not reranked, so relevance is unknown.")
            return EvidenceDecision(
                sufficient=False, checks=checks, reasons=reasons, confidence="unknown"
            )

        safe: list[RetrievedChunk] = []
        rejected_unsafe = 0
        for chunk in scored:
            partition = chunk.content_partition or "student_evidence"
            if not is_production_partition(partition) or is_boilerplate_text(chunk.text or ""):
                rejected_unsafe += 1
                continue
            safe.append(chunk)
        safe_ok = len(safe) > 0
        checks.append(
            EvidenceCheck(
                name="safe_partition",
                passed=safe_ok,
                detail=(
                    f"{len(safe)} production chunk(s); "
                    f"{rejected_unsafe} rejected as boilerplate or unsafe partition"
                ),
                value=len(safe),
            )
        )
        if not safe_ok:
            reasons.append(
                "Retrieved passages were licence/credits boilerplate or "
                "unsafe solution/practice material."
            )
            return EvidenceDecision(
                sufficient=False, checks=checks, reasons=reasons, confidence="none"
            )
        scored = safe

        # 2–3. reranker floor, with a fallback when every on-topic page scores
        # below 0 (common on NCERT maths, where the book does not use names
        # like "power rule").
        top_score = max(c.rerank_score for c in scored)  # type: ignore[type-var]
        strong = [
            c
            for c in scored
            if (c.rerank_score or float("-inf")) >= self.config.min_rerank_score
        ]
        promoted_best = False
        if not strong:
            scoped_for_best = [
                c
                for c in scored
                if (scope.grade is None and scope.subject is None)
                or self._matches_scope(c, scope)
            ]
            pool = scoped_for_best or list(scored)
            if pool:
                best = max(
                    pool,
                    key=lambda c: c.rerank_score
                    if c.rerank_score is not None
                    else float("-inf"),
                )
                strong = [best]
                promoted_best = True
        score_ok = top_score >= self.config.min_rerank_score or promoted_best
        checks.append(
            EvidenceCheck(
                name="reranker_confidence",
                passed=score_ok,
                detail=(
                    f"best reranker score {top_score:.3f} vs floor "
                    f"{self.config.min_rerank_score:.3f}"
                    + ("; accepted best in-scope chunk below floor" if promoted_best else "")
                ),
                value=round(float(top_score), 4),
            )
        )
        if not score_ok:
            reasons.append(
                f"The best retrieved passage scored {top_score:.2f}, below the "
                f"relevance floor of {self.config.min_rerank_score:.2f}."
            )

        enough_strong = len(strong) >= self.config.min_strong_chunks
        checks.append(
            EvidenceCheck(
                name="mutually_relevant_chunks",
                passed=enough_strong,
                detail=(
                    (
                        f"promoted best in-scope chunk at score "
                        f"{(strong[0].rerank_score or 0):.3f}; none met the floor "
                        f"{self.config.min_rerank_score:.3f}"
                    )
                    if promoted_best and strong
                    else (
                        f"{len(strong)} chunk(s) at or above the floor; "
                        f"{self.config.min_strong_chunks} required"
                    )
                ),
                value=len(strong),
            )
        )
        if not enough_strong:
            reasons.append(
                f"Only {len(strong)} passage(s) cleared the relevance floor; "
                f"{self.config.min_strong_chunks} required."
            )

        kept = strong if strong else []

        # 4. evidence sits inside the requested curriculum scope (or lineage)
        if self.config.require_scope_match and (
            scope.grade is not None or scope.subject is not None
        ):
            scoped = [c for c in kept if self._matches_scope(c, scope)]
            current_hits = [c for c in scoped if not self._is_prior_grade(c, scope)]
            in_scope: list[RetrievedChunk] = []
            for chunk in scoped:
                # The extra prior-grade floor is only to drop weak older
                # near-misses when this class already has a hit. If the topic
                # lives in an earlier class (Class 11 derivatives for a Class 12
                # student), that earlier page is the evidence — do not discard it
                # because bge-reranker logits sit near 0, not 1.
                if self._is_prior_grade(chunk, scope) and current_hits:
                    score = (
                        chunk.rerank_score
                        if chunk.rerank_score is not None
                        else float("-inf")
                    )
                    if score < self.config.min_prior_grade_rerank_score:
                        continue
                in_scope.append(chunk)
            scope_ok = len(in_scope) > 0
            checks.append(
                EvidenceCheck(
                    name="scope_match",
                    passed=scope_ok,
                    detail=(
                        f"{len(in_scope)}/{len(kept)} relevant chunk(s) inside "
                        f"{scope.describe()}"
                    ),
                    value=len(in_scope),
                )
            )
            if not scope_ok:
                reasons.append(
                    "No relevant passages sit in this class or a high-scoring "
                    "earlier class of the same subject."
                )
            kept = in_scope
        else:
            checks.append(
                EvidenceCheck(
                    name="scope_match",
                    passed=True,
                    detail="no grade/subject scope requested; check not applied",
                )
            )

        # 5. the evidence text shares content words with the question
        maths = (scope.subject or "").strip().lower() == "mathematics"
        query_terms = concept_terms(query, mathematics=maths)
        if maths and not query_terms:
            # Instance-only maths (a specific polynomial, a numeric sum). The
            # reranker and scope checks already judged the passages; do not
            # require the book's example to use the same coefficients.
            overlap = 1.0
            overlap_ok = True
            overlap_detail = (
                "maths instance query has no remaining concept terms; "
                "overlap check skipped"
            )
        elif query_terms and kept:
            evidence_terms = concept_terms(
                " ".join(c.text for c in kept), mathematics=maths
            )
            overlap = len(query_terms & evidence_terms) / len(query_terms)
            overlap_ok = overlap >= self.config.min_query_term_overlap
            overlap_detail = (
                f"{overlap:.0%} of the question's content words appear in the "
                f"evidence; {self.config.min_query_term_overlap:.0%} required"
            )
        else:
            overlap = 0.0
            overlap_ok = overlap >= self.config.min_query_term_overlap
            overlap_detail = (
                f"{overlap:.0%} of the question's content words appear in the "
                f"evidence; {self.config.min_query_term_overlap:.0%} required"
            )
        checks.append(
            EvidenceCheck(
                name="query_term_overlap",
                passed=overlap_ok,
                detail=overlap_detail,
                value=round(overlap, 4),
            )
        )
        if not overlap_ok:
            reasons.append(
                "The retrieved passages barely mention the terms in the question."
            )

        enough_chunks = len(kept) >= self.config.min_chunks
        checks.append(
            EvidenceCheck(
                name="minimum_chunks",
                passed=enough_chunks,
                detail=(
                    f"{len(kept)} chunk(s) survive every check; "
                    f"{self.config.min_chunks} required"
                ),
                value=len(kept),
            )
        )

        sufficient = all(check.passed for check in checks)
        decision = EvidenceDecision(
            sufficient=sufficient,
            checks=checks,
            reasons=reasons,
            kept_chunks=list(kept) if sufficient else [],
            confidence=self._confidence(sufficient, float(top_score), len(strong)),
        )

        LOGGER.info(
            "Evidence gate: %s (confidence=%s, kept=%d, top score=%.3f)%s",
            "sufficient" if sufficient else "INSUFFICIENT",
            decision.confidence,
            len(decision.kept_chunks),
            top_score,
            ""
            if sufficient
            else " -- failed: "
            + ", ".join(check.name for check in decision.failed_checks),
        )
        return decision

    @staticmethod
    def _matches_scope(chunk: RetrievedChunk, scope: RetrievalFilter) -> bool:
        return in_lineage_scope(
            chunk_grade=chunk.grade,
            chunk_subject=chunk.subject,
            current_grade=scope.grade,
            current_subject=scope.subject,
            allow_prior_grades=scope.allow_prior_grades,
        )

    @staticmethod
    def _is_prior_grade(chunk: RetrievedChunk, scope: RetrievalFilter) -> bool:
        if scope.grade is None or chunk.grade is None:
            return False
        return int(chunk.grade) < int(scope.grade)

    @staticmethod
    def _confidence(sufficient: bool, top_score: float, strong_count: int) -> str:
        """Coarse label for logs and CLI output; not a calibrated probability."""
        if not sufficient:
            return "insufficient"
        if top_score >= 4.0 and strong_count >= 3:
            return "high"
        if top_score >= 1.0:
            return "medium"
        return "low"
