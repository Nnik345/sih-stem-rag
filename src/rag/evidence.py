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
from .logging_utils import get_logger
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


def content_terms(text: str) -> set[str]:
    """Lowercased content words, with stopwords and single letters removed."""
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


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

        # 2. best reranker score clears the floor
        top_score = max(c.rerank_score for c in scored)  # type: ignore[type-var]
        score_ok = top_score >= self.config.min_rerank_score
        checks.append(
            EvidenceCheck(
                name="reranker_confidence",
                passed=score_ok,
                detail=(
                    f"best reranker score {top_score:.3f} vs floor "
                    f"{self.config.min_rerank_score:.3f}"
                ),
                value=round(float(top_score), 4),
            )
        )
        if not score_ok:
            reasons.append(
                f"The best retrieved passage scored {top_score:.2f}, below the "
                f"relevance floor of {self.config.min_rerank_score:.2f}."
            )

        # 3. enough independently relevant chunks
        strong = [
            c
            for c in scored
            if (c.rerank_score or float("-inf")) >= self.config.min_rerank_score
        ]
        enough_strong = len(strong) >= self.config.min_strong_chunks
        checks.append(
            EvidenceCheck(
                name="mutually_relevant_chunks",
                passed=enough_strong,
                detail=(
                    f"{len(strong)} chunk(s) at or above the floor; "
                    f"{self.config.min_strong_chunks} required"
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

        # 4. evidence sits inside the requested curriculum scope
        if self.config.require_scope_match and (
            scope.grade is not None or scope.subject is not None
        ):
            in_scope = [c for c in kept if self._matches_scope(c, scope)]
            scope_ok = len(in_scope) > 0 and len(in_scope) == len(kept)
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
                    "Some relevant passages fall outside the requested grade/subject."
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
        query_terms = content_terms(query)
        if query_terms and kept:
            evidence_terms = content_terms(" ".join(c.text for c in kept))
            overlap = len(query_terms & evidence_terms) / len(query_terms)
        else:
            overlap = 0.0
        overlap_ok = overlap >= self.config.min_query_term_overlap
        checks.append(
            EvidenceCheck(
                name="query_term_overlap",
                passed=overlap_ok,
                detail=(
                    f"{overlap:.0%} of the question's content words appear in the "
                    f"evidence; {self.config.min_query_term_overlap:.0%} required"
                ),
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
        if scope.grade is not None and chunk.grade != scope.grade:
            return False
        if scope.subject is not None and (chunk.subject or "").lower() != str(
            scope.subject
        ).lower():
            return False
        return True

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
