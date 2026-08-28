"""Lexical retrieval over the Neo4j full-text (Lucene) index.

    query -> Lucene query -> Neo4j full-text index -> metadata filter -> top-K

This channel recovers what dense embeddings blur: exact terminology, proper
names, curriculum phrases, numbers and specific vocabulary. No separate BM25
service is used -- Neo4j's Lucene index covers ``Chunk.text``,
``Chunk.section_title`` and ``Chunk.unit_title``.

The user's question is rewritten into a safe Lucene query: special characters
are escaped, the full phrase is included as a boosted quoted clause, and the
individual terms are OR-ed so partial matches still score.
"""

from __future__ import annotations

import re
from typing import Any

from .config import RetrievalConfig
from .graph_schema import CHUNK_FULLTEXT_INDEX
from .logging_utils import Timer, get_logger
from .neo4j_store import Neo4jStore
from .retrieval_base import build_filter_clause, chunk_from_record, where_clause
from .schemas import CHANNEL_FULLTEXT, RetrievalFilter, RetrievedChunk

LOGGER = get_logger(__name__)

# Reserved by Lucene's query parser; escaped rather than dropped so terms like
# "3+4" still match.
_LUCENE_SPECIAL = r'+-&|!(){}[]^"~*?:\\/'
_TOKEN_RE = re.compile(r"[\w'’]+", re.UNICODE)
# Boost applied to the exact-phrase clause.
_PHRASE_BOOST = 3
# Words too common to help lexical matching.
_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
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


def escape_lucene(text: str) -> str:
    out = []
    for char in text:
        if char in _LUCENE_SPECIAL:
            out.append("\\" + char)
        else:
            out.append(char)
    return "".join(out)


def build_lucene_query(query: str) -> str:
    """Turn a natural-language question into a safe Lucene query string.

    Returns an empty string when the question has no usable content words, which
    the retriever treats as "no lexical results" rather than an error.
    """
    tokens = _TOKEN_RE.findall(query.lower())
    content = [t for t in tokens if t not in _QUERY_STOPWORDS and len(t) > 1]
    if not content:
        content = [t for t in tokens if len(t) > 1]
    if not content:
        return ""

    clauses = [escape_lucene(term) for term in content]
    if len(content) > 1:
        phrase = escape_lucene(" ".join(content))
        clauses.insert(0, f'"{phrase}"^{_PHRASE_BOOST}')
    return " OR ".join(clauses)


class LexicalRetriever:
    """Full-text retrieval channel."""

    def __init__(
        self,
        store: Neo4jStore,
        config: RetrievalConfig,
        *,
        oversample: int = 5,
    ) -> None:
        self.store = store
        self.config = config
        self.oversample = max(1, oversample)
        self.last_timing_ms: float = 0.0
        self.last_lucene_query: str = ""

    @staticmethod
    def _query(filter_clause: str) -> str:
        return f"""
        CALL db.index.fulltext.queryNodes($index_name, $lucene_query, {{
            limit: $candidate_k
        }})
        YIELD node AS c, score
        {where_clause(filter_clause)}
        RETURN
            c.chunk_id        AS chunk_id,
            c.text            AS text,
            c.grade           AS grade,
            c.subject         AS subject,
            c.unit_id         AS unit_id,
            c.unit_title      AS unit_title,
            c.document_id     AS document_id,
            c.document_title  AS document_title,
            c.section_id      AS section_id,
            c.section_title   AS section_title,
            c.page_start      AS page_start,
            c.page_end        AS page_end,
            c.resource_type   AS resource_type,
            c.audience        AS audience,
            c.local_pdf_path  AS local_pdf_path,
            score             AS score
        ORDER BY score DESC
        LIMIT $top_k
        """

    def retrieve(
        self,
        query: str,
        *,
        scope: RetrievalFilter | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        scope = scope or RetrievalFilter()
        limit = top_k or self.config.fulltext_top_k
        timer = Timer()

        lucene_query = build_lucene_query(query)
        self.last_lucene_query = lucene_query
        if not lucene_query:
            LOGGER.warning(
                "Query %r contains no usable lexical terms; full-text channel "
                "returns nothing",
                query,
            )
            self.last_timing_ms = timer.stop() * 1000
            return []

        filter_clause, filter_params = build_filter_clause(scope, "c")
        candidate_k = limit * self.oversample if filter_clause else limit
        params: dict[str, Any] = {
            "index_name": CHUNK_FULLTEXT_INDEX,
            "lucene_query": lucene_query,
            "candidate_k": int(candidate_k),
            "top_k": int(limit),
            **filter_params,
        }

        records = self.store.read(self._query(filter_clause), params)

        results: list[RetrievedChunk] = []
        for rank, record in enumerate(records, start=1):
            chunk = chunk_from_record(record)
            chunk.fulltext_rank = rank
            chunk.fulltext_score = float(record["score"])
            chunk.add_source(CHANNEL_FULLTEXT)
            results.append(chunk)

        self.last_timing_ms = timer.stop() * 1000
        LOGGER.info(
            "Full-text retrieval: %d results in %.1f ms (scope=%s)",
            len(results),
            self.last_timing_ms,
            scope.describe(),
        )
        return results
