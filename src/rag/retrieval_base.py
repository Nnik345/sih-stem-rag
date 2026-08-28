"""Shared pieces for the three retrieval channels.

Holds the metadata-filter Cypher builder and the record -> :class:`RetrievedChunk`
mapping. The channel-specific Cypher itself stays in each retriever module so a
single channel can be rewritten without touching the others.
"""

from __future__ import annotations

from typing import Any

from .schemas import RetrievalFilter, RetrievedChunk

# Standard chunk projection, so every channel returns the same shape.
CHUNK_PROJECTION = """
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
    c.local_pdf_path  AS local_pdf_path
"""


def chunk_projection(alias: str) -> str:
    """The standard projection rebound to a different node alias."""
    return CHUNK_PROJECTION.replace("c.", f"{alias}.")


def build_filter_clause(
    scope: RetrievalFilter, alias: str = "c"
) -> tuple[str, dict[str, Any]]:
    """Translate a :class:`RetrievalFilter` into a Cypher predicate.

    Returns ``("", {})`` when nothing is restricted. The predicate is applied
    inside the retrieval query itself -- during the index scan, not after
    unrelated results have already been returned.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {}

    if scope.grade is not None:
        conditions.append(f"{alias}.grade = $flt_grade")
        params["flt_grade"] = int(scope.grade)
    if scope.subject is not None:
        conditions.append(f"{alias}.subject = $flt_subject")
        params["flt_subject"] = str(scope.subject).lower()
    if scope.unit_id is not None:
        conditions.append(f"{alias}.unit_id = $flt_unit_id")
        params["flt_unit_id"] = scope.unit_id
    if scope.unit_title_contains is not None:
        conditions.append(
            f"toLower({alias}.unit_title) CONTAINS toLower($flt_unit_title)"
        )
        params["flt_unit_title"] = scope.unit_title_contains
    if scope.resource_type is not None:
        conditions.append(f"{alias}.resource_type = $flt_resource_type")
        params["flt_resource_type"] = scope.resource_type
    if scope.audience is not None:
        conditions.append(f"{alias}.audience = $flt_audience")
        params["flt_audience"] = scope.audience
    if scope.document_id is not None:
        conditions.append(f"{alias}.document_id = $flt_document_id")
        params["flt_document_id"] = scope.document_id

    return " AND ".join(conditions), params


def combine_conditions(*clauses: str) -> str:
    """Join non-empty Cypher predicates with AND, returning "" if all empty."""
    active = [clause for clause in clauses if clause]
    return " AND ".join(active)


def where_clause(*clauses: str) -> str:
    """Render a WHERE keyword only when there is something to filter on."""
    combined = combine_conditions(*clauses)
    return f"WHERE {combined}" if combined else ""


def chunk_from_record(record: dict[str, Any]) -> RetrievedChunk:
    """Build a :class:`RetrievedChunk` from a projected Neo4j record."""
    return RetrievedChunk(
        chunk_id=record["chunk_id"],
        text=record.get("text") or "",
        grade=record.get("grade"),
        subject=record.get("subject"),
        unit_id=record.get("unit_id"),
        unit_title=record.get("unit_title"),
        document_id=record.get("document_id"),
        document_title=record.get("document_title"),
        section_id=record.get("section_id"),
        section_title=record.get("section_title"),
        page_start=record.get("page_start"),
        page_end=record.get("page_end"),
        resource_type=record.get("resource_type"),
        audience=record.get("audience"),
        local_pdf_path=record.get("local_pdf_path"),
    )
