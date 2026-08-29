"""Bounded diagnostic graph tracing for ignored and rejected neighbours.

Companion queries run only when a trace observer is attached. Primary retrieval
still applies metadata filters in Cypher; these diagnostics never replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .schemas import RetrievalFilter, RetrievedChunk

# Reason codes required by the visualizer.
SEED_NODE = "SEED_NODE"
SELECTED_GRAPH_CANDIDATE = "SELECTED_GRAPH_CANDIDATE"
METADATA_FILTER_MISMATCH = "METADATA_FILTER_MISMATCH"
HUB_CONCEPT_EXCLUDED = "HUB_CONCEPT_EXCLUDED"
DUPLICATE_PATH = "DUPLICATE_PATH"
PER_RELATION_LIMIT = "PER_RELATION_LIMIT"
GRAPH_TOP_K_CUTOFF = "GRAPH_TOP_K_CUTOFF"
RELATION_DISABLED_BY_DEPTH = "RELATION_DISABLED_BY_DEPTH"
TRACE_SAFETY_LIMIT = "TRACE_SAFETY_LIMIT"

REASON_CODES = (
    SEED_NODE,
    SELECTED_GRAPH_CANDIDATE,
    METADATA_FILTER_MISMATCH,
    HUB_CONCEPT_EXCLUDED,
    DUPLICATE_PATH,
    PER_RELATION_LIMIT,
    GRAPH_TOP_K_CUTOFF,
    RELATION_DISABLED_BY_DEPTH,
    TRACE_SAFETY_LIMIT,
)

REASON_EXPLANATIONS: dict[str, str] = {
    SEED_NODE: (
        "This chunk was a graph-expansion seed, so the graph channel excludes it "
        "from its own candidate list."
    ),
    SELECTED_GRAPH_CANDIDATE: (
        "This path produced a graph candidate that GraphRetriever returned."
    ),
    METADATA_FILTER_MISMATCH: (
        "The neighbouring chunk was examined but rejected because it falls outside "
        "the requested grade, subject, unit, or other metadata filters."
    ),
    HUB_CONCEPT_EXCLUDED: (
        "This concept is mentioned by more chunks than the hub threshold, so the "
        "SHARED_CONCEPT relation does not expand through it."
    ),
    DUPLICATE_PATH: (
        "Another path already reached this chunk. The extra path is recorded but "
        "does not create a second candidate."
    ),
    PER_RELATION_LIMIT: (
        "This neighbour sits beyond the per-seed, per-relation expansion cap, so "
        "the primary Cypher LIMIT never returned it."
    ),
    GRAPH_TOP_K_CUTOFF: (
        "This chunk passed graph expansion filters but ranked below graph_top_k "
        "after scores were aggregated."
    ),
    RELATION_DISABLED_BY_DEPTH: (
        "This relation's hop count exceeds graph_max_depth, so its neighbours "
        "were not examined."
    ),
    TRACE_SAFETY_LIMIT: (
        "The diagnostic trace hit a configured safety cap. Unseen neighbours were "
        "not examined and are not implied to have been ignored."
    ),
}

LOGICAL_RELATIONS = frozenset({"SAME_LESSON"})

DEFAULT_TRACE_PER_RELATION = 50
DEFAULT_TRACE_MAX_NODES = 1000

# Physical Neo4j relationship type used to draw an edge for each logical relation.
PHYSICAL_EDGE_TYPE: dict[str, str] = {
    "SAME_SECTION": "HAS_CHUNK",
    "ADJACENT": "NEXT|PREVIOUS",
    "SAME_PAGE": "ON_PAGE",
    "SHARED_CONCEPT": "MENTIONS",
}


@dataclass
class DiagnosticRow:
    """One neighbour examined by a bounded diagnostic query."""

    seed_chunk_id: str
    seed_weight: float
    relation: str
    candidate_chunk_id: str
    position: int
    text: str = ""
    via: str | None = None
    via_id: str | None = None
    via_type: str | None = None
    via_label: str | None = None
    grade: int | None = None
    subject: str | None = None
    unit_id: str | None = None
    unit_title: str | None = None
    document_id: str | None = None
    document_title: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    resource_type: str | None = None
    audience: str | None = None
    truncated_group: bool = False

    def path_key(self) -> tuple[str, str, str | None, str]:
        return (self.seed_chunk_id, self.relation, self.via, self.candidate_chunk_id)

    def metadata(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "subject": self.subject,
            "unit_id": self.unit_id,
            "unit_title": self.unit_title,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "resource_type": self.resource_type,
            "audience": self.audience,
        }


@dataclass
class HubConceptRow:
    seed_chunk_id: str
    concept_id: str
    name: str
    mention_count: int


@dataclass
class GraphPathTrace:
    path_id: str
    seed_chunk_id: str
    relation: str
    via: str | None
    via_id: str | None
    candidate_chunk_id: str
    contribution: float
    accepted: bool
    reason_code: str
    reason_detail: str
    logical: bool
    seed_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "seed_chunk_id": self.seed_chunk_id,
            "relation": self.relation,
            "via": self.via,
            "via_id": self.via_id,
            "candidate_chunk_id": self.candidate_chunk_id,
            "contribution": self.contribution,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "reason_detail": self.reason_detail,
            "logical": self.logical,
            "edge_kind": "logical" if self.logical else "physical",
            "seed_weight": self.seed_weight,
        }


@dataclass
class GraphNodeTrace:
    node_id: str
    label: str
    node_kind: str
    status: str
    text: str = ""
    display_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    seed_weight: float | None = None
    graph_score: float | None = None
    path_ids: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    entered_fusion: bool = False
    final_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "id": self.node_id,
            "label": self.label,
            "node_kind": self.node_kind,
            "status": self.status,
            "text": self.text,
            "display_label": self.display_label or self.node_id,
            "metadata": self.metadata,
            "seed_weight": self.seed_weight,
            "graph_score": self.graph_score,
            "path_ids": list(self.path_ids),
            "reason_codes": list(self.reason_codes),
            "entered_fusion": self.entered_fusion,
            "final_evidence": self.final_evidence,
        }


@dataclass
class GraphEdgeTrace:
    edge_id: str
    source: str
    target: str
    relation: str
    physical: bool
    accepted: bool
    neo4j_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "physical": self.physical,
            "logical": not self.physical,
            "accepted": self.accepted,
            "neo4j_type": self.neo4j_type,
            "label": self.relation if self.physical else f"{self.relation} (logical)",
        }


@dataclass
class RelationDecision:
    relation: str
    enabled: bool
    reason_code: str | None = None
    detail: str = ""
    hop_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "enabled": self.enabled,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "hop_count": self.hop_count,
        }


@dataclass
class GraphTrace:
    seeds: list[dict[str, Any]] = field(default_factory=list)
    enabled_relations: list[str] = field(default_factory=list)
    disabled_relations: list[RelationDecision] = field(default_factory=list)
    relation_decisions: list[RelationDecision] = field(default_factory=list)
    nodes: list[GraphNodeTrace] = field(default_factory=list)
    edges: list[GraphEdgeTrace] = field(default_factory=list)
    paths: list[GraphPathTrace] = field(default_factory=list)
    truncated: bool = False
    truncation_caps: list[str] = field(default_factory=list)
    truncation_warning: str = ""
    selected_chunk_ids: list[str] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "enabled_relations": list(self.enabled_relations),
            "disabled_relations": [d.to_dict() for d in self.disabled_relations],
            "relation_decisions": [d.to_dict() for d in self.relation_decisions],
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "paths": [p.to_dict() for p in self.paths],
            "truncated": self.truncated,
            "truncation_caps": list(self.truncation_caps),
            "truncation_warning": self.truncation_warning,
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "counters": dict(self.counters),
        }


def chunk_matches_scope(row: DiagnosticRow, scope: RetrievalFilter) -> bool:
    """Python-side copy of the Cypher metadata predicate, used only for labelling."""
    if scope.grade is not None and row.grade != scope.grade:
        return False
    if scope.subject is not None and (row.subject or "").lower() != str(scope.subject).lower():
        return False
    if scope.unit_id is not None and row.unit_id != scope.unit_id:
        return False
    if scope.unit_title_contains is not None:
        title = (row.unit_title or "").lower()
        if scope.unit_title_contains.lower() not in title:
            return False
    if scope.resource_type is not None and row.resource_type != scope.resource_type:
        return False
    if scope.audience is not None and row.audience != scope.audience:
        return False
    if scope.document_id is not None and row.document_id != scope.document_id:
        return False
    return True


def mismatch_detail(row: DiagnosticRow, scope: RetrievalFilter) -> str:
    parts: list[str] = []
    if scope.grade is not None and row.grade != scope.grade:
        parts.append(f"grade {row.grade!r} != {scope.grade}")
    if scope.subject is not None and (row.subject or "").lower() != str(scope.subject).lower():
        parts.append(f"subject {row.subject!r} != {scope.subject!r}")
    if scope.unit_id is not None and row.unit_id != scope.unit_id:
        parts.append(f"unit_id {row.unit_id!r} != {scope.unit_id!r}")
    if scope.resource_type is not None and row.resource_type != scope.resource_type:
        parts.append(f"resource_type {row.resource_type!r} != {scope.resource_type!r}")
    if scope.audience is not None and row.audience != scope.audience:
        parts.append(f"audience {row.audience!r} != {scope.audience!r}")
    if scope.document_id is not None and row.document_id != scope.document_id:
        parts.append(f"document_id {row.document_id!r} != {scope.document_id!r}")
    return "; ".join(parts) or "metadata filter mismatch"


def primary_records_to_rows(
    records: Sequence[dict[str, Any]],
    seeds: Sequence[dict[str, Any]],
) -> list[DiagnosticRow]:
    weight_by_seed = {s["chunk_id"]: float(s.get("weight") or 0.0) for s in seeds}
    grouped: dict[tuple[str, str], int] = {}
    rows: list[DiagnosticRow] = []
    for record in records:
        seed_id = record["seed_chunk_id"]
        relation = record["relation"]
        key = (seed_id, relation)
        grouped[key] = grouped.get(key, 0) + 1
        rows.append(
            DiagnosticRow(
                seed_chunk_id=seed_id,
                seed_weight=float(record.get("seed_weight") or weight_by_seed.get(seed_id, 0.0)),
                relation=relation,
                candidate_chunk_id=record["chunk_id"],
                position=grouped[key],
                text=record.get("text") or "",
                via=record.get("via"),
                via_id=_via_id_from_record(record),
                via_type=_via_type_for_relation(relation),
                via_label=record.get("via"),
                grade=record.get("grade"),
                subject=record.get("subject"),
                unit_id=record.get("unit_id"),
                unit_title=record.get("unit_title"),
                document_id=record.get("document_id"),
                document_title=record.get("document_title"),
                section_id=record.get("section_id"),
                section_title=record.get("section_title"),
                resource_type=record.get("resource_type"),
                audience=record.get("audience"),
            )
        )
    return rows


def record_to_diagnostic_row(record: dict[str, Any], position: int) -> DiagnosticRow:
    return DiagnosticRow(
        seed_chunk_id=record["seed_chunk_id"],
        seed_weight=float(record.get("seed_weight") or 0.0),
        relation=record["relation"],
        candidate_chunk_id=record["chunk_id"],
        position=position,
        text=record.get("text") or "",
        via=record.get("via"),
        via_id=record.get("via_id") or _via_id_from_record(record),
        via_type=record.get("via_type") or _via_type_for_relation(record["relation"]),
        via_label=record.get("via_label") or record.get("via"),
        grade=record.get("grade"),
        subject=record.get("subject"),
        unit_id=record.get("unit_id"),
        unit_title=record.get("unit_title"),
        document_id=record.get("document_id"),
        document_title=record.get("document_title"),
        section_id=record.get("section_id"),
        section_title=record.get("section_title"),
        resource_type=record.get("resource_type"),
        audience=record.get("audience"),
        truncated_group=bool(record.get("truncated_group")),
    )


def classify_graph_trace(
    *,
    seeds: Sequence[dict[str, Any]],
    scope: RetrievalFilter,
    selected: Sequence[RetrievedChunk],
    primary_rows: Sequence[DiagnosticRow],
    diagnostic_rows: Sequence[DiagnosticRow] = (),
    hub_concepts: Sequence[HubConceptRow] = (),
    enabled_relations: Sequence[str] = (),
    disabled_relations: Sequence[str] = (),
    relation_depth: dict[str, int] | None = None,
    relation_weight: dict[str, float] | None = None,
    per_relation_limit: int = 8,
    graph_top_k: int = 20,
    max_concept_mentions: int = 400,
    max_paths_per_seed_relation: int = DEFAULT_TRACE_PER_RELATION,
    max_nodes: int = DEFAULT_TRACE_MAX_NODES,
) -> GraphTrace:
    """Classify every examined path. Pure function: no database access."""
    from .graph_retriever import RELATION_DEPTH, RELATION_WEIGHT

    depth = relation_depth or RELATION_DEPTH
    weights = relation_weight or RELATION_WEIGHT
    selected_ids = [chunk.chunk_id for chunk in selected]
    selected_set = set(selected_ids)
    score_by_id = {chunk.chunk_id: chunk.graph_score for chunk in selected}
    seed_ids = [s["chunk_id"] for s in seeds]
    seed_set = set(seed_ids)
    seed_weight = {s["chunk_id"]: float(s.get("weight") or 0.0) for s in seeds}
    seed_text = {s["chunk_id"]: str(s.get("text") or "") for s in seeds}

    trace = GraphTrace(
        seeds=[{"chunk_id": s["chunk_id"], "weight": float(s.get("weight") or 0.0)} for s in seeds],
        enabled_relations=list(enabled_relations),
        selected_chunk_ids=list(selected_ids),
    )

    for relation, hops in depth.items():
        enabled = relation in enabled_relations
        decision = RelationDecision(
            relation=relation,
            enabled=enabled,
            hop_count=hops,
        )
        if not enabled:
            decision.reason_code = RELATION_DISABLED_BY_DEPTH
            decision.detail = (
                f"{relation} needs {hops} hop(s) but graph_max_depth excludes it. "
                "Neighbouring nodes were not examined."
            )
            trace.disabled_relations.append(decision)
        trace.relation_decisions.append(decision)

    nodes: dict[str, GraphNodeTrace] = {}
    edges: dict[str, GraphEdgeTrace] = {}
    paths: list[GraphPathTrace] = []
    accepted_chunks: set[str] = set()
    seen_path_keys: set[tuple[str, str, str | None, str]] = set()
    seen_chunk_paths: dict[str, int] = {}

    def ensure_node(node: GraphNodeTrace) -> GraphNodeTrace:
        existing = nodes.get(node.node_id)
        if existing is None:
            if len(nodes) >= max_nodes:
                trace.truncated = True
                if "max_nodes" not in trace.truncation_caps:
                    trace.truncation_caps.append("max_nodes")
                return existing  # type: ignore[return-value]
            nodes[node.node_id] = node
            return node
        if node.status == "accepted":
            existing.status = "accepted"
        if node.graph_score is not None:
            existing.graph_score = node.graph_score
        if node.seed_weight is not None:
            existing.seed_weight = node.seed_weight
        if node.text and not existing.text:
            existing.text = node.text
        existing.metadata.update({k: v for k, v in node.metadata.items() if v is not None})
        return existing

    def add_reason(node: GraphNodeTrace | None, code: str) -> None:
        if node is None:
            return
        if code not in node.reason_codes:
            node.reason_codes.append(code)

    def add_edge(
        source: str,
        target: str,
        relation: str,
        *,
        physical: bool,
        accepted: bool,
        neo4j_type: str | None,
    ) -> None:
        edge_id = f"{source}|{relation}|{target}|{int(physical)}"
        existing = edges.get(edge_id)
        if existing is None:
            edges[edge_id] = GraphEdgeTrace(
                edge_id=edge_id,
                source=source,
                target=target,
                relation=relation,
                physical=physical,
                accepted=accepted,
                neo4j_type=neo4j_type,
            )
            return
        if accepted:
            existing.accepted = True

    for seed in seeds:
        sid = seed["chunk_id"]
        ensure_node(
            GraphNodeTrace(
                node_id=sid,
                label="Chunk",
                node_kind="seed",
                status="seed",
                text=seed_text.get(sid, ""),
                display_label=_short_label(sid, "seed"),
                seed_weight=seed_weight.get(sid),
                reason_codes=[SEED_NODE],
            )
        )

    for hub in hub_concepts:
        node = ensure_node(
            GraphNodeTrace(
                node_id=hub.concept_id,
                label="Concept",
                node_kind="hub",
                status="ignored",
                display_label=hub.name or hub.concept_id,
                metadata={
                    "mention_count": hub.mention_count,
                    "max_concept_mentions": max_concept_mentions,
                    "name": hub.name,
                },
                reason_codes=[HUB_CONCEPT_EXCLUDED],
            )
        )
        add_reason(node, HUB_CONCEPT_EXCLUDED)
        add_edge(
            hub.seed_chunk_id,
            hub.concept_id,
            "MENTIONS",
            physical=True,
            accepted=False,
            neo4j_type="MENTIONS",
        )

    def classify_row(row: DiagnosticRow, *, from_primary: bool) -> None:
        nonlocal paths
        if trace.truncated and "max_nodes" in trace.truncation_caps and len(nodes) >= max_nodes:
            return

        if row.truncated_group or row.position > max_paths_per_seed_relation:
            trace.truncated = True
            if "per_seed_relation" not in trace.truncation_caps:
                trace.truncation_caps.append("per_seed_relation")
            if row.position > max_paths_per_seed_relation and not from_primary:
                reason = TRACE_SAFETY_LIMIT
                detail = (
                    f"Diagnostic cap of {max_paths_per_seed_relation} paths per "
                    f"seed/relation was reached for {row.relation}."
                )
                _append_path(row, reason, detail, accepted=False, contribution=0.0)
                return

        key = row.path_key()
        already = key in seen_path_keys
        seen_path_keys.add(key)

        contribution = row.seed_weight * weights.get(row.relation, 0.5)
        logical = row.relation in LOGICAL_RELATIONS

        if row.candidate_chunk_id in seed_set:
            reason, accepted, detail = (
                SEED_NODE,
                False,
                REASON_EXPLANATIONS[SEED_NODE],
            )
        elif already:
            reason, accepted, detail = (
                DUPLICATE_PATH,
                False,
                REASON_EXPLANATIONS[DUPLICATE_PATH],
            )
        elif not from_primary and not chunk_matches_scope(row, scope):
            reason, accepted, detail = (
                METADATA_FILTER_MISMATCH,
                False,
                f"{REASON_EXPLANATIONS[METADATA_FILTER_MISMATCH]} ({mismatch_detail(row, scope)})",
            )
        elif not from_primary and row.position > per_relation_limit:
            reason, accepted, detail = (
                PER_RELATION_LIMIT,
                False,
                (
                    f"{REASON_EXPLANATIONS[PER_RELATION_LIMIT]} "
                    f"(position {row.position} > limit {per_relation_limit} for {row.relation})"
                ),
            )
        elif row.candidate_chunk_id in selected_set:
            if seen_chunk_paths.get(row.candidate_chunk_id, 0) > 0:
                reason, accepted, detail = (
                    DUPLICATE_PATH,
                    False,
                    REASON_EXPLANATIONS[DUPLICATE_PATH],
                )
            else:
                reason, accepted, detail = (
                    SELECTED_GRAPH_CANDIDATE,
                    True,
                    REASON_EXPLANATIONS[SELECTED_GRAPH_CANDIDATE],
                )
        elif from_primary:
            reason, accepted, detail = (
                GRAPH_TOP_K_CUTOFF,
                False,
                (
                    f"{REASON_EXPLANATIONS[GRAPH_TOP_K_CUTOFF]} "
                    f"(kept top {graph_top_k})"
                ),
            )
        elif not chunk_matches_scope(row, scope):
            reason, accepted, detail = (
                METADATA_FILTER_MISMATCH,
                False,
                f"{REASON_EXPLANATIONS[METADATA_FILTER_MISMATCH]} ({mismatch_detail(row, scope)})",
            )
        elif row.position > per_relation_limit:
            reason, accepted, detail = (
                PER_RELATION_LIMIT,
                False,
                REASON_EXPLANATIONS[PER_RELATION_LIMIT],
            )
        else:
            reason, accepted, detail = (
                GRAPH_TOP_K_CUTOFF,
                False,
                REASON_EXPLANATIONS[GRAPH_TOP_K_CUTOFF],
            )

        if accepted:
            accepted_chunks.add(row.candidate_chunk_id)
        seen_chunk_paths[row.candidate_chunk_id] = seen_chunk_paths.get(row.candidate_chunk_id, 0) + 1
        _append_path(row, reason, detail, accepted=accepted, contribution=contribution, logical=logical)

    def _append_path(
        row: DiagnosticRow,
        reason: str,
        detail: str,
        *,
        accepted: bool,
        contribution: float,
        logical: bool | None = None,
    ) -> None:
        is_logical = row.relation in LOGICAL_RELATIONS if logical is None else logical
        path_id = f"p{len(paths) + 1}"
        path = GraphPathTrace(
            path_id=path_id,
            seed_chunk_id=row.seed_chunk_id,
            relation=row.relation,
            via=row.via,
            via_id=row.via_id,
            candidate_chunk_id=row.candidate_chunk_id,
            contribution=round(contribution, 6),
            accepted=accepted,
            reason_code=reason,
            reason_detail=detail,
            logical=is_logical,
            seed_weight=row.seed_weight,
        )
        paths.append(path)

        candidate_status = "accepted" if row.candidate_chunk_id in selected_set or (
            accepted and reason == SELECTED_GRAPH_CANDIDATE
        ) else "ignored"
        if row.candidate_chunk_id in selected_set:
            candidate_status = "accepted"
        if reason == METADATA_FILTER_MISMATCH:
            visual = "metadata_mismatch"
        elif reason == HUB_CONCEPT_EXCLUDED:
            visual = "hub"
        elif candidate_status == "accepted":
            visual = "accepted"
        else:
            visual = "ignored"

        node = ensure_node(
            GraphNodeTrace(
                node_id=row.candidate_chunk_id,
                label="Chunk",
                node_kind="candidate",
                status=candidate_status if visual != "metadata_mismatch" else "ignored",
                text=row.text,
                display_label=_short_label(row.candidate_chunk_id, row.section_title or "chunk"),
                metadata={**row.metadata(), "visual": visual, "reason": reason},
                graph_score=score_by_id.get(row.candidate_chunk_id),
            )
        )
        if node is None:
            return
        if candidate_status == "accepted":
            node.status = "accepted"
        node.path_ids.append(path_id)
        add_reason(node, reason)
        if visual == "metadata_mismatch":
            node.metadata["visual"] = "metadata_mismatch"

        if is_logical:
            if row.via_id:
                ensure_node(
                    GraphNodeTrace(
                        node_id=row.via_id,
                        label=row.via_type or "Context",
                        node_kind="context",
                        status="context",
                        display_label=str(row.via_label or row.via or row.via_id),
                        metadata={"via_for": row.relation},
                    )
                )
            add_edge(
                row.seed_chunk_id,
                row.candidate_chunk_id,
                row.relation,
                physical=False,
                accepted=accepted,
                neo4j_type=None,
            )
        elif row.via_id:
            ctx = ensure_node(
                GraphNodeTrace(
                    node_id=row.via_id,
                    label=row.via_type or "Context",
                    node_kind="context",
                    status="context",
                    display_label=str(row.via_label or row.via or row.via_id),
                    metadata={"via_for": row.relation},
                )
            )
            if ctx is not None:
                add_edge(
                    row.seed_chunk_id,
                    row.via_id,
                    PHYSICAL_EDGE_TYPE.get(row.relation, row.relation),
                    physical=True,
                    accepted=accepted,
                    neo4j_type=PHYSICAL_EDGE_TYPE.get(row.relation),
                )
                add_edge(
                    row.via_id,
                    row.candidate_chunk_id,
                    PHYSICAL_EDGE_TYPE.get(row.relation, row.relation),
                    physical=True,
                    accepted=accepted,
                    neo4j_type=PHYSICAL_EDGE_TYPE.get(row.relation),
                )
        else:
            add_edge(
                row.seed_chunk_id,
                row.candidate_chunk_id,
                row.relation,
                physical=True,
                accepted=accepted,
                neo4j_type=PHYSICAL_EDGE_TYPE.get(row.relation, row.relation),
            )

    for row in primary_rows:
        classify_row(row, from_primary=True)

    primary_keys = {row.path_key() for row in primary_rows}
    for row in diagnostic_rows:
        if row.path_key() in primary_keys:
            continue
        classify_row(row, from_primary=False)

    for chunk in selected:
        node = nodes.get(chunk.chunk_id)
        if node is None:
            node = ensure_node(
                GraphNodeTrace(
                    node_id=chunk.chunk_id,
                    label="Chunk",
                    node_kind="candidate",
                    status="accepted",
                    text=chunk.text,
                    display_label=_short_label(chunk.chunk_id, chunk.section_title or "chunk"),
                    metadata=chunk.provenance(),
                    graph_score=chunk.graph_score,
                    reason_codes=[SELECTED_GRAPH_CANDIDATE],
                )
            )
        if node is not None:
            node.status = "accepted"
            node.graph_score = chunk.graph_score
            add_reason(node, SELECTED_GRAPH_CANDIDATE)

    if trace.truncated:
        trace.truncation_warning = (
            "Graph trace was truncated by a safety cap "
            f"({', '.join(trace.truncation_caps)}). Neighbours beyond the cap were "
            "not examined and are not shown as ignored."
        )

    trace.nodes = list(nodes.values())
    trace.edges = list(edges.values())
    trace.paths = paths
    ignored_candidates = [
        n for n in trace.nodes if n.label == "Chunk" and n.status == "ignored" and n.node_kind == "candidate"
    ]
    context_nodes = [n for n in trace.nodes if n.node_kind == "context"]
    trace.counters = {
        "seeds": len(seed_ids),
        "context_nodes": len(context_nodes),
        "candidate_chunks_examined": len(
            [n for n in trace.nodes if n.label == "Chunk" and n.node_kind == "candidate"]
        ),
        "accepted_graph_candidates": len(selected_ids),
        "ignored_candidates": len(ignored_candidates),
        "rejected_paths": len([p for p in paths if not p.accepted]),
        "hub_concepts": len({h.concept_id for h in hub_concepts}),
        "truncated": int(trace.truncated),
    }
    return trace


def _via_type_for_relation(relation: str) -> str | None:
    return {
        "SAME_SECTION": "Section",
        "SAME_PAGE": "Page",
        "SHARED_CONCEPT": "Concept",
        "SAME_LESSON": "Section",
    }.get(relation)


def _via_id_from_record(record: dict[str, Any]) -> str | None:
    if record.get("via_id"):
        return record["via_id"]
    relation = record.get("relation")
    via = record.get("via")
    if via is None:
        return None
    if relation == "SAME_PAGE":
        return str(via)
    if relation == "SHARED_CONCEPT":
        return f"concept:{via}"
    if relation == "SAME_SECTION":
        return record.get("section_id") or f"section:{via}"
    if relation == "SAME_LESSON":
        return f"lesson:{via}"
    return str(via)


def _short_label(node_id: str, fallback: str) -> str:
    tail = node_id.rsplit(":", 1)[-1]
    return tail or fallback


def diagnostic_query(relation: str, trace_limit: int) -> str:
    """Bounded companion query: no metadata filter, hard LIMIT per seed."""
    limit = int(trace_limit)
    if relation == "SAME_SECTION":
        branch = f"""
            MATCH (seed)<-[:HAS_CHUNK]-(s:Section)-[:HAS_CHUNK]->(n:Chunk)
            WHERE n.chunk_id <> seed.chunk_id
            RETURN n AS n, s.section_id AS via_id, s.title AS via_label,
                   'Section' AS via_type, 'SAME_SECTION' AS relation
            LIMIT {limit}
        """
    elif relation == "ADJACENT":
        branch = f"""
            MATCH (seed)-[:NEXT|PREVIOUS]-(n:Chunk)
            WHERE n.chunk_id <> seed.chunk_id
            RETURN n AS n, null AS via_id, null AS via_label,
                   null AS via_type, 'ADJACENT' AS relation
            LIMIT {limit}
        """
    elif relation == "SAME_PAGE":
        branch = f"""
            MATCH (seed)-[:ON_PAGE]->(p:Page)<-[:ON_PAGE]-(n:Chunk)
            WHERE n.chunk_id <> seed.chunk_id
            RETURN n AS n, p.page_id AS via_id, toString(p.page_number) AS via_label,
                   'Page' AS via_type, 'SAME_PAGE' AS relation
            LIMIT {limit}
        """
    elif relation == "SHARED_CONCEPT":
        branch = f"""
            MATCH (seed)-[:MENTIONS]->(co:Concept)<-[:MENTIONS]-(n:Chunk)
            WHERE n.chunk_id <> seed.chunk_id
              AND coalesce(co.mention_count, 0) <= $max_concept_mentions
            RETURN n AS n, co.concept_id AS via_id, co.name AS via_label,
                   'Concept' AS via_type, 'SHARED_CONCEPT' AS relation
            LIMIT {limit}
        """
    elif relation == "SAME_LESSON":
        branch = f"""
            MATCH (n:Chunk)
            WHERE n.unit_id = seed.unit_id
              AND n.section_title = seed.section_title
              AND n.document_id <> seed.document_id
              AND n.chunk_id <> seed.chunk_id
            RETURN n AS n, seed.section_id AS via_id, seed.section_title AS via_label,
                   'Section' AS via_type, 'SAME_LESSON' AS relation
            LIMIT {limit}
        """
    else:
        raise ValueError(f"Unknown graph relation: {relation}")

    return f"""
    /* graph-trace:{relation} */
    UNWIND $seeds AS seed_row
    MATCH (seed:Chunk {{chunk_id: seed_row.chunk_id}})
    CALL (seed) {{
        {branch}
    }}
    RETURN
        n.chunk_id        AS chunk_id,
        n.text            AS text,
        n.grade           AS grade,
        n.subject         AS subject,
        n.unit_id         AS unit_id,
        n.unit_title      AS unit_title,
        n.document_id     AS document_id,
        n.document_title  AS document_title,
        n.section_id      AS section_id,
        n.section_title   AS section_title,
        n.page_start      AS page_start,
        n.page_end        AS page_end,
        n.resource_type   AS resource_type,
        n.audience        AS audience,
        relation          AS relation,
        via_label         AS via,
        via_id            AS via_id,
        via_type          AS via_type,
        via_label         AS via_label,
        seed.chunk_id     AS seed_chunk_id,
        seed_row.weight   AS seed_weight
    """


HUB_CONCEPT_QUERY = """
/* graph-trace:HUB_CONCEPT */
UNWIND $seeds AS seed_row
MATCH (seed:Chunk {chunk_id: seed_row.chunk_id})
CALL (seed) {
    MATCH (seed)-[:MENTIONS]->(co:Concept)
    WHERE coalesce(co.mention_count, 0) > $max_concept_mentions
    RETURN co
    LIMIT $trace_per_relation
}
RETURN
    seed.chunk_id AS seed_chunk_id,
    co.concept_id AS concept_id,
    co.name AS name,
    coalesce(co.mention_count, 0) AS mention_count
"""


def collect_diagnostic_rows(
    store: Any,
    *,
    seeds: Sequence[dict[str, Any]],
    enabled_relations: Sequence[str],
    max_concept_mentions: int,
    trace_per_relation: int = DEFAULT_TRACE_PER_RELATION,
) -> tuple[list[DiagnosticRow], list[HubConceptRow], list[str]]:
    """Run bounded companion queries. Caps are enforced in Cypher LIMIT clauses."""
    if not seeds:
        return [], [], []
    params = {
        "seeds": list(seeds),
        "max_concept_mentions": int(max_concept_mentions),
        "trace_per_relation": int(trace_per_relation),
    }
    rows: list[DiagnosticRow] = []
    caps: list[str] = []
    positions: dict[tuple[str, str], int] = {}
    for relation in enabled_relations:
        records = store.read(diagnostic_query(relation, trace_per_relation), params)
        if len(records) >= trace_per_relation:
            # Per-seed LIMIT means hitting the cap on any seed is possible even
            # when the total row count is below seeds * cap. Flag when any seed
            # group is exactly at the cap.
            by_seed: dict[str, int] = {}
            for record in records:
                sid = record["seed_chunk_id"]
                by_seed[sid] = by_seed.get(sid, 0) + 1
            if any(count >= trace_per_relation for count in by_seed.values()):
                caps.append("per_seed_relation")
        for record in records:
            key = (record["seed_chunk_id"], relation)
            positions[key] = positions.get(key, 0) + 1
            row = record_to_diagnostic_row(record, positions[key])
            if positions[key] >= trace_per_relation:
                row.truncated_group = True
            rows.append(row)

    hubs: list[HubConceptRow] = []
    if "SHARED_CONCEPT" in enabled_relations:
        hub_records = store.read(HUB_CONCEPT_QUERY, params)
        for record in hub_records:
            hubs.append(
                HubConceptRow(
                    seed_chunk_id=record["seed_chunk_id"],
                    concept_id=record["concept_id"],
                    name=record.get("name") or record["concept_id"],
                    mention_count=int(record.get("mention_count") or 0),
                )
            )
    return rows, hubs, caps
