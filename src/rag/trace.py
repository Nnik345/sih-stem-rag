"""Structured trace models and optional pipeline observers.

Trace collection is optional. When no observer is attached, the RAG pipeline
behaves exactly as before and performs no diagnostic graph queries.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

_SECRET_RE = re.compile(
    r"(NEO4J_PASSWORD|password|secret|api[_-]?key|authorization|credential)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return uuid.uuid4().hex


def redact_secrets(text: str) -> str:
    """Strip credential-like tokens from an error string. Never log env dumps."""
    if not text:
        return text
    redacted = re.sub(
        r"(?i)(NEO4J_PASSWORD|password|secret|api[_-]?key|authorization|credential)(\s*[=:]\s*)\S+",
        r"\1\2[redacted]",
        text,
    )
    redacted = _SECRET_RE.sub("[redacted]", redacted)
    return redacted


def safe_error_message(exc: BaseException) -> str:
    """Human-readable error without stack traces or secrets."""
    return redact_secrets(f"{type(exc).__name__}: {exc}")


def payload_contains_secrets(payload: Any) -> bool:
    """Return True if a trace payload looks like it leaked credentials."""
    try:
        blob = payload if isinstance(payload, str) else repr(payload)
    except Exception:
        return False
    return "NEO4J_PASSWORD" in blob or "os.environ" in blob


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


STAGE_ORDER = (
    "query",
    "filters",
    "dense",
    "lexical",
    "graph",
    "fusion",
    "reranker",
    "evidence",
    "prompt",
    "generator",
)

PIPELINE_EVENTS = (
    "run_started",
    "filters_applied",
    "dense_started",
    "dense_completed",
    "lexical_started",
    "lexical_completed",
    "graph_started",
    "graph_completed",
    "fusion_completed",
    "reranker_started",
    "reranker_completed",
    "evidence_completed",
    "prompt_built",
    "generation_started",
    "generation_token",
    "generation_completed",
    "run_completed",
    "run_failed",
)

EVENT_TO_STAGE = {
    "run_started": "query",
    "filters_applied": "filters",
    "dense_started": "dense",
    "dense_completed": "dense",
    "lexical_started": "lexical",
    "lexical_completed": "lexical",
    "graph_started": "graph",
    "graph_completed": "graph",
    "fusion_completed": "fusion",
    "reranker_started": "reranker",
    "reranker_completed": "reranker",
    "evidence_completed": "evidence",
    "prompt_built": "prompt",
    "generation_started": "generator",
    "generation_token": "generator",
    "generation_completed": "generator",
}


@dataclass
class StageTrace:
    name: str
    status: StageStatus = StageStatus.PENDING
    elapsed_ms: float | None = None
    summary: str = ""
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "elapsed_ms": self.elapsed_ms,
            "summary": self.summary,
            "error": self.error,
            "data": self.data,
        }


@dataclass
class CandidateTrace:
    chunk_id: str
    rank: int | None = None
    score: float | None = None
    text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    entered_fusion: bool = False
    final_evidence: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "text": self.text,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "entered_fusion": self.entered_fusion,
            "final_evidence": self.final_evidence,
        }
        payload.update(self.extra)
        return payload


@dataclass
class DenseTrace:
    model_name: str = ""
    embedding_dim: int | None = None
    query_vector_norm: float | None = None
    vector_preview: list[float] = field(default_factory=list)
    strategy: str = ""
    used_approximate_index: bool = False
    used_exact_fallback: bool = False
    candidates: list[CandidateTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "query_vector_norm": self.query_vector_norm,
            "vector_preview": self.vector_preview,
            "strategy": self.strategy,
            "used_approximate_index": self.used_approximate_index,
            "used_exact_fallback": self.used_exact_fallback,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class LexicalTrace:
    original_query: str = ""
    lucene_query: str = ""
    candidates: list[CandidateTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "lucene_query": self.lucene_query,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class FusionCandidateTrace:
    chunk_id: str
    dense_rank: int | None = None
    dense_contribution: float = 0.0
    lexical_rank: int | None = None
    lexical_contribution: float = 0.0
    graph_rank: int | None = None
    graph_contribution: float = 0.0
    channels: list[str] = field(default_factory=list)
    rrf_score: float = 0.0
    fused_rank: int = 0
    text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusionTrace:
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_fulltext: float = 1.0
    weight_graph: float = 0.5
    formula: str = "score(chunk) = Σ weight_c / (k + rank_c)"
    candidates: list[FusionCandidateTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rrf_k": self.rrf_k,
            "weight_dense": self.weight_dense,
            "weight_fulltext": self.weight_fulltext,
            "weight_graph": self.weight_graph,
            "formula": self.formula,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class RerankerCandidateTrace:
    chunk_id: str
    fused_rank: int | None = None
    reranked_rank: int | None = None
    rank_movement: int = 0
    rerank_score: float | None = None
    text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    survived_final_top_k: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankerTrace:
    candidates: list[RerankerCandidateTrace] = field(default_factory=list)
    score_kind: str = "raw_relevance_logit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "score_kind": self.score_kind,
        }


@dataclass
class EvidenceCheckTrace:
    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceTrace:
    sufficient: bool = False
    confidence: str = ""
    checks: list[EvidenceCheckTrace] = field(default_factory=list)
    kept_chunk_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "confidence": self.confidence,
            "checks": [c.to_dict() for c in self.checks],
            "kept_chunk_ids": self.kept_chunk_ids,
            "reasons": self.reasons,
        }


@dataclass
class PromptTrace:
    tutor_state: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    evidence_blocks: list[dict[str, Any]] = field(default_factory=list)
    generation_settings: dict[str, Any] = field(default_factory=dict)
    generation_skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationTrace:
    tokens: list[str] = field(default_factory=list)
    response_text: str = ""
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class RunTrace:
    run_id: str = field(default_factory=new_run_id)
    status: RunStatus = RunStatus.QUEUED
    query: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    requested_state: str | None = None
    retrieval_only: bool = False
    strict: bool = False
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    error: str | None = None
    stages: dict[str, StageTrace] = field(default_factory=dict)
    dense: DenseTrace | None = None
    lexical: LexicalTrace | None = None
    graph: dict[str, Any] | None = None
    fusion: FusionTrace | None = None
    reranker: RerankerTrace | None = None
    evidence: EvidenceTrace | None = None
    prompt: PromptTrace | None = None
    generation: GenerationTrace | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {name: StageTrace(name=name) for name in STAGE_ORDER}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "query": self.query,
            "filters": self.filters,
            "requested_state": self.requested_state,
            "retrieval_only": self.retrieval_only,
            "strict": self.strict,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "dense": self.dense.to_dict() if self.dense else None,
            "lexical": self.lexical.to_dict() if self.lexical else None,
            "graph": self.graph,
            "fusion": self.fusion.to_dict() if self.fusion else None,
            "reranker": self.reranker.to_dict() if self.reranker else None,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "prompt": self.prompt.to_dict() if self.prompt else None,
            "generation": self.generation.to_dict() if self.generation else None,
            "diagnostics": self.diagnostics,
            "events": list(self.events),
        }


@runtime_checkable
class TraceObserver(Protocol):
    """Lightweight callback surface owned by the core RAG package."""

    def on_event(self, event: str, payload: dict[str, Any]) -> None: ...


class NullTraceObserver:
    """No-op observer used when tracing is disabled."""

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        return None


class RecordingObserver:
    """Test helper that records event names and payloads."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class RunTraceCollector:
    """Builds a :class:`RunTrace` from pipeline events."""

    def __init__(self, run_id: str | None = None) -> None:
        self.trace = RunTrace(run_id=run_id or new_run_id())

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        self.trace.events.append(event)
        data = _jsonable(payload) if payload else {}

        if event == "run_started":
            self.trace.status = RunStatus.RUNNING
            self.trace.query = data.get("query", "")
            self.trace.filters = data.get("filters", {})
            self.trace.requested_state = data.get("requested_state")
            self.trace.retrieval_only = bool(data.get("retrieval_only"))
            self.trace.strict = bool(data.get("strict"))
            query_stage = self.trace.stages["query"]
            query_stage.status = StageStatus.COMPLETED
            query_stage.summary = self.trace.query
            return

        if event.endswith("_started"):
            stage_name = EVENT_TO_STAGE.get(event)
            if stage_name:
                self.trace.stages[stage_name].status = StageStatus.RUNNING
            return

        if event == "filters_applied":
            stage = self.trace.stages["filters"]
            stage.status = StageStatus.COMPLETED
            stage.summary = str(data.get("scope", ""))
            stage.data = data
            return

        if event == "dense_completed":
            raw = payload.get("dense")
            if isinstance(raw, DenseTrace):
                self.trace.dense = raw
            elif isinstance(data.get("dense"), dict):
                self.trace.dense = _dense_from_dict(data["dense"])
            self._complete_stage("dense", data)
            return

        if event == "lexical_completed":
            raw = payload.get("lexical")
            self.trace.lexical = raw if isinstance(raw, LexicalTrace) else _lexical_from_dict(
                data.get("lexical", data)
            )
            self._complete_stage("lexical", data)
            return

        if event == "graph_completed":
            raw = payload.get("graph")
            self.trace.graph = raw.to_dict() if hasattr(raw, "to_dict") else data.get("graph", data)
            self._complete_stage("graph", data)
            return

        if event == "fusion_completed":
            raw = payload.get("fusion")
            self.trace.fusion = raw if isinstance(raw, FusionTrace) else _fusion_from_dict(
                data.get("fusion", data)
            )
            self._complete_stage("fusion", data)
            return

        if event == "reranker_completed":
            raw = payload.get("reranker")
            if data.get("skipped"):
                self.trace.stages["reranker"].status = StageStatus.SKIPPED
                self.trace.stages["reranker"].summary = data.get("summary", "skipped")
                self.trace.stages["reranker"].elapsed_ms = data.get("elapsed_ms")
                return
            self.trace.reranker = raw if isinstance(raw, RerankerTrace) else _reranker_from_dict(
                data.get("reranker", data)
            )
            self._complete_stage("reranker", data)
            return

        if event == "evidence_completed":
            raw = payload.get("evidence")
            self.trace.evidence = raw if isinstance(raw, EvidenceTrace) else _evidence_from_dict(
                data.get("evidence", data)
            )
            self._complete_stage("evidence", data)
            return

        if event == "prompt_built":
            raw = payload.get("prompt")
            self.trace.prompt = raw if isinstance(raw, PromptTrace) else _prompt_from_dict(
                data.get("prompt", data)
            )
            self._complete_stage("prompt", data)
            return

        if event == "generation_token":
            if self.trace.generation is None:
                self.trace.generation = GenerationTrace()
            token = data.get("token", "")
            if token:
                self.trace.generation.tokens.append(token)
                self.trace.generation.response_text += token
            self.trace.stages["generator"].status = StageStatus.RUNNING
            return

        if event == "generation_completed":
            if self.trace.generation is None:
                self.trace.generation = GenerationTrace()
            self.trace.generation.response_text = data.get(
                "response_text", self.trace.generation.response_text
            )
            self.trace.generation.elapsed_ms = data.get("elapsed_ms", 0.0)
            if data.get("skipped"):
                self.trace.stages["generator"].status = StageStatus.SKIPPED
                self.trace.stages["generator"].summary = data.get("skip_reason", "skipped")
                self.trace.stages["generator"].elapsed_ms = data.get("elapsed_ms")
                if self.trace.prompt is not None:
                    self.trace.prompt.generation_skipped = True
                    self.trace.prompt.skip_reason = data.get("skip_reason", "")
                return
            self._complete_stage("generator", data)
            return

        if event == "run_completed":
            self.trace.status = RunStatus.COMPLETED
            self.trace.completed_at = utc_now()
            self.trace.diagnostics = data.get("diagnostics", {})
            annotated = self.trace.diagnostics.get("graph_trace")
            if isinstance(annotated, dict) and "nodes" in annotated:
                self.trace.graph = annotated
            self._mark_unused_stages_skipped()
            return

        if event == "run_failed":
            self.trace.status = RunStatus.FAILED
            self.trace.completed_at = utc_now()
            self.trace.error = redact_secrets(str(data.get("error") or "run failed"))
            stage_name = data.get("stage")
            if stage_name and stage_name in self.trace.stages:
                failed = self.trace.stages[stage_name]
                failed.status = StageStatus.FAILED
                failed.error = self.trace.error
                failed.elapsed_ms = data.get("elapsed_ms")
            return

    def snapshot(self) -> RunTrace:
        return self.trace

    def _complete_stage(self, name: str, data: dict[str, Any]) -> None:
        stage = self.trace.stages[name]
        stage.status = StageStatus(data.get("status", StageStatus.COMPLETED.value))
        if stage.status == StageStatus.PENDING:
            stage.status = StageStatus.COMPLETED
        stage.elapsed_ms = data.get("elapsed_ms")
        stage.summary = data.get("summary", stage.summary)
        stage.data = {k: v for k, v in data.items() if k not in {"dense", "lexical", "graph", "fusion", "reranker", "evidence", "prompt"}}

    def _mark_unused_stages_skipped(self) -> None:
        for stage in self.trace.stages.values():
            if stage.status == StageStatus.PENDING:
                stage.status = StageStatus.SKIPPED


def emit(observer: TraceObserver | None, event: str, **payload: Any) -> None:
    """Send an event when an observer is attached; no-op otherwise."""
    if observer is not None:
        observer.on_event(event, payload)


def attach_graph_trace(observer: TraceObserver | None, graph: dict[str, Any] | None) -> None:
    """Replace the live graph snapshot after later-stage annotation.

    ``graph_completed`` is emitted before fusion and rerank. Final-evidence
    flags are written onto ``diagnostics.graph_trace`` afterwards; copy that
    dict onto the collector so the dashboard paints green nodes.
    """
    if observer is None or not graph:
        return
    collector = getattr(observer, "collector", observer)
    trace = getattr(collector, "trace", None)
    if trace is not None:
        trace.graph = graph


def candidate_from_chunk(chunk: Any, **extra: Any) -> CandidateTrace:
    return CandidateTrace(
        chunk_id=chunk.chunk_id,
        rank=extra.pop("rank", None),
        score=extra.pop("score", None),
        text=chunk.text,
        provenance=chunk.provenance(),
        metadata={
            "grade": chunk.grade,
            "subject": chunk.subject,
            "unit_id": chunk.unit_id,
            "document_title": chunk.document_title,
            "section_title": chunk.section_title,
            "resource_type": chunk.resource_type,
            "audience": chunk.audience,
        },
        entered_fusion=extra.pop("entered_fusion", False),
        final_evidence=extra.pop("final_evidence", False),
        extra=extra,
    )


def annotate_later_status(
    candidates: list[CandidateTrace],
    fused_ids: set[str],
    evidence_ids: set[str],
) -> None:
    for candidate in candidates:
        candidate.entered_fusion = candidate.chunk_id in fused_ids
        candidate.final_evidence = candidate.chunk_id in evidence_ids


def _candidate_from_dict(data: dict[str, Any]) -> CandidateTrace:
    extra = {
        k: v
        for k, v in data.items()
        if k
        not in {
            "chunk_id",
            "rank",
            "score",
            "text",
            "provenance",
            "metadata",
            "entered_fusion",
            "final_evidence",
        }
    }
    return CandidateTrace(
        chunk_id=data.get("chunk_id", ""),
        rank=data.get("rank"),
        score=data.get("score"),
        text=data.get("text", ""),
        provenance=data.get("provenance", {}),
        metadata=data.get("metadata", {}),
        entered_fusion=bool(data.get("entered_fusion")),
        final_evidence=bool(data.get("final_evidence")),
        extra=extra,
    )


def _dense_from_dict(data: dict[str, Any]) -> DenseTrace:
    return DenseTrace(
        model_name=data.get("model_name", ""),
        embedding_dim=data.get("embedding_dim"),
        query_vector_norm=data.get("query_vector_norm"),
        vector_preview=list(data.get("vector_preview") or []),
        strategy=data.get("strategy", ""),
        used_approximate_index=bool(data.get("used_approximate_index")),
        used_exact_fallback=bool(data.get("used_exact_fallback")),
        candidates=[_candidate_from_dict(c) for c in data.get("candidates") or []],
    )


def _lexical_from_dict(data: dict[str, Any]) -> LexicalTrace:
    return LexicalTrace(
        original_query=data.get("original_query", ""),
        lucene_query=data.get("lucene_query", ""),
        candidates=[_candidate_from_dict(c) for c in data.get("candidates") or []],
    )


def _fusion_from_dict(data: dict[str, Any]) -> FusionTrace:
    return FusionTrace(
        rrf_k=int(data.get("rrf_k", 60)),
        weight_dense=float(data.get("weight_dense", 1.0)),
        weight_fulltext=float(data.get("weight_fulltext", 1.0)),
        weight_graph=float(data.get("weight_graph", 0.5)),
        formula=data.get("formula", FusionTrace.formula),
        candidates=[FusionCandidateTrace(**c) if isinstance(c, dict) else c for c in data.get("candidates") or []],
    )


def _reranker_from_dict(data: dict[str, Any]) -> RerankerTrace:
    return RerankerTrace(
        candidates=[
            RerankerCandidateTrace(**c) if isinstance(c, dict) else c
            for c in data.get("candidates") or []
        ],
        score_kind=data.get("score_kind", "raw_relevance_logit"),
    )


def _evidence_from_dict(data: dict[str, Any]) -> EvidenceTrace:
    return EvidenceTrace(
        sufficient=bool(data.get("sufficient")),
        confidence=data.get("confidence", ""),
        checks=[EvidenceCheckTrace(**c) if isinstance(c, dict) else c for c in data.get("checks") or []],
        kept_chunk_ids=list(data.get("kept_chunk_ids") or []),
        reasons=list(data.get("reasons") or []),
    )


def _prompt_from_dict(data: dict[str, Any]) -> PromptTrace:
    return PromptTrace(
        tutor_state=data.get("tutor_state", ""),
        system_prompt=data.get("system_prompt", ""),
        user_prompt=data.get("user_prompt", ""),
        evidence_blocks=list(data.get("evidence_blocks") or []),
        generation_settings=dict(data.get("generation_settings") or {}),
        generation_skipped=bool(data.get("generation_skipped")),
        skip_reason=data.get("skip_reason", ""),
    )
