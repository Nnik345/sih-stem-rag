"""End-to-end orchestration: hybrid retrieval, evidence gating, Socratic answer.

Two entry points, deliberately separate so retrieval can be studied without ever
loading the generator:

* :class:`HybridRetriever` -- metadata filter, dense + full-text + graph channels,
  RRF fusion, reranking. Returns a :class:`~rag.schemas.RetrievalResponse` whose
  ``diagnostics`` field keeps every intermediate candidate list, rank, score and
  stage timing, because later work compares retrieval architectures.
* :class:`SocraticRagPipeline` -- the above, plus the evidence gate and the
  generator. It will not call the generator when evidence is insufficient.

Nothing is hidden behind one opaque function: each stage is a public method that
returns its own intermediate results.

The embedder and reranker are loaded on demand and can be released via
:meth:`SocraticRagPipeline.release_retrieval_models` before generation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .config import RagConfig
from .dense_retriever import DenseRetriever
from .embeddings import BGEM3Embedder
from .evidence import EvidenceGate
from .fusion import fuse_standard_channels
from .generator import GenerationSettings, QwenGenerator
from .graph_retriever import GraphRetriever
from .lexical_retriever import LexicalRetriever
from .logging_utils import Timer, get_logger
from .neo4j_store import Neo4jStore
from .reranker import BGEReranker
from .schemas import (
    EvidenceDecision,
    RetrievalDiagnostics,
    RetrievalFilter,
    RetrievalResponse,
    RetrievedChunk,
)
from .socratic import SocraticController, TutorState, TutorTurn

LOGGER = get_logger(__name__)


class HybridRetriever:
    """Metadata filter + dense + full-text + graph + RRF + reranking."""

    def __init__(
        self,
        config: RagConfig,
        store: Neo4jStore,
        *,
        embedder: BGEM3Embedder | None = None,
        reranker: BGEReranker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder or BGEM3Embedder.from_config(config.models)
        self.reranker = reranker or BGEReranker.from_config(config.models)

        self.dense = DenseRetriever(
            store,
            self.embedder,
            config.retrieval,
            embedding_version=config.embedding_version,
        )
        self.lexical = LexicalRetriever(store, config.retrieval)
        self.graph = GraphRetriever(store, config.retrieval)

    # -- public API -------------------------------------------------------- #

    def retrieve(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        unit: str | None = None,
        unit_title_contains: str | None = None,
        resource_type: str | None = None,
        audience: str | None = None,
        document_id: str | None = None,
        rerank: bool = True,
        final_top_k: int | None = None,
        include_images: bool | None = None,
    ) -> RetrievalResponse:
        """Retrieve curriculum evidence for ``query``.

        ``grade`` and ``subject`` are metadata filters, used only when the caller
        supplies them. They are never inferred from the question text: the
        application already knows the student's grade, and guessing it would be
        both unreliable and pedagogically wrong. With no filters, the whole
        corpus is searched.
        """
        scope = RetrievalFilter(
            grade=grade,
            subject=subject.lower() if isinstance(subject, str) else subject,
            unit_id=unit,
            unit_title_contains=unit_title_contains,
            resource_type=resource_type,
            audience=audience,
            document_id=document_id,
        )
        diagnostics = RetrievalDiagnostics(query=query, scope=scope)
        total = Timer()

        LOGGER.info("Retrieving for %r within %s", query, scope.describe())

        # 1-2. the two primary channels
        dense_results = self.dense.retrieve(query, scope=scope)
        diagnostics.dense = dense_results
        diagnostics.timings_ms["dense"] = self.dense.last_timing_ms
        diagnostics.notes.append(f"dense strategy: {self.dense.last_strategy}")

        lexical_results = self.lexical.retrieve(query, scope=scope)
        diagnostics.fulltext = lexical_results
        diagnostics.timings_ms["fulltext"] = self.lexical.last_timing_ms
        diagnostics.notes.append(
            f"lucene query: {self.lexical.last_lucene_query or '(empty)'}"
        )

        # 3. graph expansion, seeded by the strongest primary hits
        graph_results = self.graph.retrieve(
            [dense_results, lexical_results], scope=scope
        )
        diagnostics.graph = graph_results
        diagnostics.graph_seeds = list(self.graph.last_seeds)
        diagnostics.timings_ms["graph"] = self.graph.last_timing_ms

        if not (dense_results or lexical_results or graph_results):
            diagnostics.notes.append("all channels returned zero candidates")
            diagnostics.timings_ms["total"] = total.stop() * 1000
            LOGGER.warning(
                "No candidates at all for %r within %s", query, scope.describe()
            )
            return RetrievalResponse(query, scope, [], diagnostics)

        # 4. rank fusion. Deep-copied inputs keep the per-channel lists intact
        #    for inspection, since fusion mutates candidates in place.
        fusion_timer = Timer()
        fused = fuse_standard_channels(
            copy.deepcopy(dense_results),
            copy.deepcopy(lexical_results),
            copy.deepcopy(graph_results),
            self.config.retrieval,
        )
        diagnostics.fused = fused
        diagnostics.timings_ms["fusion"] = fusion_timer.stop() * 1000

        # 5. reranking
        limit = final_top_k or self.config.retrieval.final_top_k
        if rerank and fused:
            rerank_timer = Timer()
            reranked = self.reranker.rerank(
                query, copy.deepcopy(fused), top_k=limit
            )
            diagnostics.timings_ms["rerank"] = rerank_timer.stop() * 1000
        else:
            reranked = fused[:limit]
            if not rerank:
                diagnostics.notes.append("reranking disabled by caller")
        diagnostics.reranked = reranked

        want_images = (
            self.config.multimodal_enabled if include_images is None else include_images
        )
        if want_images:
            self._attach_images(reranked)

        diagnostics.timings_ms["total"] = total.stop() * 1000
        LOGGER.info(
            "Retrieval complete in %.1f ms: %s",
            diagnostics.timings_ms["total"],
            diagnostics.channel_counts(),
        )
        return RetrievalResponse(query, scope, list(reranked), diagnostics)

    def _attach_images(self, chunks: Sequence[RetrievedChunk]) -> None:
        """Attach image metadata from the graph (no visual embeddings involved)."""
        images = self.graph.images_for_chunks([c.chunk_id for c in chunks])
        for chunk in chunks:
            chunk.images = images.get(chunk.chunk_id, [])

    def release_models(self) -> None:
        self.embedder.unload()
        self.reranker.unload()


@dataclass
class RagResult:
    """Full record of one tutoring query, for CLI display and later analysis."""

    query: str
    scope: RetrievalFilter
    retrieval: RetrievalResponse
    decision: EvidenceDecision
    turn: TutorTurn
    answered: bool
    response_text: str = ""
    generation_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope.describe(),
            "answered": self.answered,
            "tutor_state": self.turn.state.value,
            "evidence": self.decision.to_dict(),
            "provenance": self.turn.provenance,
            "response_text": self.response_text,
            "generation_ms": round(self.generation_ms, 1),
            "diagnostics": self.retrieval.diagnostics.to_dict(),
            "notes": list(self.notes),
        }


class SocraticRagPipeline:
    """Retrieval -> evidence gate -> Socratic generation."""

    def __init__(
        self,
        config: RagConfig,
        store: Neo4jStore,
        *,
        retriever: HybridRetriever | None = None,
        generator: QwenGenerator | None = None,
        controller: SocraticController | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.retriever = retriever or HybridRetriever(config, store)
        self.gate = EvidenceGate(config.evidence)
        self.controller = controller or SocraticController.from_config(config.models)
        self._generator = generator
        self._owns_generator = generator is None

    @property
    def generator(self) -> QwenGenerator:
        if self._generator is None:
            self._generator = QwenGenerator.from_config(self.config.models)
        return self._generator

    def release_retrieval_models(self) -> None:
        """Free the embedder and reranker before the generator is loaded."""
        self.retriever.release_models()

    # -- stages ------------------------------------------------------------ #

    def prepare(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        unit: str | None = None,
        requested_state: TutorState | None = None,
        **retrieval_kwargs: Any,
    ) -> RagResult:
        """Run everything except generation.

        Returns a :class:`RagResult` whose ``answered`` flag says whether the
        generator should be called at all.
        """
        retrieval = self.retriever.retrieve(
            query, grade=grade, subject=subject, unit=unit, **retrieval_kwargs
        )
        decision = self.gate.evaluate(
            query, retrieval.results, scope=retrieval.scope
        )
        turn = self.controller.build_turn(
            query,
            decision,
            scope=retrieval.scope,
            fallback_evidence=retrieval.results[:2],
            requested_state=requested_state,
        )
        return RagResult(
            query=query,
            scope=retrieval.scope,
            retrieval=retrieval,
            decision=decision,
            turn=turn,
            answered=decision.sufficient,
        )

    def stream_answer(
        self,
        result: RagResult,
        *,
        settings: GenerationSettings | None = None,
    ) -> Iterator[str]:
        """Stream the tutor's reply for an already-prepared result.

        Called for both sufficient and insufficient evidence: in the insufficient
        case the controller has already put the model into
        ``INSUFFICIENT_EVIDENCE`` state, where it must decline rather than answer
        from general knowledge.
        """
        timer = Timer()
        pieces: list[str] = []
        try:
            for piece in self.generator.stream(result.turn.messages, settings=settings):
                pieces.append(piece)
                yield piece
        finally:
            # Recorded even when generation fails partway, so whatever the
            # student already saw is not lost from the result record.
            result.response_text = "".join(pieces)
            result.generation_ms = timer.stop() * 1000
            LOGGER.info(
                "Generated %d characters in %.1f s",
                len(result.response_text),
                result.generation_ms / 1000,
            )

    def answer(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        unit: str | None = None,
        generate_on_insufficient_evidence: bool = True,
        on_token: Any = None,
        settings: GenerationSettings | None = None,
        **retrieval_kwargs: Any,
    ) -> RagResult:
        """Full pipeline. Generation is skipped when evidence is insufficient
        and ``generate_on_insufficient_evidence`` is False."""
        result = self.prepare(
            query, grade=grade, subject=subject, unit=unit, **retrieval_kwargs
        )

        if not result.answered and not generate_on_insufficient_evidence:
            result.notes.append(
                "Generation skipped: evidence insufficient and generation on "
                "insufficient evidence disabled."
            )
            return result

        for piece in self.stream_answer(result, settings=settings):
            if on_token is not None:
                on_token(piece)
        return result
