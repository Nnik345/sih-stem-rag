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
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import RagConfig
from .curriculum_catalog import validate_scope
from .dense_retriever import DenseRetriever, STRATEGY_EXACT, STRATEGY_INDEXED
from .embeddings import BGEM3Embedder
from .evidence import EvidenceGate
from .fusion import fuse_standard_channels, select_final_evidence
from .generator import GenerationSettings, QwenGenerator
from .graph_retriever import GraphRetriever
from .image_embeddings import SiglipImageEmbedder
from .image_paths import resolve_curriculum_image, select_vision_image_paths
from .image_retriever import ImageHit, ImageRetriever
from .image_serve import browser_png_cache_dir, ensure_browser_png
from .lexical_retriever import LexicalRetriever
from .logging_utils import Timer, get_logger
from .model_memory import (
    cuda_memory_gib,
    empty_cuda_cache,
    generator_should_yield_gpu,
)
from .neo4j_store import Neo4jStore
from .query_rewrite import (
    QueryRewriter,
    QueryRewriteResult,
    specialize_maths_retrieval_query,
)
from .reranker import BGEReranker
from .schemas import (
    CHANNEL_DENSE,
    CHANNEL_FULLTEXT,
    CHANNEL_GRAPH,
    EvidenceDecision,
    RetrievalDiagnostics,
    RetrievalFilter,
    RetrievalResponse,
    RetrievedChunk,
)
from .socratic import SocraticController, TutorState, TutorTurn
from .trace import (
    DenseTrace,
    EvidenceCheckTrace,
    EvidenceTrace,
    FusionCandidateTrace,
    FusionTrace,
    LexicalTrace,
    PromptTrace,
    RerankerCandidateTrace,
    RerankerTrace,
    TraceObserver,
    annotate_later_status,
    attach_graph_trace,
    candidate_from_chunk,
    emit,
    safe_error_message,
)

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
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder or BGEM3Embedder.from_config(config.models)
        self.reranker = reranker or BGEReranker.from_config(config.models)
        self.rewriter = rewriter or QueryRewriter.from_config(config.models)
        self._image_embedder: SiglipImageEmbedder | None = None

        self.dense = DenseRetriever(
            store,
            self.embedder,
            config.retrieval,
            embedding_version=config.embedding_version,
        )
        self.lexical = LexicalRetriever(store, config.retrieval)
        self.graph = GraphRetriever(store, config.retrieval)

    @property
    def image_embedder(self) -> SiglipImageEmbedder:
        if self._image_embedder is None:
            self._image_embedder = SiglipImageEmbedder.from_config(self.config.models)
        return self._image_embedder

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
        observer: TraceObserver | None = None,
        image_path: str | Path | None = None,
    ) -> RetrievalResponse:
        """Retrieve curriculum evidence for ``query``.

        ``grade`` and ``subject`` are metadata filters, used only when the caller
        supplies them. They are never inferred from the question text: the
        application already knows the student's grade, and guessing it would be
        both unreliable and pedagogically wrong. With no filters, the whole
        corpus is searched.

        ``observer`` is optional. When omitted, behaviour is unchanged and the
        graph retriever does not run diagnostic queries.
        """
        scoped_subject = subject.lower() if isinstance(subject, str) else subject
        scope = RetrievalFilter(
            grade=grade,
            subject=scoped_subject,
            unit_id=unit,
            unit_title_contains=unit_title_contains,
            resource_type=resource_type,
            audience=audience,
            document_id=document_id,
            allow_prior_grades=grade is not None and scoped_subject is not None,
        )
        diagnostics = RetrievalDiagnostics(query=query, scope=scope)
        total = Timer()
        fused_ids: set[str] = set()
        evidence_ids: set[str] = set()

        LOGGER.info("Retrieving for %r within %s", query, scope.describe())
        emit(
            observer,
            "filters_applied",
            scope=scope.describe(),
            filters={
                "grade": scope.grade,
                "subject": scope.subject,
                "unit_id": scope.unit_id,
                "unit_title_contains": scope.unit_title_contains,
                "resource_type": scope.resource_type,
                "audience": scope.audience,
                "document_id": scope.document_id,
            },
        )

        rewrite_timer = Timer()
        emit(observer, "rewrite_started")
        rewrite = self._rewrite_query(
            query,
            grade=scope.grade,
            subject=scope.subject,
            image_path=image_path,
        )
        search_query = rewrite.retrieval_query
        if (scope.subject or "").strip().lower() == "mathematics":
            specialised = specialize_maths_retrieval_query(
                search_query,
                original=query,
                transcribed=rewrite.transcribed_question,
            )
            if specialised != search_query:
                diagnostics.notes.append(
                    f"maths retrieval query specialised: {specialised!r}"
                )
            search_query = specialised
        diagnostics.retrieval_query = search_query
        diagnostics.rewrite_intent = rewrite.intent
        diagnostics.rewrite_fallback = rewrite.fallback
        diagnostics.rewrite_input_kind = rewrite.input_kind
        diagnostics.transcribed_question = rewrite.transcribed_question
        diagnostics.timings_ms["rewrite"] = rewrite_timer.stop() * 1000
        if rewrite.fallback:
            diagnostics.notes.append(
                f"query rewrite fallback: {rewrite.reason or 'using original query'}"
            )
        else:
            diagnostics.notes.append(
                f"rewritten query: {search_query!r} (intent={rewrite.intent})"
            )
        emit(
            observer,
            "rewrite_completed",
            original_query=query,
            retrieval_query=search_query,
            intent=rewrite.intent,
            input_kind=rewrite.input_kind,
            transcribed_question=rewrite.transcribed_question,
            fallback=rewrite.fallback,
            reason=rewrite.reason,
            elapsed_ms=diagnostics.timings_ms["rewrite"],
            summary=(
                "fallback to original query"
                if rewrite.fallback
                else f"intent={rewrite.intent}"
            ),
        )

        image_hits: list[ImageHit] = []
        image_chunks: list[RetrievedChunk] = []
        student_image = Path(image_path) if image_path else None
        emit(observer, "image_started")
        if student_image is not None and student_image.is_file():
            image_retriever = ImageRetriever(
                self.store,
                self.image_embedder,
                self.config.retrieval,
                embedding_version=self.config.image_embedding_version,
            )
            try:
                image_hits, image_chunks = image_retriever.retrieve(
                    student_image, scope=scope
                )
            finally:
                self.image_embedder.unload()
            diagnostics.image = image_chunks
            diagnostics.image_hits = [hit.to_dict() for hit in image_hits]
            diagnostics.timings_ms["image"] = image_retriever.last_timing_ms
            if image_retriever.last_note:
                diagnostics.notes.append(image_retriever.last_note)
            diagnostics.notes.append(
                f"image kNN: {len(image_hits)} figures, {len(image_chunks)} page chunks"
            )
            emit(
                observer,
                "image_completed",
                image_hits=diagnostics.image_hits,
                elapsed_ms=image_retriever.last_timing_ms,
                summary=f"{len(image_hits)} figure hits",
            )
        else:
            emit(
                observer,
                "image_completed",
                skipped=True,
                image_hits=[],
                elapsed_ms=0.0,
                summary="no student image",
            )

        emit(observer, "dense_started")
        dense_results = self.dense.retrieve(search_query, scope=scope)
        diagnostics.dense = dense_results
        diagnostics.timings_ms["dense"] = self.dense.last_timing_ms
        diagnostics.notes.append(f"dense strategy: {self.dense.last_strategy}")
        dense_trace = _build_dense_trace(self, dense_results)
        emit(
            observer,
            "dense_completed",
            dense=dense_trace,
            elapsed_ms=self.dense.last_timing_ms,
            summary=f"{len(dense_results)} candidates via {self.dense.last_strategy}",
        )

        emit(observer, "lexical_started")
        lexical_results = self.lexical.retrieve(search_query, scope=scope)
        diagnostics.fulltext = lexical_results
        diagnostics.timings_ms["fulltext"] = self.lexical.last_timing_ms
        diagnostics.notes.append(
            f"lucene query: {self.lexical.last_lucene_query or '(empty)'}"
        )
        lexical_trace = _build_lexical_trace(
            search_query, self.lexical.last_lucene_query, lexical_results
        )
        emit(
            observer,
            "lexical_completed",
            lexical=lexical_trace,
            elapsed_ms=self.lexical.last_timing_ms,
            summary=f"{len(lexical_results)} candidates",
        )

        emit(observer, "graph_started")
        graph_results = self.graph.retrieve(
            [dense_results, lexical_results],
            scope=scope,
            observer=observer,
        )
        diagnostics.graph = graph_results
        diagnostics.graph_seeds = list(self.graph.last_seeds)
        diagnostics.timings_ms["graph"] = self.graph.last_timing_ms
        diagnostics.graph_trace = self.graph.last_trace
        graph_summary = f"{len(graph_results)} candidates from {len(self.graph.last_seeds)} seeds"
        if observer is not None:
            # graph_completed is emitted by GraphRetriever when tracing.
            pass
        else:
            emit(
                observer,
                "graph_completed",
                graph={"selected_chunk_ids": [c.chunk_id for c in graph_results]},
                elapsed_ms=self.graph.last_timing_ms,
                summary=graph_summary,
            )

        if not (dense_results or lexical_results or graph_results or image_chunks):
            diagnostics.notes.append("all channels returned zero candidates")
            diagnostics.timings_ms["total"] = total.stop() * 1000
            LOGGER.warning(
                "No candidates at all for %r within %s", query, scope.describe()
            )
            emit(
                observer,
                "fusion_completed",
                fusion=FusionTrace(
                    rrf_k=self.config.retrieval.rrf_k,
                    weight_dense=self.config.retrieval.weight_dense,
                    weight_fulltext=self.config.retrieval.weight_fulltext,
                    weight_graph=self.config.retrieval.weight_graph,
                    weight_image=self.config.retrieval.weight_image,
                ),
                elapsed_ms=0.0,
                summary="no candidates",
            )
            emit(
                observer,
                "reranker_completed",
                skipped=True,
                summary="no candidates",
                elapsed_ms=0.0,
            )
            return RetrievalResponse(query, scope, [], diagnostics)

        fusion_timer = Timer()
        fused = fuse_standard_channels(
            copy.deepcopy(dense_results),
            copy.deepcopy(lexical_results),
            copy.deepcopy(graph_results),
            self.config.retrieval,
            image=copy.deepcopy(image_chunks),
        )
        diagnostics.fused = fused
        diagnostics.timings_ms["fusion"] = fusion_timer.stop() * 1000
        fused_ids = {c.chunk_id for c in fused}
        fusion_trace = _build_fusion_trace(fused, self.config.retrieval)
        emit(
            observer,
            "fusion_completed",
            fusion=fusion_trace,
            elapsed_ms=diagnostics.timings_ms["fusion"],
            summary=f"{len(fused)} fused candidates",
        )

        limit = final_top_k or self.config.retrieval.final_top_k
        emit(observer, "reranker_started")
        selected: list[RetrievedChunk]
        if rerank and fused:
            rerank_timer = Timer()
            scored = self.reranker.rerank(search_query, copy.deepcopy(fused), top_k=None)
            diagnostics.reranked = scored
            selected = select_final_evidence(
                scored,
                limit=limit,
                min_score=self.config.evidence.min_rerank_score,
                requested_grade=scope.grade,
            )
            reranker_trace = (
                _build_reranker_trace(scored, {c.chunk_id for c in selected})
                if observer is not None
                else None
            )
            diagnostics.timings_ms["rerank"] = rerank_timer.stop() * 1000
            emit(
                observer,
                "reranker_completed",
                reranker=reranker_trace,
                elapsed_ms=diagnostics.timings_ms["rerank"],
                summary=f"{len(selected)} selected after rerank",
            )
        else:
            diagnostics.reranked = list(fused)
            selected = select_final_evidence(
                fused, limit=limit, min_score=None, requested_grade=scope.grade
            )
            if not rerank:
                diagnostics.notes.append("reranking disabled by caller")
            emit(
                observer,
                "reranker_completed",
                skipped=True,
                summary="reranking disabled" if not rerank else "empty fusion",
                elapsed_ms=0.0,
            )

        want_images = (
            self.config.multimodal_enabled if include_images is None else include_images
        )
        if want_images:
            self._attach_images(selected)
        diagnostics.attached_figures = []
        diagnostics.notes.append(
            "textbook figures are not shown in the answer; student photos are input-only"
        )
        if scope.grade is not None and selected:
            parts = [
                f"{chunk.chunk_id}:class{chunk.grade}:d={chunk.grade_distance}"
                for chunk in selected
            ]
            diagnostics.notes.append(
                "grade proximity (requested class "
                f"{scope.grade}; closer grades preferred among comparable "
                f"relevance): {', '.join(parts)}"
            )

        if observer is not None:
            annotate_later_status(dense_trace.candidates, fused_ids, evidence_ids)
            annotate_later_status(lexical_trace.candidates, fused_ids, evidence_ids)

        diagnostics.timings_ms["total"] = total.stop() * 1000
        LOGGER.info(
            "Retrieval complete in %.1f ms: %s",
            diagnostics.timings_ms["total"],
            diagnostics.channel_counts(),
        )
        return RetrievalResponse(query, scope, list(selected), diagnostics)

    def _attach_images(self, chunks: Sequence[RetrievedChunk]) -> None:
        """Attach image metadata from the graph (no visual embeddings involved)."""
        images = self.graph.images_for_chunks([c.chunk_id for c in chunks])
        for chunk in chunks:
            chunk.images = images.get(chunk.chunk_id, [])

    def _rewrite_query(
        self,
        query: str,
        *,
        grade: int | None,
        subject: str | None,
        image_path: str | Path | None = None,
    ) -> QueryRewriteResult:
        """Load the 2B rewriter, rewrite, then unload before SigLIP/BGE-M3."""
        try:
            result = self.rewriter.rewrite(
                query, grade=grade, subject=subject, image_path=image_path
            )
        finally:
            self.rewriter.unload()
        return result

    def release_models(self) -> None:
        self.rewriter.unload()
        if self._image_embedder is not None:
            self._image_embedder.unload()
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
    trace: dict[str, Any] | None = None
    image_paths: list[str] = field(default_factory=list)
    attached_figures: list[dict[str, Any]] = field(default_factory=list)
    student_image_path: str | None = None

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
            "trace": self.trace,
            "attached_figures": list(self.attached_figures),
            "image_paths": list(self.image_paths),
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

    def _release_for_generation(self) -> None:
        """Drop BGE/reranker/SigLIP so the tutor can claim the GPU."""
        self.release_retrieval_models()
        empty_cuda_cache()

    def _maybe_unload_generator_for_retrieval(self) -> None:
        """On small GPUs, the 8B tutor must leave before the rewriter/BGE load."""
        generator = self._generator
        if generator is None or not generator.is_loaded:
            return
        stats = cuda_memory_gib()
        if stats is None:
            return
        free, total = stats
        if generator_should_yield_gpu(free, total):
            LOGGER.info(
                "Unloading the tutor (%.1f GiB free of %.1f GiB) so retrieval "
                "can use the GPU",
                free,
                total,
            )
            generator.unload()

    # -- stages ------------------------------------------------------------ #

    def prepare(
        self,
        query: str,
        *,
        grade: int | None = None,
        subject: str | None = None,
        requested_state: TutorState | None = None,
        observer: TraceObserver | None = None,
        retrieval_only: bool = False,
        strict: bool = False,
        image_path: str | Path | None = None,
        **retrieval_kwargs: Any,
    ) -> RagResult:
        """Run everything except generation.

        Returns a :class:`RagResult` whose ``answered`` flag says whether the
        generator should be called at all.
        """
        grade, subject = validate_scope(grade, subject)
        self._maybe_unload_generator_for_retrieval()
        emit(
            observer,
            "run_started",
            query=query,
            filters={
                "grade": grade,
                "subject": subject,
            },
            requested_state=requested_state.value if requested_state else None,
            retrieval_only=retrieval_only,
            strict=strict,
        )
        retrieval = self.retriever.retrieve(
            query,
            grade=grade,
            subject=subject,
            observer=observer,
            image_path=image_path,
            **retrieval_kwargs,
        )
        transcribed = retrieval.diagnostics.transcribed_question
        display_query = (query or "").strip() or transcribed or query
        tutor_question = transcribed or display_query
        # Overlap follows the student's wording (or the photo transcript), not
        # the rewritten search string. Searching "power rule / constant rule"
        # finds the chapter; requiring those English labels in NCERT then
        # false-fails the gate.
        gate_query = tutor_question or retrieval.diagnostics.retrieval_query
        decision = self.gate.evaluate(
            gate_query, retrieval.results, scope=retrieval.scope
        )
        figures: list[dict[str, Any]] = []
        evidence_trace = _build_evidence_trace(decision, self.config.evidence)
        emit(
            observer,
            "evidence_completed",
            evidence=evidence_trace,
            attached_figures=figures,
            elapsed_ms=0.0,
            summary="sufficient" if decision.sufficient else "insufficient",
        )
        turn = self.controller.build_turn(
            tutor_question,
            decision,
            scope=retrieval.scope,
            fallback_evidence=retrieval.results[:2],
            requested_state=requested_state,
            attached_figures=figures,
        )
        prompt_trace = _build_prompt_trace(turn, self.config.models)
        emit(
            observer,
            "prompt_built",
            prompt=prompt_trace,
            elapsed_ms=0.0,
            summary=turn.state.value,
        )
        result = RagResult(
            query=display_query,
            scope=retrieval.scope,
            retrieval=retrieval,
            decision=decision,
            turn=turn,
            answered=decision.sufficient,
            attached_figures=figures,
            student_image_path=str(image_path) if image_path else None,
        )
        extra_allowed = [image_path] if image_path else []
        result.image_paths = select_vision_image_paths(
            [str(image_path)] if image_path else [],
            images_dir=self.config.paths.images_dir,
            extra_allowed=extra_allowed,
        )
        if observer is not None:
            fused_ids = {c.chunk_id for c in retrieval.diagnostics.fused}
            evidence_ids = {c.chunk_id for c in decision.kept_chunks}
            _annotate_graph_later_status(retrieval, fused_ids, evidence_ids)
            attach_graph_trace(observer, retrieval.diagnostics.graph_trace)
        return result

    def stream_answer(
        self,
        result: RagResult,
        *,
        settings: GenerationSettings | None = None,
        observer: TraceObserver | None = None,
    ) -> Iterator[str]:
        """Stream the tutor's reply for an already-prepared result.

        Called for both sufficient and insufficient evidence: in the insufficient
        case the controller has already put the model into
        ``INSUFFICIENT_EVIDENCE`` state, where it must decline rather than answer
        from general knowledge.
        """
        self._release_for_generation()
        timer = Timer()
        emit(observer, "generation_started")
        try:
            from .structured_tutor import generate_structured_reply

            text = generate_structured_reply(
                self.generator,
                result.turn,
                image_paths=result.image_paths,
            )
            result.response_text = text
            yield text
        finally:
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
        generate_on_insufficient_evidence: bool = True,
        on_token: Any = None,
        settings: GenerationSettings | None = None,
        observer: TraceObserver | None = None,
        retrieval_only: bool = False,
        requested_state: TutorState | None = None,
        image_path: str | Path | None = None,
        **retrieval_kwargs: Any,
    ) -> RagResult:
        """Full pipeline. Generation is skipped when evidence is insufficient
        and ``generate_on_insufficient_evidence`` is False."""
        try:
            result = self.prepare(
                query,
                grade=grade,
                subject=subject,
                requested_state=requested_state,
                observer=observer,
                retrieval_only=retrieval_only,
                strict=not generate_on_insufficient_evidence,
                image_path=image_path,
                **retrieval_kwargs,
            )

            if retrieval_only:
                skip = "Retrieval-only mode: generation was not requested."
                result.notes.append(skip)
                emit(
                    observer,
                    "generation_completed",
                    skipped=True,
                    skip_reason=skip,
                    response_text="",
                    elapsed_ms=0.0,
                )
                _finish_run(observer, result)
                return result

            if not result.answered and not generate_on_insufficient_evidence:
                skip = (
                    "Generation skipped: evidence insufficient and generation on "
                    "insufficient evidence disabled."
                )
                result.notes.append(skip)
                emit(
                    observer,
                    "generation_completed",
                    skipped=True,
                    skip_reason=skip,
                    response_text="",
                    elapsed_ms=0.0,
                )
                _finish_run(observer, result)
                return result

            confirm = result.turn.state is TutorState.CONFIRM_ANSWER
            for piece in self.stream_answer(
                result, settings=settings, observer=observer
            ):
                if on_token is not None and not confirm:
                    on_token(piece)
            emit(
                observer,
                "generation_completed",
                response_text=result.response_text,
                elapsed_ms=result.generation_ms,
            )
            _finish_run(observer, result)
            return result
        except Exception as exc:
            emit(
                observer,
                "run_failed",
                error=safe_error_message(exc),
                stage="generator",
            )
            raise


def _select_attached_figures(
    *,
    kept: Sequence[RetrievedChunk],
    image_hits: Sequence[ImageHit],
    images_dir: Path,
    min_score: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Textbook figures only when a visual is required: image kNN, then kept-page."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _try_add(
        image_id: str,
        local_path: str,
        *,
        page_number: Any = None,
        score: float | None = None,
        source: str,
        grade: Any = None,
        subject: Any = None,
        unit_title: Any = None,
    ) -> bool:
        if not image_id or image_id in seen:
            return False
        resolved = resolve_curriculum_image(local_path, images_dir)
        if resolved is None:
            return False
        safe = ensure_browser_png(
            resolved, cache_root=browser_png_cache_dir(images_dir)
        )
        seen.add(image_id)
        selected.append(
            {
                "image_id": image_id,
                "local_path": str(safe),
                "page_number": page_number,
                "score": score,
                "source": source,
                "grade": grade,
                "subject": subject,
                "unit_title": unit_title,
            }
        )
        return True

    ordered_hits = sorted(image_hits, key=lambda hit: hit.score, reverse=True)
    for hit in ordered_hits:
        if hit.score < min_score:
            continue
        _try_add(
            hit.image_id,
            hit.local_path,
            page_number=hit.page_number,
            score=hit.score,
            source="image_knn",
            grade=hit.grade,
            subject=hit.subject,
        )
        if len(selected) >= limit:
            return selected

    for chunk in kept:
        for image in chunk.images:
            _try_add(
                str(image.get("image_id") or ""),
                str(image.get("local_path") or ""),
                page_number=image.get("page_number"),
                source="kept_page",
                grade=chunk.grade,
                subject=chunk.subject,
                unit_title=chunk.unit_title,
            )
            if len(selected) >= limit:
                return selected
    return selected


def _build_dense_trace(
    retriever: HybridRetriever, results: Sequence[RetrievedChunk]
) -> DenseTrace:
    strategy = retriever.dense.last_strategy or ""
    candidates = [
        candidate_from_chunk(
            chunk,
            rank=chunk.dense_rank,
            score=chunk.dense_score,
        )
        for chunk in results
    ]
    return DenseTrace(
        model_name=retriever.embedder.model_path.name,
        embedding_dim=retriever.dense.last_embedding_dim,
        query_vector_norm=retriever.dense.last_query_vector_norm,
        vector_preview=list(retriever.dense.last_vector_preview),
        strategy=strategy,
        used_approximate_index=strategy == STRATEGY_INDEXED,
        used_exact_fallback=strategy == STRATEGY_EXACT,
        candidates=candidates,
    )


def _build_lexical_trace(
    query: str, lucene_query: str, results: Sequence[RetrievedChunk]
) -> LexicalTrace:
    return LexicalTrace(
        original_query=query,
        lucene_query=lucene_query,
        candidates=[
            candidate_from_chunk(
                chunk,
                rank=chunk.fulltext_rank,
                score=chunk.fulltext_score,
            )
            for chunk in results
        ],
    )


def _build_fusion_trace(fused: Sequence[RetrievedChunk], config: Any) -> FusionTrace:
    candidates = []
    for chunk in fused:
        contrib = chunk.rrf_contributions or {}
        candidates.append(
            FusionCandidateTrace(
                chunk_id=chunk.chunk_id,
                dense_rank=chunk.dense_rank,
                dense_contribution=float(contrib.get(CHANNEL_DENSE, 0.0)),
                lexical_rank=chunk.fulltext_rank,
                lexical_contribution=float(contrib.get(CHANNEL_FULLTEXT, 0.0)),
                graph_rank=chunk.graph_rank,
                graph_contribution=float(contrib.get(CHANNEL_GRAPH, 0.0)),
                channels=list(chunk.retrieval_sources),
                rrf_score=float(chunk.rrf_score or 0.0),
                fused_rank=int(chunk.rrf_rank or 0),
                text=chunk.text,
                provenance=chunk.provenance(),
            )
        )
    return FusionTrace(
        rrf_k=config.rrf_k,
        weight_dense=config.weight_dense,
        weight_fulltext=config.weight_fulltext,
        weight_graph=config.weight_graph,
        weight_image=getattr(config, "weight_image", 1.0),
        candidates=candidates,
    )


def _build_reranker_trace(
    scored: Sequence[RetrievedChunk], selected_ids: set[str]
) -> RerankerTrace:
    candidates = []
    for chunk in scored:
        fused_rank = chunk.rrf_rank
        reranked_rank = chunk.rerank_rank
        movement = 0
        if fused_rank is not None and reranked_rank is not None:
            movement = fused_rank - reranked_rank
        candidates.append(
            RerankerCandidateTrace(
                chunk_id=chunk.chunk_id,
                fused_rank=fused_rank,
                reranked_rank=reranked_rank,
                rank_movement=movement,
                rerank_score=chunk.rerank_score,
                text=chunk.text,
                provenance=chunk.provenance(),
                survived_final_top_k=chunk.chunk_id in selected_ids,
            )
        )
    return RerankerTrace(candidates=candidates)


def _build_evidence_trace(decision: EvidenceDecision, config: Any) -> EvidenceTrace:
    thresholds = {
        "candidates_exist": 1,
        "reranker_confidence": config.min_rerank_score,
        "mutually_relevant_chunks": config.min_strong_chunks,
        "scope_match": True if config.require_scope_match else None,
        "query_term_overlap": config.min_query_term_overlap,
        "minimum_chunks": config.min_chunks,
        "min_prior_grade_rerank_score": config.min_prior_grade_rerank_score,
        "reranked": None,
    }
    checks = [
        EvidenceCheckTrace(
            name=check.name,
            passed=check.passed,
            value=check.value,
            threshold=thresholds.get(check.name),
            detail=check.detail,
        )
        for check in decision.checks
    ]
    return EvidenceTrace(
        sufficient=decision.sufficient,
        confidence=decision.confidence,
        checks=checks,
        kept_chunk_ids=[c.chunk_id for c in decision.kept_chunks],
        reasons=list(decision.reasons),
    )


def _build_prompt_trace(turn: TutorTurn, models: Any) -> PromptTrace:
    blocks = []
    for index, chunk in enumerate(turn.evidence, start=1):
        blocks.append(
            {
                "index": index,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "provenance": chunk.provenance(),
            }
        )
    settings = {
        "max_new_tokens": models.generator_max_new_tokens,
        "temperature": models.generator_temperature,
        "max_evidence_chars": models.generator_max_evidence_chars,
    }
    return PromptTrace(
        tutor_state=turn.state.value,
        system_prompt=turn.system_prompt,
        user_prompt=turn.user_prompt,
        evidence_blocks=blocks,
        generation_settings=settings,
    )


def _annotate_graph_later_status(
    retrieval: RetrievalResponse,
    fused_ids: set[str],
    evidence_ids: set[str],
) -> None:
    graph = retrieval.diagnostics.graph_trace
    if not graph:
        return
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        node["entered_fusion"] = node_id in fused_ids
        node["final_evidence"] = node_id in evidence_ids
        if node_id in evidence_ids and node.get("node_kind") in (None, "seed", "candidate"):
            node["status"] = "evidence"
            node.setdefault("metadata", {})["visual"] = "evidence"


def _finish_run(observer: TraceObserver | None, result: RagResult) -> None:
    diagnostics = result.retrieval.diagnostics.to_dict()
    emit(observer, "run_completed", diagnostics=diagnostics)
    if observer is not None and hasattr(observer, "snapshot"):
        result.trace = observer.snapshot().to_dict()

