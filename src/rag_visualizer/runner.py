"""Execute one traced RAG run on a worker thread."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

from rag.generator import GenerationSettings
from rag.pipeline import SocraticRagPipeline
from rag.socratic import TutorState
from rag.trace import RunTraceCollector, _jsonable, redact_secrets, safe_error_message


class PipelineLike(Protocol):
    def answer(self, query: str, **kwargs: Any) -> Any: ...


PipelineFactory = Callable[[], PipelineLike]


class FanoutObserver:
    """Collector that also forwards serialisable events to a callback."""

    def __init__(self, collector: RunTraceCollector, on_event: Callable[[str, dict[str, Any], dict[str, Any]], None]) -> None:
        self.collector = collector
        self._on_event = on_event

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        self.collector.on_event(event, payload)
        safe_payload = _jsonable(payload)
        snapshot = self.collector.snapshot().to_dict()
        self._on_event(event, safe_payload, snapshot)

    def snapshot(self):
        return self.collector.snapshot()


def _tutor_state(value: str | None) -> TutorState | None:
    if not value:
        return None
    return TutorState(value)


def _generation_settings(raw: dict[str, Any] | None) -> GenerationSettings | None:
    if not raw:
        return None
    kwargs: dict[str, Any] = {}
    if "max_new_tokens" in raw:
        kwargs["max_new_tokens"] = int(raw["max_new_tokens"])
    if "temperature" in raw:
        kwargs["temperature"] = float(raw["temperature"])
    if "top_p" in raw:
        kwargs["top_p"] = float(raw["top_p"])
    if "do_sample" in raw:
        kwargs["do_sample"] = bool(raw["do_sample"])
    return GenerationSettings(**kwargs) if kwargs else None


def execute_run(pipeline: PipelineLike, request: dict[str, Any], observer: FanoutObserver) -> None:
    """Blocking RAG execution. Called from a worker thread."""
    image_path = request.get("image_path")
    try:
        pipeline.answer(
            request.get("query") or "",
            grade=request.get("grade"),
            subject=request.get("subject"),
            requested_state=_tutor_state(request.get("tutor_state")),
            generate_on_insufficient_evidence=not bool(request.get("strict")),
            retrieval_only=bool(request.get("retrieval_only")),
            settings=_generation_settings(request.get("generation")),
            observer=observer,
            image_path=image_path,
        )
    except Exception as exc:
        observer.on_event(
            "run_failed",
            {"error": redact_secrets(safe_error_message(exc)), "stage": "generator"},
        )
    finally:
        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError:
                pass


def default_pipeline_factory() -> SocraticRagPipeline:
    from rag.config import load_config
    from rag.neo4j_store import Neo4jStore

    config = load_config()
    store = Neo4jStore(config.require_neo4j())
    store.connect()
    return SocraticRagPipeline(config, store)
