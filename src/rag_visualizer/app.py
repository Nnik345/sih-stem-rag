"""FastAPI application factory for the local visualizer."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from rag.config import PROJECT_ROOT
from rag.socratic import TutorState
from rag.trace import RunStatus, RunTraceCollector, new_run_id

from .health import check_health, frontend_dist_dir
from .runner import FanoutObserver, PipelineFactory, default_pipeline_factory, execute_run
from .store import RunStore

ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)

_VALID_STATES = {state.value for state in TutorState}
_VALID_SUBJECTS = {"science", "mathematics"}


class RunCreateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    grade: int | None = None
    subject: str | None = None
    unit: str | None = None
    resource_type: str | None = None
    audience: str | None = None
    document_id: str | None = None
    tutor_state: str | None = None
    retrieval_only: bool = False
    strict: bool = False
    generation: dict[str, Any] | None = None

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be empty.")
        return stripped

    @field_validator("grade")
    @classmethod
    def grade_range(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value not in range(1, 13):
            raise ValueError("Grade must be an integer from 1 to 12.")
        return value

    @field_validator("subject")
    @classmethod
    def subject_ok(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        lowered = value.lower()
        if lowered not in _VALID_SUBJECTS:
            raise ValueError("Subject must be science or mathematics.")
        return lowered

    @field_validator("tutor_state")
    @classmethod
    def state_ok(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in _VALID_STATES:
            raise ValueError(f"Unknown tutor state: {value}")
        return value


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    store: RunStore | None = None,
    serve_frontend: bool = True,
) -> FastAPI:
    run_store = store or RunStore()
    factory = pipeline_factory
    pipeline_holder: dict[str, Any] = {"pipeline": None, "busy": False}

    async def worker() -> None:
        run_store.bind_loop(asyncio.get_running_loop())
        while True:
            run_id = await run_store.work.get()
            record = run_store.get(run_id)
            if record is None:
                continue
            record.status = RunStatus.RUNNING.value
            collector = RunTraceCollector(run_id=run_id)
            loop = asyncio.get_running_loop()

            def _forward(event: str, payload: dict[str, Any], trace: dict[str, Any]) -> None:
                run_store.publish_sync(run_id, event, payload, trace)

            observer = FanoutObserver(collector, _forward)

            def _run() -> None:
                try:
                    pipeline = pipeline_holder["pipeline"]
                    if pipeline is None:
                        pipeline = (factory or default_pipeline_factory)()
                        pipeline_holder["pipeline"] = pipeline
                    execute_run(pipeline, record.request, observer)
                except Exception as exc:
                    observer.on_event("run_failed", {"error": str(exc), "stage": "generator"})

            await loop.run_in_executor(None, _run)
            # Ensure a terminal event exists even if the fake pipeline forgot.
            current = run_store.get(run_id)
            if current is not None and not current.done_flag.is_set():
                observer.on_event("run_completed", {"diagnostics": {}})

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(worker())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="STEM RAG Visualizer", lifespan=lifespan)
    app.state.store = run_store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return check_health()

    @app.post("/api/runs", status_code=202)
    async def create_run(body: RunCreateRequest) -> dict[str, Any]:
        run_id = new_run_id()
        record = await run_store.create(run_id, body.model_dump())
        return {"run_id": record.run_id, "status": record.status}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        record = run_store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run ID.")
        return record.snapshot()

    @app.get("/api/runs/{run_id}/trace")
    def get_trace(run_id: str) -> dict[str, Any]:
        record = run_store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run ID.")
        return record.trace

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> StreamingResponse:
        record = run_store.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run ID.")

        async def stream() -> AsyncIterator[str]:
            async for chunk in run_store.subscribe(run_id):
                if await request.is_disconnected():
                    break
                yield chunk

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    dist = frontend_dist_dir()
    if serve_frontend and (dist / "index.html").is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        def _index() -> FileResponse:
            return FileResponse(
                dist / "index.html",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @app.get("/")
        def index() -> FileResponse:
            return _index()

        @app.get("/{path:path}")
        def spa(path: str) -> FileResponse:
            candidate = dist / path
            if candidate.is_file() and PROJECT_ROOT in candidate.resolve().parents:
                return FileResponse(candidate)
            return _index()

    return app


app = create_app()
