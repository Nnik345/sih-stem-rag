"""FastAPI application factory for the local visualizer."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag.config import PROJECT_ROOT, load_config
from rag.curriculum_catalog import curriculum_options, validate_scope
from rag.image_paths import resolve_curriculum_image
from rag.image_serve import browser_png_cache_dir, ensure_browser_png
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
_ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024

CurriculumImageLookup = Callable[[str], Path | None]


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4000)
    grade: int
    subject: str
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
    def grade_range(cls, value: int) -> int:
        if value not in range(1, 13):
            raise ValueError("Grade must be an integer from 1 to 12.")
        return value

    @field_validator("subject")
    @classmethod
    def subject_ok(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Subject is required.")
        return value.strip().lower()

    @model_validator(mode="after")
    def scope_allowed(self) -> "RunCreateRequest":
        validate_scope(self.grade, self.subject)
        return self

    @field_validator("tutor_state")
    @classmethod
    def state_ok(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in _VALID_STATES:
            raise ValueError(f"Unknown tutor state: {value}")
        return value


def lookup_curriculum_image_path(image_id: str) -> Path | None:
    """Resolve a graph Image to a file under images_dir, else None."""
    try:
        from rag.neo4j_store import Neo4jStore

        config = load_config()
        store = Neo4jStore(config.require_neo4j())
        records = store.read(
            "MATCH (i:Image {image_id: $id}) RETURN i.local_path AS local_path",
            {"id": image_id},
        )
    except Exception:
        return None
    if not records:
        return None
    return resolve_curriculum_image(
        records[0].get("local_path"), config.paths.images_dir
    )


async def _save_upload(upload: Any) -> str:
    filename = str(getattr(upload, "filename", "") or "upload")
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=422, detail="Image must be JPEG, PNG, WebP, or GIF.")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Image file is empty.")
    if len(payload) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Image is larger than 8 MB.")
    config = load_config(require_neo4j=False)
    uploads = config.paths.uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(payload)
    return str(dest)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


async def parse_run_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type.lower():
        form = await request.form()
        query = str(form.get("query") or "").strip()
        upload = form.get("image")
        has_file = bool(getattr(upload, "filename", None))
        if not query and not has_file:
            raise HTTPException(
                status_code=422, detail="Enter a question or attach an image."
            )
        try:
            grade = int(str(form.get("grade") or ""))
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="Grade must be an integer from 1 to 12."
            ) from exc
        subject = str(form.get("subject") or "").strip().lower()
        tutor_state = str(form.get("tutor_state") or "") or None
        if tutor_state not in _VALID_STATES and tutor_state is not None:
            raise HTTPException(status_code=422, detail=f"Unknown tutor state: {tutor_state}")
        try:
            validate_scope(grade, subject)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        payload: dict[str, Any] = {
            "query": query,
            "grade": grade,
            "subject": subject,
            "tutor_state": tutor_state,
            "retrieval_only": _truthy(form.get("retrieval_only")),
            "strict": _truthy(form.get("strict")),
            "generation": None,
        }
        if has_file:
            payload["image_path"] = await _save_upload(upload)
        return payload

    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid JSON body.") from exc
    try:
        body = RunCreateRequest.model_validate(raw)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    return body.model_dump()


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    store: RunStore | None = None,
    serve_frontend: bool = True,
    curriculum_image_lookup: CurriculumImageLookup | None = None,
) -> FastAPI:
    run_store = store or RunStore()
    factory = pipeline_factory
    pipeline_holder: dict[str, Any] = {"pipeline": None, "busy": False}
    image_lookup = curriculum_image_lookup or lookup_curriculum_image_path

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

    @app.get("/api/curriculum-options")
    def curriculum() -> dict[str, Any]:
        return curriculum_options()

    @app.get("/api/curriculum-images/{image_id:path}")
    def curriculum_image(image_id: str) -> FileResponse:
        path = image_lookup(image_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Unknown image.")
        served = path
        try:
            config = load_config(require_neo4j=False)
            served = ensure_browser_png(
                path, cache_root=browser_png_cache_dir(config.paths.images_dir)
            )
        except Exception:
            served = path
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(served.suffix.lower(), "image/png")
        return FileResponse(
            served,
            media_type=media,
            content_disposition_type="inline",
        )

    @app.post("/api/runs", status_code=202)
    async def create_run(request: Request) -> dict[str, Any]:
        payload = await parse_run_payload(request)
        run_id = new_run_id()
        record = await run_store.create(run_id, payload)
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
