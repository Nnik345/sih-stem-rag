"""Visualizer API tests with a fake pipeline. No Neo4j, no models."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from rag.trace import TraceObserver
from rag_visualizer.app import create_app
from rag_visualizer.store import RunStore


class FakeResult:
    def __init__(self):
        self.response_text = "hello from the tutor"
        self.notes: list[str] = []


class FakePipeline:
    def __init__(self, *, fail: bool = False, delay: float = 0.0, tokens: tuple[str, ...] = ("hel", "lo")):
        self.fail = fail
        self.delay = delay
        self.tokens = tokens
        self.calls: list[dict] = []

    def answer(self, query: str, **kwargs) -> FakeResult:
        self.calls.append({"query": query, **kwargs})
        observer: TraceObserver | None = kwargs.get("observer")
        if observer is not None:
            observer.on_event("run_started", {"query": query, "filters": {}, "retrieval_only": kwargs.get("retrieval_only")})
            observer.on_event("filters_applied", {"scope": "grade=1"})
            observer.on_event("rewrite_completed", {"retrieval_query": query, "fallback": True, "elapsed_ms": 1.0, "summary": "fallback"})
            observer.on_event("dense_started", {})
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("synthetic failure")
            observer.on_event(
                "dense_completed",
                {"elapsed_ms": 1.0, "summary": "1 candidate", "dense": {"candidates": []}},
            )
            observer.on_event("lexical_started", {})
            observer.on_event("lexical_completed", {"elapsed_ms": 1.0, "summary": "1", "lexical": {"lucene_query": "weather"}})
            observer.on_event("graph_started", {})
            observer.on_event("graph_completed", {"elapsed_ms": 1.0, "graph": {"nodes": [], "truncated": False}})
            observer.on_event("fusion_completed", {"elapsed_ms": 1.0, "fusion": {"candidates": []}})
            observer.on_event("reranker_started", {})
            observer.on_event("reranker_completed", {"elapsed_ms": 1.0, "reranker": {"candidates": []}})
            observer.on_event("evidence_completed", {"elapsed_ms": 0.0, "evidence": {"sufficient": True, "checks": []}})
            observer.on_event("prompt_built", {"prompt": {"system_prompt": "sys", "user_prompt": "usr"}})
            if kwargs.get("retrieval_only"):
                observer.on_event(
                    "generation_completed",
                    {"skipped": True, "skip_reason": "retrieval-only", "elapsed_ms": 0.0},
                )
            else:
                observer.on_event("generation_started", {})
                for token in self.tokens:
                    observer.on_event("generation_token", {"token": token})
                observer.on_event("generation_completed", {"response_text": "".join(self.tokens), "elapsed_ms": 5.0})
            observer.on_event("run_completed", {"diagnostics": {}})
        return FakeResult()


@pytest.fixture
def pipeline():
    return FakePipeline()


@pytest.fixture
def client(pipeline):
    store = RunStore(max_runs=20, keepalive_seconds=0.05)

    def factory():
        return pipeline

    app = create_app(pipeline_factory=factory, store=store, serve_frontend=False)
    with TestClient(app) as test_client:
        yield test_client


def _wait_done(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish: {body}")


class TestHealth:
    def test_health_shape(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["api"] == "ok"
        for key in (
            "neo4j",
            "corpus_path_present",
            "embedding_model_path_present",
            "reranker_model_path_present",
            "generator_model_path_present",
        "rewriter_model_path_present",
        "image_embedding_model_path_present",
        "frontend_build_present",
        ):
            assert key in body
        assert body["neo4j"] in {"ok", "unavailable"}


class TestRuns:
    def test_create_run_returns_id(self, client):
        response = client.post(
            "/api/runs",
            json={"query": "how does weather change from day to day", "grade": 3, "subject": "science"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["run_id"]
        assert body["status"] in {"queued", "running"}

    def test_validation_rejects_empty_query(self, client):
        response = client.post("/api/runs", json={"query": "   ", "grade": 3, "subject": "science"})
        assert response.status_code == 422

    def test_validation_requires_grade_and_subject(self, client):
        assert client.post("/api/runs", json={"query": "how does weather change"}).status_code == 422
        assert client.post(
            "/api/runs", json={"query": "how does weather change", "grade": 3}
        ).status_code == 422
        assert client.post(
            "/api/runs", json={"query": "how does weather change", "subject": "science"}
        ).status_code == 422

    def test_validation_rejects_bad_grade(self, client):
        response = client.post(
            "/api/runs",
            json={"query": "what are the components of food", "grade": 13, "subject": "science"},
        )
        assert response.status_code == 422

    def test_validation_rejects_bad_subject(self, client):
        response = client.post(
            "/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "history"}
        )
        assert response.status_code == 422

    def test_validation_gates_pcb_by_grade(self, client):
        assert client.post(
            "/api/runs",
            json={"query": "what is electrostatics", "grade": 6, "subject": "physics"},
        ).status_code == 422
        ok = client.post(
            "/api/runs",
            json={"query": "what is electrostatics", "grade": 12, "subject": "physics"},
        )
        assert ok.status_code == 202

    def test_validation_rejects_hidden_user_fields(self, client):
        response = client.post(
            "/api/runs",
            json={
                "query": "how does weather change",
                "grade": 3,
                "subject": "science",
                "unit": "secret",
            },
        )
        assert response.status_code == 422

    def test_curriculum_options(self, client):
        response = client.get("/api/curriculum-options")
        assert response.status_code == 200
        body = response.json()
        assert body["subjects_by_grade"]["1"] == ["mathematics"]
        assert body["subjects_by_grade"]["6"] == ["mathematics", "science"]
        assert body["subjects_by_grade"]["12"] == [
            "mathematics",
            "physics",
            "chemistry",
            "biology",
        ]

    def test_unknown_run(self, client):
        assert client.get("/api/runs/does-not-exist").status_code == 404
        assert client.get("/api/runs/does-not-exist/trace").status_code == 404
        assert client.get("/api/runs/does-not-exist/events").status_code == 404

    def test_unknown_curriculum_image_404(self, client):
        assert client.get("/api/curriculum-images/missing-id").status_code == 404

    def test_curriculum_image_is_inline_png(self, tmp_path):
        png = tmp_path / "fig.png"
        png.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
                "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
            )
        )
        store = RunStore(max_runs=8, keepalive_seconds=0.05)

        def factory():
            return FakePipeline()

        app = create_app(
            pipeline_factory=factory,
            store=store,
            serve_frontend=False,
            curriculum_image_lookup=lambda image_id: png if image_id == "p1:img01" else None,
        )
        with TestClient(app) as local:
            response = local.get("/api/curriculum-images/p1:img01")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_multipart_image_without_query(self, client, tmp_path):
        response = client.post(
            "/api/runs",
            data={"grade": "3", "subject": "science"},
            files={"image": ("cell.png", b"not-a-real-png", "image/png")},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        done = _wait_done(client, run_id)
        assert done["status"] == "completed"

    def test_queued_then_completed(self, client):
        created = client.post("/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "science"})
        run_id = created.json()["run_id"]
        snapshot = client.get(f"/api/runs/{run_id}").json()
        assert snapshot["status"] in {"queued", "running", "completed"}
        done = _wait_done(client, run_id)
        assert done["status"] == "completed"
        trace = client.get(f"/api/runs/{run_id}/trace").json()
        assert trace["query"] == "how does weather change"
        assert "NEO4J_PASSWORD" not in json.dumps(trace)

    def test_history_capped_at_20(self, client):
        ids = []
        for index in range(21):
            response = client.post(
                "/api/runs",
                json={"query": f"q{index}", "grade": 3, "subject": "science"},
            )
            ids.append(response.json()["run_id"])
            _wait_done(client, ids[-1])
        assert client.get(f"/api/runs/{ids[0]}").status_code == 404
        assert client.get(f"/api/runs/{ids[-1]}").status_code == 200


class TestSSE:
    def test_sse_framing_and_event_order(self, client):
        created = client.post("/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "science"})
        run_id = created.json()["run_id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
            text = "".join(stream.iter_text())
        assert "event: run_started" in text
        assert "event: dense_completed" in text
        assert "event: run_completed" in text
        assert "data: {" in text
        names = [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("event:")
        ]
        assert names[0] == "run_started"
        assert names[-1] == "run_completed"
        dense_at = names.index("dense_completed")
        lexical_at = names.index("lexical_completed")
        assert dense_at < lexical_at

    def test_terminal_failure(self):
        pipeline = FakePipeline(fail=True)
        store = RunStore(keepalive_seconds=0.05)
        app = create_app(pipeline_factory=lambda: pipeline, store=store, serve_frontend=False)
        with TestClient(app) as client:
            run_id = client.post("/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "science"}).json()["run_id"]
            done = _wait_done(client, run_id)
            assert done["status"] == "failed"
            with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
                text = "".join(stream.iter_text())
            assert "event: run_failed" in text

    def test_late_subscriber_gets_snapshot(self, client):
        run_id = client.post("/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "science"}).json()["run_id"]
        _wait_done(client, run_id)
        with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
            text = "".join(stream.iter_text())
        assert "event: run_started" in text
        assert "event: run_completed" in text

    def test_keepalive_while_running(self):
        pipeline = FakePipeline(delay=0.2)
        store = RunStore(keepalive_seconds=0.05)
        app = create_app(pipeline_factory=lambda: pipeline, store=store, serve_frontend=False)
        with TestClient(app) as client:
            run_id = client.post("/api/runs", json={"query": "how does weather change", "grade": 3, "subject": "science"}).json()["run_id"]
            with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
                text = "".join(stream.iter_text())
        assert ": keepalive" in text
        assert "event: run_completed" in text
