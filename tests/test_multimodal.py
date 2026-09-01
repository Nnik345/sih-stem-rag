"""Multimodal helpers: rewrite JSON, path sandbox, lineage, generator images."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import PROJECT_ROOT, load_config
from rag.generator import QwenGenerator
from rag.image_paths import resolve_curriculum_image, select_vision_image_paths
from rag.image_retriever import filter_image_hits
from rag.query_rewrite import parse_rewrite_output
from rag.socratic import SocraticController, TutorState
from rag.schemas import RetrievalFilter


def test_parse_rewrite_output_reads_input_kind_and_transcript():
    result = parse_rewrite_output(
        '{"retrieval_query": "plant cell labelled diagram", "intent": "explain",'
        ' "input_kind": "diagram", "transcribed_question": "labelled plant cell"}',
        "",
    )
    assert result.fallback is False
    assert result.input_kind == "diagram"
    assert result.transcribed_question == "labelled plant cell"
    assert result.retrieval_query == "plant cell labelled diagram"


def test_parse_rewrite_math_problem_keeps_rule_query():
    result = parse_rewrite_output(
        '{"retrieval_query": "power rule and sum rule", "intent": "verify",'
        ' "input_kind": "math_problem",'
        ' "transcribed_question": "Is d/dx(x^2+3x)=2x+3?"}',
        "check my work",
    )
    assert result.input_kind == "math_problem"
    assert "x^2" not in result.retrieval_query
    assert "2x+3" in result.transcribed_question


def test_parse_rewrite_falls_back_to_transcript_when_query_missing():
    result = parse_rewrite_output(
        '{"intent": "explain", "input_kind": "other", "transcribed_question": "a leaf"}',
        "",
    )
    assert result.fallback is True
    assert result.retrieval_query == "a leaf"
    assert result.transcribed_question == "a leaf"


def test_curriculum_image_path_must_stay_under_images_dir(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    good = images / "fig.png"
    good.write_bytes(b"png")
    outside = tmp_path / "evil.png"
    outside.write_bytes(b"png")
    assert resolve_curriculum_image(str(good), images) == good.resolve()
    assert resolve_curriculum_image(str(outside), images) is None
    assert resolve_curriculum_image(str(images / ".." / "evil.png"), images) is None


def test_select_vision_image_paths_allows_student_extra(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    textbook = images / "ncert.png"
    textbook.write_bytes(b"a")
    student = tmp_path / "upload.jpg"
    student.write_bytes(b"b")
    escaped = tmp_path / "nope.png"
    escaped.write_bytes(b"c")
    kept = select_vision_image_paths(
        [str(textbook), str(escaped), str(student)],
        images_dir=images,
        extra_allowed=[str(student)],
    )
    assert str(textbook.resolve()) in kept
    assert str(student.resolve()) in kept
    assert str(escaped.resolve()) not in kept


def test_generator_message_list_keeps_only_allowed_images(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    textbook = images / "cell.png"
    textbook.write_bytes(b"x")
    student = tmp_path / "photo.jpg"
    student.write_bytes(b"y")
    sneaky = tmp_path / "other.png"
    sneaky.write_bytes(b"z")
    chat = QwenGenerator._to_chat_messages(
        [{"role": "user", "content": "what is this"}],
        [str(textbook), str(sneaky), str(student)],
        images_dir=images,
        extra_allowed=[str(student)],
    )
    blocks = chat[-1]["content"]
    image_blocks = [block for block in blocks if block.get("type") == "image"]
    assert len(image_blocks) == 2
    assert {block["image"] for block in image_blocks} == {
        str(textbook.resolve()),
        str(student.resolve()),
    }


def test_image_hit_lineage_filter_biology_lookback():
    records = [
        {"grade": 12, "subject": "biology", "image_id": "higher"},
        {"grade": 8, "subject": "science", "image_id": "ok"},
        {"grade": 8, "subject": "mathematics", "image_id": "maths"},
        {"grade": 10, "subject": "biology", "image_id": "same-line"},
    ]
    kept = filter_image_hits(
        records, grade=10, subject="biology", allow_prior_grades=True
    )
    assert {row["image_id"] for row in kept} == {"ok", "same-line"}


def test_socratic_prompt_forbids_invented_diagrams():
    controller = SocraticController()
    prompt = controller.system_prompt(
        TutorState.GIVE_HINT, RetrievalFilter(grade=9, subject="science")
    )
    blob = prompt.lower()
    assert "do not generate a drawing" in blob
    assert "do not invent" in blob
    assert "photo" in blob
    user = controller.user_prompt(
        "what is this cell",
        [],
        state=TutorState.INSUFFICIENT_EVIDENCE,
        attached_figures=(),
    )
    assert "input only" in user.lower() or "do not sketch" in user.lower()


def test_select_attached_figures_uses_files_under_images_dir(tmp_path: Path):
    from rag.image_retriever import ImageHit
    from rag.pipeline import _select_attached_figures

    fig = tmp_path / "cell.png"
    fig.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )
    )
    selected = _select_attached_figures(
        kept=[],
        image_hits=[
            ImageHit(image_id="p1:img01", local_path=str(fig), score=0.91, page_number=4)
        ],
        images_dir=tmp_path,
        min_score=0.25,
        limit=2,
    )
    assert len(selected) == 1
    assert selected[0]["image_id"] == "p1:img01"
    assert Path(selected[0]["local_path"]).is_file()


def test_ensure_browser_png_converts_jpeg(tmp_path: Path):
    import pymupdf

    from rag.image_serve import ensure_browser_png

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), 0)
    jpeg = tmp_path / "fig.jpg"
    pix.save(str(jpeg))
    cache = tmp_path / "cache"
    converted = ensure_browser_png(jpeg, cache_root=cache)
    assert converted.suffix == ".png"
    assert converted.is_file()
    assert converted != jpeg.resolve()


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "siglip-base-patch16-224" / "config.json").is_file(),
    reason="SigLIP checkpoint not downloaded",
)
def test_embed_images_job_skips_or_runs():
    from rag.image_index import embed_curriculum_images
    from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError

    try:
        config = load_config()
        store = Neo4jStore(config.require_neo4j())
        store.read("RETURN 1 AS ok")
    except (Exception, Neo4jUnavailableError) as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")
    stats = embed_curriculum_images(config, store)
    assert "scanned" in stats
    assert "embedded" in stats
    assert "skipped" in stats


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "siglip-base-patch16-224" / "config.json").is_file(),
    reason="SigLIP checkpoint not downloaded",
)
def test_image_knn_smoke_if_embeddings_present():
    from rag.image_embeddings import SiglipImageEmbedder
    from rag.image_retriever import ImageRetriever
    from rag.neo4j_store import Neo4jStore, Neo4jUnavailableError
    from rag.schemas import RetrievalFilter

    try:
        config = load_config()
        store = Neo4jStore(config.require_neo4j())
        rows = store.read(
            "MATCH (i:Image) WHERE i.embedding IS NOT NULL "
            "RETURN i.local_path AS local_path, i.grade AS grade, i.subject AS subject "
            "LIMIT 1"
        )
    except (Exception, Neo4jUnavailableError) as exc:
        pytest.skip(f"Neo4j unavailable: {exc}")
    if not rows:
        pytest.skip("no image embeddings in the graph")
    path = Path(rows[0]["local_path"])
    if not path.is_file():
        pytest.skip("embedded image file missing on disk")
    embedder = SiglipImageEmbedder.from_config(config.models)
    retriever = ImageRetriever(
        store,
        embedder,
        config.retrieval,
        embedding_version=config.image_embedding_version,
    )
    try:
        hits, chunks = retriever.retrieve(
            path,
            scope=RetrievalFilter(
                grade=int(rows[0]["grade"] or 10),
                subject=str(rows[0]["subject"] or "science"),
                allow_prior_grades=True,
            ),
        )
    finally:
        embedder.unload()
    assert isinstance(hits, list)
    assert isinstance(chunks, list)
