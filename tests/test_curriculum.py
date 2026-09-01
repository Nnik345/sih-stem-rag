"""Curriculum catalog, partitions, and download safety for CBSE/NCERT STEM."""

from __future__ import annotations

from pathlib import Path

from rag.alignment import load_alignment, outcome_ids_for, validate_alignment_row
from rag.curriculum_catalog import (
    NCERT_BOOKS,
    SOURCE_NCERT,
    SUPPORTED_GRADES,
    SUPPORTED_SUBJECTS,
    all_source_files,
    ingestible_files,
    in_lineage_scope,
    is_answers_member,
    is_chapter_pdf,
    lineage_subjects,
    ncert_zip_url,
    subjects_for_grade,
    validate_scope,
)
from rag.partitions import (
    EVALUATION_ONLY,
    EXCLUDED_BOILERPLATE,
    PRACTICE_ONLY,
    STUDENT_EVIDENCE,
    TEACHER_STRATEGY,
    classify_section,
    partition_from_filename,
    partition_from_heading,
)


def test_supported_grades_are_one_to_twelve():
    assert SUPPORTED_GRADES == tuple(range(1, 13))
    assert SUPPORTED_SUBJECTS == (
        "mathematics",
        "science",
        "physics",
        "chemistry",
        "biology",
    )
    for record in ingestible_files():
        assert record.grade in SUPPORTED_GRADES
        assert record.subject in SUPPORTED_SUBJECTS


def test_catalog_is_ncert_only_english_stem():
    records = all_source_files()
    assert records
    assert {item.source_id for item in records} == {SOURCE_NCERT}
    assert all(item.ingest and not item.alignment_only for item in records)
    assert all(item.source_role == "primary" for item in records)
    blob = " ".join(
        f"{item.source_id} {item.publisher} {item.local_path} {item.unit_title}"
        for item in records
    ).lower()
    assert "core knowledge" not in blob
    assert "cisce" not in blob
    assert "engageny" not in blob
    assert "siyavula" not in blob
    assert "utah" not in blob
    for item in records:
        assert item.direct_download_url.startswith("https://ncert.nic.in/")
        assert item.ncert_code
        assert "/ncert/" in item.local_path
        if item.subject in {"physics", "chemistry", "biology"}:
            assert "/science/" in item.local_path
        assert "copyright" in item.licence.lower()
        assert "CC BY" not in item.licence


def test_classes_1_2_are_maths_only():
    by_grade = {}
    for item in ingestible_files():
        by_grade.setdefault(item.grade, set()).add(item.subject)
    assert by_grade[1] == {"mathematics"}
    assert by_grade[2] == {"mathematics"}
    for grade in range(3, 11):
        assert "mathematics" in by_grade[grade]
        assert "science" in by_grade[grade]
    for grade in (11, 12):
        assert by_grade[grade] == {"mathematics", "physics", "chemistry", "biology"}


def test_subjects_for_grade_gates_pcb():
    assert subjects_for_grade(1) == ("mathematics",)
    assert subjects_for_grade(6) == ("mathematics", "science")
    assert subjects_for_grade(12) == ("mathematics", "physics", "chemistry", "biology")
    validate_scope(12, "physics")
    import pytest

    with pytest.raises(ValueError, match="not offered"):
        validate_scope(6, "physics")
    with pytest.raises(ValueError, match="required"):
        validate_scope(None, "science")


def test_lineage_subjects_and_prior_grade_scope():
    assert lineage_subjects("mathematics") == ("mathematics",)
    assert lineage_subjects("science") == ("science",)
    assert lineage_subjects("physics") == ("physics", "science")
    assert lineage_subjects("chemistry") == ("chemistry", "science")
    assert lineage_subjects("biology") == ("biology", "science")
    assert in_lineage_scope(
        chunk_grade=11,
        chunk_subject="mathematics",
        current_grade=12,
        current_subject="mathematics",
        allow_prior_grades=True,
    )
    assert in_lineage_scope(
        chunk_grade=10,
        chunk_subject="science",
        current_grade=12,
        current_subject="physics",
        allow_prior_grades=True,
    )
    assert not in_lineage_scope(
        chunk_grade=11,
        chunk_subject="chemistry",
        current_grade=12,
        current_subject="physics",
        allow_prior_grades=True,
    )
    assert not in_lineage_scope(
        chunk_grade=10,
        chunk_subject="mathematics",
        current_grade=12,
        current_subject="physics",
        allow_prior_grades=True,
    )
    assert not in_lineage_scope(
        chunk_grade=12,
        chunk_subject="mathematics",
        current_grade=11,
        current_subject="mathematics",
        allow_prior_grades=True,
    )
    assert not in_lineage_scope(
        chunk_grade=11,
        chunk_subject="mathematics",
        current_grade=12,
        current_subject="mathematics",
        allow_prior_grades=False,
    )


def test_senior_secondary_science_uses_pcb_subjects():
    slugs = {
        (item.grade, item.subject, item.unit_slug)
        for item in ingestible_files()
        if item.grade in (11, 12)
    }
    assert (11, "physics", "physics_part_1") in slugs
    assert (11, "chemistry", "chemistry_part_2") in slugs
    assert (11, "biology", "biology") in slugs
    assert (12, "physics", "physics_part_2") in slugs
    assert (12, "chemistry", "chemistry_part_1") in slugs
    assert (12, "biology", "biology") in slugs
    assert all(item.subject != "science" or item.grade < 11 for item in ingestible_files())


def test_class_6_science_is_curiosity_ncf():
    g6 = next(
        item
        for item in ingestible_files()
        if item.grade == 6 and item.subject == "science"
    )
    assert g6.unit_title == "Curiosity"
    assert g6.unit_slug == "curiosity"
    assert g6.ncert_code == "fecu1"


def test_book_codes_match_official_zip_pattern():
    codes = {book.code for book in NCERT_BOOKS}
    assert len(codes) == len(NCERT_BOOKS)
    for book in NCERT_BOOKS:
        assert ncert_zip_url(book.code) == f"https://ncert.nic.in/textbook/pdf/{book.code}dd.zip"


def test_chapter_pdf_helper_skips_answers_and_covers():
    assert is_chapter_pdf("aejm101.pdf", "aejm1")
    assert is_chapter_pdf("folder/keph201.pdf", "keph2")
    assert not is_chapter_pdf("aejm1an.pdf", "aejm1")
    assert not is_chapter_pdf("cover.jpg", "aejm1")
    assert not is_chapter_pdf("aejm1ps.pdf", "aejm1")
    assert is_answers_member("jemh1an.pdf")
    assert is_answers_member("answers.pdf")


def test_partition_classifier_blocks_answer_keys_and_ncert_exercises():
    assert partition_from_filename("math-g3-m1-end-of-module-assessment.pdf") == EVALUATION_ONLY
    assert partition_from_filename("jemh1an.pdf") == EVALUATION_ONLY
    assert partition_from_heading("Answer Key") == EVALUATION_ONLY
    assert partition_from_heading("Answers") == EVALUATION_ONLY
    assert partition_from_heading("Exercises") == PRACTICE_ONLY
    assert partition_from_heading("Let’s practise") == PRACTICE_ONLY
    assert partition_from_heading("Let's practise") == PRACTICE_ONLY
    assert partition_from_heading("Concept Development") == TEACHER_STRATEGY
    assert partition_from_heading("Components of food") == STUDENT_EVIDENCE
    assert (
        classify_section(
            "Credits",
            "This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.",
        ).partition
        == EXCLUDED_BOILERPLATE
    )
    assert (
        classify_section(
            "Exit Ticket",
            "Sample response: 3/4. The correct answer is 3/4.",
        ).partition
        == EVALUATION_ONLY
    )


def test_native_ncert_alignment_has_no_cisce_yaml():
    assert load_alignment() == ()
    ids, gran, status = outcome_ids_for(
        grade=6,
        subject="science",
        unit_slug="science",
        section_title="Components of food",
        text="Carbohydrates, proteins, fats, vitamins and minerals.",
    )
    assert ids == []
    assert gran == "none"
    assert status == "ncert_native"


def test_verified_alignment_requires_human_reviewer():
    import pytest

    with pytest.raises(ValueError):
        validate_alignment_row(
            {
                "outcome_id": "example",
                "alignment_status": "verified",
                "reviewer": "",
                "reviewed_at": "",
            }
        )


def test_catalog_does_not_use_archive_org_mirrors():
    for item in all_source_files():
        assert "archive.org" not in item.direct_download_url.lower()
        assert "archive.org" not in item.official_page_url.lower()
        assert "cisce.org" not in item.direct_download_url.lower()
        assert "siyavula.com" not in item.direct_download_url.lower()
        assert "nysed" not in item.direct_download_url.lower()


def test_alignment_strict_filters_to_ncert_source():
    from rag.retrieval_base import build_filter_clause
    from rag.schemas import RetrievalFilter

    clause, params = build_filter_clause(RetrievalFilter(alignment_strict=True))
    assert "source_id" in clause
    assert params.get("flt_source_ncert") == SOURCE_NCERT


def test_no_core_knowledge_in_repo_config_and_scripts():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (
        root / "src/rag/config.py",
        root / "src/rag/corpus.py",
        root / "scripts/ingest_corpus.py",
        root / ".env.example",
    ):
        text = path.read_text(encoding="utf-8").lower()
        if "core knowledge" in text or "core_knowledge" in text:
            offenders.append(str(path))
    assert offenders == []
    assert not (root / "scripts" / "download_core_knowledge_stem.py").exists()
    assert not (root / "curriculum" / "alignment" / "cisce_grade_3_5_stem.yaml").exists()
