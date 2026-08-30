"""Curriculum catalog, partitions, alignment and download safety."""

from __future__ import annotations

from pathlib import Path

from rag.alignment import load_alignment, outcome_ids_for
from rag.curriculum_catalog import (
    SOURCE_CISCE,
    SOURCE_ENGAGENY,
    SOURCE_SIYAVULA,
    SOURCE_UTAH,
    SUPPORTED_GRADES,
    all_source_files,
    ingestible_files,
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


def test_supported_grades_are_three_to_five_only():
    assert SUPPORTED_GRADES == (3, 4, 5)
    for record in ingestible_files():
        assert record.grade in SUPPORTED_GRADES


def test_catalog_has_required_sources_and_no_core_knowledge():
    records = all_source_files()
    ids = {item.source_id for item in records}
    assert {SOURCE_ENGAGENY, SOURCE_SIYAVULA, SOURCE_UTAH, SOURCE_CISCE} <= ids
    blob = " ".join(f"{item.source_id} {item.publisher} {item.local_path}" for item in records)
    assert "core knowledge" not in blob.lower()
    assert "core_knowledge" not in blob.lower()


def test_cisce_is_alignment_only_and_not_ingestible():
    cisce = [item for item in all_source_files() if item.source_id == SOURCE_CISCE]
    assert cisce
    assert all(item.alignment_only and not item.ingest for item in cisce)
    assert all(item not in ingestible_files() for item in cisce)


def test_utah_roles_and_image_policy():
    utah = [item for item in ingestible_files() if item.source_id == SOURCE_UTAH]
    g3 = next(item for item in utah if item.grade == 3)
    g4 = next(item for item in utah if item.grade == 4)
    g5 = next(item for item in utah if item.grade == 5)
    assert g3.source_role == "primary"
    assert g4.source_role == "support"
    assert g5.source_role == "support"
    assert all(not item.extract_images for item in utah)


def test_siyavula_starts_at_grade_4():
    siyavula = [item for item in ingestible_files() if item.source_id == SOURCE_SIYAVULA]
    assert {item.grade for item in siyavula} == {4, 5}
    assert all(item.source_role == "primary" for item in siyavula)
    assert all(item.file_format == "epub" for item in siyavula)
    assert all("CC-BY" in item.local_path or "CC-BY" in item.direct_download_url for item in siyavula)


def test_engageny_is_primary_math_with_nc_licence():
    math = [item for item in ingestible_files() if item.source_id == SOURCE_ENGAGENY]
    assert {item.grade for item in math} == {3, 4, 5}
    assert all(item.subject == "mathematics" for item in math)
    assert all(item.source_role == "primary" for item in math)
    assert all("NC" in item.licence for item in math)


def test_partition_classifier_blocks_answer_keys():
    assert partition_from_filename("math-g3-m1-end-of-module-assessment.pdf") == EVALUATION_ONLY
    assert partition_from_heading("Answer Key") == EVALUATION_ONLY
    assert partition_from_heading("Lesson 12 Answer Key") == EVALUATION_ONLY
    assert partition_from_heading("Mid-Module Assessment Task") == EVALUATION_ONLY
    assert partition_from_heading("End-of-Module Assessment Task") == EVALUATION_ONLY
    assert partition_from_heading("Exit Ticket (3 minutes)") == PRACTICE_ONLY
    assert partition_from_heading("Homework") == PRACTICE_ONLY
    assert partition_from_heading("Concept Development") == TEACHER_STRATEGY
    assert partition_from_heading("Fractions on a number line") == STUDENT_EVIDENCE
    assert classify_section("Credits", "This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.").partition == EXCLUDED_BOILERPLATE
    assert (
        classify_section(
            "Exit Ticket",
            "Sample response: 3/4. The correct answer is 3/4.",
        ).partition
        == EVALUATION_ONLY
    )


def test_science_mapping_is_section_level_not_whole_book():
    living, gran, status = outcome_ids_for(
        grade=3,
        subject="science",
        unit_slug="unit_01_science_oer",
        section_title="Living and Non-Living Things",
        text="A rock is a non-living thing. A bird is a living thing.",
    )
    assert gran == "section"
    assert status == "needs_human_review"
    assert "cisce_g3_sci_living_nonliving" in living
    weather, _, weather_status = outcome_ids_for(
        grade=3,
        subject="science",
        unit_slug="unit_01_science_oer",
        section_title="Weather Instruments",
        text="A thermometer measures how hot or cold the weather is.",
    )
    assert weather == []
    assert weather_status == "unmapped"
    empty, _, unmapped = outcome_ids_for(
        grade=3,
        subject="science",
        unit_slug="unit_01_science_oer",
        section_title="Publisher address",
        text="Printed in the United States. ISBN 978-0-000000-00-0.",
    )
    assert empty == []
    assert unmapped == "unmapped"


def test_verified_alignment_requires_human_reviewer():
    from rag.alignment import validate_alignment_row
    import pytest

    with pytest.raises(ValueError):
        validate_alignment_row(
            {
                "outcome_id": "cisce_g3_sci_living_nonliving",
                "alignment_status": "verified",
                "reviewer": "",
                "reviewed_at": "",
            }
        )


def test_catalog_does_not_use_archive_org_mirrors():
    for item in all_source_files():
        assert "archive.org" not in item.direct_download_url.lower()
        assert "archive.org" not in item.official_page_url.lower()


def test_alignment_file_loads_and_is_unverified():
    rows = load_alignment()
    assert rows
    assert all(row.get("alignment_status") == "needs_human_review" for row in rows)
    assert all(not row.get("reviewer") for row in rows)
    ids = {row.get("outcome_id") for row in rows}
    assert "cisce_g3_math_fractions" not in ids
    assert "cisce_g3_sci_earth_weather" not in ids
    assert "cisce_g4_sci_earth" not in ids
    data, gran, status = outcome_ids_for(
        grade=3,
        subject="mathematics",
        unit_slug="module_06_collecting",
        section_title="Tally marks and picture graphs",
        text="Record the data using tally marks and draw a pictograph.",
    )
    assert "cisce_g3_math_data" in data
    assert gran == "section"
    assert status == "needs_human_review"
    none, _, unmapped = outcome_ids_for(
        grade=3, subject="mathematics", unit_slug="module_05_fractions"
    )
    assert none == []
    assert unmapped == "unmapped"


def test_alignment_matches_truncated_and_full_slugs():
    full = "module_03_multiplication_and_division_with_units_of_0_1_6_9_and_multip"
    ids, _, _ = outcome_ids_for(
        grade=3,
        subject="mathematics",
        unit_slug=full,
        section_title="Multiplication as equal groups",
        text="Students use arrays and grouping for multiplication and division.",
    )
    assert "cisce_g3_math_number_operations" in ids
    decimals, _, unmapped = outcome_ids_for(
        grade=4,
        subject="mathematics",
        unit_slug="module_06_decimal",
        section_title="Tenths and hundredths",
        text="Write tenths as decimal fractions.",
    )
    assert decimals == []
    assert unmapped == "unmapped"
    fractions, gran, _ = outcome_ids_for(
        grade=4,
        subject="mathematics",
        unit_slug="module_05_fraction_equivalence",
        section_title="Equivalent fractions",
        text="Add like fractions that share a denominator.",
    )
    assert "cisce_g4_math_fractions" in fractions
    assert gran == "section"


def test_rounding_sections_are_excluded_from_class_iv_place_value():
    ids, _, _ = outcome_ids_for(
        grade=4,
        subject="mathematics",
        unit_slug="module_01_place_value_rounding",
        section_title="Rounding to the nearest hundred",
        text="Round 4,562 to the nearest hundred using place value.",
    )
    assert "cisce_g4_math_place_value" not in ids
    kept, _, _ = outcome_ids_for(
        grade=4,
        subject="mathematics",
        unit_slug="module_01_place_value_rounding",
        section_title="Place value of six-digit numbers",
        text="Write a six-digit number in expanded form using place value.",
    )
    assert "cisce_g4_math_place_value" in kept


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
