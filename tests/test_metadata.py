"""Metadata: deterministic IDs, filter translation and concept normalisation.

These guard the two properties ingestion depends on: IDs must be stable across
runs (idempotency), and a metadata scope must become a Cypher predicate that is
applied during retrieval rather than after it.
"""

from __future__ import annotations

import pytest

from rag.concepts import is_valid_concept_phrase, normalize_concept
from rag.retrieval_base import build_filter_clause, chunk_from_record, where_clause
from rag.schemas import (
    RetrievalFilter,
    chunk_id_for,
    concept_id_for,
    grade_id_for,
    image_id_for,
    page_id_for,
    section_id_for,
    slugify,
    subject_id_for,
    unit_id_for,
)


class TestIdentifiers:
    def test_hierarchy_ids_nest(self):
        grade = grade_id_for(2)
        subject = subject_id_for(2, "Science")
        unit = unit_id_for(2, "Science", "unit_01_matter")
        assert grade == "grade_02"
        assert subject == "grade_02:science"
        assert unit == "grade_02:science:unit_01_matter"
        assert unit.startswith(subject)
        assert subject.startswith(grade)

    def test_leaf_ids_are_zero_padded_and_sortable(self):
        document = "grade_02:science:unit_01_matter:student:student_book"
        pages = [page_id_for(document, n) for n in (2, 10, 100)]
        assert pages == [
            f"{document}:p0002",
            f"{document}:p0010",
            f"{document}:p0100",
        ]
        # Zero padding means lexical order matches numeric order.
        assert sorted(pages) == pages

    def test_chunk_id_derives_from_section(self):
        section = section_id_for("doc", 3)
        assert section == "doc:s0003"
        assert chunk_id_for(section, 7) == "doc:s0003:c0007"

    def test_image_id_derives_from_page(self):
        page = page_id_for("doc", 5)
        assert image_id_for(page, 2) == "doc:p0005:img02"

    def test_ids_are_stable_across_calls(self):
        """The whole idempotency story rests on this."""
        first = unit_id_for(3, "Mathematics", "unit_04_fractions")
        second = unit_id_for(3, "Mathematics", "unit_04_fractions")
        assert first == second

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("Plant Life Cycle", "plant_life_cycle"),
            ("Sun, Moon, and Stars", "sun_moon_and_stars"),
            ("  spaced  out  ", "spaced_out"),
            ("Café", "cafe"),
            ("2-D Shapes", "2_d_shapes"),
        ],
    )
    def test_slugify(self, value, expected):
        assert slugify(value) == expected

    def test_concept_id_is_normalisation_stable(self):
        assert concept_id_for("plant life cycle") == concept_id_for("Plant Life Cycle")


class TestConceptNormalisation:
    def test_case_and_plural_variants_collapse(self):
        assert normalize_concept("Plant Life Cycles") == normalize_concept(
            "plant life cycle"
        )

    def test_distinct_concepts_stay_distinct(self):
        assert normalize_concept("solid") != normalize_concept("liquid")

    @pytest.mark.parametrize(
        "phrase",
        [
            "plant life cycle",
            "states of matter",
            "addition within 20",
        ],
    )
    def test_valid_curriculum_phrases_accepted(self, phrase):
        assert is_valid_concept_phrase(phrase)

    @pytest.mark.parametrize(
        "phrase",
        [
            "",
            "a",
            "the",
            "3",
            "NOS4",
            "b c d e f g h i j k",
        ],
    )
    def test_noise_phrases_rejected(self, phrase):
        assert not is_valid_concept_phrase(phrase)


class TestFilterTranslation:
    def test_empty_filter_produces_no_predicate(self):
        clause, params = build_filter_clause(RetrievalFilter())
        assert clause == ""
        assert params == {}
        assert where_clause(clause) == ""

    def test_grade_and_subject_become_parameterised_predicates(self):
        clause, params = build_filter_clause(RetrievalFilter(grade=1, subject="Science"))
        assert "c.grade = $flt_grade" in clause
        assert "c.subject = $flt_subject" in clause
        assert " AND " in clause
        assert params == {"flt_grade": 1, "flt_subject": "science"}

    def test_values_are_never_inlined_into_cypher(self):
        """Everything is a bound parameter, so no query-injection surface."""
        clause, params = build_filter_clause(
            RetrievalFilter(unit_id="grade_01:science:unit_01' OR true //")
        )
        assert "OR true" not in clause
        assert params["flt_unit_id"] == "grade_01:science:unit_01' OR true //"

    def test_alias_can_be_rebound(self):
        clause, _ = build_filter_clause(RetrievalFilter(grade=3), alias="chunk")
        assert clause == "chunk.grade = $flt_grade"

    def test_all_supported_dimensions_are_translated(self):
        scope = RetrievalFilter(
            grade=2,
            subject="mathematics",
            unit_id="grade_02:mathematics:unit_03",
            unit_title_contains="Addition",
            resource_type="student_book",
            audience="student",
            document_id="doc-1",
        )
        clause, params = build_filter_clause(scope)
        assert len(params) == 7
        for key in params:
            assert f"${key}" in clause

    def test_where_clause_only_emitted_when_needed(self):
        assert where_clause("", "") == ""
        assert where_clause("a = 1", "") == "WHERE a = 1"
        assert where_clause("a = 1", "b = 2") == "WHERE a = 1 AND b = 2"

    def test_describe_reports_active_scope_only(self):
        assert "no filter" in RetrievalFilter().describe()
        described = RetrievalFilter(grade=1, subject="science").describe()
        assert "grade" in described and "science" in described


class TestRecordMapping:
    def test_projection_record_becomes_traceable_chunk(self):
        record = {
            "chunk_id": "grade_01:science:unit_01:student:book:s0002:c0001",
            "text": "The moon appears to change shape.",
            "grade": 1,
            "subject": "science",
            "unit_id": "grade_01:science:unit_01_sun_moon_and_stars",
            "unit_title": "Sun, Moon, and Stars",
            "document_id": "grade_01:science:unit_01:student:book",
            "document_title": "Sun, Moon, and Stars - Student Reader",
            "section_id": "grade_01:science:unit_01:student:book:s0002",
            "section_title": "Phases of the Moon",
            "page_start": 12,
            "page_end": 13,
            "resource_type": "student_reader",
            "audience": "student",
            "local_pdf_path": "/corpus/student_reader.pdf",
        }
        chunk = chunk_from_record(record)
        assert chunk.grade == 1
        assert chunk.page_range == "12-13"
        provenance = chunk.provenance()
        for key in ("grade", "subject", "unit_id", "document_id", "pages"):
            assert provenance[key] is not None

    def test_missing_optional_fields_do_not_raise(self):
        chunk = chunk_from_record({"chunk_id": "c1"})
        assert chunk.text == ""
        assert chunk.grade is None
        assert chunk.page_range == "?"

    def test_single_page_range_renders_without_dash(self):
        chunk = chunk_from_record(
            {"chunk_id": "c1", "page_start": 7, "page_end": 7}
        )
        assert chunk.page_range == "7"
