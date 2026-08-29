"""Approved Grade 3–5 STEM sources. CISCE is alignment-only and is never ingested."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SUPPORTED_GRADES = (3, 4, 5)
SUPPORTED_SUBJECTS = ("mathematics", "science")

SOURCE_ENGAGENY = "engageny_math"
SOURCE_SIYAVULA = "siyavula_natural_sciences"
SOURCE_UTAH = "utah_science_oer"
SOURCE_CISCE = "cisce_primary_curriculum"

PRODUCTION_PARTITIONS = ("student_evidence", "teacher_strategy")
EVALUATION_PARTITION = "evaluation_only"

NYSED_ENGAGENY_PAGE = "https://www.nysed.gov/curriculum-instruction/engageny"
NYSED_MATH_PAGE = "https://www.nysed.gov/edtech/digital-content-resources-mathematics"
NYSED_STANDARDS_PAGE = "https://www.nysed.gov/standards-instruction/standards-resources-and-supports"
# NYSED's public pages currently point at a SharePoint folder that requires a
# Microsoft login. There is no login-free nysed.gov file URL for Grade 3–5
# module PDFs. Internet Archive must not be used as a substitute.
NYSED_ENGAGENY_MATH_SHAREPOINT = (
    "https://nysed.sharepoint.com/:f:/s/P12EngageNY-Math-EXTA/"
    "En7SIs8H6v5PlQbP8fYWQbkBvFl7pdadxm5WQe2RYn6C_Q?e=aA13JQ"
)
ENGAGENY_LICENCE = "CC BY-NC-SA 4.0"
ENGAGENY_LICENCE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
# GlobalSign intermediate referenced by *.nysed.gov AIA (server omits the chain).
GLOBALSIGN_RSA_OV_SSL_CA_2018 = (
    "http://secure.globalsign.com/cacert/gsrsaovsslca2018.crt"
)

CISCE_CURRICULUM_URL = "https://cisce.org/wp-content/uploads/2025/03/PrimaryCurriculum.pdf"
CISCE_RESOURCES_URL = "https://cisce.org/rdcd-pre-school-to-class-viii-resource-material/"
CISCE_LEGAL_URL = "https://cisce.org/legal-disclaimer/"

UTAH_SCIENCE_PAGE = "https://schools.utah.gov/curr/science"
SIYAVULA_CATALOGUE = "https://www.siyavula.com/read"


@dataclass(frozen=True)
class SourceFile:
    file_id: str
    source_id: str
    publisher: str
    official_page_url: str
    direct_download_url: str
    local_path: str
    grade: int
    subject: str
    source_role: str
    licence: str
    licence_url: str
    file_format: str
    allowed_partitions: tuple[str, ...]
    unit_slug: str
    unit_title: str
    unit_number: int | None = None
    audience: str = "student"
    resource_type: str = "student_book"
    extract_images: bool = False
    ingest: bool = True
    alignment_only: bool = False
    zip_member_glob: str | None = None
    skip_member_substrings: tuple[str, ...] = ()
    retrieved_at: str = ""
    sha256: str = ""
    alignment_status: str = "needs_human_review"
    provenance_status: str = "official"

    def to_manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_partitions"] = list(self.allowed_partitions)
        data["skip_member_substrings"] = list(self.skip_member_substrings)
        return data


ENGAGENY_MODULES: dict[int, tuple[tuple[int, str], ...]] = {
    3: (
        (1, "Properties of Multiplication and Division"),
        (2, "Place Value and Problem Solving with Units of Measure"),
        (3, "Multiplication and Division with Units of 0 1 6-9 and Multiples of 10"),
        (4, "Multiplication and Area"),
        (5, "Fractions as Numbers on the Number Line"),
        (6, "Collecting and Displaying Data"),
        (7, "Geometry and Measurement Word Problems"),
    ),
    4: (
        (1, "Place Value Rounding and Algorithms for Addition and Subtraction"),
        (2, "Unit Conversions and Problem Solving with Metric Measurement"),
        (3, "Multi-Digit Multiplication and Division"),
        (4, "Angle Measure and Plane Figures"),
        (5, "Fraction Equivalence Ordering and Operations"),
        (6, "Decimal Fractions"),
        (7, "Exploring Measurement with Multiplication"),
    ),
    5: (
        (1, "Place Value and Decimal Fractions"),
        (2, "Multi-Digit Whole Number and Decimal Fraction Operations"),
        (3, "Addition and Subtraction of Fractions"),
        (4, "Multiplication and Division of Fractions and Decimal Fractions"),
        (5, "Addition and Multiplication with Volume and Area"),
        (6, "Problem Solving with the Coordinate Plane"),
    ),
}


def _engageny_files() -> list[SourceFile]:
    files: list[SourceFile] = []
    for grade, modules in ENGAGENY_MODULES.items():
        for number, title in modules:
            slug = f"module_{number:02d}_{_slug(title)}"
            files.append(
                SourceFile(
                    file_id=f"engageny_g{grade}_m{number:02d}",
                    source_id=SOURCE_ENGAGENY,
                    publisher="New York State Education Department",
                    official_page_url=NYSED_STANDARDS_PAGE,
                    # No public file on nysed.gov; SharePoint requires login.
                    direct_download_url=NYSED_ENGAGENY_MATH_SHAREPOINT,
                    local_path=(
                        f"raw/engageny/mathematics/grade_{grade:02d}/"
                        f"{slug}/student/math-g{grade}-m{number}-full-module.pdf"
                    ),
                    grade=grade,
                    subject="mathematics",
                    source_role="primary",
                    licence=ENGAGENY_LICENCE,
                    licence_url=ENGAGENY_LICENCE_URL,
                    file_format="pdf",
                    allowed_partitions=PRODUCTION_PARTITIONS,
                    unit_slug=slug,
                    unit_title=title,
                    unit_number=number,
                    audience="student",
                    resource_type="module",
                    extract_images=True,
                    provenance_status="official",
                    skip_member_substrings=(
                        "assessment",
                        "answer-key",
                        "answer_key",
                        "rubric",
                    ),
                )
            )
    return files


def _slug(title: str) -> str:
    allowed = []
    for ch in title.lower():
        if ch.isalnum():
            allowed.append(ch)
        elif ch in " -_/":
            allowed.append("_")
    collapsed = "".join(allowed)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_")[:60]


def _siyavula_files() -> list[SourceFile]:
    return [
        SourceFile(
            file_id="siyavula_g4_ns_learner",
            source_id=SOURCE_SIYAVULA,
            publisher="Siyavula Education",
            official_page_url=SIYAVULA_CATALOGUE,
            direct_download_url=(
                "https://www.siyavula.com/downloads/books/science/"
                "Gr4_NaturalSciences_Learner_Eng_CC-BY.epub"
            ),
            local_path=(
                "raw/siyavula/science/grade_04/natural_sciences/"
                "student/Gr4_NaturalSciences_Learner_Eng_CC-BY.epub"
            ),
            grade=4,
            subject="science",
            source_role="primary",
            licence="CC BY 4.0",
            licence_url="https://creativecommons.org/licenses/by/4.0/",
            file_format="epub",
            allowed_partitions=("student_evidence",),
            unit_slug="unit_01_natural_sciences",
            unit_title="Natural Sciences and Technology",
            unit_number=1,
            extract_images=True,
        ),
        SourceFile(
            file_id="siyavula_g5_ns_learner",
            source_id=SOURCE_SIYAVULA,
            publisher="Siyavula Education",
            official_page_url=SIYAVULA_CATALOGUE,
            direct_download_url=(
                "https://www.siyavula.com/downloads/books/science/"
                "Gr5_NaturalSciences_Learner_Eng_CC-BY.epub"
            ),
            local_path=(
                "raw/siyavula/science/grade_05/natural_sciences/"
                "student/Gr5_NaturalSciences_Learner_Eng_CC-BY.epub"
            ),
            grade=5,
            subject="science",
            source_role="primary",
            licence="CC BY 4.0",
            licence_url="https://creativecommons.org/licenses/by/4.0/",
            file_format="epub",
            allowed_partitions=("student_evidence",),
            unit_slug="unit_01_natural_sciences",
            unit_title="Natural Sciences and Technology",
            unit_number=1,
            extract_images=True,
        ),
    ]


def _utah_files() -> list[SourceFile]:
    # Images in these PDFs mix CK-12 and other assets; skip image extraction.
    specs = (
        (3, "primary", "3rdGradeOERSciTextbook.pdf"),
        (4, "support", "4thGradeOERSciTextbook.pdf"),
        (5, "support", "5thGradeOERSciTextbook.pdf"),
    )
    files: list[SourceFile] = []
    for grade, role, filename in specs:
        files.append(
            SourceFile(
                file_id=f"utah_g{grade}_science_oer",
                source_id=SOURCE_UTAH,
                publisher="Utah State Board of Education",
                official_page_url=UTAH_SCIENCE_PAGE,
                direct_download_url=(
                    f"https://schools.utah.gov/curr/science/_science_/{filename}"
                ),
                local_path=(
                    f"raw/utah_oer/science/grade_{grade:02d}/science_oer/"
                    f"student/{filename}"
                ),
                grade=grade,
                subject="science",
                source_role=role,
                licence="See source notices (includes CK-12 and mixed OER terms; noncommercial restrictions may apply)",
                licence_url=UTAH_SCIENCE_PAGE,
                file_format="pdf",
                allowed_partitions=("student_evidence",),
                unit_slug="unit_01_science_oer",
                unit_title="Utah Science OER Textbook",
                unit_number=1,
                extract_images=False,
            )
        )
    return files


def _cisce_alignment_file() -> SourceFile:
    return SourceFile(
        file_id="cisce_primary_curriculum_pdf",
        source_id=SOURCE_CISCE,
        publisher="Council for the Indian School Certificate Examinations",
        official_page_url=CISCE_RESOURCES_URL,
        direct_download_url=CISCE_CURRICULUM_URL,
        local_path="raw/_alignment_only/cisce/PrimaryCurriculum.pdf",
        grade=0,
        subject="alignment",
        source_role="alignment",
        licence="CISCE terms; not ingested (alignment authority only)",
        licence_url=CISCE_LEGAL_URL,
        file_format="pdf",
        allowed_partitions=(),
        unit_slug="alignment",
        unit_title="CISCE Primary Curriculum",
        ingest=False,
        alignment_only=True,
        extract_images=False,
    )


def all_source_files(*, include_alignment: bool = True) -> list[SourceFile]:
    files = _engageny_files() + _siyavula_files() + _utah_files()
    if include_alignment:
        files.append(_cisce_alignment_file())
    return files


def ingestible_files() -> list[SourceFile]:
    return [item for item in all_source_files(include_alignment=False) if item.ingest]


def files_by_id() -> dict[str, SourceFile]:
    return {item.file_id: item for item in all_source_files()}
