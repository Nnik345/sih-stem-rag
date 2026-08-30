"""Official English-medium NCERT STEM textbooks for CBSE classes 1–12.

NCERT books *are* the CBSE curriculum for this project: they are both the
ingested corpus and the alignment authority. There is no separate syllabus PDF
and no CISCE/EngageNY/Siyavula/Utah source.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_GRADES = tuple(range(1, 13))
SUPPORTED_SUBJECTS = ("mathematics", "science")

SOURCE_NCERT = "ncert_textbook"

PRODUCTION_PARTITIONS = ("student_evidence", "teacher_strategy")
EVALUATION_PARTITION = "evaluation_only"
ALIGNMENT_STATUS_NATIVE = "ncert_native"

NCERT_TEXTBOOK_PAGE = "https://ncert.nic.in/textbook.php"
NCERT_PDF_BASE = "https://ncert.nic.in/textbook/pdf"
NCERT_PUBLISHER = "National Council of Educational Research and Training"
NCERT_LICENCE = (
    "NCERT copyright; local research use of official ncert.nic.in textbooks only"
)
NCERT_LICENCE_URL = NCERT_TEXTBOOK_PAGE

_CHAPTER_NAME_RE = re.compile(r"^([a-z0-9]+?)(\d{2})\.pdf$", re.IGNORECASE)
_ANSWERS_NAME_RE = re.compile(
    r"(?:^|/)(?:.*)?(?:an\.pdf$|answers?)",
    re.IGNORECASE,
)


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
    extract_images: bool = True
    ingest: bool = True
    alignment_only: bool = False
    ncert_code: str = ""
    zip_member_glob: str | None = None
    skip_member_substrings: tuple[str, ...] = ()
    retrieved_at: str = ""
    sha256: str = ""
    alignment_status: str = ALIGNMENT_STATUS_NATIVE
    provenance_status: str = "official"

    def to_manifest_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_partitions"] = list(self.allowed_partitions)
        data["skip_member_substrings"] = list(self.skip_member_substrings)
        return data


@dataclass(frozen=True)
class NcertBook:
    """One official English student textbook (complete-book zip)."""

    grade: int
    subject: str
    title: str
    code: str
    slug: str
    unit_number: int | None = 1


# Codes transcribed from ncert.nic.in/textbook.php change1() (English STEM only).
# Current NCF titles are preferred over superseded duplicates still listed
# alongside them. Hindi/Urdu editions, Exemplars, and Lab Manuals are omitted.
NCERT_BOOKS: tuple[NcertBook, ...] = (
    NcertBook(1, "mathematics", "Joyful-Mathematics", "aejm1", "joyful_mathematics"),
    NcertBook(2, "mathematics", "Joyful-Mathematics", "bejm1", "joyful_mathematics"),
    NcertBook(3, "mathematics", "Maths Mela", "cemm1", "maths_mela"),
    NcertBook(3, "science", "Our Wondrous World", "ceev1", "our_wondrous_world"),
    NcertBook(4, "mathematics", "Math-Mela", "demm1", "maths_mela"),
    NcertBook(4, "science", "Our Wondrous World", "deev1", "our_wondrous_world"),
    NcertBook(5, "mathematics", "Math-Mela", "eemm1", "maths_mela"),
    NcertBook(5, "science", "Our Wondrous World", "eeev1", "our_wondrous_world"),
    NcertBook(6, "mathematics", "Ganita Prakash", "fegp1", "ganita_prakash"),
    NcertBook(6, "science", "Curiosity", "fecu1", "curiosity"),
    NcertBook(7, "mathematics", "Ganita Prakash Part I", "gegp1", "ganita_prakash_part_1", 1),
    NcertBook(7, "mathematics", "Ganita Prakash Part II", "gegp2", "ganita_prakash_part_2", 2),
    NcertBook(7, "science", "Curiosity", "gecu1", "curiosity"),
    NcertBook(8, "mathematics", "Ganita Prakash Part I", "hegp1", "ganita_prakash_part_1", 1),
    NcertBook(8, "mathematics", "Ganita Prakash Part II", "hegp2", "ganita_prakash_part_2", 2),
    NcertBook(8, "science", "Curiosity", "hecu1", "curiosity"),
    NcertBook(9, "mathematics", "Ganita Manjari", "iemh1", "ganita_manjari"),
    NcertBook(9, "science", "Exploration", "iesc1", "exploration"),
    NcertBook(10, "mathematics", "Mathematics", "jemh1", "mathematics"),
    NcertBook(10, "science", "Science", "jesc1", "science"),
    NcertBook(11, "mathematics", "Mathematics", "kemh1", "mathematics"),
    NcertBook(11, "science", "Physics Part I", "keph1", "physics_part_1", 1),
    NcertBook(11, "science", "Physics Part II", "keph2", "physics_part_2", 2),
    NcertBook(11, "science", "Chemistry Part I", "kech1", "chemistry_part_1", 1),
    NcertBook(11, "science", "Chemistry Part II", "kech2", "chemistry_part_2", 2),
    NcertBook(11, "science", "Biology", "kebo1", "biology"),
    NcertBook(12, "mathematics", "Mathematics Part I", "lemh1", "mathematics_part_1", 1),
    NcertBook(12, "mathematics", "Mathematics Part II", "lemh2", "mathematics_part_2", 2),
    NcertBook(12, "science", "Physics Part I", "leph1", "physics_part_1", 1),
    NcertBook(12, "science", "Physics Part II", "leph2", "physics_part_2", 2),
    NcertBook(12, "science", "Chemistry Part I", "lech1", "chemistry_part_1", 1),
    NcertBook(12, "science", "Chemistry Part II", "lech2", "chemistry_part_2", 2),
    NcertBook(12, "science", "Biology", "lebo1", "biology"),
)

_SKIP_ZIP_SUBSTRINGS = (
    "an.pdf",
    "answer",
    "answers",
    "cover",
    ".jpg",
    ".jpeg",
    ".png",
    "prelim",
)


def ncert_zip_url(code: str) -> str:
    return f"{NCERT_PDF_BASE}/{code}dd.zip"


def is_answers_member(name: str) -> bool:
    base = Path(name).name.lower()
    return bool(_ANSWERS_NAME_RE.search(base)) or base.endswith("an.pdf")


def is_chapter_pdf(name: str, code: str) -> bool:
    """True for `{code}{nn}.pdf` chapter members; false for answers/covers."""
    base = Path(name).name
    if is_answers_member(base):
        return False
    match = _CHAPTER_NAME_RE.match(base)
    if not match:
        return False
    return match.group(1).lower() == code.lower()


def chapter_number_from_filename(name: str) -> int | None:
    match = _CHAPTER_NAME_RE.match(Path(name).name)
    if not match:
        return None
    return int(match.group(2))


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


def _book_to_source(book: NcertBook) -> SourceFile:
    slug = book.slug or _slug(book.title)
    return SourceFile(
        file_id=f"ncert_g{book.grade}_{slug}",
        source_id=SOURCE_NCERT,
        publisher=NCERT_PUBLISHER,
        official_page_url=NCERT_TEXTBOOK_PAGE,
        direct_download_url=ncert_zip_url(book.code),
        local_path=(
            f"raw/ncert/{book.subject}/grade_{book.grade:02d}/{slug}/student"
        ),
        grade=book.grade,
        subject=book.subject,
        source_role="primary",
        licence=NCERT_LICENCE,
        licence_url=NCERT_LICENCE_URL,
        file_format="zip",
        allowed_partitions=PRODUCTION_PARTITIONS,
        unit_slug=slug,
        unit_title=book.title,
        unit_number=book.unit_number,
        audience="student",
        resource_type="student_book",
        extract_images=True,
        ncert_code=book.code,
        zip_member_glob=f"{book.code}*.pdf",
        skip_member_substrings=_SKIP_ZIP_SUBSTRINGS,
        alignment_status=ALIGNMENT_STATUS_NATIVE,
        provenance_status="official",
    )


def all_source_files(*, include_alignment: bool = False) -> list[SourceFile]:
    del include_alignment
    return [_book_to_source(book) for book in NCERT_BOOKS]


def ingestible_files() -> list[SourceFile]:
    return [item for item in all_source_files() if item.ingest]


def files_by_id() -> dict[str, SourceFile]:
    return {item.file_id: item for item in all_source_files()}
