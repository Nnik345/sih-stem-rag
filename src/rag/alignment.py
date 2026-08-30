"""CBSE/NCERT alignment: native textbook chunks, no separate syllabus YAML.

Official English NCERT STEM textbooks *are* the CBSE curriculum for this
project. Chunks are aligned by ``source_id == ncert_textbook``. The Neo4j
property ``cisce_outcome_ids`` is left unused/empty (no schema rename).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .curriculum_catalog import ALIGNMENT_STATUS_NATIVE, SOURCE_NCERT
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


def validate_alignment_row(row: dict[str, Any]) -> None:
    """Keep a guard if a future YAML row is marked verified without a reviewer."""
    status = str(row.get("alignment_status") or "")
    if status == "verified":
        if not str(row.get("reviewer") or "").strip():
            raise ValueError(
                f"{row.get('outcome_id')}: verified status requires a nonempty reviewer"
            )
        if not str(row.get("reviewed_at") or "").strip():
            raise ValueError(
                f"{row.get('outcome_id')}: verified status requires reviewed_at"
            )


@lru_cache(maxsize=4)
def load_alignment(path: str | None = None) -> tuple[dict[str, Any], ...]:
    del path
    LOGGER.info(
        "No separate alignment YAML; CBSE curriculum is native NCERT (%s)",
        SOURCE_NCERT,
    )
    return ()


def outcome_ids_for(
    *,
    grade: int,
    subject: str,
    unit_slug: str,
    section_title: str = "",
    text: str = "",
) -> tuple[list[str], str, str]:
    """Return empty CISCE-shaped ids; native NCERT uses ``alignment_status``."""
    del grade, subject, unit_slug, section_title, text
    return [], "none", ALIGNMENT_STATUS_NATIVE
