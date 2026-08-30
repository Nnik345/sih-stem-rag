"""Load the CISCE Grade 3–5 STEM alignment crosswalk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .logging_utils import get_logger

LOGGER = get_logger(__name__)

DEFAULT_ALIGNMENT_PATH = (
    PROJECT_ROOT / "curriculum" / "alignment" / "cisce_grade_3_5_stem.yaml"
)


def _parse_simple_yaml(text: str) -> list[dict[str, Any]]:
    """Minimal YAML list-of-maps parser for the committed alignment file."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    list_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("outcomes:"):
            continue
        if raw.startswith("  - "):
            if current:
                records.append(current)
            current = {}
            list_key = None
            rest = raw[4:]
            if ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = _scalar(value)
            continue
        if current is None:
            continue
        if raw.startswith("      - "):
            if list_key:
                current.setdefault(list_key, []).append(_scalar(raw[8:]))
            continue
        if raw.startswith("    ") and ":" in raw:
            key, value = raw.strip().split(":", 1)
            if not value.strip():
                list_key = key
                current[key] = []
            else:
                list_key = None
                current[key] = _scalar(value)
    if current:
        records.append(current)
    return records


def _scalar(value: str) -> Any:
    text = value.strip().strip('"').strip("'")
    if text in {"[]", ""}:
        return [] if text == "[]" else ""
    if text.isdigit():
        return int(text)
    return text


def validate_alignment_row(row: dict[str, Any]) -> None:
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
    alignment_path = Path(path) if path else DEFAULT_ALIGNMENT_PATH
    if not alignment_path.is_file():
        LOGGER.warning("No CISCE alignment file at %s", alignment_path)
        return ()
    records = tuple(_parse_simple_yaml(alignment_path.read_text(encoding="utf-8")))
    for row in records:
        validate_alignment_row(row)
    LOGGER.info("Loaded %d CISCE alignment outcomes from %s", len(records), alignment_path)
    return records


def _unit_matches(unit_slug: str, mapped: str) -> bool:
    """Match full slugs against truncated crosswalk entries (and vice versa)."""
    left = unit_slug.strip().lower()
    right = mapped.strip().lower()
    if not left or not right:
        return False
    return left == right or left in right or right in left


def outcome_ids_for(
    *,
    grade: int,
    subject: str,
    unit_slug: str,
    section_title: str = "",
    text: str = "",
) -> tuple[list[str], str, str]:
    """Return (outcome_ids, granularity, alignment_status).

    When an outcome lists ``section_patterns``, the section title or body must
    match at least one pattern. ``section_exclude_patterns`` drop a match even
    if an allowlist pattern also hits. Math modules without patterns stay
    unit-level. Outcomes with empty ``mapped_units`` never attach.
    """
    matches: list[str] = []
    granularities: list[str] = []
    haystack = f"{section_title}\n{text}".lower()
    for row in load_alignment():
        if int(row.get("grade") or 0) != grade:
            continue
        if str(row.get("subject") or "").lower() != subject.lower():
            continue
        mapped = row.get("mapped_units") or []
        if not isinstance(mapped, list):
            mapped = [mapped]
        if not any(_unit_matches(unit_slug, str(item)) for item in mapped):
            continue
        excludes = row.get("section_exclude_patterns") or []
        if not isinstance(excludes, list):
            excludes = [excludes]
        excludes = [str(p).strip().lower() for p in excludes if str(p).strip()]
        if excludes and any(p in haystack for p in excludes):
            continue
        patterns = row.get("section_patterns") or []
        if not isinstance(patterns, list):
            patterns = [patterns]
        patterns = [str(p).strip().lower() for p in patterns if str(p).strip()]
        if patterns:
            if not any(p in haystack for p in patterns):
                continue
            granularities.append("section")
        else:
            granularities.append("unit")
        oid = str(row.get("outcome_id") or "")
        if oid:
            matches.append(oid)
    if not matches:
        return [], "none", "unmapped"
    granularity = "section" if "section" in granularities else "unit"
    return matches, granularity, "needs_human_review"
