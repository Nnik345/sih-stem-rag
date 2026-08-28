#!/usr/bin/env python3
"""Download Core Knowledge Foundation Grade 1-3 Mathematics and Science PDFs."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

BASE_URL = "https://www.coreknowledge.org"
AJAX_URL = f"{BASE_URL}/wp-admin/admin-ajax.php"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "core_knowledge_stem"
REQUEST_DELAY = 1.0
MAX_RETRIES = 3

GRADE_SUBJECT_FILTERS = [
    (1, "mathematics", "subject-06-math", "grade1-03"),
    (1, "science", "subject-03-science", "grade1-03"),
    (2, "mathematics", "subject-06-math", "grade2-04"),
    (2, "science", "subject-03-science", "grade2-04"),
    (3, "mathematics", "subject-06-math", "grade3-05"),
    (3, "science", "subject-03-science", "grade3-05"),
]

STUDENT_TYPES = {
    "student workbook",
    "student book",
    "student reader",
    "student activity book",
    "student activity pages",
}

TEACHER_TYPES = {
    "teacher guide",
    "teacher support",
    "answer key",
    "answer keys",
}


@dataclass
class Resource:
    resource_type: str
    pdf_url: str
    category: str
    filename: str


@dataclass
class Unit:
    grade: int
    subject: str
    unit_number: int | None
    unit_title: str
    full_title: str
    unit_page: str
    unit_dir_name: str
    resources: list[Resource] = field(default_factory=list)


@dataclass
class DownloadResult:
    resource: Resource
    unit: Unit
    local_path: str
    downloaded: bool
    error: str | None = None


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_url(url: str, data: bytes | None = None, retries: int = MAX_RETRIES) -> bytes:
    headers = {
        "User-Agent": "CoreKnowledgeStemDownloader/1.0 (local research use)",
        "Accept": "text/html,application/pdf,*/*",
    }
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise
            log(f"  Retry {attempt}/{retries} for {url}: {exc}")
            time.sleep(REQUEST_DELAY * attempt)
    raise RuntimeError(f"Failed to fetch {url}")


def normalize_slug(text: str) -> str:
    text = text.lower()
    text = text.replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text


def classify_resource(resource_type: str) -> str:
    normalized = resource_type.strip().lower()
    if normalized in STUDENT_TYPES or "student" in normalized:
        return "student"
    if normalized in TEACHER_TYPES or "teacher" in normalized or "answer" in normalized:
        return "teacher"
    return "other"


def resource_type_to_filename(resource_type: str, existing: set[str]) -> str:
    base = normalize_slug(resource_type)
    if not base:
        base = "resource"
    filename = f"{base}.pdf"
    if filename not in existing:
        existing.add(filename)
        return filename
    counter = 2
    while True:
        candidate = f"{base}_{counter}.pdf"
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        counter += 1


def parse_unit_title(full_title: str) -> tuple[int | None, str]:
    unit_match = re.match(r"CK(?:Math|Sci)\s+Unit\s+(\d+)\s*:\s*(.+)", full_title, re.I)
    if unit_match:
        return int(unit_match.group(1)), unit_match.group(2).strip()
    return None, full_title.strip()


def extract_full_title(html: str) -> str:
    for pattern in (
        r"<h2>([^<]+)</h2>",
        r'<meta property="og:title" content="([^<]+?) - Core Knowledge Foundation"',
        r"<title>([^<]+?) - Core Knowledge Foundation</title>",
    ):
        match = re.search(pattern, html, re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def fetch_curriculum_units(grade: int, subject: str, subject_slug: str, grade_slug: str) -> list[tuple[str, str]]:
    data = urllib.parse.urlencode(
        [
            ("action", "dcms_ajax_directory"),
            ("elevaPostType", "library"),
            ("elevaTax", "subject"),
            ("elevaTaxII", "grade"),
            ("elevaTermArray[]", subject_slug),
            ("elevaTermIIArray[]", grade_slug),
            ("keyword", ""),
        ]
    ).encode()
    html = fetch_url(AJAX_URL, data=data).decode("utf-8", errors="replace")
    titles = re.findall(r"<h3 class=\"h4\">([^<]+)</h3>", html)
    urls = re.findall(r'href="(https://www\.coreknowledge\.org/free-resource/[^"]+)"', html)
    if len(titles) != len(urls):
        pairs = [(None, url) for url in urls]
    else:
        pairs = list(zip(titles, urls))
    return [(title.strip() if title else "", url) for title, url in pairs]


def parse_unit_page(grade: int, subject: str, listing_title: str, unit_page: str) -> Unit:
    html = fetch_url(unit_page).decode("utf-8", errors="replace")
    full_title = extract_full_title(html) or listing_title
    unit_number, unit_title = parse_unit_title(full_title)

    cards = re.findall(
        r"individual-resources-card-contain.*?<span class=\"type\">([^<]+)</span>"
        r".*?href=\"([^\"]+\.pdf)\"[^>]*>Download Free PDF Version",
        html,
        re.DOTALL | re.I,
    )

    if unit_number is None:
        dir_name = f"unit_00_{normalize_slug(unit_title)}"
    else:
        dir_name = f"unit_{unit_number:02d}_{normalize_slug(unit_title)}"

    used_filenames: set[str] = set()
    resources: list[Resource] = []
    seen_urls: set[str] = set()

    for resource_type, pdf_url in cards:
        if not pdf_url.startswith(BASE_URL):
            continue
        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)
        category = classify_resource(resource_type)
        filename = resource_type_to_filename(resource_type, used_filenames)
        resources.append(
            Resource(
                resource_type=normalize_slug(resource_type).replace("_", " "),
                pdf_url=pdf_url,
                category=category,
                filename=filename,
            )
        )

    return Unit(
        grade=grade,
        subject=subject,
        unit_number=unit_number,
        unit_title=unit_title,
        full_title=full_title,
        unit_page=unit_page,
        unit_dir_name=dir_name,
        resources=resources,
    )


def download_pdf(pdf_url: str, dest: Path) -> None:
    content = fetch_url(pdf_url)
    if not content.startswith(b"%PDF"):
        raise ValueError("Downloaded content is not a PDF")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


def discover_all_units() -> list[Unit]:
    units: list[Unit] = []
    for grade, subject, subject_slug, grade_slug in GRADE_SUBJECT_FILTERS:
        log(f"Discovering Grade {grade} {subject.title()} units...")
        listings = fetch_curriculum_units(grade, subject, subject_slug, grade_slug)
        time.sleep(REQUEST_DELAY)
        for listing_title, unit_page in listings:
            log(f"  Parsing {listing_title or unit_page}")
            unit = parse_unit_page(grade, subject, listing_title, unit_page)
            units.append(unit)
            time.sleep(REQUEST_DELAY)
    return units


def download_units(units: list[Unit]) -> tuple[list[dict], list[dict], list[dict]]:
    manifest: list[dict] = []
    failed: list[dict] = []
    no_pdfs: list[dict] = []
    seen_pdf_urls: set[str] = set()

    for unit in units:
        if not unit.resources:
            no_pdfs.append(
                {
                    "grade": unit.grade,
                    "subject": unit.subject,
                    "unit_title": unit.unit_title,
                    "unit_page": unit.unit_page,
                }
            )
            continue

        for resource in unit.resources:
            if resource.pdf_url in seen_pdf_urls:
                continue
            seen_pdf_urls.add(resource.pdf_url)

            rel_path = (
                f"grade_{unit.grade:02d}/{unit.subject}/{unit.unit_dir_name}/"
                f"{resource.category}/{resource.filename}"
            )
            dest = OUTPUT_DIR / rel_path
            entry = {
                "grade": unit.grade,
                "subject": unit.subject,
                "unit_number": unit.unit_number,
                "unit_title": unit.unit_title,
                "resource_type": resource.resource_type.replace(" ", "_"),
                "filename": resource.filename,
                "local_path": rel_path,
                "unit_page": unit.unit_page,
                "pdf_url": resource.pdf_url,
                "downloaded": False,
            }

            if dest.exists() and dest.read_bytes().startswith(b"%PDF"):
                entry["downloaded"] = True
                manifest.append(entry)
                log(f"  Skipping existing {rel_path}")
                continue

            success = False
            error_msg = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    log(f"  Downloading {rel_path}")
                    download_pdf(resource.pdf_url, dest)
                    entry["downloaded"] = True
                    manifest.append(entry)
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    error_msg = str(exc)
                    log(f"    Attempt {attempt} failed: {exc}")
                    time.sleep(REQUEST_DELAY * attempt)
            if not success:
                entry["error"] = error_msg
                failed.append(entry)
            time.sleep(REQUEST_DELAY)

    return manifest, failed, no_pdfs


def verify_completeness(units: list[Unit], manifest: list[dict]) -> dict:
    local_paths = {entry["local_path"] for entry in manifest if entry.get("downloaded")}
    pdf_urls = {entry["pdf_url"] for entry in manifest if entry.get("downloaded")}

    missing_units: list[str] = []
    missing_pdfs: list[str] = []

    for unit in units:
        unit_prefix = f"grade_{unit.grade:02d}/{unit.subject}/{unit.unit_dir_name}/"
        if unit.resources:
            if not any(path.startswith(unit_prefix) for path in local_paths):
                missing_units.append(unit.unit_page)
        for resource in unit.resources:
            rel_path = (
                f"grade_{unit.grade:02d}/{unit.subject}/{unit.unit_dir_name}/"
                f"{resource.category}/{resource.filename}"
            )
            if resource.pdf_url not in pdf_urls or rel_path not in local_paths:
                missing_pdfs.append(f"{unit.unit_page} -> {resource.pdf_url}")

    return {
        "expected_units": len(units),
        "expected_pdfs": sum(len(u.resources) for u in units),
        "downloaded_pdfs": len([m for m in manifest if m.get("downloaded")]),
        "missing_units": missing_units,
        "missing_pdfs": missing_pdfs,
        "complete": not missing_units and not missing_pdfs,
    }


def format_bytes(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def print_summary(
    units: list[Unit],
    manifest: list[dict],
    failed: list[dict],
    no_pdfs: list[dict],
    verification: dict,
) -> None:
    log("\n" + "=" * 60)
    log("DOWNLOAD SUMMARY")
    log("=" * 60)

    for grade in (1, 2, 3):
        for subject in ("mathematics", "science"):
            grade_units = [u for u in units if u.grade == grade and u.subject == subject]
            grade_manifest = [
                m
                for m in manifest
                if m["grade"] == grade and m["subject"] == subject and m.get("downloaded")
            ]
            log(f"\nGrade {grade} {subject.title()}:")
            log(f"- number of units: {len(grade_units)}")
            log(f"- number of PDFs: {len(grade_manifest)}")

    total_pdfs = len([m for m in manifest if m.get("downloaded")])
    total_size = sum(
        (OUTPUT_DIR / m["local_path"]).stat().st_size
        for m in manifest
        if m.get("downloaded") and (OUTPUT_DIR / m["local_path"]).exists()
    )

    log(f"\nTotal units: {len(units)}")
    log(f"Total PDFs: {total_pdfs}")
    log(f"Total dataset size: {format_bytes(total_size)}")

    if failed:
        log("\nFailed downloads:")
        for item in failed:
            log(f"  - {item['local_path']}: {item.get('error', 'unknown error')}")

    if no_pdfs:
        log("\nUnits with no downloadable PDFs:")
        for item in no_pdfs:
            log(f"  - Grade {item['grade']} {item['subject']}: {item['unit_title']} ({item['unit_page']})")

    auth_required = [m for m in manifest if "login" in str(m.get("error", "")).lower()]
    if auth_required:
        log("\nResources requiring authentication:")
        for item in auth_required:
            log(f"  - {item['pdf_url']}")
    else:
        log("\nResources requiring authentication: none")

    log("\nCompleteness verification:")
    log(f"  Expected units: {verification['expected_units']}")
    log(f"  Expected PDFs: {verification['expected_pdfs']}")
    log(f"  Downloaded PDFs: {verification['downloaded_pdfs']}")
    log(f"  Complete: {verification['complete']}")
    if verification["missing_units"]:
        log("  Missing unit folders:")
        for url in verification["missing_units"]:
            log(f"    - {url}")
    if verification["missing_pdfs"]:
        log("  Missing PDFs:")
        for item in verification["missing_pdfs"]:
            log(f"    - {item}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Starting Core Knowledge STEM curriculum download...")
    log(f"Output directory: {OUTPUT_DIR}")

    units = discover_all_units()
    log(f"\nDiscovered {len(units)} curriculum entries")

    manifest, failed, no_pdfs = download_units(units)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"\nWrote manifest to {manifest_path}")

    verification = verify_completeness(units, manifest)

    # Re-crawl for verification
    log("\nRe-crawling curriculum listing for completeness check...")
    rediscovered = discover_all_units()
    rediscovered_urls = {u.unit_page for u in rediscovered}
    original_urls = {u.unit_page for u in units}
    if rediscovered_urls != original_urls:
        log("WARNING: Curriculum listing changed between initial crawl and verification")

    print_summary(units, manifest, failed, no_pdfs, verification)

    if failed or not verification["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
