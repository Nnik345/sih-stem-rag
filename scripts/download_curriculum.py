#!/usr/bin/env python3
"""Download official English NCERT STEM textbooks (CBSE classes 1–12).

Only URLs in ``rag.curriculum_catalog`` on ``ncert.nic.in`` are used. There is
no mirror fallback. Each catalog row is a complete-book zip; every chapter PDF
is extracted into the book's ``student/`` directory.

    python scripts/download_curriculum.py
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.curriculum_catalog import (  # noqa: E402
    SourceFile,
    all_source_files,
    ingestible_files,
    is_answers_member,
    is_chapter_pdf,
)

USER_AGENT = "sih-stem-rag-curriculum-downloader/1.0 (local research; no login)"
TIMEOUT_S = 180
MAX_RETRIES = 4
RETRY_DELAY_S = 3.0
CHUNK = 1 << 20
PDF_MAGIC = b"%PDF"
ZIP_MAGIC = b"PK\x03\x04"
HTML_HINTS = (b"<!doctype html", b"<html", b"<head")

ALLOWED_TYPES = {
    "pdf": (
        "application/pdf",
        "application/x-pdf",
        "binary/octet-stream",
        "application/octet-stream",
    ),
    "zip": (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ),
}

ALLOWED_HOST_SUFFIX = "ncert.nic.in"


class DownloadError(RuntimeError):
    """A required source could not be fetched or validated."""


def _ssl_context() -> ssl.SSLContext:
    """Verify TLS with certifi when available. Never disable verification."""
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except Exception:
        pass
    return context


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        method="GET",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_like_html(path: Path) -> bool:
    head = path.read_bytes()[:512].lstrip().lower()
    return any(hint in head for hint in HTML_HINTS)


def _validate_payload(path: Path, *, kind: str) -> None:
    if not path.is_file() or path.stat().st_size < 32:
        raise DownloadError(f"{path} is missing or too small")
    if _looks_like_html(path):
        raise DownloadError(f"{path} is an HTML error page, not {kind}")
    magic = path.read_bytes()[:8]
    if kind == "pdf" and not magic.startswith(PDF_MAGIC):
        raise DownloadError(f"{path} does not start with %PDF")
    if kind == "zip" and not magic.startswith(ZIP_MAGIC):
        raise DownloadError(f"{path} is not a ZIP container")


def _assert_official_url(url: str, file_id: str) -> None:
    lowered = url.lower()
    if "archive.org" in lowered:
        raise DownloadError(f"{file_id}: Internet Archive is not an approved source.")
    host = urllib.parse.urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host.endswith(ALLOWED_HOST_SUFFIX):
        raise DownloadError(
            f"{file_id}: URL host {host!r} is not {ALLOWED_HOST_SUFFIX}"
        )


def fetch_to(url: str, destination: Path, *, expected_kind: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                _request(url), timeout=TIMEOUT_S, context=_ssl_context()
            ) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise DownloadError(f"{url} returned HTTP {status}")
                content_type = (
                    (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                )
                allowed = ALLOWED_TYPES[expected_kind]
                if (
                    content_type
                    and content_type not in allowed
                    and not content_type.startswith("application/")
                ):
                    raise DownloadError(
                        f"{url} content-type {content_type!r} is not a valid {expected_kind}"
                    )
                with tmp.open("wb") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
            _validate_payload(tmp, kind=expected_kind)
            tmp.replace(destination)
            return
        except (urllib.error.URLError, DownloadError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)
    raise DownloadError(f"Failed to download {url}: {last_error}")


def _existing_chapters(dest_dir: Path, record: SourceFile) -> list[Path]:
    if not dest_dir.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(dest_dir.glob("*.pdf")):
        if is_chapter_pdf(path.name, record.ncert_code):
            try:
                _validate_payload(path, kind="pdf")
            except DownloadError:
                continue
            found.append(path)
    return found


def _extract_chapter_pdfs(zip_path: Path, record: SourceFile, dest_dir: Path) -> list[Path]:
    skip = tuple(s.lower() for s in record.skip_member_substrings)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            lowered = name.lower()
            if any(token in lowered for token in skip):
                continue
            if is_answers_member(name):
                continue
            if not is_chapter_pdf(name, record.ncert_code):
                continue
            out = dest_dir / Path(name).name.lower()
            with archive.open(name) as src, out.open("wb") as handle:
                while True:
                    block = src.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
            _validate_payload(out, kind="pdf")
            written.append(out)
    if not written:
        raise DownloadError(
            f"No chapter PDFs matching {record.ncert_code}{{nn}}.pdf in {zip_path.name}"
        )
    return written


def download_record(corpus_root: Path, record: SourceFile) -> dict[str, object]:
    dest_dir = corpus_root / record.local_path
    url = record.direct_download_url or ""
    _assert_official_url(url, record.file_id)

    existing = _existing_chapters(dest_dir, record)
    if existing:
        combined = hashlib.sha256()
        for path in existing:
            combined.update(_sha256(path).encode("ascii"))
        return {
            "file_id": record.file_id,
            "status": "skipped",
            "local_path": record.local_path,
            "sha256": combined.hexdigest(),
            "bytes": sum(p.stat().st_size for p in existing),
            "chapters": [str(p.relative_to(corpus_root)) for p in existing],
        }

    zip_path = dest_dir.parent / f"{record.ncert_code}dd.zip"
    fetch_to(url, zip_path, expected_kind="zip")
    chapters = _extract_chapter_pdfs(zip_path, record, dest_dir)
    zip_digest = _sha256(zip_path)
    zip_path.unlink(missing_ok=True)
    combined = hashlib.sha256()
    for path in chapters:
        combined.update(_sha256(path).encode("ascii"))
    return {
        "file_id": record.file_id,
        "status": "downloaded",
        "local_path": record.local_path,
        "sha256": combined.hexdigest(),
        "zip_sha256": zip_digest,
        "bytes": sum(p.stat().st_size for p in chapters),
        "chapters": [str(p.relative_to(corpus_root)) for p in chapters],
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _yaml_scalar(value: object) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if any(ch in text for ch in ":#{}[]&*!?'\"\n"):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text or '""'


def write_sources_yaml(path: Path, records: list[SourceFile], results: dict[str, dict]) -> None:
    lines = [
        "# Machine-readable STEM curriculum source manifest.",
        "# Hashes are filled by scripts/download_curriculum.py.",
        "# Official English NCERT textbooks (CBSE classes 1-12).",
        "",
        f"generated_at: {_yaml_scalar(datetime.now(timezone.utc).isoformat(timespec='seconds'))}",
        "files:",
        "",
    ]
    for record in records:
        result = results.get(record.file_id, {})
        payload = record.to_manifest_dict()
        payload["sha256"] = result.get("sha256", record.sha256)
        payload["retrieved_at"] = result.get("retrieved_at", record.retrieved_at)
        payload["bytes"] = result.get("bytes", "")
        payload["download_status"] = result.get("status", "")
        payload["chapters"] = result.get("chapters", [])
        lines.append(f"  - file_id: {record.file_id}")
        for key, value in payload.items():
            if key == "file_id":
                continue
            if isinstance(value, (list, tuple)):
                lines.append(f"    {key}:")
                if not value:
                    lines.append("      []")
                else:
                    for item in value:
                        lines.append(f"      - {_yaml_scalar(item)}")
            else:
                lines.append(f"    {key}: {_yaml_scalar(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_checksums(path: Path, results: dict[str, dict]) -> None:
    rows = []
    for file_id, result in sorted(results.items()):
        digest = result.get("sha256")
        local_path = result.get("local_path")
        if digest and local_path:
            rows.append(f"{digest}  {local_path}")
        for chapter in result.get("chapters") or []:
            rows.append(f"# chapter  {chapter}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=PROJECT_ROOT / "curriculum",
        help="Curriculum root (default: ./curriculum)",
    )
    args = parser.parse_args()
    corpus_root = args.corpus_root.resolve()
    records = all_source_files()
    results: dict[str, dict] = {}
    failures: list[str] = []

    print(f"Curriculum root: {corpus_root}")
    print(f"Books to fetch : {len(records)}")
    print()

    for record in records:
        print(f"[{record.file_id}] {record.direct_download_url}")
        try:
            result = download_record(corpus_root, record)
            results[record.file_id] = result
            n_chapters = len(result.get("chapters") or [])
            print(
                f"  {result['status']}  {n_chapters} chapters  "
                f"{result['bytes']} bytes  sha256={result['sha256'][:12]}…"
            )
        except DownloadError as exc:
            failures.append(f"{record.file_id}: {exc}")
            print(f"  ERROR: {exc}")

    write_sources_yaml(corpus_root / "manifests" / "sources.yaml", records, results)
    write_checksums(corpus_root / "manifests" / "checksums.sha256", results)

    print()
    if failures:
        print("FAILED required sources:")
        for item in failures:
            print(f"  - {item}")
        print("No mirror was substituted. Fix the official URL and re-run.")
        return 1

    ingestible = ingestible_files()
    print(f"All {len(records)} book zips validated.")
    print(f"Ingestible books: {len(ingestible)}")
    print(f"Manifest: {corpus_root / 'manifests' / 'sources.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
