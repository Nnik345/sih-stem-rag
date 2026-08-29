#!/usr/bin/env python3
"""Download the approved CISCE-aligned Grade 3–5 STEM sources.

Only the URLs in ``rag.curriculum_catalog`` are used. There is no mirror
fallback. CISCE is stored under ``raw/_alignment_only/`` and is never ingested.

    python scripts/download_curriculum.py
    python scripts/download_curriculum.py --skip-alignment
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import time
import urllib.error
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
)

USER_AGENT = "sih-stem-rag-curriculum-downloader/1.0 (local research; no login)"
TIMEOUT_S = 120
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
    "epub": (
        "application/epub+zip",
        "application/zip",
        "application/octet-stream",
        "application/x-zip-compressed",
    ),
    "zip": (
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    ),
}


class DownloadError(RuntimeError):
    """A required source could not be fetched or validated."""


AIA_CACHE = PROJECT_ROOT / "curriculum" / "processed" / "tls" / "globalsign-rsa-ov-ssl-ca-2018.pem"


def _ssl_context() -> ssl.SSLContext:
    """Verify TLS. Never disable verification.

    ``*.nysed.gov`` currently omits its GlobalSign intermediate. The intermediate
    is fetched from the certificate's official AIA URL (GlobalSign), not from
    a curriculum mirror, and is combined with certifi's CA bundle.
    """
    context = ssl.create_default_context()
    try:
        import certifi

        context.load_verify_locations(certifi.where())
    except Exception:
        pass
    intermediate = _globalsign_intermediate_pem()
    if intermediate:
        context.load_verify_locations(cadata=intermediate)
    return context


def _globalsign_intermediate_pem() -> str:
    from rag.curriculum_catalog import GLOBALSIGN_RSA_OV_SSL_CA_2018

    cache = AIA_CACHE
    if cache.is_file() and cache.stat().st_size > 200:
        return cache.read_text(encoding="ascii", errors="ignore")
    try:
        req = urllib.request.Request(
            GLOBALSIGN_RSA_OV_SSL_CA_2018,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            der = response.read()
    except Exception:
        return ""
    if not der:
        return ""
    if der.lstrip().startswith(b"-----"):
        pem = der.decode("ascii", "ignore")
    else:
        import base64

        pem = (
            "-----BEGIN CERTIFICATE-----\n"
            + base64.encodebytes(der).decode("ascii")
            + "-----END CERTIFICATE-----\n"
        )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(pem, encoding="ascii")
    except OSError:
        pass
    return pem


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
    if kind in {"epub", "zip"} and not magic.startswith(ZIP_MAGIC):
        raise DownloadError(f"{path} is not a ZIP/ePUB container")
    if kind == "epub":
        with zipfile.ZipFile(path) as archive:
            if "mimetype" not in archive.namelist():
                raise DownloadError(f"{path} is a zip but not an ePUB")
            mime = archive.read("mimetype").decode("ascii", "ignore").strip()
            if mime != "application/epub+zip":
                raise DownloadError(f"{path} mimetype is {mime!r}, not ePUB")


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
                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                allowed = ALLOWED_TYPES[expected_kind]
                if content_type and content_type not in allowed and not content_type.startswith("application/"):
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


def _extract_zip_member(zip_path: Path, record: SourceFile, dest: Path) -> None:
    skip = tuple(s.lower() for s in record.skip_member_substrings)
    needle = (record.zip_member_glob or "*.pdf").replace("*", "").lower()
    with zipfile.ZipFile(zip_path) as archive:
        candidates = []
        for name in archive.namelist():
            lowered = name.lower()
            if name.endswith("/"):
                continue
            if any(token in lowered for token in skip):
                continue
            if needle and needle not in lowered:
                continue
            candidates.append(name)
        if not candidates:
            raise DownloadError(
                f"No member matching {record.zip_member_glob!r} in {zip_path.name}"
            )
        # Prefer the shortest path (usually the top-level full-module PDF).
        member = sorted(candidates, key=lambda n: (n.count("/"), len(n)))[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, dest.open("wb") as out:
            while True:
                block = src.read(CHUNK)
                if not block:
                    break
                out.write(block)
    _validate_payload(dest, kind="pdf")


def download_record(corpus_root: Path, record: SourceFile) -> dict[str, object]:
    dest = corpus_root / record.local_path
    kind = "zip" if record.zip_member_glob else record.file_format
    url = record.direct_download_url or ""
    if dest.is_file() and not _looks_like_html(dest):
        try:
            _validate_payload(
                dest, kind=record.file_format if dest.suffix != ".zip" else "pdf"
            )
            return {
                "file_id": record.file_id,
                "status": "skipped",
                "local_path": record.local_path,
                "sha256": _sha256(dest),
                "bytes": dest.stat().st_size,
            }
        except DownloadError:
            dest.unlink(missing_ok=True)
    if "archive.org" in url:
        raise DownloadError(
            f"{record.file_id}: Internet Archive is not an approved source."
        )
    if "nysed.sharepoint.com" in url:
        raise DownloadError(
            f"{record.file_id}: NYSED does not publish a login-free file URL. "
            f"Place the official full-module PDF at {record.local_path} "
            f"(from {record.official_page_url} / SharePoint) and re-run."
        )

    if record.zip_member_glob:
        zip_path = dest.with_suffix(".zip")
        fetch_to(record.direct_download_url, zip_path, expected_kind="zip")
        _extract_zip_member(zip_path, record, dest)
        zip_path.unlink(missing_ok=True)
    else:
        fetch_to(record.direct_download_url, dest, expected_kind=record.file_format)

    return {
        "file_id": record.file_id,
        "status": "downloaded",
        "local_path": record.local_path,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
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
        "# CISCE is alignment-only and must never be ingested.",
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
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=PROJECT_ROOT / "curriculum",
        help="Curriculum root (default: ./curriculum)",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="Do not download the CISCE PDF (still required for page-referenced alignment).",
    )
    args = parser.parse_args()
    corpus_root = args.corpus_root.resolve()
    records = all_source_files(include_alignment=not args.skip_alignment)
    results: dict[str, dict] = {}
    failures: list[str] = []

    print(f"Curriculum root: {corpus_root}")
    print(f"Files to fetch : {len(records)}")
    print()

    for record in records:
        label = "alignment-only" if record.alignment_only else "ingest"
        print(f"[{record.file_id}] {label} {record.direct_download_url}")
        try:
            result = download_record(corpus_root, record)
            results[record.file_id] = result
            print(
                f"  {result['status']}  {result['bytes']} bytes  sha256={result['sha256'][:12]}…"
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
    print(f"All {len(records)} sources validated.")
    print(f"Ingestible files: {len(ingestible)}")
    print(f"Manifest: {corpus_root / 'manifests' / 'sources.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
