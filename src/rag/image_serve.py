"""Turn stored textbook rasters into files a browser (and Qwen-VL) can display.

NCERT PDFs often embed JPEG2000 or CMYK JPEGs. Those decode in PyMuPDF but show
up as a black rectangle in ``<img>``. Convert once, cache next to the corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .logging_utils import get_logger

LOGGER = get_logger(__name__)

# PNG/GIF/WebP are already browser-safe. JPEG may be CMYK; JPEG2000/TIFF are not
# displayable in Chrome/Firefox at all.
_PASSTHROUGH_SUFFIXES = {".png", ".gif", ".webp"}
BROWSER_PNG_DIRNAME = ".browser-png"


def browser_png_cache_dir(images_dir: Path) -> Path:
    return images_dir / BROWSER_PNG_DIRNAME


def ensure_browser_png(src: Path, *, cache_root: Path) -> Path:
    """Return ``src`` or a cached RGB PNG the UI can render.

    The cache lives under ``images_dir`` so the vision-path sandbox still allows
    the converted file through to the tutor.
    """
    resolved = src.expanduser().resolve()
    if not resolved.is_file():
        return src
    if resolved.suffix.lower() in _PASSTHROUGH_SUFFIXES:
        return resolved
    try:
        digest = hashlib.sha256(
            f"{resolved}:{resolved.stat().st_mtime_ns}".encode()
        ).hexdigest()[:20]
        dest = cache_root / f"{digest}.png"
        if dest.is_file() and dest.stat().st_mtime_ns >= resolved.stat().st_mtime_ns:
            return dest
        payload = _render_png_bytes(resolved)
        cache_root.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        return dest
    except Exception as exc:
        LOGGER.warning("Could not convert %s for the browser (%s)", resolved, exc)
        return resolved


def _render_png_bytes(src: Path) -> bytes:
    import pymupdf

    pix = pymupdf.Pixmap(str(src))
    rgb = None
    try:
        channels = pix.colorspace.n if pix.colorspace is not None else pix.n
        if channels not in (1, 3):
            rgb = pymupdf.Pixmap(pymupdf.csRGB, pix)
            return rgb.tobytes("png")
        return pix.tobytes("png")
    finally:
        pix = None
        rgb = None
