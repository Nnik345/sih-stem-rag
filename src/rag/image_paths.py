"""Sandbox helpers for curriculum figures and student uploads.

Student photos are never stored as :Image nodes. Textbook figures are served
and passed to the tutor only when the resolved path stays under
``config.paths.images_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _resolve(path: str | Path) -> Path | None:
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def is_under_directory(path: Path, root: Path) -> bool:
    """True when ``path`` is ``root`` or a file inside it (after resolve)."""
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return False
    return True


def resolve_curriculum_image(
    path: str | Path | None, images_dir: Path
) -> Path | None:
    """Return a readable file under ``images_dir``, else None."""
    if not path:
        return None
    resolved = _resolve(path)
    if resolved is None or not resolved.is_file():
        return None
    if not is_under_directory(resolved, images_dir):
        return None
    return resolved


def select_vision_image_paths(
    paths: Sequence[str | Path],
    *,
    images_dir: Path,
    extra_allowed: Sequence[str | Path] = (),
) -> list[str]:
    """Keep textbook files under ``images_dir`` plus an optional student upload.

    Extra paths (the student photo) must exist on disk; they are never treated
    as curriculum figures. Unknown or escaped paths are dropped.
    """
    allowed_extra = {p for p in (_resolve(item) for item in extra_allowed) if p is not None}
    kept: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        resolved = _resolve(raw)
        if resolved is None or not resolved.is_file():
            continue
        textbook = is_under_directory(resolved, images_dir)
        student = resolved in allowed_extra
        if not textbook and not student:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        kept.append(key)
    return kept
