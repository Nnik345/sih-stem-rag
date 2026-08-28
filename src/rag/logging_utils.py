"""Structured, readable logging for the RAG pipeline.

Every module obtains its logger through :func:`get_logger`; entry-point scripts
call :func:`setup_logging` exactly once. Timing of retrieval and reranking stages
is recorded through :class:`Timer` so diagnostics stay comparable across runs.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure root logging once. Level falls back to $LOG_LEVEL then INFO."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # These libraries are extremely chatty at INFO and would bury our own logs.
    for noisy in ("neo4j", "urllib3", "filelock", "huggingface_hub", "fsspec"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    # Neo4j emits WARNING notifications for labels and properties that do not
    # exist yet, which is normal while ingesting into a fresh graph.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class Timer:
    """Wall-clock timer for pipeline stages."""

    __slots__ = ("_start", "elapsed_s")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.elapsed_s = 0.0

    def stop(self) -> float:
        self.elapsed_s = time.perf_counter() - self._start
        return self.elapsed_s

    @property
    def elapsed_ms(self) -> float:
        return (self.elapsed_s or (time.perf_counter() - self._start)) * 1000.0


@contextmanager
def log_stage(logger: logging.Logger, stage: str, **context: Any) -> Iterator[Timer]:
    """Log the duration of a named stage, and log-and-reraise on failure."""
    extra = " ".join(f"{k}={v}" for k, v in context.items())
    logger.debug("%s starting %s", stage, extra)
    timer = Timer()
    try:
        yield timer
    except Exception:
        timer.stop()
        logger.error("%s failed after %.1f ms %s", stage, timer.elapsed_ms, extra)
        raise
    timer.stop()
    logger.info("%s finished in %.1f ms %s", stage, timer.elapsed_ms, extra)
