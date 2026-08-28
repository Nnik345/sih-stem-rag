"""Hierarchical, token-aware chunking.

Chunks are built *inside* sections, never across them, so the
Chunk -> Section -> Page -> Document -> Unit -> Subject -> Grade lineage stays
intact. Boundaries are chosen in this order of preference:

1. section boundaries (headings) -- enforced, a chunk never spans two sections
2. paragraph boundaries -- preferred split point
3. sentence boundaries -- used only when a single paragraph exceeds the ceiling

Token counts come from the BGE-M3 tokenizer so the numbers mean the same thing
to the chunker and to the embedder. If the tokenizer is unavailable a coarse
word-based estimate is used and a warning is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import ChunkingConfig
from .logging_utils import get_logger
from .schemas import (
    Chunk,
    ParsedDocument,
    ParsedSection,
    chunk_id_for,
    page_id_for,
)

LOGGER = get_logger(__name__)

# Sentence boundary: terminator followed by whitespace and a capital/quote/digit.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
# Average XLM-R tokens per whitespace word for English curriculum prose. Only
# used when the real tokenizer cannot be loaded.
_FALLBACK_TOKENS_PER_WORD = 1.35
# Text below this is page furniture (stray page numbers, captions, single words).
_MIN_CHUNK_CHARS = 80


class TokenCounter:
    """Counts tokens with the BGE-M3 tokenizer, loaded lazily and cached."""

    def __init__(self, tokenizer_path: Path | None = None) -> None:
        self._tokenizer_path = tokenizer_path
        self._tokenizer = None
        self._load_attempted = False
        self._warned = False

    @property
    def is_exact(self) -> bool:
        self._ensure_loaded()
        return self._tokenizer is not None

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if self._tokenizer_path is None:
            return
        if not Path(self._tokenizer_path).is_dir():
            LOGGER.warning(
                "Tokenizer directory %s not found; chunk sizes will be estimated. "
                "Run: python scripts/download_retrieval_models.py",
                self._tokenizer_path,
            )
            return
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._tokenizer_path), use_fast=True
            )
            LOGGER.info("Loaded chunking tokenizer from %s", self._tokenizer_path)
        except Exception as exc:
            LOGGER.warning(
                "Could not load tokenizer from %s (%s); chunk sizes will be estimated",
                self._tokenizer_path,
                exc,
            )

    def count(self, text: str) -> int:
        self._ensure_loaded()
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        if not self._warned:
            LOGGER.warning("Using estimated token counts (tokenizer unavailable)")
            self._warned = True
        words = len(text.split())
        return max(1, round(words * _FALLBACK_TOKENS_PER_WORD))


@dataclass
class _Unit:
    """Smallest indivisible piece of text the packer works with."""

    text: str
    page_number: int
    tokens: int


@dataclass
class _Group:
    """One chunk's worth of units, plus any carried-over overlap text.

    ``prefix`` holds the tail of the previous chunk when no whole unit was small
    enough to fit the overlap budget, which is common in these curriculum PDFs
    where a single paragraph can be longer than the configured overlap.
    """

    indices: list[int]
    prefix: str = ""
    prefix_page: int | None = None


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text)]
    return [part for part in parts if part]


def _trailing_text(text: str, budget: int, counter: TokenCounter) -> str:
    """The last whole sentences of ``text`` fitting in ``budget`` tokens.

    Falls back to trailing words when even one sentence is too long, so an
    overlap is always produced.
    """
    if budget <= 0:
        return ""
    sentences = split_sentences(text)
    kept: list[str] = []
    used = 0
    for sentence in reversed(sentences):
        tokens = counter.count(sentence)
        if used + tokens > budget:
            break
        kept.insert(0, sentence)
        used += tokens
    if kept:
        return " ".join(kept)

    words = text.split()
    # Word count is an upper bound on the token budget, then trimmed to fit.
    tail = words[-budget:]
    while tail and counter.count(" ".join(tail)) > budget:
        tail = tail[1:]
    return " ".join(tail)


def _units_for_section(
    section: ParsedSection,
    counter: TokenCounter,
    config: ChunkingConfig,
) -> list[_Unit]:
    """Paragraphs as units, splitting any paragraph that exceeds the ceiling."""
    units: list[_Unit] = []
    for paragraph in section.paragraphs:
        tokens = counter.count(paragraph.text)
        if tokens <= config.max_tokens:
            units.append(
                _Unit(
                    text=paragraph.text,
                    page_number=paragraph.page_number,
                    tokens=tokens,
                )
            )
            continue

        # Oversized paragraph: fall back to sentence boundaries.
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in split_sentences(paragraph.text) or [paragraph.text]:
            sentence_tokens = counter.count(sentence)
            if buffer and buffer_tokens + sentence_tokens > config.target_tokens:
                units.append(
                    _Unit(
                        text=" ".join(buffer),
                        page_number=paragraph.page_number,
                        tokens=buffer_tokens,
                    )
                )
                buffer, buffer_tokens = [], 0
            buffer.append(sentence)
            buffer_tokens += sentence_tokens
        if buffer:
            units.append(
                _Unit(
                    text=" ".join(buffer),
                    page_number=paragraph.page_number,
                    tokens=buffer_tokens,
                )
            )
    return units


def _pack_units(
    units: list[_Unit], counter: TokenCounter, config: ChunkingConfig
) -> list[_Group]:
    """Group units into chunks of ~target_tokens with token overlap.

    Overlap prefers re-using whole trailing units, so the shared text is
    physically the same paragraphs. When the previous unit alone exceeds the
    overlap budget, its trailing sentences are carried over as text instead.
    """
    if not units:
        return []

    total = sum(unit.tokens for unit in units)
    # A short, coherent section stays whole even if slightly over target.
    if total <= config.max_tokens:
        return [_Group(indices=list(range(len(units))))]

    groups: list[_Group] = []
    current: list[int] = []
    current_tokens = 0
    prefix = ""
    prefix_page: int | None = None

    for index, unit in enumerate(units):
        would_be = current_tokens + unit.tokens
        if current and would_be > config.target_tokens:
            groups.append(_Group(indices=current, prefix=prefix, prefix_page=prefix_page))

            overlap: list[int] = []
            overlap_tokens = 0
            for prev_index in reversed(current):
                prev_tokens = units[prev_index].tokens
                if overlap_tokens + prev_tokens > config.overlap_tokens:
                    break
                overlap.insert(0, prev_index)
                overlap_tokens += prev_tokens

            if overlap:
                current, current_tokens = list(overlap), overlap_tokens
                prefix, prefix_page = "", None
            else:
                last = units[current[-1]]
                prefix = _trailing_text(last.text, config.overlap_tokens, counter)
                prefix_page = last.page_number if prefix else None
                current, current_tokens = [], counter.count(prefix) if prefix else 0

        current.append(index)
        current_tokens += unit.tokens

    if current:
        # Avoid emitting a stub tail: fold it into the previous group instead.
        if groups and current_tokens < config.min_tokens:
            previous = groups[-1]
            previous.indices = previous.indices + [
                i for i in current if i not in previous.indices
            ]
        else:
            groups.append(_Group(indices=current, prefix=prefix, prefix_page=prefix_page))

    return groups


def chunk_section(
    document: ParsedDocument,
    section: ParsedSection,
    counter: TokenCounter,
    config: ChunkingConfig,
    *,
    start_chunk_index: int,
) -> list[Chunk]:
    """Build the chunks for a single section."""
    units = _units_for_section(section, counter, config)
    metadata = document.metadata
    chunks: list[Chunk] = []

    for local_index, group in enumerate(_pack_units(units, counter, config)):
        ordered = sorted(dict.fromkeys(group.indices))
        body = "\n\n".join(units[i].text for i in ordered).strip()
        text = f"{group.prefix}\n\n{body}".strip() if group.prefix else body
        if len(text) < _MIN_CHUNK_CHARS or not any(ch.isalpha() for ch in text):
            continue

        page_numbers = {units[i].page_number for i in ordered}
        if group.prefix_page is not None:
            page_numbers.add(group.prefix_page)
        pages = sorted(page_numbers)
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(section.section_id, local_index),
                text=text,
                token_count=counter.count(text),
                chunk_index=start_chunk_index + len(chunks),
                section_id=section.section_id,
                section_title=section.title,
                section_index=section.section_index,
                page_start=pages[0],
                page_end=pages[-1],
                page_ids=[page_id_for(metadata.document_id, p) for p in pages],
                document_id=metadata.document_id,
                document_title=metadata.document_title,
                unit_id=metadata.unit_id,
                unit_title=metadata.unit_title,
                subject_id=metadata.subject_id,
                grade_id=metadata.grade_id,
                grade=metadata.grade,
                subject=metadata.subject,
                resource_type=metadata.resource_type,
                audience=metadata.audience,
                local_pdf_path=str(metadata.local_pdf_path),
            )
        )
    return chunks


def chunk_document(
    document: ParsedDocument,
    counter: TokenCounter,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Chunk every section of a parsed document, in document order."""
    chunks: list[Chunk] = []
    for section in document.sections:
        chunks.extend(
            chunk_section(
                document,
                section,
                counter,
                config,
                start_chunk_index=len(chunks),
            )
        )

    if not chunks:
        LOGGER.warning(
            "No chunks produced for %s (%d sections, %d pages without text)",
            document.metadata.relative_pdf_path,
            len(document.sections),
            document.pages_without_text,
        )
    else:
        token_counts = [c.token_count for c in chunks]
        LOGGER.info(
            "Chunked %s into %d chunks (tokens min/mean/max = %d/%.0f/%d)",
            document.metadata.relative_pdf_path,
            len(chunks),
            min(token_counts),
            sum(token_counts) / len(token_counts),
            max(token_counts),
        )
    return chunks
