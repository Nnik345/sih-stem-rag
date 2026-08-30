"""Corpus -> Neo4j ingestion.

Pipeline per document::

    discover -> parse (PyMuPDF) -> hierarchical chunks -> graph upsert
             -> conservative concept links -> BGE-M3 dense embeddings

Idempotency and resume
----------------------
Every node uses a deterministic identifier (see :mod:`rag.schemas`) and every
write is a ``MERGE``, so re-running ingestion updates in place instead of
duplicating. The ``:Document`` node stores the PDF's content hash, the chunk
count and the embedding version, which together act as the resume checkpoint: an
unchanged, fully embedded document is skipped without being reopened. The
database is never wiped implicitly -- that requires the explicit ``--reset``
flag on ``scripts/ingest_corpus.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .chunker import TokenCounter, chunk_document
from .concepts import (
    SOURCE_SECTION_TITLE,
    ConceptCandidate,
    ConceptMatcher,
    build_vocabulary,
    mentions_from_own_titles,
)
from .curriculum_catalog import ALIGNMENT_STATUS_NATIVE
from .config import RagConfig
from .corpus import discover_documents
from .embeddings import BGEM3Embedder, EmbeddingError
from .graph_schema import initialize_schema
from .logging_utils import get_logger
from .neo4j_store import Neo4jStore
from .epub_parser import EpubParseError, parse_epub_document
from .partitions import (
    EVALUATION_ONLY,
    EXCLUDED_BOILERPLATE,
    PRACTICE_ONLY,
    classify_section,
    is_production_partition,
)
from .pdf_parser import PdfParseError, compute_content_hash, parse_document
from .schemas import Chunk, ConceptMention, DocumentMetadata, ParsedDocument

LOGGER = get_logger(__name__)

# Bump when the parsing/chunking logic changes in a way that invalidates
# previously ingested chunks, so resume re-processes them.
#   v2: chunk overlap now carries trailing sentences when a paragraph is larger
#       than the overlap budget, and alpha-channel images are saved as PNG
#       instead of being skipped.
#   v4: section-aware partitions, boilerplate/practice exclusion, granular CISCE.
#   v5: NCERT chapter PDFs; native CBSE/NCERT alignment (no CISCE YAML).
INGEST_VERSION = "ingest-v5"

# Ceiling on MENTIONS edges per concept, to stop a common term from becoming a
# hub node that graph expansion would traverse into uselessly.
MAX_MENTIONS_PER_CONCEPT = 400
# Concept mentions kept per chunk, most specific (longest phrase) first.
MAX_CONCEPTS_PER_CHUNK = 8
# Concepts an image may be linked to via its page's section titles.
MAX_CONCEPTS_PER_IMAGE = 2


# --------------------------------------------------------------------------- #
# Cypher
# --------------------------------------------------------------------------- #

_UPSERT_HIERARCHY = """
MERGE (g:Grade {grade_id: $grade_id})
  ON CREATE SET g.grade = $grade, g.name = $grade_name
MERGE (s:Subject {subject_id: $subject_id})
  ON CREATE SET s.subject = $subject, s.name = $subject_name, s.grade = $grade
MERGE (g)-[:HAS_SUBJECT]->(s)
MERGE (u:Unit {unit_id: $unit_id})
  SET u.title = $unit_title,
      u.unit_slug = $unit_slug,
      u.unit_number = $unit_number,
      u.grade = $grade,
      u.subject = $subject
MERGE (s)-[:HAS_UNIT]->(u)
MERGE (d:Document {document_id: $document_id})
  SET d.title = $document_title,
      d.filename = $filename,
      d.local_pdf_path = $local_pdf_path,
      d.relative_pdf_path = $relative_pdf_path,
      d.resource_type = $resource_type,
      d.audience = $audience,
      d.source_id = $source_id,
      d.publisher = $publisher,
      d.source_role = $source_role,
      d.licence = $licence,
      d.licence_url = $licence_url,
      d.source_url = $source_url,
      d.content_partition = $content_partition,
      d.cisce_outcome_ids = $cisce_outcome_ids,
      d.alignment_status = $alignment_status,
      d.file_format = $file_format,
      d.grade = $grade,
      d.subject = $subject,
      d.unit_id = $unit_id,
      d.page_count = $page_count,
      d.content_hash = $content_hash,
      d.ingest_version = $ingest_version,
      d.unit_page_url = $unit_page_url,
      d.pdf_url = $pdf_url,
      d.from_manifest = $from_manifest,
      d.status = 'parsed',
      d.updated_at = $timestamp
MERGE (u)-[:HAS_DOCUMENT]->(d)
RETURN d.document_id AS document_id
"""

# Removes a document's derived content before it is rebuilt. Without this, a
# change to parsing or chunking would leave the previous run's Section, Chunk,
# Page and Image nodes behind as orphans, since MERGE only ever adds. The
# Document, Unit, Subject, Grade and Concept nodes are kept.
_PURGE_DOCUMENT_CONTENT = """
MATCH (d:Document {document_id: $document_id})-[:HAS_PAGE]->(p:Page)
OPTIONAL MATCH (p)-[:HAS_SECTION]->(sec:Section)
OPTIONAL MATCH (sec)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (p)-[:HAS_IMAGE]->(i:Image)
WITH collect(DISTINCT p) + collect(DISTINCT sec) + collect(DISTINCT c)
     + collect(DISTINCT i) AS nodes
UNWIND nodes AS node
DETACH DELETE node
RETURN count(node) AS deleted
"""

_UPSERT_PAGES = """
UNWIND $rows AS row
MATCH (d:Document {document_id: $document_id})
MERGE (p:Page {page_id: row.page_id})
  SET p += row.props
MERGE (d)-[:HAS_PAGE]->(p)
"""

_UPSERT_SECTIONS = """
UNWIND $rows AS row
MERGE (sec:Section {section_id: row.section_id})
  SET sec += row.props
WITH sec, row
MATCH (p:Page {page_id: row.start_page_id})
MERGE (p)-[:HAS_SECTION]->(sec)
"""

_UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (sec:Section {section_id: row.section_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
  SET c += row.props
MERGE (sec)-[:HAS_CHUNK]->(c)
WITH c, row
UNWIND row.page_ids AS page_id
MATCH (p:Page {page_id: page_id})
MERGE (c)-[:ON_PAGE]->(p)
"""

_LINK_CHUNK_ORDER = """
UNWIND $rows AS row
MATCH (a:Chunk {chunk_id: row.from_id})
MATCH (b:Chunk {chunk_id: row.to_id})
MERGE (a)-[:NEXT]->(b)
MERGE (b)-[:PREVIOUS]->(a)
"""

_UPSERT_IMAGES = """
UNWIND $rows AS row
MATCH (p:Page {page_id: row.page_id})
MERGE (i:Image {image_id: row.image_id})
  SET i += row.props
MERGE (p)-[:HAS_IMAGE]->(i)
MERGE (i)-[:APPEARS_IN]->(p)
"""

_UPSERT_CONCEPT_MENTIONS = """
UNWIND $rows AS row
MERGE (co:Concept {concept_id: row.concept_id})
  ON CREATE SET co.name = row.name,
                co.normalized_name = row.normalized_name,
                co.origin = row.source
WITH co, row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MERGE (c)-[m:MENTIONS]->(co)
  SET m.source = row.source,
      m.occurrences = row.occurrences,
      m.evidence = row.evidence
"""

_TITLES_FOR_VOCABULARY = """
MATCH (s:Section)
WHERE s.title_from_document = true
RETURN DISTINCT s.title AS title
"""

_UNIT_TITLES = "MATCH (u:Unit) RETURN DISTINCT u.title AS title"

_STREAM_CHUNKS = """
MATCH (c:Chunk)
RETURN c.chunk_id AS chunk_id,
       c.text AS text,
       c.section_title AS section_title,
       c.unit_title AS unit_title
ORDER BY c.chunk_id
SKIP $skip LIMIT $limit
"""

# Images inherit only concepts that a chunk on the same page mentions *because of
# its section title*. That keeps the association obvious from surrounding text
# rather than guessed, and no captioning model is involved.
_LINK_IMAGE_CONCEPTS = """
MATCH (i:Image)<-[:HAS_IMAGE]-(p:Page)<-[:ON_PAGE]-(c:Chunk)-[m:MENTIONS]->(co:Concept)
WHERE m.source = $section_source
WITH i, co, count(*) AS support
ORDER BY support DESC
WITH i, collect({concept: co, support: support})[0..$max_per_image] AS best
UNWIND best AS entry
// Rebound to plain variables: a map field cannot be used as a pattern node.
WITH i, entry.concept AS concept, entry.support AS support
MERGE (i)-[r:ILLUSTRATES]->(concept)
  SET r.evidence = 'concept named by a section title covering the image page',
      r.support = support
RETURN count(r) AS links
"""

_SET_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
SET c.embedding_version = $embedding_version,
    c.embedding_model = $embedding_model,
    c.embedding_dim = $embedding_dim
"""

_DOCUMENT_STATE = """
MATCH (d:Document {document_id: $document_id})
OPTIONAL MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
WITH d,
     count(c) AS chunk_count,
     count(CASE WHEN c.embedding_version = $embedding_version THEN 1 END) AS embedded
RETURN d.content_hash AS content_hash,
       d.ingest_version AS ingest_version,
       d.status AS status,
       chunk_count,
       embedded
"""

_MARK_DOCUMENT_COMPLETE = """
MATCH (d:Document {document_id: $document_id})
SET d.status = 'ingested',
    d.chunk_count = $chunk_count,
    d.image_count = $image_count,
    d.embedding_version = $embedding_version,
    d.ingested_at = $timestamp
"""

_REFRESH_ALIGNMENT = """
MATCH (d:Document {document_id: $document_id})
SET d.cisce_outcome_ids = $cisce_outcome_ids,
    d.alignment_status = $alignment_status,
    d.source_role = $source_role,
    d.licence = $licence,
    d.licence_url = $licence_url,
    d.source_id = $source_id,
    d.publisher = $publisher,
    d.source_url = $source_url
WITH d
MATCH (d)-[:HAS_PAGE]->(:Page)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
SET c.source_role = $source_role,
    c.licence = $licence,
    c.licence_url = $licence_url,
    c.source_id = $source_id,
    c.publisher = $publisher,
    c.source_url = $source_url
"""

_LIVE_GRAPH_COUNTS = """
MATCH (c:Chunk) WITH count(c) AS chunks
MATCH ()-[r]->() WITH chunks, count(r) AS relationships
MATCH (n) RETURN chunks, relationships, count(n) AS nodes
"""

_MENTION_COUNTS = """
MATCH (co:Concept)<-[m:MENTIONS]-(:Chunk)
WITH co, count(m) AS mentions
SET co.mention_count = mentions
RETURN count(co) AS concepts
"""

_PRUNE_HUB_CONCEPTS = """
MATCH (co:Concept)
WHERE co.mention_count > $max_mentions
RETURN co.concept_id AS concept_id, co.name AS name, co.mention_count AS mentions
ORDER BY mentions DESC
"""


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


@dataclass
class IngestStats:
    documents_total: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    pages: int = 0
    pages_without_text: int = 0
    sections: int = 0
    chunks: int = 0
    chunks_parsed: int = 0
    chunks_excluded_partition: int = 0
    chunks_excluded_boilerplate: int = 0
    chunks_excluded_practice: int = 0
    chunks_excluded_evaluation: int = 0
    chunks_written: int = 0
    nodes_deleted: int = 0
    live_chunk_count: int = 0
    live_node_count: int = 0
    live_relationship_count: int = 0
    images: int = 0
    concept_mentions: int = 0
    image_concept_links: int = 0
    concepts: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    elapsed_s: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    hub_concepts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_total": self.documents_total,
            "documents_ingested": self.documents_ingested,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "pages": self.pages,
            "pages_without_text": self.pages_without_text,
            "sections": self.sections,
            "chunks": self.chunks,
            "chunks_parsed": self.chunks_parsed,
            "chunks_excluded_partition": self.chunks_excluded_partition,
            "chunks_excluded_boilerplate": self.chunks_excluded_boilerplate,
            "chunks_excluded_practice": self.chunks_excluded_practice,
            "chunks_excluded_evaluation": self.chunks_excluded_evaluation,
            "chunks_written": self.chunks_written,
            "nodes_deleted": self.nodes_deleted,
            "live_chunk_count": self.live_chunk_count,
            "live_node_count": self.live_node_count,
            "live_relationship_count": self.live_relationship_count,
            "images": self.images,
            "concepts": self.concepts,
            "concept_mentions": self.concept_mentions,
            "image_concept_links": self.image_concept_links,
            "embeddings_computed": self.embeddings_computed,
            "embeddings_reused": self.embeddings_reused,
            "elapsed_s": round(self.elapsed_s, 1),
            "failures": [{"document": d, "error": e} for d, e in self.failures],
            "warnings": [{"document": d, "warning": w} for d, w in self.warnings],
            "hub_concepts": self.hub_concepts,
        }

    def describe(self) -> str:
        lines = [
            "Ingestion summary",
            "-" * 60,
            f"  documents        : {self.documents_ingested} ingested, "
            f"{self.documents_skipped} skipped (up to date), "
            f"{self.documents_failed} failed "
            f"(of {self.documents_total} discovered)",
            f"  pages            : {self.pages} "
            f"({self.pages_without_text} without extractable text)",
            f"  sections         : {self.sections}",
            f"  chunks           : {self.chunks_written} written "
            f"({self.chunks_parsed} parsed; "
            f"{self.chunks_excluded_evaluation} evaluation, "
            f"{self.chunks_excluded_practice} practice, "
            f"{self.chunks_excluded_boilerplate} boilerplate excluded)",
            f"  live graph       : {self.live_chunk_count} chunks, "
            f"{self.live_node_count} nodes, "
            f"{self.live_relationship_count} relationships",
            f"  images           : {self.images}",
            f"  concepts         : {self.concepts}",
            f"  concept mentions : {self.concept_mentions}",
            f"  image->concept   : {self.image_concept_links}",
            f"  embeddings       : {self.embeddings_computed} computed, "
            f"{self.embeddings_reused} reused",
            f"  elapsed          : {self.elapsed_s:.1f}s",
        ]
        if self.failures:
            lines.append(f"  FAILED PDFs      : {len(self.failures)}")
            for document, error in self.failures:
                lines.append(f"    - {document}: {error}")
        if self.warnings:
            lines.append(f"  warnings         : {len(self.warnings)}")
            for document, warning in self.warnings[:20]:
                lines.append(f"    - {document}: {warning}")
            if len(self.warnings) > 20:
                lines.append(f"    ... and {len(self.warnings) - 20} more")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ingestor
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CorpusIngestor:
    """Drives parsing, graph loading and embedding for the whole corpus."""

    def __init__(
        self,
        config: RagConfig,
        store: Neo4jStore,
        *,
        embedder: BGEM3Embedder | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.stats = IngestStats()
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self._token_counter = TokenCounter(config.models.embedding_model_path)
        self._vocabulary: dict[str, ConceptCandidate] = {}
        self._vocabulary_concept_ids: set[str] = set()
        self._matcher: ConceptMatcher | None = None

    # -- embedder lifecycle ------------------------------------------------ #

    @property
    def embedder(self) -> BGEM3Embedder:
        if self._embedder is None:
            self._embedder = BGEM3Embedder.from_config(self.config.models)
        return self._embedder

    def release_embedder(self) -> None:
        """Free memory after indexing, before a reranker/generator is loaded."""
        if self._embedder is not None and self._owns_embedder:
            self._embedder.unload()

    # -- checkpointing ----------------------------------------------------- #

    def _document_is_current(
        self, metadata: DocumentMetadata, content_hash: str, *, need_embeddings: bool
    ) -> bool:
        records = self.store.read(
            _DOCUMENT_STATE,
            {
                "document_id": metadata.document_id,
                "embedding_version": self.config.embedding_version,
            },
        )
        if not records:
            return False
        state = records[0]
        if state.get("content_hash") != content_hash:
            LOGGER.info(
                "%s changed on disk since last ingest; re-processing",
                metadata.relative_pdf_path,
            )
            return False
        if state.get("ingest_version") != INGEST_VERSION:
            return False
        if state.get("status") != "ingested":
            return False
        chunk_count = int(state.get("chunk_count") or 0)
        if chunk_count == 0:
            return False
        if need_embeddings and int(state.get("embedded") or 0) < chunk_count:
            return False
        return True

    def _refresh_alignment_metadata(self, metadata: DocumentMetadata) -> None:
        """Update provenance on skipped documents so crosswalk fixes apply."""
        self.store.execute_write(
            _REFRESH_ALIGNMENT,
            {
                "document_id": metadata.document_id,
                "cisce_outcome_ids": list(metadata.cisce_outcome_ids),
                "alignment_status": metadata.alignment_status,
                "source_role": metadata.source_role,
                "licence": metadata.licence,
                "licence_url": metadata.licence_url,
                "source_id": metadata.source_id,
                "publisher": metadata.publisher,
                "source_url": metadata.source_url,
            },
        )

    # -- graph writes ------------------------------------------------------ #

    def _upsert_hierarchy(self, parsed: ParsedDocument) -> None:
        metadata = parsed.metadata
        self.store.execute_write(
            _UPSERT_HIERARCHY,
            {
                "grade_id": metadata.grade_id,
                "grade": metadata.grade,
                "grade_name": f"Grade {metadata.grade}",
                "subject_id": metadata.subject_id,
                "subject": metadata.subject,
                "subject_name": metadata.subject.replace("_", " ").title(),
                "unit_id": metadata.unit_id,
                "unit_slug": metadata.unit_slug,
                "unit_title": metadata.unit_title,
                "unit_number": metadata.unit_number,
                "document_id": metadata.document_id,
                "document_title": metadata.document_title,
                "filename": metadata.filename,
                "local_pdf_path": str(metadata.local_pdf_path),
                "relative_pdf_path": metadata.relative_pdf_path,
                "resource_type": metadata.resource_type,
                "audience": metadata.audience,
                "source_id": metadata.source_id,
                "publisher": metadata.publisher,
                "source_role": metadata.source_role,
                "licence": metadata.licence,
                "licence_url": metadata.licence_url,
                "source_url": metadata.source_url,
                "content_partition": metadata.content_partition,
                "cisce_outcome_ids": list(metadata.cisce_outcome_ids),
                "alignment_status": metadata.alignment_status,
                "file_format": metadata.file_format,
                "page_count": parsed.page_count,
                "content_hash": parsed.content_hash,
                "ingest_version": INGEST_VERSION,
                "unit_page_url": metadata.unit_page_url,
                "pdf_url": metadata.pdf_url,
                "from_manifest": metadata.from_manifest,
                "timestamp": _now(),
            },
        )

    def _purge_document_content(self, metadata: DocumentMetadata) -> None:
        """Drop a document's pages/sections/chunks/images before rebuilding them."""
        records = self.store.execute_write(
            _PURGE_DOCUMENT_CONTENT, {"document_id": metadata.document_id}
        )
        deleted = int(records[0]["deleted"]) if records else 0
        if deleted:
            LOGGER.info(
                "Removed %d stale node(s) from a previous ingest of %s",
                deleted,
                metadata.relative_pdf_path,
            )

    def _upsert_pages(self, parsed: ParsedDocument) -> None:
        rows = [
            {
                "page_id": page.page_id,
                "props": page.to_properties(parsed.metadata),
            }
            for page in parsed.pages
        ]
        self.store.execute_write_batches(
            _UPSERT_PAGES,
            rows,
            batch_size=self.config.ingest.neo4j_batch_size,
            extra_parameters={"document_id": parsed.metadata.document_id},
        )

    def _upsert_sections(self, parsed: ParsedDocument) -> None:
        from .schemas import page_id_for

        rows = [
            {
                "section_id": section.section_id,
                "props": section.to_properties(parsed.metadata),
                "start_page_id": page_id_for(
                    parsed.metadata.document_id, section.page_start
                ),
            }
            for section in parsed.sections
        ]
        self.store.execute_write_batches(
            _UPSERT_SECTIONS,
            rows,
            batch_size=self.config.ingest.neo4j_batch_size,
        )

    def _upsert_chunks(self, chunks: Sequence[Chunk]) -> None:
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "page_ids": chunk.page_ids,
                "props": chunk.to_properties(),
            }
            for chunk in chunks
        ]
        self.store.execute_write_batches(
            _UPSERT_CHUNKS,
            rows,
            batch_size=self.config.ingest.neo4j_batch_size,
        )

        order_rows = [
            {"from_id": a.chunk_id, "to_id": b.chunk_id}
            for a, b in zip(chunks, chunks[1:])
        ]
        self.store.execute_write_batches(
            _LINK_CHUNK_ORDER,
            order_rows,
            batch_size=self.config.ingest.neo4j_batch_size,
        )

    def _upsert_images(self, parsed: ParsedDocument) -> int:
        rows = [
            {
                "image_id": image.image_id,
                "page_id": image.page_id,
                "props": image.to_properties(),
            }
            for image in parsed.images
        ]
        return self.store.execute_write_batches(
            _UPSERT_IMAGES,
            rows,
            batch_size=self.config.ingest.neo4j_batch_size,
        )

    # -- concepts ---------------------------------------------------------- #

    def link_concepts(self) -> None:
        """Build the concept graph over the whole ingested corpus.

        This runs after all documents are loaded, for a specific reason: section
        titles are only known once a PDF has been parsed, so linking during
        per-document ingestion would let a document see only the concepts of the
        documents parsed before it. Running it globally gives every chunk the
        same vocabulary and produces the cross-document ``Concept`` links that
        graph expansion depends on.

        Evidence for each ``MENTIONS`` edge is either the chunk's own section/unit
        title or a verbatim phrase occurrence in the chunk text. The pass is
        idempotent (all MERGE) and cheap enough to re-run.
        """
        unit_titles = [
            record["title"]
            for record in self.store.read(_UNIT_TITLES)
            if record.get("title")
        ]
        section_titles = [
            record["title"]
            for record in self.store.read(_TITLES_FOR_VOCABULARY)
            if record.get("title")
        ]
        LOGGER.info(
            "Building concept vocabulary from %d unit titles and %d section titles",
            len(unit_titles),
            len(section_titles),
        )
        self._vocabulary = build_vocabulary(unit_titles, section_titles)
        self._vocabulary_concept_ids = {
            candidate.concept_id for candidate in self._vocabulary.values()
        }
        self._matcher = ConceptMatcher(self._vocabulary)

        if not self._vocabulary:
            LOGGER.warning("Concept vocabulary is empty; no MENTIONS edges created")
            return

        batch = 500
        skip = 0
        total_rows = 0
        while True:
            records = self.store.read(_STREAM_CHUNKS, {"skip": skip, "limit": batch})
            if not records:
                break
            rows: list[dict[str, Any]] = []
            for record in records:
                rows.extend(self._mention_rows(record))
            self.store.execute_write_batches(
                _UPSERT_CONCEPT_MENTIONS,
                rows,
                batch_size=self.config.ingest.neo4j_batch_size,
            )
            total_rows += len(rows)
            skip += len(records)
            LOGGER.info(
                "Concept linking: %d chunks processed, %d MENTIONS upserted",
                skip,
                total_rows,
            )

        self.stats.concept_mentions = total_rows

        image_records = self.store.execute_write(
            _LINK_IMAGE_CONCEPTS,
            {
                "section_source": SOURCE_SECTION_TITLE,
                "max_per_image": MAX_CONCEPTS_PER_IMAGE,
            },
        )
        self.stats.image_concept_links = (
            int(image_records[0]["links"]) if image_records else 0
        )
        LOGGER.info(
            "Linked %d image->concept edges from section titles",
            self.stats.image_concept_links,
        )

    def _mention_rows(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """MENTIONS rows for one chunk record streamed from the graph."""
        assert self._matcher is not None
        mentions: dict[str, ConceptMention] = {}

        # Structural evidence: the chunk's own section and unit titles.
        for mention in mentions_from_own_titles(
            record.get("section_title") or "", record.get("unit_title") or ""
        ):
            if mention.concept_id in self._vocabulary_concept_ids:
                mentions[mention.concept_id] = mention

        # Textual evidence: verbatim vocabulary phrases in the chunk text.
        for mention in self._matcher.find_all(record.get("text") or ""):
            mentions.setdefault(mention.concept_id, mention)

        return [
            {
                "chunk_id": record["chunk_id"],
                "concept_id": mention.concept_id,
                "name": mention.name,
                "normalized_name": mention.normalized_name,
                "source": mention.source,
                "occurrences": mention.occurrences,
                "evidence": mention.evidence,
            }
            for mention in list(mentions.values())[:MAX_CONCEPTS_PER_CHUNK]
        ]

    # -- embeddings -------------------------------------------------------- #

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.encode_documents([chunk.text for chunk in chunks])
        if vectors.shape[0] != len(chunks):
            raise EmbeddingError(
                f"Embedder returned {vectors.shape[0]} vectors for "
                f"{len(chunks)} chunks"
            )
        rows = [
            {"chunk_id": chunk.chunk_id, "embedding": vector.tolist()}
            for chunk, vector in zip(chunks, vectors)
        ]
        self.store.execute_write_batches(
            _SET_EMBEDDINGS,
            rows,
            batch_size=self.config.ingest.neo4j_batch_size,
            extra_parameters={
                "embedding_version": self.config.embedding_version,
                "embedding_model": str(self.config.models.embedding_model_path.name),
                "embedding_dim": int(vectors.shape[1]),
            },
        )
        return len(rows)

    # -- per-document driver ----------------------------------------------- #

    def ingest_document(
        self,
        metadata: DocumentMetadata,
        *,
        skip_embeddings: bool = False,
        force: bool = False,
    ) -> str:
        """Ingest one source file. Returns "ingested", "skipped" or "failed"."""
        if metadata.content_partition == EVALUATION_ONLY:
            LOGGER.info(
                "Skipping evaluation-only file %s (not stored in production graph)",
                metadata.relative_pdf_path,
            )
            return "skipped"

        try:
            content_hash = compute_content_hash(metadata.local_pdf_path)
        except OSError as exc:
            self.stats.failures.append((metadata.relative_pdf_path, str(exc)))
            LOGGER.error("Cannot read %s: %s", metadata.relative_pdf_path, exc)
            return "failed"

        if not force and self._document_is_current(
            metadata, content_hash, need_embeddings=not skip_embeddings
        ):
            self._refresh_alignment_metadata(metadata)
            LOGGER.info("Skipping %s (already up to date)", metadata.relative_pdf_path)
            return "skipped"

        try:
            if metadata.file_format == "epub":
                parsed = parse_epub_document(
                    metadata,
                    images_root=self.config.paths.images_dir,
                    config=self.config.ingest,
                )
            else:
                parsed = parse_document(
                    metadata,
                    images_root=self.config.paths.images_dir,
                    config=self.config.ingest,
                )
        except (PdfParseError, EpubParseError) as exc:
            self.stats.failures.append((metadata.relative_pdf_path, str(exc)))
            LOGGER.error("Failed to parse %s: %s", metadata.relative_pdf_path, exc)
            return "failed"

        for warning in parsed.warnings:
            self.stats.warnings.append((metadata.relative_pdf_path, warning))

        chunks = chunk_document(parsed, self._token_counter, self.config.chunking)
        self.stats.chunks_parsed += len(chunks)
        production_chunks: list[Chunk] = []
        previous_title = ""
        for chunk in chunks:
            decision = classify_section(
                chunk.section_title,
                chunk.text,
                previous_title=previous_title,
                default=metadata.content_partition,
            )
            previous_title = chunk.section_title
            chunk.content_partition = decision.partition
            chunk.cisce_outcome_ids = []
            chunk.mapping_granularity = "none"
            chunk.alignment_status = ALIGNMENT_STATUS_NATIVE
            if decision.partition == EVALUATION_ONLY:
                self.stats.chunks_excluded_evaluation += 1
                self.stats.chunks_excluded_partition += 1
                continue
            if decision.partition == PRACTICE_ONLY:
                self.stats.chunks_excluded_practice += 1
                self.stats.chunks_excluded_partition += 1
                continue
            if decision.partition == EXCLUDED_BOILERPLATE:
                self.stats.chunks_excluded_boilerplate += 1
                self.stats.chunks_excluded_partition += 1
                continue
            if not is_production_partition(decision.partition):
                self.stats.chunks_excluded_partition += 1
                continue
            production_chunks.append(chunk)
        skipped = len(chunks) - len(production_chunks)
        if skipped:
            self.stats.warnings.append(
                (
                    metadata.relative_pdf_path,
                    f"skipped {skipped} non-production chunk(s) "
                    f"(evaluation/practice/boilerplate)",
                )
            )
        chunks = production_chunks
        if not chunks:
            self.stats.warnings.append(
                (metadata.relative_pdf_path, "produced no chunks; nothing to retrieve")
            )

        records = self.store.execute_write(
            _PURGE_DOCUMENT_CONTENT, {"document_id": metadata.document_id}
        )
        deleted = int(records[0]["deleted"]) if records else 0
        self.stats.nodes_deleted += deleted
        if deleted:
            LOGGER.info(
                "Removed %d stale node(s) from a previous ingest of %s",
                deleted,
                metadata.relative_pdf_path,
            )
        self._upsert_hierarchy(parsed)
        self._upsert_pages(parsed)
        self._upsert_sections(parsed)
        self._upsert_chunks(chunks)
        self.stats.chunks_written += len(chunks)
        image_count = self._upsert_images(parsed)

        if not skip_embeddings and chunks:
            self.stats.embeddings_computed += self._embed_chunks(chunks)

        self.store.execute_write(
            _MARK_DOCUMENT_COMPLETE,
            {
                "document_id": metadata.document_id,
                "chunk_count": len(chunks),
                "image_count": image_count,
                "embedding_version": (
                    self.config.embedding_version if not skip_embeddings else None
                ),
                "timestamp": _now(),
            },
        )

        self.stats.pages += parsed.page_count
        self.stats.pages_without_text += parsed.pages_without_text
        self.stats.sections += len(parsed.sections)
        self.stats.chunks += len(chunks)
        self.stats.images += image_count
        return "ingested"

    # -- corpus driver ----------------------------------------------------- #

    def run(
        self,
        *,
        grades: tuple[int, ...] | None = None,
        subjects: tuple[str, ...] | None = None,
        limit: int | None = None,
        skip_embeddings: bool = False,
        force: bool = False,
        link_concepts: bool = True,
    ) -> IngestStats:
        started = time.perf_counter()

        documents, corpus_stats = discover_documents(
            self.config.paths.corpus_path,
            grades=grades,
            subjects=subjects,
            limit=limit,
        )
        self.stats.documents_total = len(documents)
        self.config.paths.ensure_processed_dirs()

        dimension = self.embedder.dimension
        LOGGER.info("BGE-M3 dense dimension detected from checkpoint: %d", dimension)
        schema = initialize_schema(self.store, dimension)
        LOGGER.info("Graph schema ready\n%s", schema.describe())

        for index, metadata in enumerate(documents, start=1):
            LOGGER.info(
                "[%d/%d] %s", index, len(documents), metadata.relative_pdf_path
            )
            try:
                outcome = self.ingest_document(
                    metadata, skip_embeddings=skip_embeddings, force=force
                )
            except KeyboardInterrupt:
                LOGGER.warning(
                    "Interrupted after %d/%d documents. Re-run the same command to "
                    "resume; completed documents are skipped.",
                    index - 1,
                    len(documents),
                )
                raise
            except Exception as exc:
                self.stats.documents_failed += 1
                self.stats.failures.append((metadata.relative_pdf_path, repr(exc)))
                LOGGER.exception(
                    "Unexpected failure ingesting %s", metadata.relative_pdf_path
                )
                continue

            if outcome == "ingested":
                self.stats.documents_ingested += 1
            elif outcome == "skipped":
                self.stats.documents_skipped += 1
            else:
                self.stats.documents_failed += 1

        if link_concepts:
            self.link_concepts()
        self._finalize_concepts()
        self._record_live_graph_counts()
        self.stats.elapsed_s = time.perf_counter() - started
        self._write_report(corpus_stats)
        return self.stats

    def _record_live_graph_counts(self) -> None:
        records = self.store.read(_LIVE_GRAPH_COUNTS)
        if not records:
            return
        row = records[0]
        self.stats.live_chunk_count = int(row.get("chunks") or 0)
        self.stats.live_node_count = int(row.get("nodes") or 0)
        self.stats.live_relationship_count = int(row.get("relationships") or 0)
        LOGGER.info(
            "Live graph: %d chunks, %d nodes, %d relationships",
            self.stats.live_chunk_count,
            self.stats.live_node_count,
            self.stats.live_relationship_count,
        )

    def _finalize_concepts(self) -> None:
        records = self.store.execute_write(_MENTION_COUNTS)
        self.stats.concepts = int(records[0]["concepts"]) if records else 0
        hubs = self.store.read(
            _PRUNE_HUB_CONCEPTS, {"max_mentions": MAX_MENTIONS_PER_CONCEPT}
        )
        self.stats.hub_concepts = hubs
        if hubs:
            LOGGER.info(
                "%d concept(s) exceed %d mentions and are excluded from graph "
                "expansion at query time: %s",
                len(hubs),
                MAX_MENTIONS_PER_CONCEPT,
                ", ".join(f"{h['name']} ({h['mentions']})" for h in hubs[:10]),
            )

    def _write_report(self, corpus_stats: Any) -> None:
        report_path = self.config.paths.manifests_dir / "ingest_report.json"
        payload = {
            "generated_at": _now(),
            "ingest_version": INGEST_VERSION,
            "embedding_version": self.config.embedding_version,
            "corpus": {
                "path": str(self.config.paths.corpus_path),
                "pdfs_discovered": corpus_stats.pdf_count,
                "units": corpus_stats.unit_count,
                "manifest_records": corpus_stats.manifest_records,
            },
            "stats": self.stats.to_dict(),
        }
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            LOGGER.info("Wrote ingestion report to %s", report_path)
        except OSError as exc:
            LOGGER.warning("Could not write ingestion report: %s", exc)
