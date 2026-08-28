"""Curriculum graph schema: labels, relationships, constraints and indexes.

Structural hierarchy (the only relationships created from document
structure, never from model inference)::

    (:Grade)-[:HAS_SUBJECT]->(:Subject)
    (:Subject)-[:HAS_UNIT]->(:Unit)
    (:Unit)-[:HAS_DOCUMENT]->(:Document)
    (:Document)-[:HAS_PAGE]->(:Page)
    (:Page)-[:HAS_SECTION]->(:Section)
    (:Section)-[:HAS_CHUNK]->(:Chunk)
    (:Page)-[:HAS_IMAGE]->(:Image)

Supported additional edges::

    (:Chunk)-[:MENTIONS]->(:Concept)      verbatim phrase / own title evidence
    (:Chunk)-[:ON_PAGE]->(:Page)          chunk to each page it covers
    (:Image)-[:APPEARS_IN]->(:Page)       inverse of HAS_IMAGE
    (:Image)-[:ILLUSTRATES]->(:Concept)   only via the image's own page text
    (:Chunk)-[:NEXT]->(:Chunk)            document reading order
    (:Chunk)-[:PREVIOUS]->(:Chunk)        inverse of NEXT

Semantic relationships such as CAUSES, PREREQUISITE_OF, DEPENDS_ON, PROVES and
IMPLIES are intentionally absent: they cannot be established reliably without
model inference, and this system is meant to be hallucination-resistant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .logging_utils import get_logger
from .neo4j_store import Neo4jStore

LOGGER = get_logger(__name__)

NODE_LABELS = (
    "Grade",
    "Subject",
    "Unit",
    "Document",
    "Page",
    "Section",
    "Chunk",
    "Concept",
    "Image",
)

RELATIONSHIP_TYPES = (
    "HAS_SUBJECT",
    "HAS_UNIT",
    "HAS_DOCUMENT",
    "HAS_PAGE",
    "HAS_SECTION",
    "HAS_CHUNK",
    "HAS_IMAGE",
    "ON_PAGE",
    "APPEARS_IN",
    "MENTIONS",
    "ILLUSTRATES",
    "NEXT",
    "PREVIOUS",
)

CHUNK_VECTOR_INDEX = "chunk_embedding_index"
CHUNK_FULLTEXT_INDEX = "chunk_fulltext_index"
CONCEPT_FULLTEXT_INDEX = "concept_fulltext_index"

# One uniqueness constraint per label; each also provides a backing index.
_CONSTRAINTS: tuple[tuple[str, str, str], ...] = (
    ("grade_id_unique", "Grade", "grade_id"),
    ("subject_id_unique", "Subject", "subject_id"),
    ("unit_id_unique", "Unit", "unit_id"),
    ("document_id_unique", "Document", "document_id"),
    ("page_id_unique", "Page", "page_id"),
    ("section_id_unique", "Section", "section_id"),
    ("chunk_id_unique", "Chunk", "chunk_id"),
    ("concept_id_unique", "Concept", "concept_id"),
    ("image_id_unique", "Image", "image_id"),
)

# Property indexes supporting metadata filtering during retrieval.
_PROPERTY_INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("chunk_grade_subject", "Chunk", ("grade", "subject")),
    ("chunk_unit", "Chunk", ("unit_id",)),
    ("chunk_document", "Chunk", ("document_id",)),
    ("chunk_resource_type", "Chunk", ("resource_type",)),
    ("chunk_audience", "Chunk", ("audience",)),
    ("chunk_embedding_version", "Chunk", ("embedding_version",)),
    ("chunk_section", "Chunk", ("section_id",)),
    ("document_hash", "Document", ("content_hash",)),
    ("page_document", "Page", ("document_id",)),
    ("image_document", "Image", ("document_id",)),
    ("concept_normalized", "Concept", ("normalized_name",)),
)

# Lexical retrieval fields. Chunk.text carries the content; the two titles add
# curriculum phrasing that often matches a student's wording.
_CHUNK_FULLTEXT_FIELDS = ("text", "section_title", "unit_title")


@dataclass
class SchemaReport:
    constraints_created: list[str]
    property_indexes_created: list[str]
    vector_index: str
    vector_dimension: int
    fulltext_indexes: list[str]

    def describe(self) -> str:
        lines = [
            f"constraints        : {len(self.constraints_created)}",
            f"property indexes   : {len(self.property_indexes_created)}",
            f"vector index       : {self.vector_index} "
            f"(dim={self.vector_dimension}, cosine)",
            f"full-text indexes  : {', '.join(self.fulltext_indexes)}",
        ]
        return "\n".join(lines)


def create_constraints(store: Neo4jStore) -> list[str]:
    created: list[str] = []
    for name, label, prop in _CONSTRAINTS:
        store.run_ddl(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:`{label}`) REQUIRE n.`{prop}` IS UNIQUE"
        )
        created.append(name)
    LOGGER.info("Ensured %d uniqueness constraints", len(created))
    return created


def create_property_indexes(store: Neo4jStore) -> list[str]:
    created: list[str] = []
    for name, label, props in _PROPERTY_INDEXES:
        prop_list = ", ".join(f"n.`{p}`" for p in props)
        store.run_ddl(
            f"CREATE INDEX {name} IF NOT EXISTS FOR (n:`{label}`) ON ({prop_list})"
        )
        created.append(name)
    LOGGER.info("Ensured %d property indexes for metadata filtering", len(created))
    return created


def create_vector_index(store: Neo4jStore, dimension: int) -> str:
    """Create the Chunk embedding vector index (cosine similarity).

    ``dimension`` must come from the embedding model's own config, never a
    hardcoded guess. An existing index with a different dimension is reported
    rather than silently dropped.
    """
    if dimension <= 0:
        raise ValueError(f"Vector dimension must be positive, got {dimension}")

    existing = _vector_index_dimension(store)
    if existing is not None and existing != dimension:
        raise RuntimeError(
            f"Vector index {CHUNK_VECTOR_INDEX} already exists with dimension "
            f"{existing}, but the embedding model produces {dimension}. Drop the "
            f"index explicitly before re-indexing:\n"
            f"  DROP INDEX {CHUNK_VECTOR_INDEX}"
        )

    store.run_ddl(
        f"CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{ indexConfig: {{ "
        f"`vector.dimensions`: {int(dimension)}, "
        f"`vector.similarity_function`: 'cosine' }} }}"
    )
    LOGGER.info(
        "Ensured vector index %s (dimension=%d, cosine)",
        CHUNK_VECTOR_INDEX,
        dimension,
    )
    return CHUNK_VECTOR_INDEX


def _vector_index_dimension(store: Neo4jStore) -> int | None:
    records = store.read(
        "SHOW INDEXES YIELD name, type, options "
        "WHERE name = $name AND type = 'VECTOR' RETURN options",
        {"name": CHUNK_VECTOR_INDEX},
    )
    if not records:
        return None
    options = records[0].get("options") or {}
    index_config = options.get("indexConfig") or {}
    value = index_config.get("vector.dimensions")
    return int(value) if value is not None else None


def create_fulltext_indexes(store: Neo4jStore) -> list[str]:
    fields = ", ".join(f"c.`{field}`" for field in _CHUNK_FULLTEXT_FIELDS)
    store.run_ddl(
        f"CREATE FULLTEXT INDEX {CHUNK_FULLTEXT_INDEX} IF NOT EXISTS "
        f"FOR (c:Chunk) ON EACH [{fields}]"
    )
    store.run_ddl(
        f"CREATE FULLTEXT INDEX {CONCEPT_FULLTEXT_INDEX} IF NOT EXISTS "
        f"FOR (c:Concept) ON EACH [c.`name`, c.`normalized_name`]"
    )
    LOGGER.info(
        "Ensured full-text indexes %s (%s) and %s",
        CHUNK_FULLTEXT_INDEX,
        ", ".join(_CHUNK_FULLTEXT_FIELDS),
        CONCEPT_FULLTEXT_INDEX,
    )
    return [CHUNK_FULLTEXT_INDEX, CONCEPT_FULLTEXT_INDEX]


def wait_for_indexes(store: Neo4jStore, timeout_seconds: int = 300) -> bool:
    """Block until all indexes are ONLINE. Returns False on timeout."""
    records = store.read(
        "SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN name, state"
    )
    if not records:
        return True
    LOGGER.info("Waiting for %d index(es) to come online", len(records))
    try:
        store.read(
            "CALL db.awaitIndexes($timeout)", {"timeout": int(timeout_seconds)}
        )
    except Exception as exc:
        LOGGER.warning("db.awaitIndexes failed: %s", exc)
        return False
    remaining = store.read(
        "SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN name, state"
    )
    if remaining:
        LOGGER.warning(
            "Indexes still not online after %ds: %s",
            timeout_seconds,
            ", ".join(f"{r['name']}={r['state']}" for r in remaining),
        )
        return False
    return True


def initialize_schema(store: Neo4jStore, embedding_dimension: int) -> SchemaReport:
    """Create every constraint and index the pipeline needs. Idempotent."""
    report = SchemaReport(
        constraints_created=create_constraints(store),
        property_indexes_created=create_property_indexes(store),
        vector_index=create_vector_index(store, embedding_dimension),
        vector_dimension=embedding_dimension,
        fulltext_indexes=create_fulltext_indexes(store),
    )
    wait_for_indexes(store)
    return report


def reset_graph(store: Neo4jStore, *, drop_indexes: bool = False) -> dict[str, int]:
    """Delete all curriculum data. Only ever called behind an explicit flag.

    Nodes are removed in batches so a large graph does not exhaust the heap.
    """
    LOGGER.warning("Resetting the curriculum graph: deleting all nodes")
    deleted = 0
    while True:
        records = store.execute_write(
            "MATCH (n) WITH n LIMIT 20000 DETACH DELETE n RETURN count(n) AS removed"
        )
        removed = int(records[0]["removed"]) if records else 0
        deleted += removed
        if removed == 0:
            break

    dropped = 0
    if drop_indexes:
        for name in (
            CHUNK_VECTOR_INDEX,
            CHUNK_FULLTEXT_INDEX,
            CONCEPT_FULLTEXT_INDEX,
        ):
            try:
                store.run_ddl(f"DROP INDEX {name} IF EXISTS")
                dropped += 1
            except Exception as exc:
                LOGGER.warning("Could not drop index %s: %s", name, exc)
        for name, _, _ in _PROPERTY_INDEXES:
            try:
                store.run_ddl(f"DROP INDEX {name} IF EXISTS")
                dropped += 1
            except Exception as exc:
                LOGGER.warning("Could not drop index %s: %s", name, exc)

    LOGGER.warning("Graph reset complete: %d nodes deleted", deleted)
    return {"nodes_deleted": deleted, "indexes_dropped": dropped}
