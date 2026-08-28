"""Thin wrapper around the official Neo4j Python driver.

Deliberately thin: Cypher lives in the module that owns the query (ingestion,
each retriever, graph inspection) so individual retrieval strategies can be
rewritten without touching a shared query layer.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ClientError, Neo4jError, ServiceUnavailable

from .config import Neo4jConfig
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


class Neo4jUnavailableError(RuntimeError):
    """Raised when Neo4j cannot be reached or authentication fails."""


class Neo4jStore:
    """Connection holder plus small read/write helpers."""

    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver: Driver | None = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self.connect()
        assert self._driver is not None
        return self._driver

    def connect(self) -> None:
        if self._driver is not None:
            return
        LOGGER.info("Connecting to Neo4j at %s", self.config.describe())
        try:
            driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.user, self.config.password),
            )
            driver.verify_connectivity()
        except AuthError as exc:
            raise Neo4jUnavailableError(
                f"Neo4j rejected the credentials for user {self.config.user!r} at "
                f"{self.config.uri}. Check NEO4J_USER / NEO4J_PASSWORD in .env."
            ) from exc
        except (ServiceUnavailable, OSError) as exc:
            raise Neo4jUnavailableError(
                f"Cannot reach Neo4j at {self.config.uri} ({exc}). Start the "
                f"database first: scripts/neo4j_local.sh start "
                f"(see README.md -> 'Neo4j setup')."
            ) from exc
        except Neo4jError as exc:
            raise Neo4jUnavailableError(
                f"Neo4j connection to {self.config.uri} failed: {exc}"
            ) from exc

        self._driver = driver
        LOGGER.info("Neo4j connection established (%s)", self.server_version())

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            LOGGER.debug("Neo4j connection closed")

    def __enter__(self) -> "Neo4jStore":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator[Any]:
        with self.driver.session(database=self.config.database) as session:
            yield session

    # -- queries ----------------------------------------------------------- #

    def read(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read query and materialise the records as dictionaries."""
        try:
            with self.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except ClientError as exc:
            raise Neo4jUnavailableError(f"Neo4j rejected a read query: {exc}") from exc
        except (ServiceUnavailable, OSError) as exc:
            raise Neo4jUnavailableError(
                f"Lost connection to Neo4j during a read: {exc}"
            ) from exc

    def run_ddl(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        """Run a schema statement (CREATE INDEX / CONSTRAINT) in auto-commit.

        Index and constraint creation cannot run inside a managed transaction, so
        this uses an implicit transaction and discards the result.
        """
        try:
            with self.session() as session:
                session.run(query, parameters or {}).consume()
        except ClientError as exc:
            raise Neo4jUnavailableError(
                f"Neo4j rejected a schema statement: {exc}\nStatement: {query}"
            ) from exc
        except (ServiceUnavailable, OSError) as exc:
            raise Neo4jUnavailableError(
                f"Lost connection to Neo4j during schema creation: {exc}"
            ) from exc

    def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a write query in a managed transaction and return its records."""

        def _work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(query, parameters or {})
            return [record.data() for record in result]

        try:
            with self.session() as session:
                return session.execute_write(_work)
        except ClientError as exc:
            raise Neo4jUnavailableError(f"Neo4j rejected a write query: {exc}") from exc
        except (ServiceUnavailable, OSError) as exc:
            raise Neo4jUnavailableError(
                f"Lost connection to Neo4j during a write: {exc}"
            ) from exc

    def execute_write_batches(
        self,
        query: str,
        rows: Sequence[dict[str, Any]],
        *,
        batch_size: int = 200,
        parameter_name: str = "rows",
        extra_parameters: dict[str, Any] | None = None,
    ) -> int:
        """Apply an UNWIND-style write query over ``rows`` in batches.

        Returns the number of rows submitted. Batching keeps transactions small
        enough that an interrupted ingest leaves a consistent graph.
        """
        if not rows:
            return 0
        extra = extra_parameters or {}
        submitted = 0
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start : start + batch_size])
            self.execute_write(query, {parameter_name: batch, **extra})
            submitted += len(batch)
        return submitted

    # -- introspection ----------------------------------------------------- #

    def server_version(self) -> str:
        try:
            records = self.read(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            )
        except Neo4jUnavailableError:
            return "unknown"
        if not records:
            return "unknown"
        record = records[0]
        return f"{record['name']} {record['version']} ({record['edition']})"

    def list_indexes(self) -> list[dict[str, Any]]:
        return self.read(
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, "
            "state RETURN name, type, entityType, labelsOrTypes, properties, state "
            "ORDER BY name"
        )

    def list_constraints(self) -> list[dict[str, Any]]:
        return self.read(
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
            "RETURN name, type, labelsOrTypes, properties ORDER BY name"
        )

    def node_counts(self, labels: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in labels:
            records = self.read(f"MATCH (n:`{label}`) RETURN count(n) AS total")
            counts[label] = int(records[0]["total"]) if records else 0
        return counts

    def relationship_counts(self, types: Sequence[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rel_type in types:
            records = self.read(
                f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS total"
            )
            counts[rel_type] = int(records[0]["total"]) if records else 0
        return counts

    def is_empty(self) -> bool:
        records = self.read("MATCH (n) RETURN count(n) AS total")
        return not records or int(records[0]["total"]) == 0
