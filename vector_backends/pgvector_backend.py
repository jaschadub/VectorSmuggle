# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""pgvector research backend.

pgvector is the de-facto vector store for teams that already operate
PostgreSQL and want to bolt embedding search onto an existing OLTP
database rather than stand up a dedicated vector service. It is
relevant to the threat model for a very specific reason: a row in a
``CREATE TABLE foo (id TEXT, embedding vector(N), metadata JSONB)`` is
indistinguishable from any other row to the surrounding application
code, RBAC, backup, and replication machinery. The vector is just
another column. If a steganographic payload survives a round-trip
through pgvector, every existing Postgres governance control still
treats the poisoned row as ordinary data.

The implementation matches the contract of the other research backends:

  - ``open(dim)`` (re)creates a research table sized to ``dim``.
  - ``insert(records)`` L2-normalizes vectors and ``COPY``-equivalents
    them into the table along with their metadata.
  - ``get_by_id(record_id)`` returns the stored vector verbatim. For
    pgvector with the default ``vector`` column type, this round-trip
    is lossless (single-precision float on the wire and on disk).
  - ``search(query, k)`` uses pgvector's cosine-distance operator
    ``<=>`` and converts it to a similarity score so the cross-backend
    study's monotonicity contract is preserved.

Connection parameters come from ``PGVECTOR_URL`` or default to the
local docker-compose service in ``test_vector_dbs_docker/`` (image:
``pgvector/pgvector:pg16``, db ``vectordb``, user ``postgres``).

``is_available()`` performs a single short connection attempt so the
cross-backend study can silently skip pgvector when no docker is
running.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from vector_backends.base import (
    BackendUnavailable,
    InsertRecord,
    SearchResult,
    VectorBackend,
)

_DEFAULT_URL = "postgresql://postgres:mypassword@localhost:5432/vectordb"
_DEFAULT_TABLE = "vectorsmuggle_research"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier. The caller is responsible for ensuring
    only trusted, validated identifiers reach this function — pgvector
    has no parameterized form for table/column names.
    """
    if not name or any(c in name for c in '"\x00') or "\n" in name or "\r" in name:
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return '"' + name + '"'


class PgVectorBackend(VectorBackend):
    """pgvector-backed research backend.

    The default table is dropped and recreated on each ``open()`` so
    repeated runs don't accumulate state. ``close()`` drops the table
    and closes the connection.
    """

    name = "pgvector"

    def __init__(
        self,
        url: str | None = None,
        table_name: str = _DEFAULT_TABLE,
    ) -> None:
        self._url = url or os.getenv("PGVECTOR_URL", _DEFAULT_URL)
        # Validate eagerly so a bad table name surfaces at construction,
        # not at the first query.
        _quote_ident(table_name)
        self._table = table_name
        self._conn: Any = None
        self._dim: int | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import pgvector.psycopg  # noqa: F401
            import psycopg  # noqa: F401
        except ImportError as e:
            return False, f"psycopg / pgvector not installed: {e}"
        url = os.getenv("PGVECTOR_URL", _DEFAULT_URL)
        try:
            import psycopg
            with psycopg.connect(url, connect_timeout=2) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
        except Exception as e:  # noqa: BLE001 - psycopg surfaces many error types
            return False, f"pgvector unreachable at {url}: {e}"
        return True, ""

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"PgVectorBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as e:
            raise BackendUnavailable(f"psycopg / pgvector not installed: {e}") from e

        self._conn = psycopg.connect(self._url, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # register_vector requires the extension to exist already.
        register_vector(self._conn)

        table = _quote_ident(self._table)
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cur.execute(
                f"""
                CREATE TABLE {table} (
                    id TEXT PRIMARY KEY,
                    embedding vector({int(dim)}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )
            # HNSW index with cosine distance — pgvector's recommended
            # default for production cosine-similarity workloads.
            cur.execute(
                f"""
                CREATE INDEX ON {table}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._conn is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        import json

        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)

        table = _quote_ident(self._table)
        rows = [
            (rec.id, vec, json.dumps(dict(rec.metadata)))
            for rec, vec in zip(records_list, normalized, strict=True)
        ]
        with self._conn.cursor() as cur:
            # ON CONFLICT (id) DO UPDATE so re-inserting the same id is
            # well-defined (research scripts sometimes re-pin).
            cur.executemany(
                f"""
                INSERT INTO {table} (id, embedding, metadata)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                  SET embedding = EXCLUDED.embedding,
                      metadata = EXCLUDED.metadata
                """,
                rows,
            )

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        table = _quote_ident(self._table)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding, metadata FROM {table} WHERE id = %s",
                (record_id,),
            )
            row = cur.fetchone()
        if row is None:
            raise KeyError(record_id)
        vec, metadata = row
        return np.asarray(vec, dtype=np.float32), dict(metadata or {})

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._conn is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))[0]
        table = _quote_ident(self._table)
        with self._conn.cursor() as cur:
            # pgvector's <=> operator is cosine *distance* in [0, 2].
            # For L2-normalized inputs it lies in [0, 1] and similarity
            # is 1 - distance, monotonically consistent with the other
            # backends.
            cur.execute(
                f"""
                SELECT id, embedding <=> %s AS distance
                FROM {table}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q, q, int(k)),
            )
            rows = cur.fetchall()
        return [SearchResult(id=row[0], score=float(1.0 - row[1])) for row in rows]

    def close(self) -> None:
        if self._conn is not None:
            try:
                table = _quote_ident(self._table)
                with self._conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        self._conn = None
        self._dim = None
