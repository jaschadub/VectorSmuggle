# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""AWS MemoryDB for Redis (vector search) research backend.

MemoryDB for Redis is included because Amazon's vector-search story
on Redis runs the same Redis Search engine that ships in
``redis/redis-stack-server`` locally. Identical FT.SEARCH command
surface, identical HNSW indexing, identical float32 storage. The
local Redis Stack container is the offline test target; AWS MemoryDB
is the production target. From the cross-backend study's perspective
the empirical question is the same in both cases: does a
steganographic payload survive Redis Search's HNSW index after the
DB normalizes and re-stores the vector?

The answer (spoiler) is yes — Redis Search stores the embedding as
the literal float32 bytes in a HASH field; the HNSW index is built
over those bytes and search returns the originals. Quantization is
not a default option in Redis Search 2.x. Same shape as Chroma's
HNSW path.

Connection parameters come from ``REDIS_URL`` (default
``redis://localhost:6379``) for local Redis Stack, or a TLS-enabled
URL for MemoryDB:

    rediss://app:<auth>@<memorydb-endpoint>:6379/0

``is_available()`` does a short PING probe so the cross-backend study
can skip silently when no Redis is running.
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

_DEFAULT_URL = "redis://localhost:6379"
_DEFAULT_INDEX = "vectorsmuggle_research"
_DEFAULT_PREFIX = "vs:"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


def _f32_to_bytes(v: np.ndarray) -> bytes:
    """Little-endian float32 byte representation. Redis Search
    expects the raw bytes packed in the same order as the index
    declared."""
    return v.astype(np.float32).tobytes(order="C")


def _bytes_to_f32(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)


class MemoryDBBackend(VectorBackend):
    """Redis Search-backed research backend.

    Wraps an ``FT.CREATE`` index over a HASH-prefixed key space. The
    same code targets local ``redis/redis-stack-server`` (no auth,
    plaintext) and AWS MemoryDB (TLS + auth token, IAM, or
    user-and-password).
    """

    name = "memorydb"

    def __init__(
        self,
        url: str | None = None,
        index_name: str = _DEFAULT_INDEX,
        key_prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        self._url = url or os.getenv("REDIS_URL", _DEFAULT_URL)
        self._index = index_name
        self._prefix = key_prefix
        self._client: Any = None
        self._dim: int | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import redis  # noqa: F401
        except ImportError as e:
            return False, f"redis not installed: {e}"
        url = os.getenv("REDIS_URL", _DEFAULT_URL)
        try:
            import redis as redis_mod
            client = redis_mod.from_url(url, socket_timeout=2)
            client.ping()
        except Exception as e:  # noqa: BLE001
            return False, f"redis unreachable at {url}: {e}"
        return True, ""

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"MemoryDBBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            import redis
        except ImportError as e:
            raise BackendUnavailable(f"redis not installed: {e}") from e
        self._client = redis.from_url(self._url, decode_responses=False)

        # Drop any prior data + index so repeated test runs start fresh.
        self._drop_index_and_keys()

        # FT.CREATE ON HASH PREFIX 1 <prefix> SCHEMA <field> VECTOR HNSW ...
        # Spaces between the integer constants matter: redis-py passes
        # each arg as a separate token to the server.
        try:
            self._client.execute_command(
                "FT.CREATE",
                self._index,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._prefix,
                "SCHEMA",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(int(dim)),
                "DISTANCE_METRIC",
                "COSINE",
            )
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(
                f"FT.CREATE failed (is Redis Search loaded? "
                f"required for AWS MemoryDB + redis-stack): {e}"
            ) from e
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)

        # One HSET per record. Pipeline so we get one network round-trip
        # for the batch. Don't use MSET: Redis Search needs the key to
        # be a HASH with named fields, which MSET doesn't support.
        pipe = self._client.pipeline(transaction=False)
        for rec, vec in zip(records_list, normalized, strict=True):
            key = f"{self._prefix}{rec.id}".encode()
            fields = {
                b"embedding": _f32_to_bytes(vec),
                b"__research_id": rec.id.encode(),
            }
            for k, v in rec.metadata.items():
                # Redis HASH stores bytes; coerce metadata values
                # through repr so the round-trip is unambiguous.
                fields[f"meta_{k}".encode()] = str(v).encode()
            pipe.hset(key, mapping=fields)
        pipe.execute()

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        if self._dim is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        key = f"{self._prefix}{record_id}".encode()
        raw = self._client.hgetall(key)
        if not raw:
            raise KeyError(record_id)
        emb_bytes = raw.get(b"embedding")
        if emb_bytes is None:
            raise KeyError(f"{record_id!r} has no 'embedding' field")
        vec = _bytes_to_f32(emb_bytes, self._dim).copy()
        metadata = {}
        for k_b, v_b in raw.items():
            k = k_b.decode()
            if k in ("embedding", "__research_id"):
                continue
            if k.startswith("meta_"):
                metadata[k[len("meta_"):]] = v_b.decode()
        return vec, metadata

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))[0]
        q_bytes = _f32_to_bytes(q)

        # FT.SEARCH <index> "*=>[KNN k @embedding $vec AS score]"
        # PARAMS 2 vec <bytes> RETURN 2 __research_id score DIALECT 2
        try:
            raw = self._client.execute_command(
                "FT.SEARCH",
                self._index,
                f"*=>[KNN {int(k)} @embedding $vec AS score]",
                "PARAMS",
                "2",
                "vec",
                q_bytes,
                "RETURN",
                "2",
                "__research_id",
                "score",
                "SORTBY",
                "score",
                "DIALECT",
                "2",
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"FT.SEARCH failed: {e}") from e

        # Response shape: [total, key1, [field, value, ...], key2, [...]].
        if not raw or len(raw) < 1:
            return []
        out: list[SearchResult] = []
        # Pairs after raw[0] are (key, fields_list).
        for i in range(1, len(raw), 2):
            if i + 1 >= len(raw):
                break
            fields = raw[i + 1]
            field_map: dict[bytes, bytes] = {}
            for j in range(0, len(fields), 2):
                if j + 1 >= len(fields):
                    break
                field_map[fields[j]] = fields[j + 1]
            research_id = field_map.get(b"__research_id", b"").decode()
            score_b = field_map.get(b"score", b"0")
            try:
                score = float(score_b)
            except (TypeError, ValueError):
                score = 0.0
            # Redis Search returns COSINE *distance* (0 = identical).
            # Convert to similarity for the cross-backend monotonicity
            # contract.
            out.append(SearchResult(id=research_id, score=float(1.0 - score)))
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                self._drop_index_and_keys()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._dim = None

    # ---- internals ---------------------------------------------------------

    def _drop_index_and_keys(self) -> None:
        """Best-effort: drop the FT index (DD = delete docs) and any
        leftover keys under the prefix. Used on open() so repeated
        runs start fresh, and on close() to leave nothing behind."""
        try:
            self._client.execute_command("FT.DROPINDEX", self._index, "DD")
        except Exception:  # noqa: BLE001 - index might not exist yet
            pass
        # Belt-and-suspenders: scan for any remaining keys.
        try:
            for k in self._client.scan_iter(match=f"{self._prefix}*", count=500):
                self._client.delete(k)
        except Exception:  # noqa: BLE001
            pass
