# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Chroma research backend.

Chroma is included because it is the default vector store for many
LangChain.js / LlamaIndex tutorials and a natural production candidate
for small-to-medium RAG deployments. It uses HNSW under the hood and
stores embeddings as float32, so a round-trip is lossless --- which is
the empirical point the paper makes about Chroma. If steganography
survives Chroma's round-trip (and it does), the gap is not specific to
any particular ANN library; it is the lack of integrity inspection.

We use Chroma's in-process (in-memory) client for CI; no docker, no
network. The actual production deployment (chroma-server) uses the
same code path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from vector_backends.base import (
    BackendUnavailable,
    InsertRecord,
    SearchResult,
    VectorBackend,
)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


class ChromaBackend(VectorBackend):
    """In-process Chroma backend with cosine similarity."""

    name = "chroma"

    def __init__(self, collection_name: str = "vectorsmuggle_research") -> None:
        self._collection_name = collection_name
        self._client = None
        self._collection = None
        self._dim: int | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            return False, f"chromadb not installed: {e}"
        return True, ""

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"ChromaBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            import chromadb
        except ImportError as e:
            raise BackendUnavailable(f"chromadb not installed: {e}") from e
        self._client = chromadb.EphemeralClient()
        # Drop any leftover collection from a prior run in the same
        # process. EphemeralClient is supposed to start fresh but we
        # clear defensively.
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - chroma raises various error types here
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._collection is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        records_list = list(records)
        if not records_list:
            return
        ids = [r.id for r in records_list]
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)
        # Chroma requires non-empty metadata for every record; substitute
        # a marker dict for empty ones rather than letting it fail.
        metadatas = [r.metadata if r.metadata else {"_": ""} for r in records_list]
        self._collection.add(
            ids=ids,
            embeddings=normalized.tolist(),
            metadatas=metadatas,
        )

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._collection is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        result = self._collection.get(
            ids=[record_id],
            include=["embeddings", "metadatas"],
        )
        if not result["ids"]:
            raise KeyError(record_id)
        embedding = np.asarray(result["embeddings"][0], dtype=np.float32)
        metadata = result["metadatas"][0] if result["metadatas"] else {}
        # Strip the placeholder if we added one.
        if metadata == {"_": ""}:
            metadata = {}
        return embedding, dict(metadata or {})

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._collection is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))
        result = self._collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
        )
        out: list[SearchResult] = []
        ids = result["ids"][0] if result["ids"] else []
        # Chroma returns cosine *distance* in "distances"; convert to
        # similarity so all backends are monotonic on the score field.
        distances = result["distances"][0] if result["distances"] else []
        for record_id, dist in zip(ids, distances, strict=True):
            out.append(SearchResult(id=record_id, score=float(1.0 - dist)))
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._collection = None
        self._dim = None
