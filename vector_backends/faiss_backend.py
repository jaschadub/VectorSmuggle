# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""FAISS-backed research backends.

Two variants ship in this module because they cover the two ends of the
``does the index alter the stored vector?`` spectrum:

  - ``FaissFlatBackend``: ``IndexFlatIP`` over an L2-normalized
    embedding. The index stores vectors verbatim. A round-trip is
    lossless to floating-point precision. This is the control case ---
    if the steganographic payload doesn't survive flat FAISS, it won't
    survive anything.

  - ``FaissHNSWBackend``: ``IndexHNSWFlat`` over an L2-normalized
    embedding. The HNSW graph is built over the same float32 vectors
    so storage is also lossless, but search is approximate. This is
    the realistic case for production-FAISS deployments.

Neither requires a server. Both work in a single process. Suitable for
CI without docker.

Quantizing variants (``IndexIVFPQ``, ``IndexBinaryHash``) are
deliberately not exported here. They aggressively destroy direction
information and the empirical question they answer is ``can
quantization kill steganography'', not ``does the universal gap
exist''. The cross-backend study targets the latter; quantization is
covered separately in ``scripts/preprint_extensions.py``.
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
    """L2-normalize each row so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


class _FaissBase(VectorBackend):
    """Shared insert/get/search machinery for both FAISS variants."""

    def __init__(self) -> None:
        self._faiss = None
        self._index = None
        self._dim: int | None = None
        # Map id <-> ordinal position in the FAISS index. FAISS doesn't
        # store string ids natively; we maintain a parallel dict.
        self._id_to_pos: dict[str, int] = {}
        self._positions: list[str] = []
        self._stored_vectors: list[np.ndarray] = []
        self._metadata: dict[str, dict[str, Any]] = {}

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            return False, f"faiss-cpu not installed: {e}"
        return True, ""

    # Subclass hook: build and return the index for ``dim`` dimensions.
    def _build_index(self, dim: int) -> Any:  # noqa: ANN401 - faiss types
        raise NotImplementedError

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"{type(self).__name__} only supports metric='cosine'; got {metric!r}"
            )
        try:
            import faiss
        except ImportError as e:
            raise BackendUnavailable(f"faiss-cpu not installed: {e}") from e
        self._faiss = faiss
        self._index = self._build_index(dim)
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._index is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)
        self._index.add(normalized)
        for rec, vec in zip(records_list, normalized, strict=True):
            self._id_to_pos[rec.id] = len(self._positions)
            self._positions.append(rec.id)
            # Keep a copy so we can return the stored vector via id lookup
            # without round-tripping through the index (FAISS exposes
            # vectors via ``reconstruct(i)`` on flat indices but the
            # interface differs across index types).
            self._stored_vectors.append(vec.copy())
            self._metadata[rec.id] = dict(rec.metadata)

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if record_id not in self._id_to_pos:
            raise KeyError(record_id)
        pos = self._id_to_pos[record_id]
        return self._stored_vectors[pos].copy(), dict(self._metadata[record_id])

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._index is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))
        scores, indices = self._index.search(q, k)
        out: list[SearchResult] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0 or idx >= len(self._positions):
                continue
            out.append(SearchResult(id=self._positions[idx], score=float(score)))
        return out

    def close(self) -> None:
        # FAISS in-process indices have no resources to release; reset
        # state so a subsequent open() is clean.
        self._index = None
        self._dim = None
        self._id_to_pos.clear()
        self._positions.clear()
        self._stored_vectors.clear()
        self._metadata.clear()


class FaissFlatBackend(_FaissBase):
    """Lossless flat (exact-search) FAISS index. The control case."""

    name = "faiss_flat"

    def _build_index(self, dim: int) -> Any:  # noqa: ANN401
        # IndexFlatIP gives us inner product. Combined with
        # _normalize() this is cosine similarity.
        return self._faiss.IndexFlatIP(dim)


class FaissHNSWBackend(_FaissBase):
    """HNSW-indexed FAISS. Approximate search; storage is still float32."""

    name = "faiss_hnsw"

    def __init__(self, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        super().__init__()
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

    def _build_index(self, dim: int) -> Any:  # noqa: ANN401
        # IndexHNSWFlat with METRIC_INNER_PRODUCT. The graph is built
        # over normalized vectors so search is effectively cosine.
        index = self._faiss.IndexHNSWFlat(dim, self._m, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search
        return index
