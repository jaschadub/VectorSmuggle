# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Round-trip backend interface.

A backend's only job is: take a batch of vectors, store them, and let us
read them back out (both by id and by similarity search). Everything we
care about for the research --- whether the attacker's hidden bits
survived the DB's quantization, whether retrieval-utility changed,
whether the DB's index introduced its own distortion --- is measurable
from those three operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class BackendUnavailable(RuntimeError):  # noqa: N818 - "Unavailable" reads better than "UnavailableError" here
    """Raised when a backend's optional dependency is missing or its
    service is unreachable. Callers should catch this and skip the
    backend rather than failing the whole study."""


@dataclass(frozen=True)
class InsertRecord:
    """One row to insert: stable id, dense vector, optional metadata."""

    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """One hit from a similarity search."""

    id: str
    score: float
    """Similarity score. Higher = more similar. Backends that return a
    distance natively must convert (e.g. cosine = 1 - cosine_distance)
    so this field is monotonic across backends."""


class VectorBackend(ABC):
    """Minimal contract every research backend implements.

    Lifecycle:
      1. Construct.
      2. ``open(dim, metric="cosine")`` to allocate the index.
      3. ``insert([...])`` one or more times.
      4. Any number of ``get_by_id`` and ``search`` calls.
      5. ``close()`` to release resources.

    Subclasses MUST be safe to ``close()`` even if ``open()`` was never
    called or raised --- this lets test cleanup be unconditional.
    """

    name: str = "abstract"

    @classmethod
    @abstractmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(available, reason)``.

        ``available`` is True iff this backend can be used right now in
        this process (deps importable, service reachable). ``reason`` is
        a short human string explaining why not, when False --- safe to
        log to the user.
        """

    @abstractmethod
    def open(self, dim: int, *, metric: str = "cosine") -> None:
        """Allocate the index for ``dim``-dimensional vectors."""

    @abstractmethod
    def insert(self, records: Iterable[InsertRecord]) -> None:
        """Write the records into the backing index."""

    @abstractmethod
    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Fetch a single record by id.

        The returned vector is whatever the backend has stored, which
        may differ from what was inserted due to quantization or
        normalization. Callers compare it to the original to measure
        round-trip fidelity.
        """

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Top-k similarity search."""

    def close(self) -> None:  # noqa: B027
        """Release resources. Default is a no-op; backends override."""

    # Convenience methods. Defaults in terms of the abstract operations.

    def insert_arrays(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert from parallel arrays --- the form the empirical
        scripts work in."""
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D vectors, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"id count {len(ids)} != vector count {vectors.shape[0]}"
            )
        meta_iter = metadatas if metadatas is not None else [{}] * len(ids)
        records = [
            InsertRecord(id=str(i), vector=v, metadata=m)
            for i, v, m in zip(ids, vectors, meta_iter, strict=True)
        ]
        self.insert(records)

    def __enter__(self) -> VectorBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def list_backends() -> dict[str, type[VectorBackend]]:
    """Return every concrete backend class shipped in this package."""
    # Imported here to avoid pulling optional deps at package import.
    from vector_backends.faiss_backend import (
        FaissFlatBackend,
        FaissHNSWBackend,
        FaissPQBackend,
    )

    backends: dict[str, type[VectorBackend]] = {
        FaissFlatBackend.name: FaissFlatBackend,
        FaissHNSWBackend.name: FaissHNSWBackend,
        FaissPQBackend.name: FaissPQBackend,
    }

    # Optional backends are wrapped in try/except so importing one with
    # a missing dep does not break the whole list.
    try:
        from vector_backends.chroma_backend import ChromaBackend
        backends[ChromaBackend.name] = ChromaBackend
    except ImportError:
        pass
    try:
        from vector_backends.qdrant_backend import QdrantBackend
        backends[QdrantBackend.name] = QdrantBackend
    except ImportError:
        pass
    try:
        from vector_backends.pgvector_backend import PgVectorBackend
        backends[PgVectorBackend.name] = PgVectorBackend
    except ImportError:
        pass

    return backends


def available_backends() -> dict[str, type[VectorBackend]]:
    """Subset of ``list_backends()`` whose ``is_available()`` is True.

    Convenient for the cross-backend study: skip silently, run on
    whatever's installed and reachable.
    """
    return {
        name: cls
        for name, cls in list_backends().items()
        if cls.is_available()[0]
    }
