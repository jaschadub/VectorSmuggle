# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Qdrant research backend.

Two construction modes are exposed because the empirical question
"does Qdrant's quantization scrub the steganographic payload?" needs a
controlled comparison:

  - ``QdrantBackend(quantize=False)`` --- HNSW over float32, no
    quantization. The lossless reference for this backend.
  - ``QdrantBackend(quantize=True)``  --- HNSW + scalar int8
    quantization (Qdrant's recommended production default). The
    realistic case for memory-constrained deployments.

The quantizing variant is the one a reviewer asks about under the
"quantization depth" feedback. Running the cross-backend study with
both modes lets us directly report the cosine drop attributable to
quantization.

An empirical nuance worth flagging: Qdrant stores both the float32
original and the int8 quantized form. ``retrieve()`` returns the
original; the quantization only affects ANN search behavior. The
implication for the threat model: an attacker with read access to
vectors recovers the bits losslessly even when quantization is on.
Quantization is a search-side artifact, not a storage-side defense.
This is the right finding for the paper --- it means scalar
quantization does not narrow the attack surface for adversaries
under threat models A or B.

Requires a reachable Qdrant instance. ``is_available()`` does a
single-shot connection check; if it fails, the cross-backend study
skips this backend silently.
"""

from __future__ import annotations

import os
import uuid
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


def _stable_uuid_from(record_id: str) -> str:
    """Qdrant point ids must be int or UUID strings; map our string ids
    deterministically to a UUID so callers can keep using their own
    string ids without a separate translation table."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, record_id))


class QdrantBackend(VectorBackend):
    """Qdrant-backed research backend with optional int8 quantization."""

    name = "qdrant"

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "vectorsmuggle_research",
        quantize: bool = False,
    ) -> None:
        self._url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._api_key = api_key or os.getenv("QDRANT_API_KEY")
        # Suffix the collection name when quantizing so a side-by-side
        # run does not collide with the lossless variant.
        self._collection_name = (
            f"{collection_name}_q8" if quantize else collection_name
        )
        self._quantize = quantize
        self._client = None
        self._dim: int | None = None
        self._id_lookup: dict[str, str] = {}

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as e:
            return False, f"qdrant-client not installed: {e}"
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url, api_key=api_key, timeout=2.0)
            # Lightweight ping: list collections returns quickly when up.
            client.get_collections()
        except Exception as e:  # noqa: BLE001 - qdrant raises a wide variety
            return False, f"qdrant unreachable at {url}: {e}"
        return True, ""

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"QdrantBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError as e:
            raise BackendUnavailable(f"qdrant-client not installed: {e}") from e
        self._client = QdrantClient(url=self._url, api_key=self._api_key)

        # Recreate to start clean for each test run.
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - missing collection is fine
            pass

        quantization_config = None
        if self._quantize:
            quantization_config = qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    always_ram=True,
                ),
            )

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
            quantization_config=quantization_config,
        )
        self._dim = dim
        self._id_lookup.clear()

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        from qdrant_client.http import models as qmodels

        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)

        points: list[qmodels.PointStruct] = []
        for rec, vec in zip(records_list, normalized, strict=True):
            point_uuid = _stable_uuid_from(rec.id)
            self._id_lookup[rec.id] = point_uuid
            payload = dict(rec.metadata)
            payload["__research_id"] = rec.id
            points.append(
                qmodels.PointStruct(
                    id=point_uuid,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        point_uuid = self._id_lookup.get(record_id)
        if point_uuid is None:
            raise KeyError(record_id)
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_uuid],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            raise KeyError(record_id)
        point = results[0]
        vec = np.asarray(point.vector, dtype=np.float32)
        payload = dict(point.payload or {})
        payload.pop("__research_id", None)
        return vec, payload

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))[0]
        # Use the modern query_points API.
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=q.tolist(),
                limit=k,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            # qdrant-client < 1.10: fall back to the search() method.
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=q.tolist(),
                limit=k,
                with_payload=True,
            )
        out: list[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            research_id = payload.get("__research_id") or str(hit.id)
            out.append(SearchResult(id=research_id, score=float(hit.score)))
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._dim = None
        self._id_lookup.clear()
