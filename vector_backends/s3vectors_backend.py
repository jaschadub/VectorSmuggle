# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""AWS S3 Vectors research backend.

S3 Vectors (announced at AWS re:Invent 2024, GA mid-2025) is the most
distinctive AWS vector store from the threat model's perspective: it
stores vectors as native S3 objects under a new ``s3vectors:`` API
surface, with similarity search built into the S3 data plane. The
quantization story is opaque to the customer (like Pinecone, unlike
pgvector). The empirical question the cross-backend study asks of
S3 Vectors is therefore "does an object-storage-backed vector store
preserve the steganographic payload?" — and the answer materially
extends the paper's claim of universality across the OSS / on-prem /
hyperscaler boundary.

The data plane is pure AWS — no local emulator exists. The unit
tests in ``tests/unit/test_vector_backends.py`` mock the boto3
client via ``moto.s3vectors``. The integration tests in
``tests/integration/test_s3vectors_backend.py`` are gated on
``AWS_S3VECTORS_TEST_BUCKET`` (a vector bucket the test can write
to) plus standard AWS credentials in the environment.

Connection parameters:

- ``AWS_REGION`` — the region the vector bucket lives in.
- ``AWS_S3VECTORS_BUCKET`` (optional) — bucket name to use. When
  unset the backend creates a UUID-suffixed bucket on ``open()`` and
  deletes it on ``close()``.
- ``AWS_S3VECTORS_INDEX`` (optional) — index name within the
  bucket. Defaults to ``vectorsmuggle-research``.
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

_DEFAULT_INDEX = "vectorsmuggle-research"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


class S3VectorsBackend(VectorBackend):
    """S3 Vectors-backed research backend.

    Construct, call ``open(dim)``, then insert / get / search /
    close. ``open()`` will create the vector bucket if the backend
    was constructed without an explicit one, and the index in either
    case; ``close()`` tears the index down, and also the bucket if
    we created it.
    """

    name = "s3vectors"

    def __init__(
        self,
        bucket: str | None = None,
        index_name: str = _DEFAULT_INDEX,
        region: str | None = None,
    ) -> None:
        self._region = region or os.environ.get("AWS_REGION") or "us-east-1"
        # Bucket: caller-provided > env-var > auto-create.
        env_bucket = os.environ.get("AWS_S3VECTORS_BUCKET")
        self._bucket = bucket or env_bucket
        self._bucket_owned = self._bucket is None
        self._index = index_name
        self._client: Any = None
        self._dim: int | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import boto3  # noqa: F401
        except ImportError as e:
            return False, f"boto3 not installed: {e}"
        # We don't probe AWS here — that costs an HTTP call and may
        # surface IAM / credential errors that aren't really about
        # "is the backend usable." Defer to open() for the real check.
        return True, ""

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"S3VectorsBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            import boto3
        except ImportError as e:
            raise BackendUnavailable(f"boto3 not installed: {e}") from e
        self._client = boto3.client("s3vectors", region_name=self._region)

        if self._bucket_owned:
            self._bucket = f"vs-research-{uuid.uuid4().hex[:10]}"
            self._client.create_vector_bucket(vectorBucketName=self._bucket)

        # Drop any prior index of the same name so repeated runs start
        # clean (only matters when caller provided a persistent bucket).
        try:
            self._client.delete_index(
                vectorBucketName=self._bucket,
                indexName=self._index,
            )
        except Exception:  # noqa: BLE001 - missing index is fine
            pass

        self._client.create_index(
            vectorBucketName=self._bucket,
            indexName=self._index,
            dataType="float32",
            dimension=int(dim),
            distanceMetric="cosine",
        )
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)

        # S3 Vectors accepts up to 500 vectors per put_vectors call;
        # batch defensively at 250 so we never hit the limit.
        batch_size = 250
        for start in range(0, len(records_list), batch_size):
            chunk = records_list[start : start + batch_size]
            chunk_vecs = normalized[start : start + batch_size]
            payload = []
            for rec, vec in zip(chunk, chunk_vecs, strict=True):
                payload.append(
                    {
                        "key": rec.id,
                        "data": {"float32": vec.tolist()},
                        "metadata": _coerce_metadata(rec.metadata),
                    }
                )
            self._client.put_vectors(
                vectorBucketName=self._bucket,
                indexName=self._index,
                vectors=payload,
            )

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        resp = self._client.get_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            keys=[record_id],
            returnData=True,
            returnMetadata=True,
        )
        vectors = resp.get("vectors") or []
        if not vectors:
            raise KeyError(record_id)
        entry = vectors[0]
        data = entry.get("data") or {}
        floats = data.get("float32")
        if floats is None:
            raise KeyError(f"{record_id!r} has no float32 data")
        vec = np.asarray(floats, dtype=np.float32)
        metadata = dict(entry.get("metadata") or {})
        return vec, metadata

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))[0]
        resp = self._client.query_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            topK=int(k),
            queryVector={"float32": q.tolist()},
            returnMetadata=False,
            returnDistance=True,
        )
        out: list[SearchResult] = []
        for hit in resp.get("vectors") or []:
            distance = float(hit.get("distance") or 0.0)
            # S3 Vectors returns cosine *distance* in [0, 2]; for
            # L2-normalized inputs it's in [0, 1] and similarity is
            # 1 - distance, monotonic across backends.
            out.append(
                SearchResult(id=str(hit.get("key")), score=float(1.0 - distance))
            )
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_index(
                    vectorBucketName=self._bucket,
                    indexName=self._index,
                )
            except Exception:  # noqa: BLE001
                pass
            if self._bucket_owned and self._bucket is not None:
                try:
                    self._client.delete_vector_bucket(vectorBucketName=self._bucket)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._dim = None


def _coerce_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """S3 Vectors metadata is a Document type — JSON-able values
    only. Coerce non-JSON-able values to repr() so the round-trip is
    unambiguous; bool/int/float/str pass through.
    """
    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = v
        else:
            out[k] = repr(v)
    return out
