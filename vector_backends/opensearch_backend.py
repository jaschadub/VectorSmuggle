# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""AWS OpenSearch (Service + Serverless) research backend.

OpenSearch is included because it is the AWS-native vector option
most enterprise customers reach for: existing Elasticsearch
familiarity, fully managed, supports both the FAISS and Lucene
engines via the k-NN plugin. The empirical question for the cross-
backend study is the same as for any other HNSW-based backend: does
a steganographic payload survive the round-trip through k-NN's
index? With ``ef_search`` and ``ef_construction`` at OpenSearch's
defaults, and the default FAISS engine storing float32, the answer
is yes — the index is built over the same float32 bytes the caller
indexed, and ``_source`` retrieval round-trips losslessly.

The backend uses the open ``opensearch-py`` data plane, which is
identical between the local ``opensearchproject/opensearch:latest``
container, AWS OpenSearch Service (provisioned), and AWS OpenSearch
Serverless. The auth path differs:

- **Local**: basic-auth (``admin``/``admin``) or anonymous.
- **AWS OpenSearch Service**: HTTPS + IAM SigV4 via
  ``requests-aws4auth``. Service code ``es``.
- **AWS OpenSearch Serverless**: HTTPS + IAM SigV4 with service code
  ``aoss``. The Serverless data plane only allows certain index
  settings (no ``index.knn`` toggle — k-NN is on by default).

Connection parameters come from:

- ``OPENSEARCH_URL`` (default ``https://localhost:9200``)
- ``OPENSEARCH_USERNAME`` / ``OPENSEARCH_PASSWORD`` (default ``admin``/
  ``admin`` for the local container; ignored when SigV4 is enabled)
- ``OPENSEARCH_VERIFY_TLS`` (default ``false`` for the local container's
  self-signed cert; set to ``true`` for AWS)
- ``AWS_OPENSEARCH_REGION`` — when set, SigV4 auth is enabled and the
  ``aoss`` service code is used unless ``AWS_OPENSEARCH_SERVICE`` is
  set to ``es`` explicitly.

``is_available()`` does a short cluster-info probe so the cross-
backend study can skip silently when no OpenSearch is running.
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

_DEFAULT_URL = "https://localhost:9200"
_DEFAULT_INDEX = "vectorsmuggle_research"
_DEFAULT_USER = "admin"
# OpenSearch 2.12+ requires an initial admin password; the local
# docker image accepts it via the OPENSEARCH_INITIAL_ADMIN_PASSWORD
# env var. Tests should pass a known value; we default to the same
# constant that the test compose file uses.
_DEFAULT_PASS = "vectorsmuggle-Admin-2026"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return (vectors / norms).astype(np.float32)


def _is_serverless() -> bool:
    """True iff caller asked for the AWS OpenSearch Serverless data
    plane (service code 'aoss'). Service code 'es' is the
    provisioned variant."""
    if "AWS_OPENSEARCH_REGION" not in os.environ:
        return False
    svc = os.environ.get("AWS_OPENSEARCH_SERVICE", "aoss").lower()
    return svc == "aoss"


class OpenSearchBackend(VectorBackend):
    """OpenSearch k-NN-backed research backend.

    Wraps a single index. ``open()`` recreates the index so repeated
    runs start clean; ``close()`` drops it.
    """

    name = "opensearch"

    def __init__(
        self,
        url: str | None = None,
        index_name: str = _DEFAULT_INDEX,
    ) -> None:
        self._url = url or os.getenv("OPENSEARCH_URL", _DEFAULT_URL)
        self._index = index_name
        self._client: Any = None
        self._dim: int | None = None
        # Auto-suffix the index name on Serverless because Serverless
        # requires unique names per call (the data plane is shared
        # across collections of a given type).
        if _is_serverless():
            self._index = f"{index_name}-{uuid.uuid4().hex[:8]}"

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import opensearchpy  # noqa: F401
        except ImportError as e:
            return False, f"opensearch-py not installed: {e}"
        url = os.getenv("OPENSEARCH_URL", _DEFAULT_URL)
        try:
            client = cls._build_client(url)
            client.info()
        except Exception as e:  # noqa: BLE001
            return False, f"opensearch unreachable at {url}: {e}"
        return True, ""

    @staticmethod
    def _build_client(url: str) -> Any:
        from opensearchpy import OpenSearch, RequestsHttpConnection

        verify_certs = os.environ.get("OPENSEARCH_VERIFY_TLS", "false").lower() == "true"

        # SigV4 path (AWS).
        region = os.environ.get("AWS_OPENSEARCH_REGION")
        if region:
            try:
                import boto3
                from requests_aws4auth import AWS4Auth
            except ImportError as e:  # noqa: BLE001
                raise BackendUnavailable(
                    f"boto3 + requests-aws4auth required for AWS SigV4: {e}"
                ) from e
            svc = os.environ.get("AWS_OPENSEARCH_SERVICE", "aoss")
            creds = boto3.Session().get_credentials()
            auth = AWS4Auth(
                creds.access_key,
                creds.secret_key,
                region,
                svc,
                session_token=creds.token,
            )
            # opensearch-py expects host+port split from the URL.
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or 443
            return OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )

        # Local / non-AWS path: basic auth, optional TLS.
        username = os.environ.get("OPENSEARCH_USERNAME", _DEFAULT_USER)
        password = os.environ.get("OPENSEARCH_PASSWORD", _DEFAULT_PASS)
        return OpenSearch(
            hosts=[url],
            http_auth=(username, password),
            use_ssl=url.startswith("https"),
            verify_certs=verify_certs,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
        )

    def open(self, dim: int, *, metric: str = "cosine") -> None:
        if metric != "cosine":
            raise ValueError(
                f"OpenSearchBackend only supports metric='cosine'; got {metric!r}"
            )
        try:
            self._client = self._build_client(self._url)
        except BackendUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(f"opensearch client build: {e}") from e

        # Drop any prior index so repeated runs start clean. Serverless
        # has its own lifecycle so the delete is best-effort.
        try:
            self._client.indices.delete(index=self._index)
        except Exception:  # noqa: BLE001 - 404 on first run is fine
            pass

        # k-NN index mapping. OS 2.18+ removed FAISS + cosinesimil
        # support — FAISS now offers l2 / innerproduct only. Because
        # insert() L2-normalizes every vector before indexing, inner
        # product on normalized inputs IS cosine similarity (and the
        # AWS-default engine for OpenSearch Service / Serverless).
        # The space_type lives at the mapping level (top of the method
        # config); engine: faiss with innerproduct is the production
        # AWS-typical configuration.
        body: dict[str, Any] = {
            "settings": {},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": int(dim),
                        "space_type": "innerproduct",
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                        },
                    },
                    "__research_id": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }
        if not _is_serverless():
            # Provisioned cluster: explicitly enable k-NN on the index.
            body["settings"]["index"] = {"knn": True}

        self._client.indices.create(index=self._index, body=body)
        self._dim = dim

    def insert(self, records: Iterable[InsertRecord]) -> None:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        from opensearchpy.helpers import bulk as os_bulk

        records_list = list(records)
        if not records_list:
            return
        vectors = np.stack([r.vector for r in records_list]).astype(np.float32)
        normalized = _normalize(vectors)

        actions = []
        for rec, vec in zip(records_list, normalized, strict=True):
            actions.append(
                {
                    "_index": self._index,
                    "_id": rec.id,
                    "_source": {
                        "embedding": vec.tolist(),
                        "__research_id": rec.id,
                        "metadata": dict(rec.metadata),
                    },
                }
            )
        # refresh="true" makes the docs immediately searchable, which
        # the round-trip test needs. Don't use refresh in production.
        os_bulk(self._client, actions, refresh=True)

    def get_by_id(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        try:
            doc = self._client.get(index=self._index, id=record_id)
        except Exception as e:  # noqa: BLE001
            # opensearchpy raises NotFoundError; we surface as KeyError
            # to match the contract.
            if "NotFoundError" in type(e).__name__ or "404" in str(e):
                raise KeyError(record_id) from e
            raise
        src = doc.get("_source") or {}
        emb = src.get("embedding")
        if emb is None:
            raise KeyError(f"{record_id!r} has no embedding")
        vec = np.asarray(emb, dtype=np.float32)
        metadata = dict(src.get("metadata") or {})
        return vec, metadata

    def search(self, query: np.ndarray, k: int = 10) -> list[SearchResult]:
        if self._client is None:
            raise RuntimeError("backend not opened; call .open(dim) first")
        q = _normalize(query.reshape(1, -1))[0]
        body = {
            "size": int(k),
            "_source": ["__research_id"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": q.tolist(),
                        "k": int(k),
                    }
                }
            },
        }
        resp = self._client.search(index=self._index, body=body)
        out: list[SearchResult] = []
        hits = (resp.get("hits") or {}).get("hits") or []
        for hit in hits:
            rid = ((hit.get("_source") or {}).get("__research_id")) or hit.get("_id")
            # OpenSearch k-NN returns 1 / (1 + distance) as _score for
            # cosinesimil, monotonically increasing in similarity.
            score = float(hit.get("_score") or 0.0)
            out.append(SearchResult(id=str(rid), score=score))
        return out

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.indices.delete(index=self._index)
            except Exception:  # noqa: BLE001
                pass
        self._client = None
        self._dim = None
