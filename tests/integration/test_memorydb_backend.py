# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the MemoryDB / Redis Search research backend.

Runs against a local ``redis/redis-stack-server`` container which
runs the same Redis Search engine that AWS MemoryDB exposes
(``FT.SEARCH ... VECTOR HNSW``). The data plane is identical; only
auth + TLS differ on AWS, and those are covered by separate live
tests under valid AWS credentials.

To run locally:

    cd test_vector_dbs_docker
    docker compose up -d redis-stack
    cd ..
    REDIS_URL=redis://localhost:6379 \
      pytest tests/integration/test_memorydb_backend.py -m "docker"

Skips silently when Redis isn't reachable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]


def _import_or_skip():
    try:
        from vector_backends.memorydb_backend import MemoryDBBackend
    except ImportError as e:
        pytest.skip(f"memorydb backend not importable: {e}")
    return MemoryDBBackend


def _skip_if_unavailable(cls):
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    avail, reason = cls.is_available()
    if not avail:
        pytest.skip(f"redis unavailable: {reason}")


@pytest.fixture(scope="module")
def memorydb_cls():
    cls = _import_or_skip()
    _skip_if_unavailable(cls)
    return cls


@pytest.fixture
def backend(memorydb_cls):
    import uuid
    suffix = uuid.uuid4().hex[:8]
    b = memorydb_cls(
        index_name=f"vs_test_idx_{suffix}",
        key_prefix=f"vs_test_{suffix}:",
    )
    yield b
    b.close()


@pytest.fixture
def small_corpus():
    rng = np.random.default_rng(42)
    vectors = rng.normal(0, 1, size=(20, 32)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def test_open_then_insert_then_get_round_trip(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus, [{"src": "test"}] * len(ids))
    got, metadata = backend.get_by_id("r5")
    cos = float(
        np.dot(got, small_corpus[5])
        / (np.linalg.norm(got) * np.linalg.norm(small_corpus[5]))
    )
    assert cos > 0.999
    assert metadata.get("src") == "test"


def test_search_finds_self(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[7], k=1)
    assert hits, "search returned no hits"
    assert hits[0].id == "r7"


def test_search_returns_correct_k(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[0], k=5)
    assert len(hits) == 5
    scores = [h.score for h in hits]
    # Similarity (1 - cosine_distance) is monotonically non-increasing.
    assert scores == sorted(scores, reverse=True)


def test_get_by_id_raises_for_unknown_id(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a", "b"], small_corpus[:2])
    with pytest.raises(KeyError):
        backend.get_by_id("nope")


def test_close_is_idempotent(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a"], small_corpus[:1])
    backend.close()
    backend.close()


def test_metric_other_than_cosine_rejected(memorydb_cls):
    b = memorydb_cls(index_name="vs_never_opened")
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()
