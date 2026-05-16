# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the OpenSearch research backend.

Runs the same round-trip / search / shape-validation contract that
``tests/unit/test_vector_backends.py`` exercises against the
in-process backends, but against a live OpenSearch container
(``opensearchproject/opensearch``). The same code path is what
production AWS OpenSearch Service / Serverless deployments use; the
AWS-only difference is SigV4 auth (covered by the unit tests).

To run locally:

    cd test_vector_dbs_docker
    docker compose up -d opensearch
    cd ..
    OPENSEARCH_URL=http://localhost:9200 \
      pytest tests/integration/test_opensearch_backend.py -m "docker"

The tests skip silently if OpenSearch is not reachable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.docker]


def _import_or_skip():
    try:
        from vector_backends.opensearch_backend import OpenSearchBackend
    except ImportError as e:
        pytest.skip(f"opensearch backend not importable: {e}")
    return OpenSearchBackend


def _skip_if_unavailable(cls):
    # The local docker-compose image runs with the security plugin
    # disabled, so HTTP on port 9200 with no auth is the default.
    os.environ.setdefault("OPENSEARCH_URL", "http://localhost:9200")
    avail, reason = cls.is_available()
    if not avail:
        pytest.skip(f"opensearch unavailable: {reason}")


@pytest.fixture(scope="module")
def opensearch_cls():
    cls = _import_or_skip()
    _skip_if_unavailable(cls)
    return cls


@pytest.fixture
def backend(opensearch_cls):
    import uuid
    index = f"vectorsmuggle_test_{uuid.uuid4().hex[:8]}"
    b = opensearch_cls(index_name=index)
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
    # k-NN _score is monotonically decreasing in distance.
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
    backend.close()  # must not raise


def test_metric_other_than_cosine_rejected(opensearch_cls):
    b = opensearch_cls(index_name="vectorsmuggle_test_never_opened")
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()
