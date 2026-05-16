# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the S3 Vectors research backend.

S3 Vectors has no local emulator, so this file is gated on real AWS
credentials being present in the environment. It skips silently
otherwise — same pattern as ``tests/test_adapter_pinecone.py`` in
the VectorPin OSS repo.

To run locally:

    export AWS_REGION=us-east-1
    export AWS_PROFILE=<your-profile>           # or use AWS_ACCESS_KEY_ID etc
    export AWS_S3VECTORS_TEST=1                  # explicit opt-in
    pytest tests/integration/test_s3vectors_backend.py -m "integration"

The opt-in env var is intentional — S3 Vectors test runs cost real
money (well under a cent per run, but non-zero) and write to a
bucket in your account. We make the operator type the opt-in.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration]


def _import_or_skip():
    try:
        from vector_backends.s3vectors_backend import S3VectorsBackend
    except ImportError as e:
        pytest.skip(f"s3vectors backend not importable: {e}")
    return S3VectorsBackend


def _skip_unless_opt_in():
    if os.environ.get("AWS_S3VECTORS_TEST", "").lower() not in ("1", "true", "yes"):
        pytest.skip(
            "set AWS_S3VECTORS_TEST=1 to opt in to live S3 Vectors tests "
            "(creates and deletes a bucket in the configured AWS account)"
        )


@pytest.fixture(scope="module")
def s3vectors_cls():
    _skip_unless_opt_in()
    return _import_or_skip()


@pytest.fixture
def backend(s3vectors_cls):
    region = os.environ.get("AWS_REGION", "us-east-1")
    b = s3vectors_cls(region=region)
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
    # S3 Vectors stores float32 verbatim; round-trip is lossless.
    assert cos > 0.999
    assert metadata.get("src") == "test"


def test_search_finds_self(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[7], k=1)
    assert hits
    assert hits[0].id == "r7"


def test_search_returns_correct_k(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[0], k=5)
    assert len(hits) == 5
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_get_by_id_raises_for_unknown_id(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a", "b"], small_corpus[:2])
    with pytest.raises(KeyError):
        backend.get_by_id("nope")
