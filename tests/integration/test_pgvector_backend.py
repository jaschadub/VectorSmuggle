# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the pgvector research backend.

These run the same round-trip / search / shape-validation contract that
``tests/unit/test_vector_backends.py`` exercises against the in-process
backends, but against an actual ``ankane/pgvector`` container.

To run locally:

    cd test_vector_dbs_docker
    docker compose up -d postgres
    cd ..
    pytest tests/integration/test_pgvector_backend.py -m "docker"

The tests skip silently if pgvector is not reachable (no docker
running, no ``psycopg`` installed, etc.) so this file is safe to
collect on a developer laptop without docker.
"""

from __future__ import annotations

import numpy as np
import pytest

from vector_backends.base import BackendUnavailable

pytestmark = [pytest.mark.integration, pytest.mark.docker]


def _import_or_skip():
    try:
        from vector_backends.pgvector_backend import PgVectorBackend
    except ImportError as e:  # pragma: no cover - exercised only when deps missing
        pytest.skip(f"pgvector backend not importable: {e}")
    return PgVectorBackend


def _skip_if_unavailable(cls):
    available, reason = cls.is_available()
    if not available:
        pytest.skip(f"pgvector unavailable: {reason}")


@pytest.fixture(scope="module")
def pgvector_cls():
    cls = _import_or_skip()
    _skip_if_unavailable(cls)
    return cls


@pytest.fixture
def backend(pgvector_cls):
    # Use a per-test table name so parallel pytest-xdist workers don't
    # collide on the same DROP TABLE / CREATE TABLE sequence.
    import uuid
    table = f"vectorsmuggle_test_{uuid.uuid4().hex[:8]}"
    b = pgvector_cls(table_name=table)
    yield b
    b.close()


@pytest.fixture
def small_corpus():
    rng = np.random.default_rng(42)
    vectors = rng.normal(0, 1, size=(20, 32)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def test_is_available_returns_tuple(pgvector_cls):
    avail, reason = pgvector_cls.is_available()
    assert avail is True
    assert isinstance(reason, str)


def test_registered_in_list_backends(pgvector_cls):
    from vector_backends import list_backends
    assert pgvector_cls.name in list_backends()
    assert list_backends()[pgvector_cls.name] is pgvector_cls


def test_open_then_insert_then_get_round_trip(backend, small_corpus):
    """The point of the round-trip test: pgvector stores float32 in the
    `vector` column type, so a normalized embedding read back must be
    cosine-equivalent to the original.
    """
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    got, metadata = backend.get_by_id("r5")
    cos = float(
        np.dot(got, small_corpus[5])
        / (np.linalg.norm(got) * np.linalg.norm(small_corpus[5]))
    )
    # pgvector round-trips at fp32 precision — cosine should be ~1.0.
    assert cos > 0.999
    assert metadata == {}


def test_search_finds_self(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[7], k=1)
    assert hits, "search returned no hits"
    assert hits[0].id == "r7"
    # cosine similarity score should be monotonic with closeness; the
    # self-query should be very close to 1.0.
    assert hits[0].score > 0.99


def test_search_returns_correct_k(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[0], k=5)
    assert len(hits) == 5
    # First hit is the query itself.
    assert hits[0].id == "r0"
    # Scores are monotonically non-increasing.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_metadata_round_trips(backend, small_corpus):
    """Metadata survives the JSONB column unchanged."""
    from vector_backends import InsertRecord

    backend.open(dim=small_corpus.shape[1])
    records = [
        InsertRecord(
            id="r0",
            vector=small_corpus[0],
            metadata={"source": "test", "ord": 0, "tags": ["a", "b"]},
        ),
        InsertRecord(id="r1", vector=small_corpus[1], metadata={"source": "test", "ord": 1}),
    ]
    backend.insert(records)
    _, md0 = backend.get_by_id("r0")
    _, md1 = backend.get_by_id("r1")
    assert md0 == {"source": "test", "ord": 0, "tags": ["a", "b"]}
    assert md1 == {"source": "test", "ord": 1}


def test_insert_with_conflict_updates(backend, small_corpus):
    """ON CONFLICT (id) DO UPDATE — re-inserting the same id is
    well-defined and replaces the vector + metadata."""
    from vector_backends import InsertRecord

    backend.open(dim=small_corpus.shape[1])
    backend.insert([InsertRecord(id="dup", vector=small_corpus[0], metadata={"v": 1})])
    backend.insert([InsertRecord(id="dup", vector=small_corpus[3], metadata={"v": 2})])
    got, md = backend.get_by_id("dup")
    cos = float(
        np.dot(got, small_corpus[3])
        / (np.linalg.norm(got) * np.linalg.norm(small_corpus[3]))
    )
    assert cos > 0.999
    assert md == {"v": 2}


def test_get_by_id_raises_for_unknown_id(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a", "b"], small_corpus[:2])
    with pytest.raises(KeyError):
        backend.get_by_id("nope")


def test_insert_before_open_raises(pgvector_cls):
    b = pgvector_cls(table_name="vectorsmuggle_test_unopened")
    try:
        with pytest.raises(RuntimeError):
            b.insert_arrays(["a"], np.zeros((1, 4), dtype=np.float32))
    finally:
        b.close()


def test_insert_arrays_validates_shapes(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    with pytest.raises(ValueError, match="2-D"):
        backend.insert_arrays(["a"], np.zeros(small_corpus.shape[1], dtype=np.float32))
    with pytest.raises(ValueError, match="id count"):
        backend.insert_arrays(["a", "b"], small_corpus[:3])


def test_close_is_idempotent(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a"], small_corpus[:1])
    backend.close()
    backend.close()  # must not raise


def test_context_manager_closes(pgvector_cls, small_corpus):
    import uuid
    table = f"vectorsmuggle_test_{uuid.uuid4().hex[:8]}"
    with pgvector_cls(table_name=table) as b:
        b.open(dim=small_corpus.shape[1])
        b.insert_arrays(["a"], small_corpus[:1])
    # After exit, a fresh open() against a new table must work.
    table2 = f"vectorsmuggle_test_{uuid.uuid4().hex[:8]}"
    fresh = pgvector_cls(table_name=table2)
    fresh.open(dim=small_corpus.shape[1])
    fresh.close()


def test_metric_other_than_cosine_rejected(pgvector_cls):
    import uuid
    table = f"vectorsmuggle_test_{uuid.uuid4().hex[:8]}"
    b = pgvector_cls(table_name=table)
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()


def test_invalid_table_name_rejected_at_construction(pgvector_cls):
    """SQL identifier validation runs in __init__ so the bad-name failure
    surfaces immediately, before any database operations."""
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        pgvector_cls(table_name='bad"name')
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        pgvector_cls(table_name="line\nbreak")


def test_close_without_open_does_not_raise(pgvector_cls):
    """The base contract requires close() to be safe even if open()
    never ran or raised. The cross-backend study relies on this for
    unconditional cleanup."""
    b = pgvector_cls(table_name="vectorsmuggle_test_never_opened")
    b.close()  # must not raise


def test_backend_unavailable_is_runtime_error():
    """Catchable as RuntimeError for the cross-backend study's
    skip-on-unavailable logic."""
    assert issubclass(BackendUnavailable, RuntimeError)
