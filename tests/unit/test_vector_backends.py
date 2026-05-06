"""Contract tests for the vector_backends package.

Each backend implements the same VectorBackend interface, so the test
contract is the same: open -> insert -> get_by_id round-trips at high
fidelity, search returns the right id at top-1, close cleans up.

Backends with optional deps or external services skip silently rather
than failing CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from vector_backends import (
    BackendUnavailable,
    FaissFlatBackend,
    FaissHNSWBackend,
    InsertRecord,
    available_backends,
    list_backends,
)
from vector_backends.base import SearchResult, VectorBackend

# --- registry ---------------------------------------------------------------


@pytest.mark.unit
def test_list_backends_returns_at_least_faiss():
    backends = list_backends()
    assert "faiss_flat" in backends
    assert "faiss_hnsw" in backends


@pytest.mark.unit
def test_available_backends_subset_of_list_backends():
    avail = available_backends()
    listed = list_backends()
    assert set(avail).issubset(set(listed))


@pytest.mark.unit
def test_search_result_is_immutable():
    r = SearchResult(id="x", score=0.5)
    with pytest.raises((AttributeError, Exception)):
        r.id = "y"  # type: ignore[misc]


# --- contract tests parameterized across always-available backends ----------


def _is_skippable(cls: type[VectorBackend]) -> bool:
    available, _ = cls.is_available()
    return not available


@pytest.fixture(
    params=[FaissFlatBackend, FaissHNSWBackend],
    ids=["faiss_flat", "faiss_hnsw"],
)
def backend_cls(request):
    cls = request.param
    if _is_skippable(cls):
        pytest.skip(f"{cls.name} not available in this environment")
    return cls


@pytest.fixture
def backend(backend_cls):
    b = backend_cls()
    yield b
    b.close()


@pytest.fixture
def small_corpus():
    rng = np.random.default_rng(42)
    vectors = rng.normal(0, 1, size=(20, 32)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.mark.unit
def test_open_then_insert_then_get_round_trip(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    got, _ = backend.get_by_id("r5")
    cos = float(np.dot(got, small_corpus[5]) / (np.linalg.norm(got) * np.linalg.norm(small_corpus[5])))
    # FAISS round-trips losslessly for both flat and HNSW indices.
    assert cos > 0.999


@pytest.mark.unit
def test_search_finds_self(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    ids = [f"r{i}" for i in range(small_corpus.shape[0])]
    backend.insert_arrays(ids, small_corpus)
    hits = backend.search(small_corpus[7], k=1)
    assert hits, "search returned no hits"
    assert hits[0].id == "r7"
    assert hits[0].score > 0.99


@pytest.mark.unit
def test_get_by_id_raises_for_unknown_id(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a", "b"], small_corpus[:2])
    with pytest.raises(KeyError):
        backend.get_by_id("nope")


@pytest.mark.unit
def test_insert_before_open_raises(backend_cls):
    b = backend_cls()
    try:
        with pytest.raises(RuntimeError):
            b.insert_arrays(["a"], np.zeros((1, 4), dtype=np.float32))
    finally:
        b.close()


@pytest.mark.unit
def test_insert_arrays_validates_shapes(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    # 1-D not allowed
    with pytest.raises(ValueError, match="2-D"):
        backend.insert_arrays(["a"], np.zeros(small_corpus.shape[1], dtype=np.float32))
    # id/vector count mismatch
    with pytest.raises(ValueError, match="id count"):
        backend.insert_arrays(["a", "b"], small_corpus[:3])


@pytest.mark.unit
def test_close_is_idempotent(backend, small_corpus):
    backend.open(dim=small_corpus.shape[1])
    backend.insert_arrays(["a"], small_corpus[:1])
    backend.close()
    backend.close()  # second close must not raise


@pytest.mark.unit
def test_context_manager_closes(backend_cls, small_corpus):
    with backend_cls() as b:
        b.open(dim=small_corpus.shape[1])
        b.insert_arrays(["a"], small_corpus[:1])
    # After exit, a fresh open() should work cleanly.
    fresh = backend_cls()
    fresh.open(dim=small_corpus.shape[1])
    fresh.close()


@pytest.mark.unit
def test_metric_other_than_cosine_rejected(backend_cls):
    b = backend_cls()
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()


# --- non-fixture-based: BackendUnavailable contract -------------------------


@pytest.mark.unit
def test_backend_unavailable_is_runtime_error():
    """BackendUnavailable should be a RuntimeError so callers can catch
    either type when handling missing deps."""
    assert issubclass(BackendUnavailable, RuntimeError)


@pytest.mark.unit
def test_insert_record_metadata_defaults_to_empty_dict():
    r = InsertRecord(id="a", vector=np.zeros(4, dtype=np.float32))
    assert r.metadata == {}
