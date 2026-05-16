# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""moto-backed unit tests for the AWS vector backends.

These tests stub the relevant AWS services with ``moto`` so they
exercise the backend's client-construction + request-shape paths
without making real API calls. They are CI-safe (no network, no
credentials, no money) and fast (no docker).

The integration tests in ``tests/integration/`` exercise the real
data plane against a live OpenSearch / Redis Stack container (for
the open-source equivalents) or a live AWS account (for S3 Vectors,
which has no local emulator).

Test scope at this layer:

- The backend's client construction works.
- The expected boto3 calls fire with the expected payload shape.
- ``is_available()`` returns True when deps are installed.
- Error paths (open-before-insert, unknown id) raise the right
  exception types.

We DO NOT exercise actual similarity search at this layer — moto
records calls, it does not implement the FAISS / Redis Search /
S3 Vectors index engine. Search correctness lives in the live
integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip everything in this file when boto3 / moto is missing rather
# than failing import-time for users who haven't installed them.
boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from moto import mock_aws

# ----- S3 Vectors -----------------------------------------------------------


@pytest.fixture
def aws_env(monkeypatch):
    """Set the minimum env vars boto3 needs to keep moto happy."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    yield


@pytest.mark.unit
def test_s3vectors_is_available_when_boto3_installed():
    from vector_backends.s3vectors_backend import S3VectorsBackend
    avail, reason = S3VectorsBackend.is_available()
    assert avail is True, reason


@pytest.mark.unit
@mock_aws
def test_s3vectors_open_creates_bucket_and_index(aws_env):
    from vector_backends.s3vectors_backend import S3VectorsBackend

    backend = S3VectorsBackend(region="us-east-1")
    backend.open(dim=16)
    # Verify the bucket + index actually exist after open().
    client = boto3.client("s3vectors", region_name="us-east-1")
    buckets = client.list_vector_buckets()["vectorBuckets"]
    bucket_names = [b["vectorBucketName"] for b in buckets]
    assert backend._bucket in bucket_names, f"got: {bucket_names}"

    indexes = client.list_indexes(vectorBucketName=backend._bucket)["indexes"]
    index_names = [i["indexName"] for i in indexes]
    assert "vectorsmuggle-research" in index_names

    backend.close()


@pytest.mark.unit
@mock_aws
def test_s3vectors_roundtrip_via_moto(aws_env):
    """End-to-end through moto: open, insert, get_by_id, close.

    moto's s3vectors implementation stores vectors in-memory under
    the bucket+index, so get_by_id returns what put_vectors received.
    Search behavior is implementation-dependent; we don't assert on
    rank order here (that's the live test's job).
    """
    from vector_backends.s3vectors_backend import S3VectorsBackend

    backend = S3VectorsBackend(region="us-east-1")
    backend.open(dim=4)
    rng = np.random.default_rng(0)
    vectors = rng.normal(0, 1, size=(3, 4)).astype(np.float32)
    backend.insert_arrays(["a", "b", "c"], vectors, [{"src": "x"}] * 3)

    got, metadata = backend.get_by_id("b")
    # Cosine-close round trip (we normalize on insert).
    norm_in = vectors[1] / np.linalg.norm(vectors[1])
    cos = float(np.dot(got, norm_in))
    assert cos > 0.999, f"got cos={cos}"
    assert metadata["src"] == "x"

    backend.close()


@pytest.mark.unit
@mock_aws
def test_s3vectors_get_unknown_raises_keyerror(aws_env):
    from vector_backends.s3vectors_backend import S3VectorsBackend

    backend = S3VectorsBackend(region="us-east-1")
    backend.open(dim=4)
    with pytest.raises(KeyError):
        backend.get_by_id("never-inserted")
    backend.close()


@pytest.mark.unit
def test_s3vectors_insert_before_open_raises():
    from vector_backends.s3vectors_backend import S3VectorsBackend

    b = S3VectorsBackend(region="us-east-1")
    try:
        with pytest.raises(RuntimeError, match="not opened"):
            b.insert_arrays(["a"], np.zeros((1, 4), dtype=np.float32))
    finally:
        b.close()


@pytest.mark.unit
def test_s3vectors_metric_other_than_cosine_rejected():
    from vector_backends.s3vectors_backend import S3VectorsBackend

    b = S3VectorsBackend(region="us-east-1")
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()


@pytest.mark.unit
@mock_aws
def test_s3vectors_owned_bucket_is_deleted_on_close(aws_env):
    """When the backend creates its own bucket (no bucket= or env
    var supplied), close() must clean it up so test runs don't leak
    persistent state."""
    from vector_backends.s3vectors_backend import S3VectorsBackend

    backend = S3VectorsBackend(region="us-east-1")
    backend.open(dim=4)
    bucket = backend._bucket
    backend.close()

    client = boto3.client("s3vectors", region_name="us-east-1")
    buckets = client.list_vector_buckets()["vectorBuckets"]
    bucket_names = [b["vectorBucketName"] for b in buckets]
    assert bucket not in bucket_names, "owned bucket should have been deleted"


@pytest.mark.unit
@mock_aws
def test_s3vectors_caller_owned_bucket_is_preserved(aws_env):
    """When the caller passes bucket=..., close() must NOT delete it
    (it might belong to a longer-lived resource managed elsewhere)."""
    from vector_backends.s3vectors_backend import S3VectorsBackend

    # Pre-create the bucket as a caller would.
    client = boto3.client("s3vectors", region_name="us-east-1")
    client.create_vector_bucket(vectorBucketName="caller-owned-bucket")

    backend = S3VectorsBackend(bucket="caller-owned-bucket", region="us-east-1")
    backend.open(dim=4)
    backend.close()

    # Bucket still there.
    buckets = client.list_vector_buckets()["vectorBuckets"]
    bucket_names = [b["vectorBucketName"] for b in buckets]
    assert "caller-owned-bucket" in bucket_names


# ----- OpenSearch (offline checks) -----------------------------------------


@pytest.mark.unit
def test_opensearch_is_available_when_libs_installed():
    """is_available() returns (True, '') iff opensearch-py is
    importable AND the configured endpoint answers info(). Without
    a running OpenSearch the second part fails — this test just
    confirms the *library* check passes."""
    from vector_backends.opensearch_backend import OpenSearchBackend

    # We don't assert on the bool because there might or might not
    # be an OpenSearch running locally; we only assert that the
    # method runs without raising.
    avail, reason = OpenSearchBackend.is_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)


@pytest.mark.unit
def test_opensearch_metric_other_than_cosine_rejected():
    from vector_backends.opensearch_backend import OpenSearchBackend

    b = OpenSearchBackend(url="http://127.0.0.1:1")  # nothing on port 1
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()


@pytest.mark.unit
def test_opensearch_insert_before_open_raises():
    from vector_backends.opensearch_backend import OpenSearchBackend

    b = OpenSearchBackend(url="http://127.0.0.1:1")
    try:
        with pytest.raises(RuntimeError, match="not opened"):
            b.insert_arrays(["a"], np.zeros((1, 4), dtype=np.float32))
    finally:
        b.close()


# ----- MemoryDB / Redis (offline checks) -----------------------------------


@pytest.mark.unit
def test_memorydb_is_available_returns_tuple():
    from vector_backends.memorydb_backend import MemoryDBBackend
    avail, reason = MemoryDBBackend.is_available()
    assert isinstance(avail, bool)
    assert isinstance(reason, str)


@pytest.mark.unit
def test_memorydb_metric_other_than_cosine_rejected():
    from vector_backends.memorydb_backend import MemoryDBBackend
    # Point at a port nothing is listening on so open() fails fast
    # if the metric check passes (we want the metric check to fire
    # FIRST).
    b = MemoryDBBackend(url="redis://127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="cosine"):
            b.open(dim=8, metric="l2")
    finally:
        b.close()


@pytest.mark.unit
def test_memorydb_insert_before_open_raises():
    from vector_backends.memorydb_backend import MemoryDBBackend
    b = MemoryDBBackend(url="redis://127.0.0.1:1")
    try:
        with pytest.raises(RuntimeError, match="not opened"):
            b.insert_arrays(["a"], np.zeros((1, 4), dtype=np.float32))
    finally:
        b.close()


# ----- Registry registration ----------------------------------------------


@pytest.mark.unit
def test_aws_backends_listed_when_deps_present():
    """The three new AWS-targeted backends register in list_backends()
    when their client libraries are importable."""
    from vector_backends import list_backends
    listed = list_backends()
    assert "opensearch" in listed
    assert "memorydb" in listed
    assert "s3vectors" in listed
