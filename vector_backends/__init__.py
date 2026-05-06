# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Round-trip-focused vector store backends for empirical research.

This package is distinct from the legacy ``database_adapters/`` package
in this repo. The legacy package is oriented toward subprocess-driven
effectiveness tests for an earlier framework. This package is oriented
toward the question that matters for the steganography research:

    "Does an obfuscated vector survive a round-trip through a real
    vector database with its default index/quantization, and does the
    attacker's recovery still work afterwards?"

The point of running the same battery against multiple backends is not
benchmark comparison --- it is to demonstrate that the underlying gap
(no production vector store inspects or attests to its embedding
contents) is universal. If the attack survives every backend tested,
the paper's framing is "this is a class-wide vulnerability of the
deployed vector-store category" rather than "a Qdrant-specific bug."
Backends that happen to filter the attack would be incidental defenses,
not principled ones.

Every backend implements the same minimal interface (``VectorBackend``)
focused on three operations:

  - ``insert(records)`` - write a batch of (id, vector, metadata)
  - ``get_by_id(id) -> (vector, metadata)`` - exact lookup, used to
    measure how the DB altered the stored vector
  - ``search(query, k) -> [(id, score), ...]`` - similarity search,
    used to measure retrieval utility before/after obfuscation

Backends are lazily imported. Constructing a ``FaissBackend`` does not
require ``chromadb`` to be installed; importing this package does not
require any optional dependency at all.

Available backends:

  +------------+-----------+----------------+----------------------+
  | Backend    | Docker?   | Default index  | Default quantization |
  +============+===========+================+======================+
  | FaissFlat  | no        | IndexFlatIP    | none (float32)       |
  | FaissHNSW  | no        | IndexHNSWFlat  | none (float32)       |
  | Chroma     | no        | HNSW           | none (float32)       |
  | Qdrant     | yes       | HNSW           | scalar int8          |
  +------------+-----------+----------------+----------------------+

Pinecone is intentionally excluded because (a) it requires cloud
credentials, (b) its quantization is opaque to the user and changes
without notice, and (c) it would not run in CI. It can be added as a
research backend later.
"""

from vector_backends.base import (
    BackendUnavailable,
    InsertRecord,
    SearchResult,
    VectorBackend,
    available_backends,
    list_backends,
)
from vector_backends.faiss_backend import (
    FaissFlatBackend,
    FaissHNSWBackend,
    FaissPQBackend,
)

__all__ = [
    "BackendUnavailable",
    "FaissFlatBackend",
    "FaissHNSWBackend",
    "FaissPQBackend",
    "InsertRecord",
    "SearchResult",
    "VectorBackend",
    "available_backends",
    "list_backends",
]
