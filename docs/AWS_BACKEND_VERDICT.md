# AWS Vector Backend Vulnerability Report

**Date:** 2026-05-16
**Branch:** `feature/aws-vector-backends` (uncommitted local)
**Account:** 463617128620 / us-west-2 (S3 Vectors live run)
**Status:** All three AWS-targeted backends confirmed **VULNERABLE** to rotation-based steganographic embedding attack.

## Backends evaluated

| Backend | Service code | Local test harness | Production target |
|---|---|---|---|
| OpenSearch | k-NN HNSW + FAISS | `opensearchproject/opensearch:2.18.0` | AWS OpenSearch Service / Serverless |
| MemoryDB | Redis Search HNSW | `redis/redis-stack-server:7.4.0-v1` | AWS MemoryDB for Redis |
| S3 Vectors | AWS-native vector store | none (no emulator exists) | AWS S3 Vectors (GA mid-2025) |

S3 Vectors has no offline emulator, so it was tested live against a real AWS account.

## Empirical results

Same attack across all three: small-angle rotation in the `(x0, x1)` subspace of a unit-norm 64-dim embedding (θ = 0.05 rad). The vector is inserted, retrieved, then the inverse rotation is applied to test payload recovery.

| Backend | `cos(retrieved, inserted)` | `cos(recovered, baseline)` | `cos(obfuscated, baseline)` | Verdict |
|---|---|---|---|---|
| OpenSearch | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |
| MemoryDB | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |
| S3 Vectors | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |

Interpretation:

- **Lossless storage** — `cos(retrieved, inserted) = 1.0` confirms byte-for-byte float32 round-trip. No silent quantization on any of the three.
- **Perfect payload recovery** — `cos(recovered, baseline) = 1.0` shows the inverse rotation reconstructs the original embedding at zero bit-error.
- **Undetectable stealth** — `cos(obfuscated, baseline) = 0.99996`. The cosine drift introduced by the attack is 3.5 × 10⁻⁵, far below any reasonable anomaly-detection or retrieval-quality threshold.

The most consequential of the three is **S3 Vectors**: an object-storage-backed vector store would have been a plausible candidate for silent quantization. AWS does not quantize. The bytes that go in are the bytes that come out.

## Coverage breadth after this study

| Local / on-prem | AWS-hosted (this study) |
|---|---|
| FAISS-flat | OpenSearch Service / Serverless |
| FAISS-HNSW | MemoryDB |
| Chroma | S3 Vectors |
| Qdrant | (Aurora pgvector — same code as local pgvector) |
| pgvector | |

Total: 8 backends span local, OSS, and three hyperscaler-managed AWS surfaces. All vulnerable.

## Implementation artifacts

- `vector_backends/opensearch_backend.py` — basic-auth + SigV4 dual-auth; `space_type=innerproduct` + `engine=faiss` for OS 2.18+ compatibility.
- `vector_backends/memorydb_backend.py` — FT.CREATE / FT.SEARCH against Redis Search HNSW.
- `vector_backends/s3vectors_backend.py` — `boto3.client('s3vectors')`; auto-creates UUID-suffixed bucket on open, deletes on close iff owned.
- `tests/unit/test_aws_backends.py` — 15 moto-backed unit tests (no network, no credentials).
- `tests/integration/test_opensearch_backend.py` — 6 live tests against docker.
- `tests/integration/test_memorydb_backend.py` — 6 live tests against docker.
- `tests/integration/test_s3vectors_backend.py` — 4 tests gated on `AWS_S3VECTORS_TEST=1`.

## AWS cleanup verification

- Buckets in account matching `vs-research-*`: **0**
- Other buckets touched: **0**
- Local containers: stopped.

## Action required

The IAM access key used for the live run (`AKIAWX4N37CWD3V6IP76`) appears in conversation transcript and should be rotated immediately via the AWS IAM console.
