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

Run 1 (rotation primitive, all three backends, live S3 Vectors). Small-angle Givens rotation in the `(x0, x1)` subspace of a unit-norm 64-dim embedding (θ = 0.05 rad). Insert → retrieve → inverse-rotate → compare.

| Backend | `cos(retrieved, inserted)` | `cos(recovered, baseline)` | `cos(obfuscated, baseline)` | Verdict |
|---|---|---|---|---|
| OpenSearch | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |
| MemoryDB | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |
| S3 Vectors | 1.0000000000 | 1.0000000000 | 0.9999647737 | **VULNERABLE** |

Run 2 (full primitive matrix). Harness: [`scripts/aws_backend_steg_matrix.py`](../scripts/aws_backend_steg_matrix.py). Exercises every primitive in `steganography.obfuscation.EmbeddingObfuscator` plus the composed-pipeline path. Each cell reports `(lossless, recovery)` — see harness header for definitions.

| Primitive | OpenSearch (lossless / recovery) | MemoryDB (lossless / recovery) | S3 Vectors |
|---|---|---|---|
| noise | 1.0000000 / 0.9999999 | 1.0000000 / 0.9999999 | * |
| rotation | 1.0000000 / 0.9999999 | 1.0000000 / 0.9999999 | 1.0 / 1.0 (Run 1) |
| scaling | 1.0000001 / 1.0000000 | 1.0000001 / 1.0000000 | * |
| offset | 1.0000000 / 0.9999999 | 1.0000000 / 0.9999999 | * |
| fragmentation (4 sparse fragments → separate records) | 1.0000000 / 0.9996010 | 1.0000000 / 0.9996010 | * |
| decoy interleaving (1 real + 4 decoys) | 1.0000000 / 1.0000000 | 1.0000000 / 1.0000000 | * |
| composed (noise + rotation + scaling + offset) | 1.0000000 / 0.9999999 | 1.0000000 / 0.9999999 | * |

\* S3 Vectors not re-run in this matrix — the IAM key was rotated between Run 1 and Run 2 (correctly, per security advice). The remaining six primitives are not directly observed on S3 Vectors but follow by reduction: Run 1 established `cos(retrieved, inserted) = 1.0` on S3 Vectors, i.e. byte-for-byte lossless float32 round-trip. Every primitive other than fragmentation is a deterministic per-vector transformation in float32 space, so lossless storage is a sufficient condition for survival. Fragmentation requires per-fragment storage + retrieval-by-ID, both of which S3 Vectors supports per the integration tests in `tests/integration/test_s3vectors_backend.py`.

Interpretation of the full matrix:

- **Storage is lossless on both backends across all primitives** — every cell shows `cos(retrieved, inserted) ≈ 1.0` (deviations at the 8th decimal are float32 renormalization noise).
- **Recovery succeeds for every primitive** — direction-reconstruction cosine is ≥ 0.9996 in every cell; 5/7 primitives reach ≥ 0.9999999. The fragmentation cell is the lowest, at 0.9996, because non-overlapping fragments are renormalized independently before being summed, which introduces a small reconstruction error. Still well above any anomaly-detector threshold.
- **The composed pipeline (noise + rotation + scaling + offset stacked) recovers at 0.9999999** — degradation from stacking four primitives is negligible.
- **The L2-normalize-on-insert step that all three backends perform does not defeat any primitive.** Magnitude-only attacks (scaling) survive because cosine retrieval is invariant to magnitude; magnitude-and-direction attacks (offset, noise) survive because the perturbations are small enough that the renormalized vector remains close to the recoverable target.

The most consequential single finding is still **S3 Vectors**: AWS's brand-new object-storage-backed vector store stores float32 verbatim. The bytes that go in are the bytes that come out — and that's all the rotation attack needs.

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
- `scripts/aws_backend_steg_matrix.py` — full 7-primitive × N-backend matrix harness.
- `scripts/aws_steg_matrix_results.json` — recorded Run 2 results.

## AWS cleanup verification

- Buckets in account matching `vs-research-*`: **0**
- Other buckets touched: **0**
- Local containers: stopped.

## Action required

The IAM access key used for the live run (`AKIAWX4N37CWD3V6IP76`) appears in conversation transcript and should be rotated immediately via the AWS IAM console. **Confirmed rotated** between Run 1 and Run 2 (Run 2's S3 Vectors row returned `InvalidToken`).
