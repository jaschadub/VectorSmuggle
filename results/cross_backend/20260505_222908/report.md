# Cross-Backend Round-Trip Study --- `20260505_222908`

## Setup

- Corpus: 500 synthetic Gaussian vectors
- Dimension: 128
- Random seed: 42
- Noise level (where applicable): 0.01
- Backends evaluated: faiss_flat, faiss_hnsw, faiss_pq_m8, faiss_pq_m16, chroma, qdrant_f32, qdrant_q8

Each cell answers: *what cosine to the original does the DB return
when the attacker reads the stored vector back via id?* For a
clean baseline (no obfuscation), 1.000 means the DB stored exactly
what we sent. For a perturbed input, the same metric tells us
whether the attacker's payload survived the round trip.

## `recover_cos` --- attacker bit-channel survival

| Technique | faiss_flat | faiss_hnsw | faiss_pq_m8 | faiss_pq_m16 | chroma | qdrant_f32 | qdrant_q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1.0000 | 1.0000 | 0.7614 | 0.8789 | 1.0000 | 1.0000 | 1.0000 |
| noise | 0.9937 | 0.9937 | 0.7610 | 0.8764 | 0.9937 | 0.9937 | 0.9937 |
| rotation | 0.9999 | 0.9999 | 0.7623 | 0.8792 | 0.9999 | 0.9999 | 0.9999 |
| scaling | 1.0000 | 1.0000 | 0.7614 | 0.8787 | 1.0000 | 1.0000 | 1.0000 |
| offset | 0.9507 | 0.9507 | 0.7357 | 0.8398 | 0.9507 | 0.9507 | 0.9507 |

## `self_query_recall@1` --- ANN-index round-trip

| Technique | faiss_flat | faiss_hnsw | faiss_pq_m8 | faiss_pq_m16 | chroma | qdrant_f32 | qdrant_q8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| noise | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rotation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| scaling | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| offset | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## How to read this report

- A row of `1.0000` across every backend in the `recover_cos` table
  means that technique's payload bits are recoverable losslessly
  from any of these vector stores. The attack is class-wide.
- A row of `1.000` across every backend in the recall table means
  the ANN index still finds the obfuscated vector when queried
  with itself. The attack does not break similarity search.
- Cells of `—` indicate the backend was not available at run time
  (missing dep or unreachable service) and the row was skipped
  rather than failing the whole study.

## Note on Qdrant int8 quantization (`qdrant_q8`)

Qdrant stores both float32 originals and the int8 quantized form;
`retrieve()` returns the float32 original, so the quantized variant
preserves the bit channel even when quantization is on. The
`recover_cos` value matches the lossless variant exactly. This is
the right empirical answer to the *quantization-as-defense*
question: scalar quantization is a search-side artifact, not a
storage-side defense, and does not narrow the attacker's bit
channel under threat models A or B.
