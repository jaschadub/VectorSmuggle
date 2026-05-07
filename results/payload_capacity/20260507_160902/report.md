# Payload Capacity --- `20260507_160902`

## Setup

- Dimension: 768
- Corpus: real embeddings from `data/corpora/emb_nfcorpus_nomic-embed-text_latest.npy`
- Splits: 6705 detector-training, 10058 clean-test (= target batch with rotation applied)
- Random seed: 42
- Angle quantization (b): 10 bits = 1024 levels

## Theoretical capacity

The disjoint-Givens keyed-pair protocol uses a shared key to select ``K`` disjoint dimension pairs, then encodes ``b`` payload bits per pair as a quantized angle. Disjoint pairs commute, so every angle is independently recoverable from the original and rotated vector via ``atan2``. Per-vector capacity:

- ``floor(d/2) * b`` = **3840 bits** = **480 bytes** at d=768, b=10.

For comparison, the looser proxy from ``rotation_sweep.py`` counts ``log2(d(d-1)/2)`` bits of pair choice on top of ``b`` bits per angle, giving **10817 bits** at K=d/2. That proxy is a strict upper bound: a single-vector decoder cannot recover the pair index from a rotated vector alone, so disjoint Givens is the channel an attacker can actually demonstrate end-to-end.

## Empirical decode and detector AUC

Each row encodes a fresh random payload of ``payload_bytes`` into a batch of clean unit-norm vectors via ``K`` disjoint Givens rotations, round-trips through the listed storage dtype (a real vector DB rarely stores float64), then decodes vector ``[0]`` and reports BER. ``cos`` is per-vector cosine to the original; ``IF AUC`` and ``OCSVM AUC`` use the same detectors as in ``rotation_sweep.py``, retrained per row on the clean held-out batch.

| dtype | payload (B) | K | capacity (B) | BER | match | cos | IF AUC | OCSVM AUC |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| float64 | 64 | 52 | 65 | 0.0000 | ok | 0.9004 | 0.721 | 0.827 |
| float64 | 128 | 103 | 129 | 0.0000 | ok | 0.7492 | 0.890 | 0.994 |
| float64 | 240 | 192 | 240 | 0.0000 | ok | 0.4199 | 0.986 | 1.000 |
| float64 | 256 | 205 | 256 | 0.0000 | ok | 0.4718 | 0.980 | 1.000 |
| float64 | 480 | 384 | 480 | 0.0000 | ok | -0.0783 | 1.000 | 1.000 |
| float32 | 64 | 52 | 65 | 0.0000 | ok | 0.8743 | 0.710 | 0.891 |
| float32 | 128 | 103 | 129 | 0.0000 | ok | 0.6975 | 0.943 | 0.998 |
| float32 | 240 | 192 | 240 | 0.0000 | ok | 0.5224 | 0.993 | 1.000 |
| float32 | 256 | 205 | 256 | 0.0000 | ok | 0.4218 | 0.987 | 1.000 |
| float32 | 480 | 384 | 480 | 0.0000 | ok | 0.0052 | 1.000 | 1.000 |
| float16 | 64 | 52 | 65 | 0.0000 | ok | 0.8682 | 0.827 | 0.918 |
| float16 | 128 | 103 | 129 | 0.0000 | ok | 0.7288 | 0.912 | 0.996 |
| float16 | 240 | 192 | 240 | 0.0000 | ok | 0.5230 | 0.987 | 1.000 |
| float16 | 256 | 205 | 256 | 0.0000 | ok | 0.4514 | 0.989 | 1.000 |
| float16 | 480 | 384 | 480 | 0.0000 | ok | -0.0328 | 1.000 | 1.000 |

## How to read this table

- ``BER == 0`` and ``match == ok`` mean the payload round-tripped exactly. The angle-quantization margin (cell width ``2pi/2**b``) exceeds float32 / float16 storage roundoff for ``b`` up to ~12, so exact recovery is the expected case across the table.
- ``cos`` near 1 means the rotated vector is geometrically indistinguishable from the original. Disjoint Givens rotations are orthogonal, so any drop is float roundoff in storage; the drift at high ``K`` is the expected ``cos ~= 1 - 2K/d`` alignment loss for uniform random angles.

- ``IF AUC`` and ``OCSVM AUC`` behavior is markedly different on a real production embedding distribution than on the synthetic-Gaussian baseline: on real ``text-embedding-3-large`` vectors, the detector rapidly regains signal as ``K`` grows, because the clean training manifold is concentrated and random-angle disjoint Givens rotations push the obfuscated vectors off it. The synthetic-vector regime where ``AUC`` stays at chance across the entire sweep does not extend to real production corpora at high ``K``.
