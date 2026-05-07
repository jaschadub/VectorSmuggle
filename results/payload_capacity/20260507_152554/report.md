# Payload Capacity --- `20260507_152554`

## Setup

- Dimension: 3072
- Corpus: real embeddings from `results/empirical/20260505_144136/embeddings_baseline.npy`
- Splits: 27 detector-training, 41 clean-test (= target batch with rotation applied)
- Random seed: 42
- Angle quantization (b): 10 bits = 1024 levels

## Theoretical capacity

The disjoint-Givens keyed-pair protocol uses a shared key to select ``K`` disjoint dimension pairs, then encodes ``b`` payload bits per pair as a quantized angle. Disjoint pairs commute, so every angle is independently recoverable from the original and rotated vector via ``atan2``. Per-vector capacity:

- ``floor(d/2) * b`` = **15360 bits** = **1920 bytes** at d=3072, b=10.

For comparison, the looser proxy from ``rotation_sweep.py`` counts ``log2(d(d-1)/2)`` bits of pair choice on top of ``b`` bits per angle, giving **49412 bits** at K=d/2. That proxy is a strict upper bound: a single-vector decoder cannot recover the pair index from a rotated vector alone, so disjoint Givens is the channel an attacker can actually demonstrate end-to-end.

## Empirical decode and detector AUC

Each row encodes a fresh random payload of ``payload_bytes`` into a batch of clean unit-norm vectors via ``K`` disjoint Givens rotations, round-trips through the listed storage dtype (a real vector DB rarely stores float64), then decodes vector ``[0]`` and reports BER. ``cos`` is per-vector cosine to the original; ``IF AUC`` and ``OCSVM AUC`` use the same detectors as in ``rotation_sweep.py``, retrained per row on the clean held-out batch.

| dtype | payload (B) | K | capacity (B) | BER | match | cos | IF AUC | OCSVM AUC |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| float64 | 64 | 52 | 65 | 0.0000 | ok | 0.9788 | 0.494 | 0.581 |
| float64 | 128 | 103 | 129 | 0.0000 | ok | 0.9387 | 0.561 | 0.695 |
| float64 | 256 | 205 | 256 | 0.0000 | ok | 0.8433 | 0.597 | 0.922 |
| float64 | 512 | 410 | 512 | 0.0000 | ok | 0.7093 | 0.793 | 1.000 |
| float64 | 960 | 768 | 960 | 0.0000 | ok | 0.5121 | 0.821 | 1.000 |
| float64 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4739 | 0.813 | 1.000 |
| float64 | 1920 | 1536 | 1920 | 0.0000 | ok | -0.0102 | 0.996 | 1.000 |
| float32 | 64 | 52 | 65 | 0.0000 | ok | 0.9714 | 0.508 | 0.612 |
| float32 | 128 | 103 | 129 | 0.0000 | ok | 0.9358 | 0.579 | 0.707 |
| float32 | 256 | 205 | 256 | 0.0000 | ok | 0.8628 | 0.625 | 0.895 |
| float32 | 512 | 410 | 512 | 0.0000 | ok | 0.7241 | 0.724 | 1.000 |
| float32 | 960 | 768 | 960 | 0.0000 | ok | 0.4984 | 0.786 | 1.000 |
| float32 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4430 | 0.823 | 1.000 |
| float32 | 1920 | 1536 | 1920 | 0.0000 | ok | 0.0065 | 0.973 | 1.000 |
| float16 | 64 | 52 | 65 | 0.0000 | ok | 0.9744 | 0.544 | 0.595 |
| float16 | 128 | 103 | 129 | 0.0000 | ok | 0.9299 | 0.560 | 0.743 |
| float16 | 256 | 205 | 256 | 0.0000 | ok | 0.8701 | 0.591 | 0.859 |
| float16 | 512 | 410 | 512 | 0.0000 | ok | 0.6947 | 0.743 | 1.000 |
| float16 | 960 | 768 | 960 | 0.0000 | ok | 0.5031 | 0.765 | 1.000 |
| float16 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4476 | 0.871 | 1.000 |
| float16 | 1920 | 1536 | 1920 | 0.0000 | ok | -0.0121 | 1.000 | 1.000 |

## How to read this table

- ``BER == 0`` and ``match == ok`` mean the payload round-tripped exactly. The angle-quantization margin (cell width ``2pi/2**b``) exceeds float32 / float16 storage roundoff for ``b`` up to ~12, so exact recovery is the expected case across the table.
- ``cos`` near 1 means the rotated vector is geometrically indistinguishable from the original. Disjoint Givens rotations are orthogonal, so any drop is float roundoff in storage; the drift at high ``K`` is the expected ``cos ~= 1 - 2K/d`` alignment loss for uniform random angles.

- ``IF AUC`` and ``OCSVM AUC`` behavior is markedly different on a real production embedding distribution than on the synthetic-Gaussian baseline: on real ``text-embedding-3-large`` vectors, the detector rapidly regains signal as ``K`` grows, because the clean training manifold is concentrated and random-angle disjoint Givens rotations push the obfuscated vectors off it. The synthetic-vector regime where ``AUC`` stays at chance across the entire sweep does not extend to real production corpora at high ``K``.
