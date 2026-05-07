# Payload Capacity --- `20260507_125102`

## Setup

- Dimension: 3072
- Per-split corpus size: 300 unit-norm Gaussians
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
| float64 | 64 | 52 | 65 | 0.0000 | ok | 0.9733 | 0.449 | 0.510 |
| float64 | 128 | 103 | 129 | 0.0000 | ok | 0.9365 | 0.449 | 0.511 |
| float64 | 256 | 205 | 256 | 0.0000 | ok | 0.8529 | 0.445 | 0.517 |
| float64 | 512 | 410 | 512 | 0.0000 | ok | 0.7246 | 0.452 | 0.514 |
| float64 | 960 | 768 | 960 | 0.0000 | ok | 0.4857 | 0.445 | 0.504 |
| float64 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4798 | 0.425 | 0.511 |
| float64 | 1920 | 1536 | 1920 | 0.0000 | ok | -0.0161 | 0.459 | 0.527 |
| float32 | 64 | 52 | 65 | 0.0000 | ok | 0.9677 | 0.459 | 0.512 |
| float32 | 128 | 103 | 129 | 0.0000 | ok | 0.9323 | 0.447 | 0.508 |
| float32 | 256 | 205 | 256 | 0.0000 | ok | 0.8689 | 0.453 | 0.503 |
| float32 | 512 | 410 | 512 | 0.0000 | ok | 0.7382 | 0.434 | 0.520 |
| float32 | 960 | 768 | 960 | 0.0000 | ok | 0.4904 | 0.443 | 0.517 |
| float32 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4535 | 0.452 | 0.501 |
| float32 | 1920 | 1536 | 1920 | 0.0000 | ok | 0.0324 | 0.473 | 0.542 |
| float16 | 64 | 52 | 65 | 0.0000 | ok | 0.9705 | 0.455 | 0.513 |
| float16 | 128 | 103 | 129 | 0.0000 | ok | 0.9304 | 0.453 | 0.510 |
| float16 | 256 | 205 | 256 | 0.0000 | ok | 0.8732 | 0.464 | 0.510 |
| float16 | 512 | 410 | 512 | 0.0000 | ok | 0.7224 | 0.439 | 0.508 |
| float16 | 960 | 768 | 960 | 0.0000 | ok | 0.5094 | 0.476 | 0.510 |
| float16 | 1024 | 820 | 1025 | 0.0000 | ok | 0.4702 | 0.447 | 0.530 |
| float16 | 1920 | 1536 | 1920 | 0.0000 | ok | -0.0097 | 0.490 | 0.528 |

## How to read this table

- ``BER == 0`` and ``match == ok`` mean the payload round-tripped exactly. The angle-quantization margin (cell width ``2pi/2**b``) exceeds float32 / float16 storage roundoff for ``b`` up to ~12, so exact recovery is the expected case across the table.
- ``cos`` near 1 means the rotated vector is geometrically indistinguishable from the original. Disjoint Givens rotations are orthogonal, so any drop is float roundoff in storage.
- ``IF AUC`` and ``OCSVM AUC`` near 0.5 confirm the punchline: raising the payload (and therefore ``K``) does not move the off-the-shelf detectors off chance, even at the full ``floor(d/2)`` rotation count where the channel is carrying ~1920 bytes per vector. Capacity grows; detectability does not.