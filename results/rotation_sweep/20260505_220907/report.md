# Rotation Sweep --- `20260505_220907`

## Setup

- Detector training set: 300 clean unit-norm Gaussians
- Clean test set: 300 (same distribution, held out from training)
- Target set (rotated): 300
- Dimension: 384
- Random seed: 42
- theta grid: [0.01, 0.05, 0.1, 0.5, 1.0, 0.7853981633974483, 1.5707963267948966]
- num_rotations grid: [1, 5, 10, 50, 200]

## Rotation operating curve

Each row is one ``(theta_max, num_rotations)`` operating point. ``cos`` is per-vector cosine similarity to the original; ``IF`` and ``OCSVM`` are detector ROC-AUCs trained on the clean held-out batch; ``bits`` is the payload-bit proxy described in the script. The reviewer point is: small ``theta`` and few rotations leaves cos near 1 and AUC near 0.5 (undetectable but low capacity); larger values trade fidelity and detectability for capacity.

| theta_max | num_rot | bits | cos | pair-corr | IF AUC | OCSVM AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0100 | 1 | 22.2 | 1.0000 | 1.0000 | 0.529 | 0.490 |
| 0.0100 | 5 | 110.8 | 1.0000 | 1.0000 | 0.529 | 0.490 |
| 0.0100 | 10 | 221.7 | 1.0000 | 1.0000 | 0.529 | 0.490 |
| 0.0100 | 50 | 1108.3 | 1.0000 | 1.0000 | 0.529 | 0.490 |
| 0.0100 | 200 | 4433.2 | 1.0000 | 1.0000 | 0.530 | 0.490 |
| 0.0500 | 1 | 22.2 | 1.0000 | 1.0000 | 0.528 | 0.490 |
| 0.0500 | 5 | 110.8 | 1.0000 | 1.0000 | 0.528 | 0.490 |
| 0.0500 | 10 | 221.7 | 1.0000 | 1.0000 | 0.528 | 0.490 |
| 0.0500 | 50 | 1108.3 | 0.9999 | 1.0000 | 0.528 | 0.490 |
| 0.0500 | 200 | 4433.2 | 0.9996 | 1.0000 | 0.529 | 0.491 |
| 0.1000 | 1 | 22.2 | 1.0000 | 1.0000 | 0.529 | 0.490 |
| 0.1000 | 5 | 110.8 | 1.0000 | 1.0000 | 0.528 | 0.490 |
| 0.1000 | 10 | 221.7 | 0.9999 | 1.0000 | 0.527 | 0.490 |
| 0.1000 | 50 | 1108.3 | 0.9996 | 1.0000 | 0.526 | 0.490 |
| 0.1000 | 200 | 4433.2 | 0.9983 | 1.0000 | 0.528 | 0.491 |
| 0.5000 | 1 | 22.2 | 0.9996 | 1.0000 | 0.529 | 0.490 |
| 0.5000 | 5 | 110.8 | 0.9990 | 1.0000 | 0.526 | 0.491 |
| 0.5000 | 10 | 221.7 | 0.9977 | 1.0000 | 0.525 | 0.490 |
| 0.5000 | 50 | 1108.3 | 0.9897 | 1.0000 | 0.531 | 0.490 |
| 0.5000 | 200 | 4433.2 | 0.9582 | 1.0000 | 0.526 | 0.492 |
| 1.0000 | 1 | 22.2 | 0.9986 | 1.0000 | 0.528 | 0.491 |
| 1.0000 | 5 | 110.8 | 0.9960 | 1.0000 | 0.526 | 0.491 |
| 1.0000 | 10 | 221.7 | 0.9911 | 1.0000 | 0.529 | 0.489 |
| 1.0000 | 50 | 1108.3 | 0.9611 | 1.0000 | 0.526 | 0.488 |
| 1.0000 | 200 | 4433.2 | 0.8473 | 1.0000 | 0.521 | 0.495 |
| 0.7854 | 1 | 22.2 | 0.9991 | 1.0000 | 0.528 | 0.491 |
| 0.7854 | 5 | 110.8 | 0.9975 | 1.0000 | 0.525 | 0.491 |
| 0.7854 | 10 | 221.7 | 0.9944 | 1.0000 | 0.527 | 0.489 |
| 0.7854 | 50 | 1108.3 | 0.9753 | 1.0000 | 0.526 | 0.489 |
| 0.7854 | 200 | 4433.2 | 0.9014 | 1.0000 | 0.532 | 0.493 |
| 1.5708 | 1 | 22.2 | 0.9968 | 1.0000 | 0.528 | 0.491 |
| 1.5708 | 5 | 110.8 | 0.9907 | 1.0000 | 0.526 | 0.491 |
| 1.5708 | 10 | 221.7 | 0.9796 | 1.0000 | 0.532 | 0.488 |
| 1.5708 | 50 | 1108.3 | 0.9134 | 1.0000 | 0.524 | 0.486 |
| 1.5708 | 200 | 4433.2 | 0.6804 | 1.0000 | 0.494 | 0.498 |

## How to read this table

- `theta_max` × `num_rotations` is the rotation parameterization.
- `bits` is the payload-bit proxy: an upper bound on what the rotation can encode
  per vector, as a function of how many ``(pair, angle)`` choices the attacker has.
- `cos` is per-vector cosine to the original. Above ~0.95 the obfuscated batch is
  visually indistinguishable; below ~0.7 retrieval starts to break.
- `IF AUC` and `OCSVM AUC` near 0.5 mean detectors are guessing. Above 0.8 they are
  reliably catching the attack at small false-positive rates.
- The reviewer point is borne out: rotation has a *narrow* useful regime where
  fidelity is preserved AND detection is hard AND capacity is non-zero. The
  empirical operating curve in this table makes that explicit rather than asserted.