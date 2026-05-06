# Adaptive-Attacker Detector Evaluation --- `20260505_224350`

## Setup

- Detector training set: 200 unit-norm Gaussians at d=128
- Clean test set: 200 held-out vectors
- Target set (perturbed): 200 held-out vectors
- L2 perturbation budgets: [0.05, 0.1, 0.2, 0.5]
- Adaptive attack steps: 30

## Naive vs adaptive attack at each budget

Each row is one perturbation strategy at one L2 budget. ``IF AUC`` and
``OCSVM AUC`` are the AUCs of the named detector trained on clean data,
scoring clean test vectors against perturbed test vectors. Lower AUC =
better evasion. The naive baseline is random Gaussian noise scaled to
the budget; the adaptive rows give the attacker white-box access to
the named detector and run a budget-bounded optimizer against it.

| Budget | Attack | cos | IF AUC | OCSVM AUC | Notes |
|---:|---|---:|---:|---:|---|
| 0.0500 | naive_gaussian | 0.9988 | 0.470 | 0.580 | random unit-direction noise, no detector knowledge |
| 0.0500 | adaptive_iforest | 1.0000 | 0.404 | 0.468 | finite-difference greedy walk against trained IF |
| 0.0500 | adaptive_ocsvm | 1.0000 | 0.223 | 0.000 | PGD against OCSVM RBF score (analytic gradient) |
| 0.1000 | naive_gaussian | 0.9951 | 0.484 | 0.691 | random unit-direction noise, no detector knowledge |
| 0.1000 | adaptive_iforest | 0.9995 | 0.327 | 0.437 | finite-difference greedy walk against trained IF |
| 0.1000 | adaptive_ocsvm | 1.0000 | 0.069 | 0.000 | PGD against OCSVM RBF score (analytic gradient) |
| 0.2000 | naive_gaussian | 0.9807 | 0.556 | 0.881 | random unit-direction noise, no detector knowledge |
| 0.2000 | adaptive_iforest | 0.9957 | 0.215 | 0.342 | finite-difference greedy walk against trained IF |
| 0.2000 | adaptive_ocsvm | 0.9999 | 0.001 | 0.000 | PGD against OCSVM RBF score (analytic gradient) |
| 0.5000 | naive_gaussian | 0.8941 | 0.890 | 1.000 | random unit-direction noise, no detector knowledge |
| 0.5000 | adaptive_iforest | 0.9696 | 0.103 | 0.383 | finite-difference greedy walk against trained IF |
| 0.5000 | adaptive_ocsvm | 0.9986 | 0.000 | 0.000 | PGD against OCSVM RBF score (analytic gradient) |

## How to read this table

The interesting comparison is naive vs adaptive at the same budget.
If adaptive substantially lowers the targeted detector's AUC
(IF AUC for adaptive_iforest, OCSVM AUC for adaptive_ocsvm) while
preserving cosine similarity, an attacker who knows the detector
can defeat it.

**A separate fact** that no adaptive evaluation can change: under
orthogonal rotation, both detectors fail at chance regardless of
attacker awareness, because rotation preserves every density feature
the detectors fit on. The adaptive evaluation here is only
meaningful for distribution-shifting attacks; the rotation result
from the main paper stands without modification.