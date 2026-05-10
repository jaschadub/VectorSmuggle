#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Adaptive-attacker detector evaluation.

The reviewer feedback observes that the paper's detection results
treat the detector as a fixed baseline rather than as an adaptive
opponent. An attacker with read access to the clean training set
could in principle craft perturbations that stay within dense
regions of the detector's score function, evading detection while
still encoding payload.

This script measures how much that adaptive freedom buys the
attacker. We grant the attacker white-box access to the detector
(both the trained Isolation Forest and the One-Class SVM) and let
them perform projected-gradient descent on the score function:

  - Start from a clean test vector.
  - Add a fixed-magnitude perturbation budget (typical noise scale).
  - Iterate: perturb -> evaluate detector score -> step downhill
    on score (toward "more clean-looking") within an L2 ball of
    the chosen budget.

We compare the resulting AUCs against the non-adaptive (random
Gaussian noise) baseline at the same budget. The empirical question:
how much can an attacker buy by knowing the detector?

The structural counterpart, which this script does not need to
demonstrate but the paper restates: even a detector-aware attacker
gains nothing against orthogonal rotation, because rotation
preserves every density feature the detector can fit on. So the
adaptive evaluation is only meaningful for distribution-shifting
techniques.

Output: results/adaptive_attacker/<timestamp>/{summary.json, report.md}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("adaptive")


@dataclass
class AttackResult:
    name: str
    budget: float
    cos_orig_obf: float
    if_auc: float
    ocsvm_auc: float
    notes: str = ""


# --- baseline (non-adaptive) attacks ---------------------------------------


def gaussian_noise_attack(
    vectors: np.ndarray,
    budget: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Naive Gaussian additive noise scaled to a target L2 budget."""
    noise = rng.normal(0, 1, size=vectors.shape)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True) * budget
    return vectors + noise


# --- adaptive attacks -------------------------------------------------------


def adaptive_iforest_attack(
    vectors: np.ndarray,
    budget: float,
    iforest: IsolationForest,
    n_steps: int = 50,
    step_size: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """White-box attack against an Isolation Forest detector.

    Vectorized finite-difference: at each step, for every vector
    sample K random unit-norm directions, score each candidate, and
    pick whichever one increases the detector's "normal" score. Then
    project back to an L2 ball of the chosen budget.

    Isolation Forest doesn't expose a clean analytic gradient (it's
    tree-based), but its ``score_samples`` is vectorized over
    candidates --- so we batch all (vector, direction) candidates
    into one call rather than looping per-vector.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if step_size is None:
        step_size = budget / 10
    n, d = vectors.shape

    perturbed = vectors.copy()
    initial = vectors.copy()
    n_directions = 8
    fd_step = step_size * 0.5

    for _ in range(n_steps):
        directions = rng.normal(0, 1, size=(n_directions, d))
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        # Build all candidate vectors at once: shape (n, n_directions, d)
        candidates = perturbed[:, None, :] + fd_step * directions[None, :, :]
        # Flatten to (n*n_directions, d) for one batched scoring call
        flat = candidates.reshape(-1, d)
        scores = iforest.score_samples(flat).reshape(n, n_directions)
        # For each vector, pick the direction with highest score (most
        # normal-looking). Compare to the base score of perturbed[i] so
        # we only step if at least one direction beats the current point.
        base_scores = iforest.score_samples(perturbed)
        best_dir_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(n), best_dir_idx]
        improved = best_scores > base_scores
        chosen_directions = directions[best_dir_idx]
        chosen_directions[~improved] = 0.0
        proposed = perturbed + step_size * chosen_directions
        # Project back into L2-budget ball around the original
        offsets = proposed - initial
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        scale = np.minimum(1.0, budget / (norms + 1e-12))
        perturbed = initial + offsets * scale

    return perturbed


def adaptive_ocsvm_attack(
    vectors: np.ndarray,
    budget: float,
    ocsvm: OneClassSVM,
    n_steps: int = 50,
    step_size: float | None = None,
) -> np.ndarray:
    """White-box attack against a One-Class SVM with RBF kernel.

    OCSVM does have an analytic gradient w.r.t. the input (the kernel
    expansion has a closed-form derivative), so we use proper PGD
    here rather than finite differences.
    """
    if step_size is None:
        step_size = budget / 10
    initial = vectors.copy()
    perturbed = vectors.copy()

    # Extract support vectors and dual coefs once.
    sv = ocsvm.support_vectors_
    dual = ocsvm.dual_coef_[0]
    gamma = ocsvm._gamma

    for _ in range(n_steps):
        # Score function: sum_i alpha_i * K(x, sv_i) - rho
        # Gradient w.r.t. x: sum_i alpha_i * dK/dx
        # For RBF: dK(x, sv_i)/dx = -2 * gamma * (x - sv_i) * exp(-gamma * |x - sv_i|^2)
        diffs = perturbed[:, None, :] - sv[None, :, :]
        sq = np.sum(diffs ** 2, axis=2)
        kvals = np.exp(-gamma * sq)
        grad = -2 * gamma * np.einsum("ij,ijk->ik", dual * kvals, diffs)
        # We want to *increase* the score (more normal-looking). Step in +grad direction.
        norms = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        proposed = perturbed + step_size * grad / norms
        # Project back to budget ball.
        offsets = proposed - initial
        offset_norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        clip = np.minimum(1.0, budget / (offset_norms + 1e-12))
        perturbed = initial + offsets * clip

    return perturbed


# --- evaluation -------------------------------------------------------------


def cosine_per_pair(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float(np.einsum("ij,ij->i", a_n, b_n).mean())


def detector_aucs(
    iforest: IsolationForest,
    ocsvm: OneClassSVM,
    clean_test: np.ndarray,
    obfuscated: np.ndarray,
) -> tuple[float, float]:
    if_clean = -iforest.score_samples(clean_test)
    if_obf = -iforest.score_samples(obfuscated)
    oc_clean = -ocsvm.decision_function(clean_test)
    oc_obf = -ocsvm.decision_function(obfuscated)
    y = np.concatenate([np.zeros(len(clean_test)), np.ones(len(obfuscated))])
    return (
        float(roc_auc_score(y, np.concatenate([if_clean, if_obf]))),
        float(roc_auc_score(y, np.concatenate([oc_clean, oc_obf]))),
    )


def run(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "results" / "adaptive_attacker" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out_dir)

    rng = np.random.default_rng(args.seed)
    n = args.n_per_split
    d = args.dim
    raw = rng.normal(0, 1, size=(3 * n, d)).astype(np.float64)
    raw = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    train, clean_test, target = raw[:n], raw[n : 2 * n], raw[2 * n :]
    log.info("corpus: train=%d clean_test=%d target=%d dim=%d", n, n, n, d)

    iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iforest.fit(train)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(train)

    results: list[AttackResult] = []

    for budget in args.budgets:
        log.info("--- L2 budget: %.4f ---", budget)

        # 1. Naive Gaussian noise at this budget
        baseline = gaussian_noise_attack(target, budget, rng)
        cos_b = cosine_per_pair(target, baseline)
        if_auc_b, oc_auc_b = detector_aucs(iforest, ocsvm, clean_test, baseline)
        results.append(
            AttackResult(
                name="naive_gaussian",
                budget=budget,
                cos_orig_obf=cos_b,
                if_auc=if_auc_b,
                ocsvm_auc=oc_auc_b,
                notes="random unit-direction noise, no detector knowledge",
            )
        )
        log.info(
            "  naive_gaussian        cos=%.4f IF=%.3f OCSVM=%.3f",
            cos_b, if_auc_b, oc_auc_b,
        )

        # 2. Adaptive against IF (finite-difference)
        adaptive_if = adaptive_iforest_attack(
            target, budget, iforest, n_steps=args.n_steps, rng=rng,
        )
        cos_if = cosine_per_pair(target, adaptive_if)
        if_auc_if, oc_auc_if = detector_aucs(iforest, ocsvm, clean_test, adaptive_if)
        results.append(
            AttackResult(
                name="adaptive_iforest",
                budget=budget,
                cos_orig_obf=cos_if,
                if_auc=if_auc_if,
                ocsvm_auc=oc_auc_if,
                notes="finite-difference greedy walk against trained IF",
            )
        )
        log.info(
            "  adaptive_iforest      cos=%.4f IF=%.3f OCSVM=%.3f",
            cos_if, if_auc_if, oc_auc_if,
        )

        # 3. Adaptive against OCSVM (analytic PGD)
        adaptive_oc = adaptive_ocsvm_attack(
            target, budget, ocsvm, n_steps=args.n_steps,
        )
        cos_oc = cosine_per_pair(target, adaptive_oc)
        if_auc_oc, oc_auc_oc = detector_aucs(iforest, ocsvm, clean_test, adaptive_oc)
        results.append(
            AttackResult(
                name="adaptive_ocsvm",
                budget=budget,
                cos_orig_obf=cos_oc,
                if_auc=if_auc_oc,
                ocsvm_auc=oc_auc_oc,
                notes="PGD against OCSVM RBF score (analytic gradient)",
            )
        )
        log.info(
            "  adaptive_ocsvm        cos=%.4f IF=%.3f OCSVM=%.3f",
            cos_oc, if_auc_oc, oc_auc_oc,
        )

    summary = {
        "timestamp": timestamp,
        "n_per_split": n,
        "dim": d,
        "budgets": args.budgets,
        "n_steps": args.n_steps,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    write_report(out_dir, summary, results)
    log.info("done.")
    return 0


def write_report(out_dir: Path, summary: dict, results: list[AttackResult]) -> None:
    md = [f"# Adaptive-Attacker Detector Evaluation --- `{summary['timestamp']}`\n"]
    md.append("## Setup\n")
    md.append(f"- Detector training set: {summary['n_per_split']} unit-norm Gaussians at d={summary['dim']}")
    md.append(f"- Clean test set: {summary['n_per_split']} held-out vectors")
    md.append(f"- Target set (perturbed): {summary['n_per_split']} held-out vectors")
    md.append(f"- L2 perturbation budgets: {summary['budgets']}")
    md.append(f"- Adaptive attack steps: {summary['n_steps']}\n")

    md.append("## Naive vs adaptive attack at each budget\n")
    md.append("Each row is one perturbation strategy at one L2 budget. ``IF AUC`` and")
    md.append("``OCSVM AUC`` are the AUCs of the named detector trained on clean data,")
    md.append("scoring clean test vectors against perturbed test vectors. Lower AUC =")
    md.append("better evasion. The naive baseline is random Gaussian noise scaled to")
    md.append("the budget; the adaptive rows give the attacker white-box access to")
    md.append("the named detector and run a budget-bounded optimizer against it.\n")

    md.append("| Budget | Attack | cos | IF AUC | OCSVM AUC | Notes |")
    md.append("|---:|---|---:|---:|---:|---|")
    for r in results:
        md.append(
            f"| {r.budget:.4f} | {r.name} | {r.cos_orig_obf:.4f} "
            f"| {r.if_auc:.3f} | {r.ocsvm_auc:.3f} | {r.notes} |"
        )
    md.append("")

    md.append("## How to read this table\n")
    md.append("The interesting comparison is naive vs adaptive at the same budget.")
    md.append("If adaptive substantially lowers the targeted detector's AUC")
    md.append("(IF AUC for adaptive_iforest, OCSVM AUC for adaptive_ocsvm) while")
    md.append("preserving cosine similarity, an attacker who knows the detector")
    md.append("can defeat it.\n")
    md.append("**A separate fact** that no adaptive evaluation can change: under")
    md.append("orthogonal rotation, both detectors fail at chance regardless of")
    md.append("attacker awareness, because rotation preserves every density feature")
    md.append("the detectors fit on. The adaptive evaluation here is only")
    md.append("meaningful for distribution-shifting attacks; the rotation result")
    md.append("from the main paper stands without modification.")

    (out_dir / "report.md").write_text("\n".join(md))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-per-split", type=int, default=300, dest="n_per_split")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5],
        help="L2 perturbation budgets to evaluate at",
    )
    parser.add_argument("--n-steps", type=int, default=30, dest="n_steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
