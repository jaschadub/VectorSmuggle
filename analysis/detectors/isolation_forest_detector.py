# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Isolation Forest baseline detector.

This is the simplest, most widely cited statistical anomaly detector in
the literature. Including it as a defensive baseline answers reviewer
questions about "would a *real* defender catch this?" with a concrete
ROC curve rather than a hand-wavy DLP bypass rate.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score


class IsolationForestDetector:
    """Wraps sklearn's IsolationForest with a steganography-aware API."""

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, clean_embeddings: np.ndarray) -> IsolationForestDetector:
        """Fit on a batch of known-clean embeddings."""
        self.model.fit(clean_embeddings)
        self._fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (negative of sklearn's score_samples)."""
        if not self._fitted:
            raise RuntimeError("Detector must be fit before scoring")
        # score_samples returns "the lower, the more abnormal"; flip sign so
        # higher = more anomalous, matching the rest of the codebase.
        return -self.model.score_samples(embeddings)

    def decide(self, embeddings: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Return per-vector anomaly decisions (True = flagged)."""
        scores = self.score(embeddings)
        if threshold is None:
            # Use sklearn's built-in threshold (offset_)
            return self.model.predict(embeddings) == -1
        return scores > threshold


def evaluate(
    detector: IsolationForestDetector,
    clean: np.ndarray,
    obfuscated: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC, F1, and TPR@1%FPR for a fitted detector.

    Args:
        detector: An already-fit detector.
        clean: Held-out clean embeddings (label 0).
        obfuscated: Obfuscated embeddings (label 1).

    Returns:
        Dictionary with auc, f1, tpr_at_1fpr, tpr_at_5fpr.
    """
    scores_clean = detector.score(clean)
    scores_obf = detector.score(obfuscated)

    y_true = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_obf))])
    y_score = np.concatenate([scores_clean, scores_obf])

    auc = float(roc_auc_score(y_true, y_score))

    # F1 at sklearn's default threshold
    decisions_clean = detector.decide(clean)
    decisions_obf = detector.decide(obfuscated)
    y_pred = np.concatenate([decisions_clean, decisions_obf]).astype(int)
    f1 = float(f1_score(y_true, y_pred))

    # TPR at fixed FPR thresholds
    sorted_clean = np.sort(scores_clean)
    n = len(sorted_clean)

    def tpr_at_fpr(target_fpr: float) -> float:
        # Threshold at the (1-target_fpr) quantile of clean scores
        thr_idx = int(np.ceil((1.0 - target_fpr) * n)) - 1
        thr_idx = max(0, min(thr_idx, n - 1))
        threshold = sorted_clean[thr_idx]
        return float(np.mean(scores_obf > threshold))

    return {
        "auc": auc,
        "f1": f1,
        "tpr_at_1fpr": tpr_at_fpr(0.01),
        "tpr_at_5fpr": tpr_at_fpr(0.05),
        "n_clean": int(len(clean)),
        "n_obfuscated": int(len(obfuscated)),
    }
