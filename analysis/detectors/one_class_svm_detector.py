# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""One-class SVM detector — second baseline for triangulation.

We pair this with IsolationForest because:
  - Isolation Forest is tree-based and density-agnostic
  - One-class SVM is kernel-based and assumes a smooth boundary
A technique that evades both is harder to dismiss as fitting one
detector's blind spot.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSVMDetector:
    """One-class SVM with an RBF kernel for embedding anomaly detection."""

    def __init__(
        self,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str | float = "scale",
    ):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self._fitted = False

    def fit(self, clean_embeddings: np.ndarray) -> OneClassSVMDetector:
        """Fit on a batch of known-clean embeddings."""
        self.model.fit(clean_embeddings)
        self._fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (sign-flipped distance to boundary)."""
        if not self._fitted:
            raise RuntimeError("Detector must be fit before scoring")
        return -self.model.decision_function(embeddings)

    def decide(self, embeddings: np.ndarray, threshold: float | None = None) -> np.ndarray:
        scores = self.score(embeddings)
        if threshold is None:
            return self.model.predict(embeddings) == -1
        return scores > threshold
