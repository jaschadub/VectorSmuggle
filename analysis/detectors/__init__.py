# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
"""Concrete statistical detectors used as defensive baselines.

Each detector implements a uniform interface:
    fit(clean_embeddings)
    score(embeddings) -> per-vector anomaly score (higher = more anomalous)
    decide(embeddings, threshold) -> per-vector boolean decision

Detectors are intentionally simple and reproducible so they can serve as
peer-review-friendly comparison baselines for steganographic evasion claims.
"""

from .isolation_forest_detector import IsolationForestDetector
from .one_class_svm_detector import OneClassSVMDetector

__all__ = ["IsolationForestDetector", "OneClassSVMDetector"]
