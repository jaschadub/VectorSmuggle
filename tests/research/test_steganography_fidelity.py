# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0

"""Research validation tests for VectorSmuggle steganographic technique fidelity.

These tests validate the core research claims:
1. Steganographic obfuscation preserves enough semantic information for retrieval
2. Different techniques produce measurably different distortion profiles
3. Detection signatures (statistical fingerprints) emerge predictably
4. Recovery is feasible from obfuscated embeddings

Run with: pytest tests/research/ -m research -v
"""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from steganography.obfuscation import EmbeddingObfuscator


class TestSemanticPreservation:
    """Validate that obfuscation preserves semantic relationships."""

    @pytest.fixture
    def synthetic_clustered_embeddings(self):
        """Generate clustered embeddings simulating semantic groups."""
        np.random.seed(42)
        # Create 3 distinct clusters of 20 vectors each
        cluster_centers = [
            np.array([1.0] * 64 + [0.0] * 64),
            np.array([0.0] * 64 + [1.0] * 64),
            np.array([0.5] * 128),
        ]
        embeddings = []
        labels = []
        for i, center in enumerate(cluster_centers):
            cluster = np.random.normal(0, 0.1, (20, 128)) + center
            embeddings.append(cluster)
            labels.extend([i] * 20)
        embeddings = np.vstack(embeddings).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings, np.array(labels)

    @pytest.mark.research
    def test_noise_preserves_cluster_structure(self, synthetic_clustered_embeddings):
        """Light noise should preserve cluster relationships."""
        embeddings, labels = synthetic_clustered_embeddings
        obfuscator = EmbeddingObfuscator(noise_level=0.01, seed=42)
        noisy = obfuscator.inject_noise(embeddings)

        # Compute pairwise cosine similarity for original and noisy
        orig_sim = cosine_similarity(embeddings)
        noisy_sim = cosine_similarity(noisy)

        # Correlation between similarity matrices indicates structure preservation
        correlation = np.corrcoef(orig_sim.flatten(), noisy_sim.flatten())[0, 1]
        assert correlation > 0.95, f"Cluster structure degraded: correlation={correlation}"

    @pytest.mark.research
    def test_high_noise_degrades_retrieval(self, synthetic_clustered_embeddings):
        """High noise should measurably degrade retrieval fidelity."""
        embeddings, labels = synthetic_clustered_embeddings
        low_obf = EmbeddingObfuscator(noise_level=0.01, seed=42)
        high_obf = EmbeddingObfuscator(noise_level=0.5, seed=42)

        low_noisy = low_obf.inject_noise(embeddings)
        high_noisy = high_obf.inject_noise(embeddings)

        orig_sim = cosine_similarity(embeddings)
        low_corr = np.corrcoef(orig_sim.flatten(), cosine_similarity(low_noisy).flatten())[0, 1]
        high_corr = np.corrcoef(orig_sim.flatten(), cosine_similarity(high_noisy).flatten())[0, 1]

        assert low_corr > high_corr, "Higher noise should degrade structure more"

    @pytest.mark.research
    def test_rotation_preserves_pairwise_similarities(self, synthetic_clustered_embeddings):
        """Rotation should perfectly preserve pairwise relationships."""
        embeddings, _ = synthetic_clustered_embeddings
        obfuscator = EmbeddingObfuscator(seed=42)
        rotated, _ = obfuscator.apply_rotation(embeddings)

        orig_sim = cosine_similarity(embeddings)
        rotated_sim = cosine_similarity(rotated)

        # Rotation is orthogonal, preserves all inner products
        assert np.allclose(orig_sim, rotated_sim, atol=1e-4)


class TestTechniqueDistinguishability:
    """Validate that different techniques have distinct statistical signatures."""

    @pytest.fixture
    def base_embeddings(self):
        """Generate base test embeddings."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (200, 128)).astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    @pytest.mark.research
    def test_techniques_produce_different_distortion_profiles(self, base_embeddings):
        """Different obfuscation techniques produce measurably distinct distortions."""
        obf = EmbeddingObfuscator(seed=42)

        noisy = obf.inject_noise(base_embeddings)
        scaled = obf.apply_scaling(base_embeddings)
        rotated, _ = obf.apply_rotation(base_embeddings)

        # MSE from original
        noise_mse = np.mean((base_embeddings - noisy) ** 2)
        scale_mse = np.mean((base_embeddings - scaled) ** 2)
        rotation_mse = np.mean((base_embeddings - rotated) ** 2)

        # Each technique should produce non-zero distortion
        assert noise_mse > 0
        assert scale_mse > 0
        assert rotation_mse > 0

        # Profiles should be distinguishable
        profiles = [noise_mse, scale_mse, rotation_mse]
        assert max(profiles) / min(profiles) > 1.5, "Techniques too similar"

    @pytest.mark.research
    def test_norm_preservation_signature(self, base_embeddings):
        """Different techniques have characteristic effects on vector norms."""
        obf = EmbeddingObfuscator(seed=42)

        original_norms = np.linalg.norm(base_embeddings, axis=1)

        # Rotation preserves norms exactly
        rotated, _ = obf.apply_rotation(base_embeddings)
        rotation_norm_ratio = np.mean(np.linalg.norm(rotated, axis=1) / original_norms)
        assert abs(rotation_norm_ratio - 1.0) < 0.001

        # Scaling deviates from 1.0
        scaled = obf.apply_scaling(base_embeddings)
        scaling_norm_ratio = np.mean(np.linalg.norm(scaled, axis=1) / original_norms)
        assert abs(scaling_norm_ratio - 1.0) > 0.01


class TestStatisticalDetectability:
    """Test the detectability properties of obfuscated embeddings."""

    @pytest.mark.research
    def test_noise_increases_dimensional_variance(self):
        """Adding noise should increase per-dimension variance."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 0.5, (500, 128)).astype(np.float32)

        obf = EmbeddingObfuscator(noise_level=0.1, seed=42)
        noisy = obf.inject_noise(embeddings)

        original_var = np.var(embeddings, axis=0)
        noisy_var = np.var(noisy, axis=0)

        # Noise injection increases variance
        assert np.mean(noisy_var) > np.mean(original_var)

    @pytest.mark.research
    def test_obfuscate_metadata_capture(self):
        """The obfuscate() method should capture sufficient metadata for analysis."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (50, 128)).astype(np.float32)

        obf = EmbeddingObfuscator(seed=42)
        result = obf.obfuscate(embeddings, techniques=["noise", "rotation", "scaling"])

        # All applied techniques should be recorded
        assert "noise" in result["techniques_applied"]
        assert "rotation" in result["techniques_applied"]
        assert "scaling" in result["techniques_applied"]

        # Metadata should include reversibility info — the actual realizations
        # needed for exact deobfuscation, not just the config scalars.
        assert "rotation_matrix" in result["metadata"]
        assert "noise" in result["metadata"]
        assert "scale_factors" in result["metadata"]


class TestRecoveryFeasibility:
    """Test that obfuscated embeddings can be partially recovered."""

    @pytest.mark.research
    def test_rotation_fully_recoverable(self):
        """Rotation should be exactly reversible given the rotation matrix."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (50, 128)).astype(np.float32)

        obf = EmbeddingObfuscator(seed=42)
        rotated, rotation_matrix = obf.apply_rotation(embeddings)
        recovered = rotated @ rotation_matrix

        assert np.allclose(embeddings, recovered, atol=1e-4)

    @pytest.mark.research
    def test_combined_obfuscation_partial_recovery(self):
        """Combined obfuscation can be partially recovered using metadata."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (50, 128)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        obf = EmbeddingObfuscator(noise_level=0.005, seed=42)
        result = obf.obfuscate(embeddings, techniques=["rotation", "scaling"])
        recovered = obf.deobfuscate(result)

        # Should recover most of the structure (noise & scaling have residuals)
        sim = np.mean([
            np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r))
            for o, r in zip(embeddings, recovered, strict=False)
        ])
        assert sim > 0.95, f"Recovery quality too low: {sim}"
