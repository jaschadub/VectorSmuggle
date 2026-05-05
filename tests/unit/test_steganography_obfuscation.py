"""Unit tests for VectorSmuggle steganography obfuscation module."""

import pytest
import numpy as np

from steganography.obfuscation import EmbeddingObfuscator


class TestEmbeddingObfuscator:
    """Test embedding obfuscation techniques."""

    @pytest.fixture
    def obfuscator(self):
        """Create obfuscator instance for testing."""
        return EmbeddingObfuscator(noise_level=0.01, seed=42)

    @pytest.fixture
    def test_embeddings(self):
        """Create test embeddings."""
        np.random.seed(42)
        embeddings = np.random.normal(0, 1, (100, 128)).astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    @pytest.mark.unit
    def test_obfuscator_initialization(self):
        """Test obfuscator initialization with various parameters."""
        obfuscator = EmbeddingObfuscator()
        assert obfuscator.noise_level == 0.01
        assert obfuscator.rotation_angle == 0.1
        assert obfuscator.scaling_factor == 0.95

        obfuscator = EmbeddingObfuscator(noise_level=0.05, rotation_angle=0.2, seed=123)
        assert obfuscator.noise_level == 0.05
        assert obfuscator.rotation_angle == 0.2

    @pytest.mark.unit
    def test_noise_injection(self, obfuscator, test_embeddings, assert_helpers):
        """Test noise injection technique."""
        original_embeddings = test_embeddings.copy()
        noisy_embeddings = obfuscator.inject_noise(test_embeddings)

        assert_helpers.assert_embeddings_valid(noisy_embeddings)
        assert noisy_embeddings.shape == test_embeddings.shape
        assert noisy_embeddings.dtype in [np.float32, np.float64]
        assert not np.array_equal(original_embeddings, noisy_embeddings)

        mse = np.mean((original_embeddings - noisy_embeddings) ** 2)
        assert mse > 0
        assert mse < 0.1

        assert np.array_equal(original_embeddings, test_embeddings)

    @pytest.mark.unit
    def test_noise_reproducibility(self, test_embeddings):
        """Test that noise injection is reproducible when seed is reset before each call."""
        obfuscator = EmbeddingObfuscator(noise_level=0.01, seed=42)
        noisy1 = obfuscator.inject_noise(test_embeddings)

        # Reset seed to same value to get same result
        obfuscator2 = EmbeddingObfuscator(noise_level=0.01, seed=42)
        noisy2 = obfuscator2.inject_noise(test_embeddings)

        assert np.array_equal(noisy1, noisy2)

    @pytest.mark.unit
    def test_noise_level_impact(self, test_embeddings):
        """Test that higher noise levels produce more distortion."""
        low_noise = EmbeddingObfuscator(noise_level=0.001, seed=42)
        high_noise = EmbeddingObfuscator(noise_level=0.1, seed=42)

        low_noisy = low_noise.inject_noise(test_embeddings)
        high_noisy = high_noise.inject_noise(test_embeddings)

        low_mse = np.mean((test_embeddings - low_noisy) ** 2)
        high_mse = np.mean((test_embeddings - high_noisy) ** 2)

        assert high_mse > low_mse

    @pytest.mark.unit
    def test_rotation_technique(self, obfuscator, test_embeddings, assert_helpers):
        """Test embedding rotation technique."""
        original_embeddings = test_embeddings.copy()

        rotated_embeddings, rotation_matrix = obfuscator.apply_rotation(test_embeddings)

        assert_helpers.assert_embeddings_valid(rotated_embeddings)
        assert rotated_embeddings.shape == test_embeddings.shape
        assert rotation_matrix.shape == (test_embeddings.shape[1], test_embeddings.shape[1])
        assert not np.array_equal(original_embeddings, rotated_embeddings)

        det = np.linalg.det(rotation_matrix)
        assert abs(abs(det) - 1.0) < 1e-5

    @pytest.mark.unit
    def test_scaling_technique(self, obfuscator, test_embeddings, assert_helpers):
        """Test embedding scaling technique."""
        original_embeddings = test_embeddings.copy()
        scaled_embeddings = obfuscator.apply_scaling(test_embeddings)

        assert_helpers.assert_embeddings_valid(scaled_embeddings)
        assert scaled_embeddings.shape == test_embeddings.shape
        assert not np.array_equal(original_embeddings, scaled_embeddings)

        original_norms = np.linalg.norm(test_embeddings, axis=1)
        scaled_norms = np.linalg.norm(scaled_embeddings, axis=1)
        ratios = scaled_norms / original_norms
        assert np.all(ratios > 0.7)
        assert np.all(ratios < 1.3)

    @pytest.mark.unit
    def test_offset_technique(self, obfuscator, test_embeddings, assert_helpers):
        """Test embedding offset technique."""
        original_embeddings = test_embeddings.copy()
        offset_embeddings = obfuscator.apply_offset(test_embeddings)

        assert_helpers.assert_embeddings_valid(offset_embeddings)
        assert offset_embeddings.shape == test_embeddings.shape
        assert not np.array_equal(original_embeddings, offset_embeddings)

    @pytest.mark.unit
    def test_fragmentation_technique(self, obfuscator, test_embeddings):
        """Test embedding fragmentation technique."""
        fragments = obfuscator.fragment_embeddings(test_embeddings)

        assert isinstance(fragments, list)
        assert len(fragments) > 0

        for fragment in fragments:
            assert isinstance(fragment, np.ndarray)
            assert fragment.shape == (test_embeddings.shape[1],)

        total_fragments = len(fragments)
        assert total_fragments >= test_embeddings.shape[0]

    @pytest.mark.unit
    def test_empty_embeddings_handling(self, obfuscator):
        """Test handling of empty embeddings array."""
        empty_embeddings = np.array([]).reshape(0, 128).astype(np.float32)

        noisy = obfuscator.inject_noise(empty_embeddings)
        assert noisy.shape == (0, 128)

        rotated, _ = obfuscator.apply_rotation(empty_embeddings)
        assert rotated.shape == (0, 128)

    @pytest.mark.unit
    def test_single_embedding_handling(self, obfuscator):
        """Test handling of single embedding vector."""
        single_embedding = np.random.normal(0, 1, (1, 128)).astype(np.float32)

        noisy = obfuscator.inject_noise(single_embedding)
        assert noisy.shape == (1, 128)

        rotated, rotation_matrix = obfuscator.apply_rotation(single_embedding)
        assert rotated.shape == (1, 128)
        assert rotation_matrix.shape == (128, 128)

    @pytest.mark.unit
    @pytest.mark.parametrize("noise_level", [0.001, 0.01, 0.1])
    def test_noise_levels(self, test_embeddings, noise_level):
        """Test different noise levels produce valid output."""
        obfuscator = EmbeddingObfuscator(noise_level=noise_level, seed=42)
        noisy = obfuscator.inject_noise(test_embeddings)

        mse = np.mean((test_embeddings - noisy) ** 2)
        assert mse > 0
        assert mse < noise_level * 100

    @pytest.mark.unit
    def test_rotation_preserves_magnitude(self, test_embeddings):
        """Test that rotation preserves embedding magnitudes."""
        obfuscator = EmbeddingObfuscator()
        rotated, rotation_matrix = obfuscator.apply_rotation(test_embeddings)

        original_norms = np.linalg.norm(test_embeddings, axis=1)
        rotated_norms = np.linalg.norm(rotated, axis=1)
        assert np.allclose(original_norms, rotated_norms, rtol=1e-4)

    @pytest.mark.unit
    def test_obfuscation_reversibility_via_rotation(self, obfuscator, test_embeddings):
        """Test that rotation can be reversed using the returned matrix."""
        rotated, rotation_matrix = obfuscator.apply_rotation(test_embeddings)

        # rotated = embeddings @ R.T, so inverse is rotated @ R
        inverse_rotated = rotated @ rotation_matrix
        assert np.allclose(test_embeddings, inverse_rotated, atol=1e-5)

    @pytest.mark.unit
    def test_technique_combination(self, obfuscator, test_embeddings):
        """Test combining multiple obfuscation techniques."""
        step1 = obfuscator.inject_noise(test_embeddings)
        step2, _ = obfuscator.apply_rotation(step1)
        step3 = obfuscator.apply_scaling(step2)

        assert not np.array_equal(test_embeddings, step3)
        assert step3.shape == test_embeddings.shape
        assert step3.dtype in [np.float32, np.float64]
        assert not np.isnan(step3).any()

    @pytest.mark.unit
    def test_obfuscate_method_all_techniques(self, test_embeddings):
        """Test the comprehensive obfuscate() method with all techniques."""
        obfuscator = EmbeddingObfuscator(seed=42)
        result = obfuscator.obfuscate(test_embeddings, techniques=["noise", "rotation", "scaling"])

        assert "embeddings" in result
        assert "techniques_applied" in result
        assert "metadata" in result
        assert result["embeddings"].shape[1] == test_embeddings.shape[1]
        assert "noise" in result["techniques_applied"]

    @pytest.mark.unit
    def test_obfuscate_method_returns_different_embeddings(self, test_embeddings):
        """Test that obfuscate returns modified embeddings."""
        obfuscator = EmbeddingObfuscator(seed=42)
        result = obfuscator.obfuscate(test_embeddings, techniques=["noise"])

        assert not np.array_equal(test_embeddings, result["embeddings"])

    @pytest.mark.unit
    def test_performance_large_embeddings(self, obfuscator, large_embeddings):
        """Test performance with large embedding datasets."""
        import time

        start_time = time.time()
        noisy = obfuscator.inject_noise(large_embeddings)
        noise_time = time.time() - start_time

        start_time = time.time()
        rotated, _ = obfuscator.apply_rotation(large_embeddings)
        rotation_time = time.time() - start_time

        assert noise_time < 1.0
        assert rotation_time < 5.0

        assert noisy.shape == large_embeddings.shape
        assert rotated.shape == large_embeddings.shape

    @pytest.mark.unit
    def test_memory_efficiency(self, obfuscator, large_embeddings):
        """Test memory efficiency of obfuscation techniques."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        noisy = obfuscator.inject_noise(large_embeddings)

        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        data_size = large_embeddings.nbytes
        assert memory_increase < data_size * 10

    @pytest.mark.unit
    def test_statistical_properties(self, test_embeddings):
        """Test statistical properties of noisy embeddings."""
        obfuscator = EmbeddingObfuscator(noise_level=0.01, seed=42)
        original_mean = np.mean(test_embeddings, axis=0)

        noisy = obfuscator.inject_noise(test_embeddings)
        noisy_mean = np.mean(noisy, axis=0)

        assert np.allclose(original_mean, noisy_mean, atol=0.05)

    @pytest.mark.unit
    def test_interleave_with_decoys(self, obfuscator, test_embeddings):
        """Test interleaving embeddings with decoys."""
        np.random.seed(42)
        decoy_embeddings = np.random.normal(0, 1, (50, 128)).astype(np.float32)

        interleaved, real_indices = obfuscator.interleave_with_decoys(test_embeddings, decoy_embeddings)

        assert isinstance(interleaved, np.ndarray)
        assert isinstance(real_indices, list)
        assert len(real_indices) == test_embeddings.shape[0]
        assert interleaved.shape[0] >= test_embeddings.shape[0]
        assert interleaved.shape[1] == test_embeddings.shape[1]
