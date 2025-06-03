"""Embedding obfuscation techniques for steganographic data hiding."""

import logging
import random
from typing import Any

import numpy as np


class EmbeddingObfuscator:
    """Implements various embedding obfuscation techniques to hide data patterns."""

    def __init__(
        self,
        noise_level: float = 0.01,
        rotation_angle: float = 0.1,
        scaling_factor: float = 0.95,
        offset_range: float = 0.05,
        fragment_size: int = 128,
        interleave_ratio: float = 0.3,
        seed: int | None = None
    ):
        """
        Initialize embedding obfuscator.

        Args:
            noise_level: Standard deviation for Gaussian noise injection
            rotation_angle: Maximum rotation angle in radians for embedding transformation
            scaling_factor: Scaling factor for embedding magnitude adjustment
            offset_range: Range for random offset addition
            fragment_size: Size of embedding fragments for fragmentation
            interleave_ratio: Ratio of decoy data to interleave with real data
            seed: Random seed for reproducible obfuscation
        """
        self.noise_level = noise_level
        self.rotation_angle = rotation_angle
        self.scaling_factor = scaling_factor
        self.offset_range = offset_range
        self.fragment_size = fragment_size
        self.interleave_ratio = interleave_ratio

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.logger = logging.getLogger(__name__)

    def inject_noise(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Inject Gaussian noise into embeddings to mask patterns.

        Args:
            embeddings: Input embeddings array of shape (n_vectors, embedding_dim)

        Returns:
            Noise-injected embeddings
        """
        if embeddings.size == 0:
            return embeddings

        noise = np.random.normal(0, self.noise_level, embeddings.shape)
        noisy_embeddings = embeddings + noise

        self.logger.debug(f"Injected noise with std={self.noise_level} to {embeddings.shape[0]} embeddings")
        return noisy_embeddings

    def apply_rotation(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply random rotation to embedding vectors.

        Args:
            embeddings: Input embeddings array

        Returns:
            Tuple of (rotated_embeddings, rotation_matrix)
        """
        if embeddings.size == 0:
            return embeddings, np.eye(embeddings.shape[1] if len(embeddings.shape) > 1 else 1)

        embedding_dim = embeddings.shape[1]

        # Generate random rotation matrix using Givens rotations
        rotation_matrix = np.eye(embedding_dim)

        # Apply multiple small rotations to avoid detection
        num_rotations = min(5, embedding_dim // 2)
        for _ in range(num_rotations):
            i, j = random.sample(range(embedding_dim), 2)
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)

            givens_rotation = np.eye(embedding_dim)
            givens_rotation[i, i] = np.cos(angle)
            givens_rotation[i, j] = -np.sin(angle)
            givens_rotation[j, i] = np.sin(angle)
            givens_rotation[j, j] = np.cos(angle)

            rotation_matrix = rotation_matrix @ givens_rotation

        rotated_embeddings = embeddings @ rotation_matrix.T

        self.logger.debug(f"Applied rotation to {embeddings.shape[0]} embeddings")
        return rotated_embeddings, rotation_matrix

    def apply_scaling(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply random scaling to embedding magnitudes.

        Args:
            embeddings: Input embeddings array

        Returns:
            Scaled embeddings
        """
        if embeddings.size == 0:
            return embeddings

        # Apply per-vector scaling with small random variations
        scale_factors = np.random.normal(
            self.scaling_factor,
            self.scaling_factor * 0.1,
            embeddings.shape[0]
        )
        scale_factors = np.clip(scale_factors, 0.8, 1.2)  # Prevent extreme scaling

        scaled_embeddings = embeddings * scale_factors[:, np.newaxis]

        self.logger.debug(f"Applied scaling to {embeddings.shape[0]} embeddings")
        return scaled_embeddings

    def apply_offset(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply random offset to embeddings.

        Args:
            embeddings: Input embeddings array

        Returns:
            Offset embeddings
        """
        if embeddings.size == 0:
            return embeddings

        offset = np.random.uniform(
            -self.offset_range,
            self.offset_range,
            embeddings.shape
        )
        offset_embeddings = embeddings + offset

        self.logger.debug(f"Applied offset to {embeddings.shape[0]} embeddings")
        return offset_embeddings

    def fragment_embeddings(self, embeddings: np.ndarray) -> list[np.ndarray]:
        """
        Fragment embeddings into smaller chunks across multiple vectors.

        Args:
            embeddings: Input embeddings array

        Returns:
            List of embedding fragments
        """
        if embeddings.size == 0:
            return [embeddings]

        fragments = []
        embedding_dim = embeddings.shape[1]

        for embedding in embeddings:
            # Split each embedding into fragments
            num_fragments = max(1, embedding_dim // self.fragment_size)
            fragment_indices = np.array_split(np.arange(embedding_dim), num_fragments)

            for indices in fragment_indices:
                if len(indices) > 0:
                    fragment = np.zeros(embedding_dim)
                    fragment[indices] = embedding[indices]
                    fragments.append(fragment)

        self.logger.debug(f"Fragmented {embeddings.shape[0]} embeddings into {len(fragments)} fragments")
        return fragments

    def interleave_with_decoys(
        self,
        embeddings: np.ndarray,
        decoy_embeddings: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        """
        Interleave real embeddings with decoy embeddings.

        Args:
            embeddings: Real embeddings to hide
            decoy_embeddings: Decoy embeddings for camouflage

        Returns:
            Tuple of (interleaved_embeddings, real_indices)
        """
        if embeddings.size == 0:
            return embeddings, []

        num_real = embeddings.shape[0]
        num_decoys = int(num_real * self.interleave_ratio)

        if decoy_embeddings.shape[0] < num_decoys:
            # Repeat decoys if not enough available
            repeat_factor = (num_decoys // decoy_embeddings.shape[0]) + 1
            decoy_embeddings = np.tile(decoy_embeddings, (repeat_factor, 1))

        selected_decoys = decoy_embeddings[:num_decoys]

        # Create interleaved array
        total_embeddings = np.vstack([embeddings, selected_decoys])

        # Generate random permutation
        indices = np.random.permutation(total_embeddings.shape[0])
        interleaved_embeddings = total_embeddings[indices]

        # Track positions of real embeddings
        real_indices = [i for i, idx in enumerate(indices) if idx < num_real]

        self.logger.debug(
            f"Interleaved {num_real} real embeddings with {num_decoys} decoys"
        )
        return interleaved_embeddings, real_indices

    def obfuscate(
        self,
        embeddings: np.ndarray,
        decoy_embeddings: np.ndarray | None = None,
        techniques: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Apply comprehensive obfuscation to embeddings.

        Args:
            embeddings: Input embeddings to obfuscate
            decoy_embeddings: Optional decoy embeddings for interleaving
            techniques: List of techniques to apply. If None, applies all techniques.

        Returns:
            Dictionary containing obfuscated embeddings and metadata
        """
        if techniques is None:
            techniques = ["noise", "rotation", "scaling", "offset", "fragmentation"]
            if decoy_embeddings is not None:
                techniques.append("interleaving")

        result = {
            "original_shape": embeddings.shape,
            "techniques_applied": [],
            "metadata": {}
        }

        obfuscated = embeddings.copy()

        # Apply noise injection
        if "noise" in techniques:
            obfuscated = self.inject_noise(obfuscated)
            result["techniques_applied"].append("noise")
            result["metadata"]["noise_level"] = self.noise_level

        # Apply rotation
        if "rotation" in techniques:
            obfuscated, rotation_matrix = self.apply_rotation(obfuscated)
            result["techniques_applied"].append("rotation")
            result["metadata"]["rotation_matrix"] = rotation_matrix

        # Apply scaling
        if "scaling" in techniques:
            obfuscated = self.apply_scaling(obfuscated)
            result["techniques_applied"].append("scaling")
            result["metadata"]["scaling_factor"] = self.scaling_factor

        # Apply offset
        if "offset" in techniques:
            obfuscated = self.apply_offset(obfuscated)
            result["techniques_applied"].append("offset")
            result["metadata"]["offset_range"] = self.offset_range

        # Apply fragmentation
        if "fragmentation" in techniques:
            fragments = self.fragment_embeddings(obfuscated)
            result["techniques_applied"].append("fragmentation")
            result["metadata"]["fragment_count"] = len(fragments)
            result["fragments"] = fragments

        # Apply interleaving with decoys
        if "interleaving" in techniques and decoy_embeddings is not None:
            interleaved, real_indices = self.interleave_with_decoys(obfuscated, decoy_embeddings)
            result["techniques_applied"].append("interleaving")
            result["metadata"]["real_indices"] = real_indices
            result["embeddings"] = interleaved
        else:
            result["embeddings"] = obfuscated

        self.logger.info(f"Applied obfuscation techniques: {result['techniques_applied']}")
        return result

    def deobfuscate(
        self,
        obfuscated_data: dict[str, Any]
    ) -> np.ndarray:
        """
        Reverse obfuscation to recover original embeddings.

        Args:
            obfuscated_data: Dictionary containing obfuscated embeddings and metadata

        Returns:
            Recovered embeddings (approximate due to noise and other irreversible operations)
        """
        embeddings = obfuscated_data["embeddings"]
        metadata = obfuscated_data["metadata"]
        techniques = obfuscated_data["techniques_applied"]

        # Reverse interleaving
        if "interleaving" in techniques:
            real_indices = metadata["real_indices"]
            embeddings = embeddings[real_indices]

        # Reverse fragmentation (reconstruct from fragments)
        if "fragmentation" in techniques and "fragments" in obfuscated_data:
            fragments = obfuscated_data["fragments"]
            original_shape = obfuscated_data["original_shape"]

            # Reconstruct embeddings from fragments
            reconstructed = []
            fragments_per_embedding = len(fragments) // original_shape[0]

            for i in range(original_shape[0]):
                embedding = np.zeros(original_shape[1])
                start_idx = i * fragments_per_embedding
                end_idx = start_idx + fragments_per_embedding

                for fragment in fragments[start_idx:end_idx]:
                    embedding += fragment

                reconstructed.append(embedding)

            embeddings = np.array(reconstructed)

        # Reverse rotation
        if "rotation" in techniques and "rotation_matrix" in metadata:
            rotation_matrix = metadata["rotation_matrix"]
            embeddings = embeddings @ rotation_matrix  # Apply inverse rotation

        # Note: Noise, scaling, and offset are not perfectly reversible
        # but we can attempt approximate reversal

        if "scaling" in techniques and "scaling_factor" in metadata:
            scaling_factor = metadata["scaling_factor"]
            embeddings = embeddings / scaling_factor

        self.logger.info("Deobfuscation completed (approximate recovery)")
        return embeddings
