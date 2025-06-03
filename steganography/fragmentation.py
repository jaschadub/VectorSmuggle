"""Multi-model fragmentation for distributing data across different embedding models."""

import hashlib
import logging
from typing import Any

import numpy as np
from langchain_openai import OpenAIEmbeddings


class MultiModelFragmenter:
    """Fragments data across multiple embedding models for enhanced steganography."""

    def __init__(
        self,
        models: list[dict[str, Any]] | None = None,
        fragment_strategy: str = "round_robin",
        integrity_check: bool = True,
        checksum_algorithm: str = "sha256"
    ):
        """
        Initialize multi-model fragmenter.

        Args:
            models: List of model configurations with 'name', 'type', and 'config' keys
            fragment_strategy: Strategy for distributing fragments ('round_robin', 'random', 'weighted')
            integrity_check: Whether to include integrity checks for fragments
            checksum_algorithm: Algorithm for computing checksums
        """
        self.models = models or self._get_default_models()
        self.fragment_strategy = fragment_strategy
        self.integrity_check = integrity_check
        self.checksum_algorithm = checksum_algorithm
        self.logger = logging.getLogger(__name__)

        # Initialize embedding models
        self.embedding_models = {}
        self._initialize_models()

    def _get_default_models(self) -> list[dict[str, Any]]:
        """Get default model configurations."""
        return [
            {
                "name": "openai_ada_002",
                "type": "openai",
                "config": {"model": "text-embedding-ada-002"}
            },
            {
                "name": "openai_3_small",
                "type": "openai",
                "config": {"model": "text-embedding-3-small"}
            },
            {
                "name": "openai_3_large",
                "type": "openai",
                "config": {"model": "text-embedding-3-large"}
            }
        ]

    def _initialize_models(self) -> None:
        """Initialize embedding model instances."""
        for model_config in self.models:
            try:
                if model_config["type"] == "openai":
                    self.embedding_models[model_config["name"]] = OpenAIEmbeddings(
                        **model_config["config"]
                    )
                    self.logger.debug(f"Initialized OpenAI model: {model_config['name']}")
                else:
                    self.logger.warning(f"Unsupported model type: {model_config['type']}")
            except Exception as e:
                self.logger.error(f"Failed to initialize model {model_config['name']}: {e}")

    def _compute_checksum(self, data: bytes) -> str:
        """Compute checksum for data integrity verification."""
        if self.checksum_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.checksum_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {self.checksum_algorithm}")

    def _fragment_text(self, text: str, num_fragments: int) -> list[str]:
        """Fragment text into specified number of pieces."""
        if num_fragments <= 1:
            return [text]

        # Calculate fragment size
        text_length = len(text)
        base_size = text_length // num_fragments
        remainder = text_length % num_fragments

        fragments = []
        start = 0

        for i in range(num_fragments):
            # Add extra character to first 'remainder' fragments
            fragment_size = base_size + (1 if i < remainder else 0)
            end = start + fragment_size

            fragments.append(text[start:end])
            start = end

        return fragments

    def _distribute_fragments(self, fragments: list[str]) -> list[tuple[str, str]]:
        """Distribute fragments across models based on strategy."""
        model_names = list(self.embedding_models.keys())
        if not model_names:
            raise ValueError("No embedding models available")

        distribution = []

        if self.fragment_strategy == "round_robin":
            for i, fragment in enumerate(fragments):
                model_name = model_names[i % len(model_names)]
                distribution.append((fragment, model_name))

        elif self.fragment_strategy == "random":
            import random
            for fragment in fragments:
                model_name = random.choice(model_names)
                distribution.append((fragment, model_name))

        elif self.fragment_strategy == "weighted":
            # Simple weighted distribution - can be enhanced with actual weights
            weights = [1.0] * len(model_names)  # Equal weights for now
            for fragment in fragments:
                model_name = np.random.choice(model_names, p=np.array(weights)/sum(weights))
                distribution.append((fragment, model_name))

        else:
            raise ValueError(f"Unsupported fragment strategy: {self.fragment_strategy}")

        return distribution

    def fragment_and_embed(self, text: str, num_fragments: int | None = None) -> dict[str, Any]:
        """
        Fragment text and create embeddings across multiple models.

        Args:
            text: Input text to fragment and embed
            num_fragments: Number of fragments to create. If None, uses number of available models

        Returns:
            Dictionary containing fragmented embeddings and metadata
        """
        if num_fragments is None:
            num_fragments = len(self.embedding_models)

        if num_fragments <= 0:
            raise ValueError("Number of fragments must be positive")

        # Fragment the text
        text_fragments = self._fragment_text(text, num_fragments)

        # Distribute fragments across models
        fragment_distribution = self._distribute_fragments(text_fragments)

        # Create embeddings for each fragment
        fragmented_embeddings = []
        fragment_metadata = []

        for i, (fragment, model_name) in enumerate(fragment_distribution):
            try:
                model = self.embedding_models[model_name]
                embedding = model.embed_query(fragment)

                metadata = {
                    "fragment_id": i,
                    "model_name": model_name,
                    "fragment_text": fragment,
                    "fragment_length": len(fragment)
                }

                if self.integrity_check:
                    checksum = self._compute_checksum(fragment.encode('utf-8'))
                    metadata["checksum"] = checksum

                fragmented_embeddings.append(embedding)
                fragment_metadata.append(metadata)

                self.logger.debug(f"Created embedding for fragment {i} using model {model_name}")

            except Exception as e:
                self.logger.error(f"Failed to create embedding for fragment {i}: {e}")
                raise

        result = {
            "embeddings": fragmented_embeddings,
            "metadata": fragment_metadata,
            "original_text": text,
            "num_fragments": num_fragments,
            "fragment_strategy": self.fragment_strategy,
            "total_checksum": self._compute_checksum(text.encode('utf-8')) if self.integrity_check else None
        }

        self.logger.info(f"Successfully fragmented text into {num_fragments} embeddings across {len({md['model_name'] for md in fragment_metadata})} models")
        return result

    def reconstruct_from_fragments(self, fragmented_data: dict[str, Any]) -> str:
        """
        Reconstruct original text from fragmented embeddings.

        Args:
            fragmented_data: Dictionary containing fragmented embeddings and metadata

        Returns:
            Reconstructed text
        """
        metadata = fragmented_data["metadata"]

        # Sort fragments by fragment_id to maintain order
        sorted_metadata = sorted(metadata, key=lambda x: x["fragment_id"])

        # Reconstruct text by concatenating fragments
        reconstructed_fragments = []
        for fragment_meta in sorted_metadata:
            fragment_text = fragment_meta["fragment_text"]

            # Verify integrity if enabled
            if self.integrity_check and "checksum" in fragment_meta:
                computed_checksum = self._compute_checksum(fragment_text.encode('utf-8'))
                if computed_checksum != fragment_meta["checksum"]:
                    self.logger.warning(f"Checksum mismatch for fragment {fragment_meta['fragment_id']}")

            reconstructed_fragments.append(fragment_text)

        reconstructed_text = "".join(reconstructed_fragments)

        # Verify total integrity
        if self.integrity_check and fragmented_data.get("total_checksum"):
            total_checksum = self._compute_checksum(reconstructed_text.encode('utf-8'))
            if total_checksum != fragmented_data["total_checksum"]:
                self.logger.warning("Total checksum mismatch during reconstruction")

        self.logger.info(f"Successfully reconstructed text from {len(sorted_metadata)} fragments")
        return reconstructed_text

    def validate_fragments(self, fragmented_data: dict[str, Any]) -> dict[str, Any]:
        """
        Validate fragment integrity and completeness.

        Args:
            fragmented_data: Dictionary containing fragmented embeddings and metadata

        Returns:
            Validation results
        """
        metadata = fragmented_data["metadata"]
        validation_results = {
            "is_valid": True,
            "fragment_count": len(metadata),
            "expected_fragments": fragmented_data["num_fragments"],
            "missing_fragments": [],
            "checksum_failures": [],
            "model_distribution": {}
        }

        # Check fragment count
        if len(metadata) != fragmented_data["num_fragments"]:
            validation_results["is_valid"] = False
            validation_results["missing_fragments"] = [
                i for i in range(fragmented_data["num_fragments"])
                if i not in [meta["fragment_id"] for meta in metadata]
            ]

        # Check individual fragment integrity
        for fragment_meta in metadata:
            model_name = fragment_meta["model_name"]
            validation_results["model_distribution"][model_name] = validation_results["model_distribution"].get(model_name, 0) + 1

            if self.integrity_check and "checksum" in fragment_meta:
                fragment_text = fragment_meta["fragment_text"]
                computed_checksum = self._compute_checksum(fragment_text.encode('utf-8'))
                if computed_checksum != fragment_meta["checksum"]:
                    validation_results["is_valid"] = False
                    validation_results["checksum_failures"].append(fragment_meta["fragment_id"])

        # Check total integrity
        if self.integrity_check and fragmented_data.get("total_checksum"):
            reconstructed_text = self.reconstruct_from_fragments(fragmented_data)
            total_checksum = self._compute_checksum(reconstructed_text.encode('utf-8'))
            if total_checksum != fragmented_data["total_checksum"]:
                validation_results["is_valid"] = False
                validation_results["total_checksum_failure"] = True

        self.logger.info(f"Fragment validation completed. Valid: {validation_results['is_valid']}")
        return validation_results

    def get_model_statistics(self) -> dict[str, Any]:
        """Get statistics about available models and their usage."""
        stats = {
            "total_models": len(self.embedding_models),
            "available_models": list(self.embedding_models.keys()),
            "fragment_strategy": self.fragment_strategy,
            "integrity_check_enabled": self.integrity_check
        }

        return stats
