"""Seed management utilities for deterministic operations across the VectorSmuggle codebase."""

import hashlib
import os
from typing import Any

import numpy as np


class SeedManager:
    """Manages random seeds for deterministic operations across all modules."""

    def __init__(self, base_seed: int | None = None):
        """
        Initialize seed manager.

        Args:
            base_seed: Base seed for all operations. If None, uses RANDOM_SEED environment variable.
        """
        self.base_seed = base_seed or self._get_env_seed()

    def _get_env_seed(self) -> int | None:
        """Get seed from RANDOM_SEED environment variable."""
        seed_str = os.getenv("RANDOM_SEED")
        if seed_str:
            try:
                return int(seed_str)
            except ValueError as e:
                raise ValueError(f"RANDOM_SEED must be an integer, got: {seed_str}") from e
        return None

    def get_seeded_random_state(self, module_name: str = "", additional_entropy: str = "") -> np.random.RandomState:
        """
        Get a seeded RandomState for a specific module or operation.

        Args:
            module_name: Name of the module requesting the random state
            additional_entropy: Additional entropy to mix into the seed

        Returns:
            Seeded RandomState instance
        """
        if self.base_seed is None:
            return np.random.RandomState()

        # Create deterministic seed from base seed, module name, and additional entropy
        combined = f"{self.base_seed}_{module_name}_{additional_entropy}"
        seed_hash = int(hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        seed = (self.base_seed + seed_hash) % (2**32)

        return np.random.RandomState(seed)

    def seed_global_generators(self) -> None:
        """Seed global random number generators."""
        if self.base_seed is not None:
            import random
            random.seed(self.base_seed)
            np.random.seed(self.base_seed)
            os.environ["PYTHONHASHSEED"] = str(self.base_seed)

    def create_seeded_instance(self, cls: type, module_name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a class with seeded random state.

        Args:
            cls: Class to instantiate
            module_name: Module name for seed generation
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor

        Returns:
            Instance with seeded random state
        """
        if 'random_state' not in kwargs and self.base_seed is not None:
            kwargs['random_state'] = self.get_seeded_random_state(module_name)
        return cls(*args, **kwargs)


# Global seed manager instance
_global_seed_manager: SeedManager | None = None


def get_global_seed_manager() -> SeedManager:
    """Get the global seed manager instance."""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager()
    return _global_seed_manager


def set_global_seed(seed: int) -> None:
    """Set the global seed for all operations."""
    global _global_seed_manager
    _global_seed_manager = SeedManager(seed)
    _global_seed_manager.seed_global_generators()


def get_seeded_random_state(module_name: str = "", additional_entropy: str = "") -> np.random.RandomState:
    """Convenience function to get a seeded random state."""
    return get_global_seed_manager().get_seeded_random_state(module_name, additional_entropy)


def create_seeded_instance(cls: type, module_name: str, *args, **kwargs) -> Any:
    """Convenience function to create a seeded instance."""
    return get_global_seed_manager().create_seeded_instance(cls, module_name, *args, **kwargs)
