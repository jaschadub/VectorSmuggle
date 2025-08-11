# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Database adapter registry for managing multiple database adapters."""

import logging

from .base_adapter import DatabaseAdapter


class DatabaseAdapterRegistry:
    """Registry for managing database adapters."""

    def __init__(self, logger: logging.Logger):
        """Initialize the registry.

        Args:
            logger: Logger instance
        """
        self.logger = logger
        self._adapters: dict[str, type[DatabaseAdapter]] = {}
        self._instances: dict[str, DatabaseAdapter] = {}

    def register_adapter(self, name: str, adapter_class: type[DatabaseAdapter]) -> None:
        """Register a database adapter class.

        Args:
            name: Name of the database
            adapter_class: Adapter class to register
        """
        self.logger.debug(f"Registering adapter for database: {name}")
        self._adapters[name] = adapter_class

    def get_adapter(self, name: str) -> DatabaseAdapter | None:
        """Get an adapter instance for a database.

        Args:
            name: Name of the database

        Returns:
            Adapter instance or None if not found
        """
        if name not in self._adapters:
            self.logger.warning(f"No adapter registered for database: {name}")
            return None

        if name not in self._instances:
            try:
                adapter_class = self._adapters[name]
                self._instances[name] = adapter_class(name, self.logger)
                self.logger.debug(f"Created adapter instance for: {name}")
            except Exception as e:
                self.logger.error(f"Failed to create adapter for {name}: {e}")
                return None

        return self._instances[name]

    def get_available_databases(self) -> list[str]:
        """Get list of available database names.

        Returns:
            List of registered database names
        """
        return list(self._adapters.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a database adapter is registered.

        Args:
            name: Name of the database

        Returns:
            True if registered, False otherwise
        """
        return name in self._adapters

    def cleanup_all(self) -> None:
        """Clean up all adapter instances."""
        for name, instance in self._instances.items():
            try:
                instance.cleanup()
                self.logger.debug(f"Cleaned up adapter for: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup adapter {name}: {e}")

        self._instances.clear()

    def validate_adapters(self) -> dict[str, bool]:
        """Validate all registered adapters by testing connections.

        Returns:
            Dictionary mapping database names to connection status
        """
        results = {}
        for name in self._adapters.keys():
            adapter = self.get_adapter(name)
            if adapter:
                try:
                    results[name] = adapter.test_connection()
                except Exception as e:
                    self.logger.error(f"Connection test failed for {name}: {e}")
                    results[name] = False
            else:
                results[name] = False

        return results
