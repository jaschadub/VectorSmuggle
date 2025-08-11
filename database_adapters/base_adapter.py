# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Base database adapter interface for multi-database effectiveness testing."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from config import Config


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""

    def __init__(self, name: str, logger: logging.Logger):
        """Initialize the database adapter.

        Args:
            name: Name of the database
            logger: Logger instance
        """
        self.name = name
        self.logger = logger

    @abstractmethod
    def get_connection_config(self) -> dict[str, Any]:
        """Get connection configuration for this database.

        Returns:
            Dictionary containing connection parameters
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the database connection is working.

        Returns:
            True if connection is successful, False otherwise
        """
        pass

    @abstractmethod
    def run_effectiveness_test(self, config: Config) -> dict[str, Any]:
        """Run the effectiveness test against this database.

        Args:
            config: Configuration object

        Returns:
            Dictionary containing test results
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by this adapter."""
        pass

    @abstractmethod
    def get_setup_instructions(self) -> str:
        """Get setup instructions for this database.

        Returns:
            String containing setup instructions
        """
        pass

    def get_health_check_info(self) -> dict[str, Any]:
        """Get health check information for this database.

        Returns:
            Dictionary containing health check details
        """
        try:
            connection_ok = self.test_connection()
            return {
                'status': 'healthy' if connection_ok else 'unhealthy',
                'connection_test': 'passed' if connection_ok else 'failed',
                'config': self.get_connection_config()
            }
        except Exception as e:
            return {
                'status': 'error',
                'connection_test': 'error',
                'error': str(e),
                'config': self.get_connection_config()
            }
