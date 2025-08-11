# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Database adapters for multi-database effectiveness testing."""

from .base_adapter import DatabaseAdapter
from .registry import DatabaseAdapterRegistry

__all__ = ['DatabaseAdapter', 'DatabaseAdapterRegistry']
