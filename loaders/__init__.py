# Copyright 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the VectorSmuggle project.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

"""Document loaders for VectorSmuggle multi-format support."""

from .database_loader import DatabaseLoader
from .document_factory import DocumentLoaderFactory
from .email_loader import EmailLoader
from .office_loader import OfficeLoader
from .preprocessors import ContentPreprocessor
from .structured_loader import StructuredLoader

__all__ = [
    "DocumentLoaderFactory",
    "OfficeLoader",
    "StructuredLoader",
    "EmailLoader",
    "DatabaseLoader",
    "ContentPreprocessor"
]
