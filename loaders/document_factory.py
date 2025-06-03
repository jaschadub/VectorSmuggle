"""Document loader factory for multi-format support."""

import logging
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from .email_loader import EmailLoader
from .office_loader import OfficeLoader
from .structured_loader import StructuredLoader


class DocumentLoaderFactory:
    """Factory for creating appropriate document loaders based on file type."""

    # Supported file extensions mapped to loader classes
    LOADER_MAPPING = {
        # PDF documents
        '.pdf': PyPDFLoader,

        # Office documents
        '.docx': OfficeLoader,
        '.xlsx': OfficeLoader,
        '.pptx': OfficeLoader,
        '.doc': OfficeLoader,
        '.xls': OfficeLoader,
        '.ppt': OfficeLoader,

        # Structured data
        '.csv': StructuredLoader,
        '.json': StructuredLoader,
        '.xml': StructuredLoader,
        '.yaml': StructuredLoader,
        '.yml': StructuredLoader,
        '.txt': StructuredLoader,
        '.md': StructuredLoader,
        '.html': StructuredLoader,
        '.htm': StructuredLoader,

        # Email formats
        '.eml': EmailLoader,
        '.msg': EmailLoader,
        '.mbox': EmailLoader,
    }

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the document loader factory.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns:
            List of supported file extensions
        """
        return list(self.LOADER_MAPPING.keys())

    def is_supported_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if format is supported, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.LOADER_MAPPING

    def detect_format(self, file_path: str | Path) -> str | None:
        """Detect file format based on extension.

        Args:
            file_path: Path to the file

        Returns:
            File extension if supported, None otherwise
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension in self.LOADER_MAPPING:
            return extension

        self.logger.warning(f"Unsupported file format: {extension}")
        return None

    def create_loader(self, file_path: str | Path, **kwargs) -> Any:
        """Create appropriate document loader for the file.

        Args:
            file_path: Path to the document file
            **kwargs: Additional arguments for the loader

        Returns:
            Document loader instance

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Detect format
        format_ext = self.detect_format(path)
        if not format_ext:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Get loader class
        loader_class = self.LOADER_MAPPING[format_ext]

        # Create loader instance
        try:
            if loader_class == PyPDFLoader:
                # PyPDFLoader only takes file path
                loader = loader_class(str(path))
            else:
                # Custom loaders take file path and additional kwargs
                loader = loader_class(str(path), **kwargs)

            self.logger.info(f"Created {loader_class.__name__} for {path}")
            return loader

        except Exception as e:
            self.logger.error(f"Failed to create loader for {path}: {e}")
            raise

    def load_documents(self, file_paths: str | Path | list[str | Path], **kwargs) -> list[Document]:
        """Load documents from one or more files.

        Args:
            file_paths: Single file path or list of file paths
            **kwargs: Additional arguments for loaders

        Returns:
            List of loaded documents

        Raises:
            ValueError: If no valid files provided or unsupported formats
        """
        # Normalize input to list
        if isinstance(file_paths, str | Path):
            file_paths = [file_paths]

        if not file_paths:
            raise ValueError("No file paths provided")

        all_documents = []
        failed_files = []

        for file_path in file_paths:
            try:
                # Create loader and load documents
                loader = self.create_loader(file_path, **kwargs)
                documents = loader.load()

                # Add source metadata
                for doc in documents:
                    if 'source' not in doc.metadata:
                        doc.metadata['source'] = str(Path(file_path).resolve())
                    doc.metadata['file_type'] = self.detect_format(file_path)

                all_documents.extend(documents)
                self.logger.info(f"Loaded {len(documents)} documents from {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                failed_files.append(str(file_path))
                continue

        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        if not all_documents:
            raise ValueError("No documents could be loaded from provided files")

        self.logger.info(f"Successfully loaded {len(all_documents)} total documents")
        return all_documents

    def load_directory(self, directory_path: str | Path, recursive: bool = True, **kwargs) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            directory_path: Path to directory
            recursive: Whether to search subdirectories
            **kwargs: Additional arguments for loaders

        Returns:
            List of loaded documents

        Raises:
            ValueError: If directory doesn't exist or no supported files found
        """
        dir_path = Path(directory_path)

        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")

        # Find all supported files
        supported_files = []

        if recursive:
            # Search recursively
            for ext in self.LOADER_MAPPING.keys():
                pattern = f"**/*{ext}"
                supported_files.extend(dir_path.glob(pattern))
        else:
            # Search only in current directory
            for ext in self.LOADER_MAPPING.keys():
                pattern = f"*{ext}"
                supported_files.extend(dir_path.glob(pattern))

        if not supported_files:
            raise ValueError(f"No supported files found in {dir_path}")

        self.logger.info(f"Found {len(supported_files)} supported files in {dir_path}")

        # Load all documents
        return self.load_documents(supported_files, **kwargs)

    def get_format_statistics(self, file_paths: list[str | Path]) -> dict[str, int]:
        """Get statistics about file formats in the provided paths.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file extensions to counts
        """
        format_counts = {}

        for file_path in file_paths:
            format_ext = self.detect_format(file_path)
            if format_ext:
                format_counts[format_ext] = format_counts.get(format_ext, 0) + 1

        return format_counts
