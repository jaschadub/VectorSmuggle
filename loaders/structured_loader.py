"""Structured data loader for CSV, JSON, XML, TXT files."""

import csv
import json
import logging
from pathlib import Path

from langchain.schema import Document


class StructuredLoader:
    """Loader for structured data formats (CSV, JSON, XML, TXT, etc.)."""

    def __init__(self, file_path: str, logger: logging.Logger | None = None):
        """Initialize the structured data loader.

        Args:
            file_path: Path to the structured data file
            logger: Optional logger instance
        """
        self.file_path = Path(file_path)
        self.logger = logger or logging.getLogger(__name__)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file_type = self.file_path.suffix.lower()

        # Validate supported formats
        supported_formats = ['.csv', '.json', '.xml', '.yaml', '.yml', '.txt', '.md', '.html', '.htm']
        if self.file_type not in supported_formats:
            raise ValueError(f"Unsupported structured format: {self.file_type}")

    def load(self) -> list[Document]:
        """Load the structured document and return Document objects.

        Returns:
            List of Document objects with extracted content

        Raises:
            Exception: If document loading fails
        """
        try:
            if self.file_type == '.csv':
                return self._load_csv()
            elif self.file_type == '.json':
                return self._load_json()
            elif self.file_type in ['.xml', '.html', '.htm']:
                return self._load_xml_html()
            elif self.file_type in ['.yaml', '.yml']:
                return self._load_yaml()
            elif self.file_type in ['.txt', '.md']:
                return self._load_text()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

        except Exception as e:
            self.logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def _load_csv(self) -> list[Document]:
        """Load CSV file.

        Returns:
            List of Document objects
        """
        try:
            documents = []

            with open(self.file_path, encoding='utf-8', newline='') as file:
                # Try to detect delimiter
                sample = file.read(1024)
                file.seek(0)

                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter

                reader = csv.DictReader(file, delimiter=delimiter)

                # Get headers
                headers = reader.fieldnames
                if not headers:
                    raise ValueError("CSV file has no headers")

                # Process rows
                rows = list(reader)

                # Create content as formatted table
                content_parts = []
                content_parts.append("Headers: " + " | ".join(headers))
                content_parts.append("-" * 50)

                for i, row in enumerate(rows):
                    row_text = " | ".join(str(row.get(header, "")) for header in headers)
                    content_parts.append(f"Row {i+1}: {row_text}")

                full_content = "\n".join(content_parts)

                metadata = {
                    'source': str(self.file_path.resolve()),
                    'file_type': self.file_type,
                    'loader': 'StructuredLoader',
                    'headers': headers,
                    'row_count': len(rows),
                    'column_count': len(headers)
                }

                documents.append(Document(page_content=full_content, metadata=metadata))

                # Also create individual documents for each row if there are many rows
                if len(rows) > 10:
                    for i, row in enumerate(rows):
                        row_content = "\n".join([f"{header}: {row.get(header, '')}" for header in headers])
                        row_metadata = metadata.copy()
                        row_metadata.update({
                            'row_number': i + 1,
                            'content_type': 'csv_row'
                        })
                        documents.append(Document(
                            page_content=f"[CSV Row {i+1}]\n{row_content}",
                            metadata=row_metadata
                        ))

                self.logger.info(f"Loaded CSV with {len(rows)} rows and {len(headers)} columns")
                return documents

        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            raise

    def _load_json(self) -> list[Document]:
        """Load JSON file.

        Returns:
            List of Document objects
        """
        try:
            with open(self.file_path, encoding='utf-8') as file:
                data = json.load(file)

            documents = []

            # Convert JSON to readable text
            formatted_json = json.dumps(data, indent=2, ensure_ascii=False)

            metadata = {
                'source': str(self.file_path.resolve()),
                'file_type': self.file_type,
                'loader': 'StructuredLoader',
                'json_type': type(data).__name__
            }

            # Add structure information
            if isinstance(data, dict):
                metadata['keys'] = list(data.keys())
                metadata['key_count'] = len(data.keys())
            elif isinstance(data, list):
                metadata['item_count'] = len(data)
                if data and isinstance(data[0], dict):
                    metadata['item_keys'] = list(data[0].keys())

            documents.append(Document(page_content=formatted_json, metadata=metadata))

            # If it's a list of objects, create individual documents
            if isinstance(data, list) and len(data) > 1:
                for i, item in enumerate(data):
                    item_content = json.dumps(item, indent=2, ensure_ascii=False)
                    item_metadata = metadata.copy()
                    item_metadata.update({
                        'item_number': i + 1,
                        'content_type': 'json_item'
                    })
                    documents.append(Document(
                        page_content=f"[JSON Item {i+1}]\n{item_content}",
                        metadata=item_metadata
                    ))

            self.logger.info(f"Loaded JSON with {len(documents)} document(s)")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load JSON: {e}")
            raise

    def _load_xml_html(self) -> list[Document]:
        """Load XML or HTML file.

        Returns:
            List of Document objects
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for XML/HTML files. Install with: pip install beautifulsoup4")

        try:
            with open(self.file_path, encoding='utf-8') as file:
                content = file.read()

            # Parse with BeautifulSoup
            if self.file_type == '.xml':
                soup = BeautifulSoup(content, 'xml')
            else:
                soup = BeautifulSoup(content, 'html.parser')

            # Extract text content
            text_content = soup.get_text(separator='\n', strip=True)

            # Extract structure information
            tags = [tag.name for tag in soup.find_all()]
            unique_tags = list(set(tags))

            metadata = {
                'source': str(self.file_path.resolve()),
                'file_type': self.file_type,
                'loader': 'StructuredLoader',
                'unique_tags': unique_tags,
                'total_tags': len(tags)
            }

            # Add title if it's HTML
            if self.file_type in ['.html', '.htm']:
                title_tag = soup.find('title')
                if title_tag:
                    metadata['title'] = title_tag.get_text(strip=True)

            documents = [Document(page_content=text_content, metadata=metadata)]

            self.logger.info(f"Loaded {self.file_type.upper()} with {len(unique_tags)} unique tags")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load {self.file_type.upper()}: {e}")
            raise

    def _load_yaml(self) -> list[Document]:
        """Load YAML file.

        Returns:
            List of Document objects
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML files. Install with: pip install PyYAML")

        try:
            with open(self.file_path, encoding='utf-8') as file:
                data = yaml.safe_load(file)

            # Convert YAML to readable text
            formatted_yaml = yaml.dump(data, default_flow_style=False, allow_unicode=True)

            metadata = {
                'source': str(self.file_path.resolve()),
                'file_type': self.file_type,
                'loader': 'StructuredLoader',
                'yaml_type': type(data).__name__
            }

            # Add structure information
            if isinstance(data, dict):
                metadata['keys'] = list(data.keys())
                metadata['key_count'] = len(data.keys())
            elif isinstance(data, list):
                metadata['item_count'] = len(data)

            documents = [Document(page_content=formatted_yaml, metadata=metadata)]

            self.logger.info("Loaded YAML document")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load YAML: {e}")
            raise

    def _load_text(self) -> list[Document]:
        """Load plain text or markdown file.

        Returns:
            List of Document objects
        """
        try:
            with open(self.file_path, encoding='utf-8') as file:
                content = file.read()

            metadata = {
                'source': str(self.file_path.resolve()),
                'file_type': self.file_type,
                'loader': 'StructuredLoader',
                'character_count': len(content),
                'line_count': len(content.splitlines())
            }

            documents = [Document(page_content=content, metadata=metadata)]

            self.logger.info(f"Loaded text file with {len(content)} characters")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load text file: {e}")
            raise
