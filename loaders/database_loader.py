"""Database loader for query results and database exports."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from langchain.schema import Document


class DatabaseLoader:
    """Loader for database files and query results."""

    def __init__(self, file_path: str, query: str | None = None, logger: logging.Logger | None = None):
        """Initialize the database loader.

        Args:
            file_path: Path to the database file
            query: Optional SQL query to execute
            logger: Optional logger instance
        """
        self.file_path = Path(file_path)
        self.query = query
        self.logger = logger or logging.getLogger(__name__)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.file_path}")

        self.file_type = self.file_path.suffix.lower()

        # Validate supported formats
        supported_formats = ['.db', '.sqlite', '.sqlite3']
        if self.file_type not in supported_formats:
            raise ValueError(f"Unsupported database format: {self.file_type}")

    def load(self) -> list[Document]:
        """Load the database and return Document objects.

        Returns:
            List of Document objects with extracted content

        Raises:
            Exception: If database loading fails
        """
        try:
            if self.file_type in ['.db', '.sqlite', '.sqlite3']:
                return self._load_sqlite()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

        except Exception as e:
            self.logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def _load_sqlite(self) -> list[Document]:
        """Load SQLite database.

        Returns:
            List of Document objects
        """
        try:
            conn = sqlite3.connect(str(self.file_path))
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            documents = []

            if self.query:
                # Execute custom query
                documents.extend(self._execute_query(cursor, self.query))
            else:
                # Get all tables and their data
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]

                    # Skip system tables
                    if table_name.startswith('sqlite_'):
                        continue

                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    schema = cursor.fetchall()

                    # Get table data
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000;")  # Limit for safety
                    rows = cursor.fetchall()

                    if rows:
                        doc = self._create_table_document(table_name, schema, rows)
                        documents.append(doc)

            conn.close()

            self.logger.info(f"Loaded {len(documents)} document(s) from SQLite database")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load SQLite database: {e}")
            raise

    def _execute_query(self, cursor: sqlite3.Cursor, query: str) -> list[Document]:
        """Execute a custom SQL query and return documents.

        Args:
            cursor: Database cursor
            query: SQL query to execute

        Returns:
            List of Document objects
        """
        try:
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                return []

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Create content
            content_parts = []
            content_parts.append(f"Query: {query}")
            content_parts.append(f"Results: {len(rows)} rows")
            content_parts.append("")

            # Add header
            content_parts.append(" | ".join(columns))
            content_parts.append("-" * 50)

            # Add rows
            for row in rows:
                row_data = [str(value) if value is not None else "NULL" for value in row]
                content_parts.append(" | ".join(row_data))

            full_content = "\n".join(content_parts)

            metadata = {
                'source': str(self.file_path.resolve()),
                'file_type': self.file_type,
                'loader': 'DatabaseLoader',
                'query': query,
                'row_count': len(rows),
                'column_count': len(columns),
                'columns': columns
            }

            return [Document(page_content=full_content, metadata=metadata)]

        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise

    def _create_table_document(self, table_name: str, schema: list, rows: list) -> Document:
        """Create a document for a database table.

        Args:
            table_name: Name of the table
            schema: Table schema information
            rows: Table rows

        Returns:
            Document object
        """
        content_parts = []

        # Table header
        content_parts.append(f"Table: {table_name}")
        content_parts.append(f"Rows: {len(rows)}")
        content_parts.append("")

        # Schema information
        content_parts.append("Schema:")
        for col in schema:
            col_info = f"  {col[1]} ({col[2]})"
            if col[3]:  # NOT NULL
                col_info += " NOT NULL"
            if col[5]:  # PRIMARY KEY
                col_info += " PRIMARY KEY"
            content_parts.append(col_info)
        content_parts.append("")

        # Data
        if rows:
            # Get column names from schema
            columns = [col[1] for col in schema]

            # Add header
            content_parts.append(" | ".join(columns))
            content_parts.append("-" * 50)

            # Add rows (limit display for readability)
            display_rows = rows[:100]  # Show first 100 rows
            for row in display_rows:
                row_data = [str(value) if value is not None else "NULL" for value in row]
                content_parts.append(" | ".join(row_data))

            if len(rows) > 100:
                content_parts.append(f"... and {len(rows) - 100} more rows")

        full_content = "\n".join(content_parts)

        # Extract column information
        columns = [col[1] for col in schema]
        column_types = [col[2] for col in schema]

        metadata = {
            'source': str(self.file_path.resolve()),
            'file_type': self.file_type,
            'loader': 'DatabaseLoader',
            'table_name': table_name,
            'row_count': len(rows),
            'column_count': len(columns),
            'columns': columns,
            'column_types': column_types
        }

        return Document(page_content=full_content, metadata=metadata)

    def get_table_names(self) -> list[str]:
        """Get list of table names in the database.

        Returns:
            List of table names
        """
        try:
            conn = sqlite3.connect(str(self.file_path))
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            conn.close()

            # Filter out system tables
            table_names = [table[0] for table in tables if not table[0].startswith('sqlite_')]

            return table_names

        except Exception as e:
            self.logger.error(f"Failed to get table names: {e}")
            return []

    def get_table_schema(self, table_name: str) -> list[dict[str, Any]]:
        """Get schema information for a specific table.

        Args:
            table_name: Name of the table

        Returns:
            List of column information dictionaries
        """
        try:
            conn = sqlite3.connect(str(self.file_path))
            cursor = conn.cursor()

            cursor.execute(f"PRAGMA table_info({table_name});")
            schema = cursor.fetchall()

            conn.close()

            # Convert to more readable format
            schema_info = []
            for col in schema:
                schema_info.append({
                    'column_id': col[0],
                    'name': col[1],
                    'type': col[2],
                    'not_null': bool(col[3]),
                    'default_value': col[4],
                    'primary_key': bool(col[5])
                })

            return schema_info

        except Exception as e:
            self.logger.error(f"Failed to get table schema: {e}")
            return []
