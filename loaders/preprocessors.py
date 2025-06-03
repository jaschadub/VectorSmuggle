"""Content preprocessors for document sanitization and enhancement."""

import logging
import re
from typing import Any

from langchain.schema import Document


class ContentPreprocessor:
    """Preprocessor for content sanitization, normalization, and enhancement."""

    # Patterns for sensitive data detection
    SENSITIVE_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'ssn_no_dash': r'\b\d{9}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'bank_account': r'\b\d{8,17}\b',
        'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
        'license_plate': r'\b[A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b',
        'api_key': r'\b[A-Za-z0-9]{32,}\b',
        'password_field': r'(?i)(password|pwd|pass)\s*[:=]\s*\S+',
        'salary': r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
        'date_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    }

    # Financial keywords that might indicate sensitive content
    FINANCIAL_KEYWORDS = [
        'salary', 'wage', 'income', 'revenue', 'profit', 'loss', 'budget',
        'investment', 'portfolio', 'dividend', 'interest', 'loan', 'debt',
        'mortgage', 'credit', 'debit', 'account', 'balance', 'transaction'
    ]

    # HR/Personal keywords
    HR_KEYWORDS = [
        'employee', 'staff', 'personnel', 'hire', 'fire', 'terminate',
        'performance', 'review', 'evaluation', 'disciplinary', 'promotion',
        'demotion', 'resignation', 'retirement', 'benefits', 'healthcare'
    ]

    # Technical/IP keywords
    TECHNICAL_KEYWORDS = [
        'proprietary', 'confidential', 'secret', 'patent', 'trademark',
        'copyright', 'algorithm', 'source code', 'database', 'schema',
        'api', 'endpoint', 'credentials', 'token', 'key', 'certificate'
    ]

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the content preprocessor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_documents(self, documents: list[Document],
                           sanitize: bool = False,
                           normalize: bool = True,
                           detect_sensitive: bool = True,
                           chunk_strategy: str = "auto") -> list[Document]:
        """Preprocess a list of documents.

        Args:
            documents: List of documents to preprocess
            sanitize: Whether to sanitize sensitive content
            normalize: Whether to normalize text content
            detect_sensitive: Whether to detect and mark sensitive data
            chunk_strategy: Chunking strategy ("auto", "fixed", "semantic")

        Returns:
            List of preprocessed documents
        """
        processed_docs = []

        for doc in documents:
            try:
                processed_doc = self.preprocess_document(
                    doc, sanitize, normalize, detect_sensitive, chunk_strategy
                )
                processed_docs.extend(processed_doc if isinstance(processed_doc, list) else [processed_doc])
            except Exception as e:
                self.logger.error(f"Failed to preprocess document: {e}")
                # Include original document if preprocessing fails
                processed_docs.append(doc)

        self.logger.info(f"Preprocessed {len(documents)} documents into {len(processed_docs)} chunks")
        return processed_docs

    def preprocess_document(self, document: Document,
                          sanitize: bool = False,
                          normalize: bool = True,
                          detect_sensitive: bool = True,
                          chunk_strategy: str = "auto") -> list[Document]:
        """Preprocess a single document.

        Args:
            document: Document to preprocess
            sanitize: Whether to sanitize sensitive content
            normalize: Whether to normalize text content
            detect_sensitive: Whether to detect and mark sensitive data
            chunk_strategy: Chunking strategy

        Returns:
            List of processed document chunks
        """
        content = document.page_content
        metadata = document.metadata.copy()

        # Normalize content
        if normalize:
            content = self._normalize_content(content)

        # Detect sensitive data
        sensitive_data = {}
        if detect_sensitive:
            sensitive_data = self._detect_sensitive_data(content)
            metadata.update(sensitive_data)

        # Sanitize content if requested
        if sanitize:
            content = self._sanitize_content(content)
            metadata['sanitized'] = True

        # Apply chunking strategy
        chunks = self._apply_chunking_strategy(content, metadata, chunk_strategy)

        return chunks

    def _normalize_content(self, content: str) -> str:
        """Normalize text content.

        Args:
            content: Raw text content

        Returns:
            Normalized text content
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)

        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)

        # Strip leading/trailing whitespace
        content = content.strip()

        return content

    def _detect_sensitive_data(self, content: str) -> dict[str, Any]:
        """Detect sensitive data patterns in content.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with sensitive data information
        """
        sensitive_info = {
            'has_sensitive_data': False,
            'sensitive_patterns': [],
            'sensitive_keywords': [],
            'risk_level': 'low'
        }

        # Check for sensitive patterns
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                sensitive_info['sensitive_patterns'].append({
                    'type': pattern_name,
                    'count': len(matches),
                    'examples': matches[:3]  # First 3 examples
                })
                sensitive_info['has_sensitive_data'] = True

        # Check for sensitive keywords
        content_lower = content.lower()

        financial_matches = [kw for kw in self.FINANCIAL_KEYWORDS if kw in content_lower]
        hr_matches = [kw for kw in self.HR_KEYWORDS if kw in content_lower]
        technical_matches = [kw for kw in self.TECHNICAL_KEYWORDS if kw in content_lower]

        if financial_matches:
            sensitive_info['sensitive_keywords'].append({
                'category': 'financial',
                'keywords': financial_matches
            })

        if hr_matches:
            sensitive_info['sensitive_keywords'].append({
                'category': 'hr_personal',
                'keywords': hr_matches
            })

        if technical_matches:
            sensitive_info['sensitive_keywords'].append({
                'category': 'technical_ip',
                'keywords': technical_matches
            })

        # Determine risk level
        pattern_count = len(sensitive_info['sensitive_patterns'])
        keyword_categories = len(sensitive_info['sensitive_keywords'])

        if pattern_count >= 3 or keyword_categories >= 2:
            sensitive_info['risk_level'] = 'high'
        elif pattern_count >= 1 or keyword_categories >= 1:
            sensitive_info['risk_level'] = 'medium'

        return sensitive_info

    def _sanitize_content(self, content: str) -> str:
        """Sanitize sensitive content by masking or removing it.

        Args:
            content: Text content to sanitize

        Returns:
            Sanitized content
        """
        sanitized = content

        # Mask sensitive patterns
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern_name == 'ssn':
                sanitized = re.sub(pattern, 'XXX-XX-XXXX', sanitized)
            elif pattern_name == 'credit_card':
                sanitized = re.sub(pattern, 'XXXX-XXXX-XXXX-XXXX', sanitized)
            elif pattern_name == 'phone':
                sanitized = re.sub(pattern, 'XXX-XXX-XXXX', sanitized)
            elif pattern_name == 'email':
                sanitized = re.sub(pattern, '[EMAIL_REDACTED]', sanitized)
            elif pattern_name == 'ip_address':
                sanitized = re.sub(pattern, 'XXX.XXX.XXX.XXX', sanitized)
            elif pattern_name == 'api_key':
                sanitized = re.sub(pattern, '[API_KEY_REDACTED]', sanitized)
            elif pattern_name == 'password_field':
                sanitized = re.sub(pattern, r'\1: [REDACTED]', sanitized, flags=re.IGNORECASE)
            else:
                # Generic masking for other patterns
                sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        return sanitized

    def _apply_chunking_strategy(self, content: str, metadata: dict[str, Any],
                               strategy: str) -> list[Document]:
        """Apply chunking strategy to content.

        Args:
            content: Text content to chunk
            metadata: Document metadata
            strategy: Chunking strategy

        Returns:
            List of document chunks
        """
        if strategy == "fixed":
            return self._fixed_chunking(content, metadata)
        elif strategy == "semantic":
            return self._semantic_chunking(content, metadata)
        else:  # auto
            return self._auto_chunking(content, metadata)

    def _fixed_chunking(self, content: str, metadata: dict[str, Any],
                       chunk_size: int = 512, overlap: int = 50) -> list[Document]:
        """Apply fixed-size chunking.

        Args:
            content: Text content
            metadata: Document metadata
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of document chunks
        """
        chunks = []
        start = 0
        chunk_num = 1

        while start < len(content):
            end = start + chunk_size
            chunk_content = content[start:end]

            # Try to break at word boundary
            if end < len(content):
                last_space = chunk_content.rfind(' ')
                if last_space > chunk_size * 0.8:  # If we can find a space in the last 20%
                    end = start + last_space
                    chunk_content = content[start:end]

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_number': chunk_num,
                'chunk_start': start,
                'chunk_end': end,
                'chunking_strategy': 'fixed'
            })

            chunks.append(Document(page_content=chunk_content.strip(), metadata=chunk_metadata))

            start = end - overlap
            chunk_num += 1

        return chunks

    def _semantic_chunking(self, content: str, metadata: dict[str, Any]) -> list[Document]:
        """Apply semantic chunking based on content structure.

        Args:
            content: Text content
            metadata: Document metadata

        Returns:
            List of document chunks
        """
        # Simple semantic chunking based on paragraphs and sections
        chunks = []

        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        current_chunk = ""
        chunk_num = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would make chunk too large, start new chunk
            if len(current_chunk) + len(paragraph) > 800 and current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_number': chunk_num,
                    'chunking_strategy': 'semantic'
                })

                chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))
                current_chunk = paragraph
                chunk_num += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_number': chunk_num,
                'chunking_strategy': 'semantic'
            })
            chunks.append(Document(page_content=current_chunk.strip(), metadata=chunk_metadata))

        return chunks if chunks else [Document(page_content=content, metadata=metadata)]

    def _auto_chunking(self, content: str, metadata: dict[str, Any]) -> list[Document]:
        """Apply automatic chunking based on content characteristics.

        Args:
            content: Text content
            metadata: Document metadata

        Returns:
            List of document chunks
        """
        # Choose strategy based on content characteristics
        if len(content) < 1000:
            # Small content, no chunking needed
            return [Document(page_content=content, metadata=metadata)]
        elif '\n\n' in content and content.count('\n\n') > 2:
            # Content has clear paragraph structure, use semantic chunking
            return self._semantic_chunking(content, metadata)
        else:
            # Use fixed chunking for other content
            return self._fixed_chunking(content, metadata)

    def extract_metadata(self, document: Document) -> dict[str, Any]:
        """Extract enhanced metadata from document.

        Args:
            document: Document to analyze

        Returns:
            Enhanced metadata dictionary
        """
        content = document.page_content
        metadata = document.metadata.copy()

        # Basic statistics
        metadata.update({
            'character_count': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.splitlines()),
            'paragraph_count': len([p for p in content.split('\n\n') if p.strip()])
        })

        # Content analysis
        sensitive_data = self._detect_sensitive_data(content)
        metadata.update(sensitive_data)

        # Language detection (simple heuristic)
        if re.search(r'[а-яё]', content, re.IGNORECASE):
            metadata['detected_language'] = 'russian'
        elif re.search(r'[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]', content, re.IGNORECASE):
            metadata['detected_language'] = 'european'
        else:
            metadata['detected_language'] = 'english'

        return metadata
