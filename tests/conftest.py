"""Pytest configuration and shared fixtures for VectorSmuggle tests."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Test Configuration Fixtures
@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "temp_dir": tempfile.gettempdir(),
        "mock_external_apis": True
    }


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Mock configuration object for testing."""
    from config import Config

    config = Config()
    config.document.chunk_size = 512
    config.document.chunk_overlap = 50
    config.document.enable_preprocessing = True
    config.vector_store.type = "faiss"
    config.openai.api_key = "test-api-key"
    config.openai.model = "text-embedding-ada-002"

    return config


# Document and Data Fixtures
@pytest.fixture
def sample_text():
    """Sample text content for testing."""
    return """
    This is a sample document for testing VectorSmuggle functionality.
    It contains multiple sentences and paragraphs to test text processing.

    This document is used for unit testing and should not contain any
    real sensitive information. It's designed to test document loading,
    preprocessing, and embedding generation capabilities.
    """


@pytest.fixture
def sample_documents():
    """Sample Document objects for testing."""
    documents = [
        Document(
            page_content="This is a public document about machine learning.",
            metadata={
                "source": "public_doc.pdf",
                "file_type": "pdf",
                "classification": "public",
                "has_sensitive_data": False
            }
        ),
        Document(
            page_content="Internal company policy on data handling procedures.",
            metadata={
                "source": "internal_policy.docx",
                "file_type": "docx",
                "classification": "internal",
                "has_sensitive_data": False
            }
        ),
        Document(
            page_content="SSN: 123-45-6789, Credit Card: 4532-1234-5678-9012",
            metadata={
                "source": "sensitive_data.txt",
                "file_type": "txt",
                "classification": "confidential",
                "has_sensitive_data": True,
                "sensitive_patterns": [
                    {"type": "ssn", "count": 1},
                    {"type": "credit_card", "count": 1}
                ]
            }
        )
    ]
    return documents


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing."""
    # Create realistic embeddings similar to OpenAI's text-embedding-ada-002
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.normal(0, 1, (10, 1536)).astype(np.float32)
    # Normalize to unit vectors (typical for embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def large_embeddings():
    """Large embedding dataset for performance testing."""
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (1000, 1536)).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


# Mock External Services
@pytest.fixture
def mock_openai_embeddings(monkeypatch):
    """Mock OpenAI embeddings API."""
    def mock_embed_documents(texts: list[str]) -> list[list[float]]:
        # Return deterministic embeddings for testing
        return [
            [0.1] * 1536 for _ in texts
        ]

    def mock_embed_query(text: str) -> list[float]:
        return [0.1] * 1536

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents = mock_embed_documents
    mock_embeddings.embed_query = mock_embed_query

    # Mock the embeddings creation
    def mock_create_embeddings(*args, **kwargs):
        return mock_embeddings

    monkeypatch.setattr("utils.embedding_factory.create_embeddings", mock_create_embeddings)
    return mock_embeddings


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = MagicMock()

    def mock_add_texts(texts, metadatas=None):
        return [f"doc_{i}" for i in range(len(texts))]

    def mock_similarity_search(query, k=4, **kwargs):
        return [
            Document(
                page_content=f"Mock result {i} for query: {query}",
                metadata={"score": 0.9 - i * 0.1, "mock": True}
            )
            for i in range(k)
        ]

    mock_store.add_texts = mock_add_texts
    mock_store.similarity_search = mock_similarity_search
    mock_store.as_retriever = lambda: mock_store

    return mock_store


@pytest.fixture
def mock_ollama_server(monkeypatch):
    """Mock Ollama server for testing fallback."""
    def mock_ollama_response(*args, **kwargs):
        return {"embedding": [0.2] * 384}  # Different dimension for testing

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: MagicMock(
        json=lambda: mock_ollama_response(),
        status_code=200
    ))


# File System Fixtures
@pytest.fixture
def sample_pdf_file(temp_directory):
    """Create a sample PDF file for testing."""
    pdf_path = temp_directory / "test_document.pdf"

    # Create a simple text file as a placeholder
    # In a real implementation, you'd create actual PDF content
    pdf_path.write_text("Sample PDF content for testing")

    return pdf_path


@pytest.fixture
def sample_docx_file(temp_directory):
    """Create a sample DOCX file for testing."""
    docx_path = temp_directory / "test_document.docx"

    # Create a simple text file as a placeholder
    docx_path.write_text("Sample DOCX content for testing")

    return docx_path


@pytest.fixture
def sensitive_test_file(temp_directory):
    """Create a file with sensitive data for testing."""
    sensitive_path = temp_directory / "sensitive.txt"
    content = """
    Employee ID: EMP001
    SSN: 123-45-6789
    Credit Card: 4532-1234-5678-9012
    Email: john.doe@example.com
    Phone: (555) 123-4567
    """
    sensitive_path.write_text(content)
    return sensitive_path


# Performance Testing Fixtures
@pytest.fixture
def performance_baseline():
    """Baseline performance metrics for regression testing."""
    return {
        "document_processing_ms_per_doc": 50,
        "embedding_generation_ms_per_1k": 100,
        "query_latency_ms": 10,
        "memory_usage_mb_per_1k_docs": 50
    }


# Security Testing Fixtures
@pytest.fixture
def malicious_inputs():
    """Collection of malicious inputs for security testing."""
    return {
        "sql_injection": ["'; DROP TABLE users; --", "1' OR '1'='1"],
        "command_injection": ["; cat /etc/passwd", "$(whoami)"],
        "path_traversal": ["../../../etc/passwd", "..\\windows\\system32"],
        "xss": ["<script>alert('XSS')</script>", "javascript:alert(1)"],
        "large_payload": "A" * (10 * 1024 * 1024),  # 10MB
        "unicode_overflow": "\x00" * 1000,
        "format_string": "%s%s%s%s%s%n%n%n%n%n"
    }


@pytest.fixture
def steganography_test_vectors():
    """Test vectors for steganography validation."""
    return {
        "noise_levels": [0.001, 0.01, 0.1],
        "rotation_angles": [0.1, 0.5, 1.0],  # radians
        "scaling_factors": [0.9, 1.0, 1.1],
        "fragmentation_sizes": [2, 4, 8],
        "payload_sizes": [1, 10, 100, 1000]  # bits
    }


# Research Validation Fixtures
@pytest.fixture
def research_datasets():
    """Datasets for research validation."""
    return {
        "public_documents": [
            "Wikipedia article about machine learning",
            "Open source software documentation",
            "Public research paper abstract"
        ],
        "corporate_documents": [
            "Internal company memo",
            "Project requirements document",
            "Meeting notes and action items"
        ],
        "sensitive_documents": [
            "Financial quarterly report",
            "Employee personal information",
            "Customer data and analytics"
        ]
    }


# Test Utilities
@pytest.fixture
def assert_helpers():
    """Helper functions for common test assertions."""
    class AssertHelpers:
        @staticmethod
        def assert_embeddings_valid(embeddings: np.ndarray):
            """Assert embeddings are valid."""
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.dtype in [np.float32, np.float64]
            assert embeddings.shape[1] > 0  # Has dimensions
            assert not np.isnan(embeddings).any()  # No NaN values
            assert not np.isinf(embeddings).any()  # No infinite values

        @staticmethod
        def assert_documents_valid(documents: list[Document]):
            """Assert documents are valid."""
            assert isinstance(documents, list)
            assert all(isinstance(doc, Document) for doc in documents)
            assert all(len(doc.page_content) > 0 for doc in documents)
            assert all(isinstance(doc.metadata, dict) for doc in documents)

        @staticmethod
        def assert_performance_acceptable(actual_ms: float, baseline_ms: float, tolerance: float = 0.2):
            """Assert performance is within acceptable range."""
            max_allowed = baseline_ms * (1 + tolerance)
            assert actual_ms <= max_allowed, f"Performance degraded: {actual_ms}ms > {max_allowed}ms"

    return AssertHelpers()


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically clean up test artifacts."""
    yield
    # Cleanup logic here if needed
    pass


# Markers for test categorization
pytest_plugins = []

# Standalone research/CLI scripts that share the test_*.py naming convention
# but are not pytest tests. They are invoked by run_research_tests.sh
# (Docker-based). Skip pytest collection to keep `pytest tests/` clean.
collect_ignore = [
    "test_api_connectivity.py",
    "test_baseline_generation.py",
    "test_detection_systems.py",
    "test_steganography.py",
    "generate_comprehensive_report.py",
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "research: Research validation tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "docker: Tests requiring Docker")
    config.addinivalue_line("markers", "external: Tests requiring external services")
