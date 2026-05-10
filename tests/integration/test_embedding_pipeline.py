# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for VectorSmuggle embedding pipeline."""


import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.document_factory import DocumentLoaderFactory
from loaders.preprocessors import ContentPreprocessor
from steganography.obfuscation import EmbeddingObfuscator
from utils.embedding_factory import create_embeddings


class TestEmbeddingPipeline:
    """Test end-to-end embedding pipeline integration."""

    @pytest.fixture
    def sample_documents_on_disk(self, temp_directory):
        """Create sample documents on disk for testing.

        Uses real text files (.txt/.md) since fake .pdf/.docx content
        fails real parsers (pypdf needs real PDF headers, docx needs ZIP).
        """
        documents = {}

        # Plain text file
        text_file = temp_directory / "sample.txt"
        text_file.write_text("This is sample text content for testing the embedding pipeline.")
        documents["text"] = str(text_file)

        # Markdown file
        md_file = temp_directory / "sample.md"
        md_file.write_text("# Business Information\n\nSample markdown content with business notes.")
        documents["markdown"] = str(md_file)

        # Sensitive data file
        sensitive_file = temp_directory / "sensitive.txt"
        sensitive_content = """
        Employee: John Doe
        SSN: 123-45-6789
        Credit Card: 4532-1234-5678-9012
        Email: john.doe@company.com
        """
        sensitive_file.write_text(sensitive_content)
        documents["sensitive"] = str(sensitive_file)

        return documents

    @pytest.mark.integration
    def test_full_pipeline_without_steganography(
        self,
        sample_documents_on_disk,
        sample_config,
        mock_openai_embeddings
    ):
        """Test complete pipeline: load → process → embed → store."""
        # 1. Document Loading
        factory = DocumentLoaderFactory()
        documents = factory.load_documents(list(sample_documents_on_disk.values()))

        assert len(documents) >= 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(len(doc.page_content) > 0 for doc in documents)

        # 2. Content Preprocessing
        preprocessor = ContentPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)

        # Should have detected sensitive data
        sensitive_docs = [
            doc for doc in processed_documents
            if doc.metadata.get("has_sensitive_data", False)
        ]
        assert len(sensitive_docs) > 0

        # 3. Text Splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=sample_config.document.chunk_size,
            chunk_overlap=sample_config.document.chunk_overlap
        )
        chunks = splitter.split_documents(processed_documents)

        assert len(chunks) >= len(documents)  # Should have at least as many chunks as docs

        # 4. Embedding Generation
        embeddings = create_embeddings(sample_config, None)
        texts = [chunk.page_content for chunk in chunks]
        embedding_vectors = embeddings.embed_documents(texts)

        assert len(embedding_vectors) == len(texts)
        assert all(len(vec) > 0 for vec in embedding_vectors)

        # 5. Verify pipeline integrity
        for _i, (text, vector) in enumerate(zip(texts, embedding_vectors, strict=False)):
            assert len(text) > 0
            assert len(vector) > 0
            assert isinstance(vector, list)
            assert all(isinstance(x, float) for x in vector)

    @pytest.mark.integration
    def test_pipeline_with_steganography(
        self,
        sample_documents_on_disk,
        sample_config,
        mock_openai_embeddings
    ):
        """Test pipeline with steganographic obfuscation."""
        # Load and process documents
        factory = DocumentLoaderFactory()
        documents = factory.load_documents(list(sample_documents_on_disk.values()))

        preprocessor = ContentPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=sample_config.document.chunk_size,
            chunk_overlap=sample_config.document.chunk_overlap
        )
        chunks = splitter.split_documents(processed_documents)

        # Generate embeddings
        embeddings = create_embeddings(sample_config, None)
        texts = [chunk.page_content for chunk in chunks]
        embedding_vectors = np.array(embeddings.embed_documents(texts))

        # Apply steganographic obfuscation
        obfuscator = EmbeddingObfuscator(noise_level=0.01, seed=42)

        # Test different techniques
        techniques = ["noise", "rotation", "scaling"]
        obfuscated_embeddings = embedding_vectors.copy()

        for technique in techniques:
            if technique == "noise":
                obfuscated_embeddings = obfuscator.inject_noise(obfuscated_embeddings)
            elif technique == "rotation":
                obfuscated_embeddings, _ = obfuscator.apply_rotation(obfuscated_embeddings)
            elif technique == "scaling":
                obfuscated_embeddings = obfuscator.apply_scaling(obfuscated_embeddings)

        # Verify obfuscation was applied
        assert not np.array_equal(embedding_vectors, obfuscated_embeddings)
        assert obfuscated_embeddings.shape == embedding_vectors.shape

        # Verify embeddings have changed but remain bounded
        # (Note: mocked embeddings are all identical so similarity correlation
        # cannot be meaningfully tested here. Real embedding fidelity tests
        # live in tests/research/.)
        diff = np.abs(embedding_vectors - obfuscated_embeddings).mean()
        assert diff > 0  # Obfuscation produced non-zero changes
        assert diff < 1.0  # But changes are bounded

    @pytest.mark.integration
    def test_error_handling_in_pipeline(self, temp_directory, sample_config):
        """Test error handling throughout the pipeline."""
        # Create problematic files
        empty_file = temp_directory / "empty.txt"
        empty_file.write_text("")

        binary_file = temp_directory / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03" * 1000)

        large_file = temp_directory / "large.txt"
        large_file.write_text("A" * (10 * 1024 * 1024))  # 10MB file

        problematic_files = [str(empty_file), str(binary_file), str(large_file)]

        # Test document loading error handling
        factory = DocumentLoaderFactory()
        try:
            documents = factory.load_documents(problematic_files)
            # Should either handle gracefully or provide informative errors
            assert isinstance(documents, list)
        except Exception as e:
            assert isinstance(e, (ValueError, IOError, UnicodeDecodeError))

    @pytest.mark.integration
    def test_pipeline_performance(self, sample_documents_on_disk, mock_openai_embeddings):
        """Test pipeline performance with realistic data sizes."""
        import time

        # Create multiple document batches
        all_files = list(sample_documents_on_disk.values()) * 10  # 30 files total

        start_time = time.time()

        # Load documents
        factory = DocumentLoaderFactory()
        documents = factory.load_documents(all_files)

        load_time = time.time() - start_time

        # Process documents
        start_time = time.time()
        preprocessor = ContentPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)
        process_time = time.time() - start_time

        # Generate embeddings
        start_time = time.time()
        embeddings = create_embeddings(None, None)
        texts = [doc.page_content for doc in processed_documents]
        embeddings.embed_documents(texts[:100])  # Limit for mock
        embed_time = time.time() - start_time

        # Performance assertions (adjust thresholds based on requirements)
        assert load_time < 5.0  # Should load 30 files in <5 seconds
        assert process_time < 2.0  # Should process in <2 seconds
        assert embed_time < 3.0  # Should generate embeddings in <3 seconds

        print(f"Performance: Load={load_time:.2f}s, Process={process_time:.2f}s, Embed={embed_time:.2f}s")

    @pytest.mark.integration
    def test_pipeline_memory_usage(self, sample_documents_on_disk, mock_openai_embeddings):
        """Test memory usage throughout the pipeline."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run pipeline
        factory = DocumentLoaderFactory()
        documents = factory.load_documents(list(sample_documents_on_disk.values()))

        after_load_memory = process.memory_info().rss

        preprocessor = ContentPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)

        after_process_memory = process.memory_info().rss

        embeddings = create_embeddings(None, None)
        texts = [doc.page_content for doc in processed_documents]
        embeddings.embed_documents(texts)

        final_memory = process.memory_info().rss

        # Memory increases should be reasonable
        load_increase = after_load_memory - initial_memory
        process_increase = after_process_memory - after_load_memory
        embed_increase = final_memory - after_process_memory

        # Assert memory increases are reasonable (adjust based on requirements)
        assert load_increase < 100 * 1024 * 1024  # <100MB for document loading
        assert process_increase < 50 * 1024 * 1024  # <50MB for processing
        assert embed_increase < 200 * 1024 * 1024  # <200MB for embeddings

    @pytest.mark.integration
    def test_concurrent_pipeline_execution(self, sample_documents_on_disk, mock_openai_embeddings):
        """Test pipeline execution with concurrent requests."""
        import threading

        results = {}
        errors = {}

        def run_pipeline(thread_id):
            try:
                factory = DocumentLoaderFactory()
                documents = factory.load_documents(list(sample_documents_on_disk.values()))

                preprocessor = ContentPreprocessor()
                processed_documents = preprocessor.preprocess_documents(documents)

                embeddings = create_embeddings(None, None)
                texts = [doc.page_content for doc in processed_documents]
                embedding_vectors = embeddings.embed_documents(texts)

                results[thread_id] = {
                    "documents": len(documents),
                    "embeddings": len(embedding_vectors),
                    "success": True
                }
            except Exception as e:
                errors[thread_id] = str(e)

        # Run multiple threads concurrently
        threads = []
        for i in range(3):  # 3 concurrent executions
            thread = threading.Thread(target=run_pipeline, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3, "Not all threads completed successfully"

        # Verify consistent results
        first_result = results[0]
        for result in results.values():
            assert result["documents"] == first_result["documents"]
            assert result["embeddings"] == first_result["embeddings"]

    @pytest.mark.integration
    def test_pipeline_data_integrity(self, sample_documents_on_disk, mock_openai_embeddings):
        """Test data integrity throughout the pipeline."""
        # Run pipeline twice with same inputs
        factory = DocumentLoaderFactory()

        # First run
        documents1 = factory.load_documents(list(sample_documents_on_disk.values()))
        preprocessor1 = ContentPreprocessor()
        processed1 = preprocessor1.preprocess_documents(documents1)

        # Second run
        documents2 = factory.load_documents(list(sample_documents_on_disk.values()))
        preprocessor2 = ContentPreprocessor()
        processed2 = preprocessor2.preprocess_documents(documents2)

        # Compare results
        assert len(documents1) == len(documents2)
        assert len(processed1) == len(processed2)

        # Content should be identical
        for doc1, doc2 in zip(processed1, processed2, strict=False):
            assert doc1.page_content == doc2.page_content
            # Metadata might differ in timestamps, but core data should match
            assert doc1.metadata.get("source") == doc2.metadata.get("source")
            assert doc1.metadata.get("has_sensitive_data") == doc2.metadata.get("has_sensitive_data")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_scale_pipeline(self, temp_directory, mock_openai_embeddings):
        """Test pipeline with large-scale data."""
        # Create many documents
        large_documents = []
        for i in range(100):
            doc_file = temp_directory / f"large_doc_{i}.txt"
            content = f"Document {i} content. " * 100  # ~2KB per document
            doc_file.write_text(content)
            large_documents.append(str(doc_file))

        # Run pipeline
        factory = DocumentLoaderFactory()
        documents = factory.load_documents(large_documents)

        preprocessor = ContentPreprocessor()
        processed_documents = preprocessor.preprocess_documents(documents)

        # Should handle large scale gracefully
        # Preprocessor may chunk documents, so result count is >= input count
        assert len(processed_documents) >= 100
        assert all(len(doc.page_content) > 0 for doc in processed_documents)
        # Verify all 100 source files are represented
        sources = {doc.metadata.get("source") for doc in processed_documents}
        assert len(sources) == 100

    @pytest.mark.integration
    def test_pipeline_with_malformed_data(self, temp_directory):
        """Test pipeline robustness with malformed data."""
        # Create malformed files
        malformed_files = []

        # File with invalid encoding
        invalid_encoding_file = temp_directory / "invalid_encoding.txt"
        with open(invalid_encoding_file, 'wb') as f:
            f.write(b'\xff\xfe\x00\x00invalid\x00content\x00')
        malformed_files.append(str(invalid_encoding_file))

        # File with mixed line endings
        mixed_endings_file = temp_directory / "mixed_endings.txt"
        mixed_endings_file.write_text("Line 1\rLine 2\nLine 3\r\nLine 4")
        malformed_files.append(str(mixed_endings_file))

        # Test pipeline resilience
        factory = DocumentLoaderFactory()
        try:
            documents = factory.load_documents(malformed_files)
            # If it succeeds, verify it handled the data appropriately
            assert isinstance(documents, list)
            for doc in documents:
                assert isinstance(doc, Document)
                assert len(doc.page_content) >= 0  # May be empty if couldn't parse
        except Exception as e:
            # Acceptable to fail gracefully with informative error
            assert isinstance(e, (UnicodeDecodeError, ValueError, IOError))
