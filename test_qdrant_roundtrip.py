#!/usr/bin/env python3
"""
VectorSmuggle Qdrant Round-trip Testing Script

This script tests the complete embedding and retrieval pipeline by:
1. Embedding 100 Enron emails and 10 exfiltrate docs to Qdrant
2. Retrieving and reassembling the documents
3. Comparing with originals to validate data integrity

Usage:
    python test_qdrant_roundtrip.py [--enron-path PATH] [--output-dir DIR]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from loaders import DocumentLoaderFactory, EmailLoader
from utils.embedding_factory import EmbeddingFactory


class QdrantRoundtripTester:
    """Test Qdrant embedding and retrieval pipeline."""

    def __init__(self, config, logger: logging.Logger):
        """Initialize the tester with configuration and logger."""
        self.config = config
        self.logger = logger
        self.qdrant_client = None
        self.embedding_factory = None
        self.embeddings = None
        self.collection_name = f"test_roundtrip_{int(time.time())}"

        # Test data storage
        self.original_documents = []
        self.embedded_documents = []
        self.retrieved_documents = []

        # Validation results
        self.validation_results = {
            'total_documents': 0,
            'successful_embeddings': 0,
            'successful_retrievals': 0,
            'content_matches': 0,
            'metadata_matches': 0,
            'hash_matches': 0,
            'errors': []
        }

    def setup_qdrant_connection(self) -> None:
        """Setup connection to Qdrant database."""
        try:
            self.logger.info(f"Connecting to Qdrant at {self.config.vector_store.qdrant_url}")
            self.qdrant_client = QdrantClient(url=self.config.vector_store.qdrant_url)

            # Test connection
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"Connected to Qdrant. Found {len(collections.collections)} existing collections")

        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def setup_embeddings(self) -> None:
        """Setup embedding model."""
        try:
            self.logger.info("Initializing embedding factory")
            self.embedding_factory = EmbeddingFactory(logger=self.logger)
            self.embeddings = self.embedding_factory.create_embeddings(
                config=self.config,
                prefer_ollama=False
            )

            # Test embedding generation
            test_embedding = self.embeddings.embed_query("test")
            self.logger.info(f"Embeddings initialized. Dimension: {len(test_embedding)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def create_test_collection(self) -> None:
        """Create a test collection in Qdrant."""
        try:
            # Get embedding dimension
            test_embedding = self.embeddings.embed_query("test")
            dimension = len(test_embedding)

            self.logger.info(f"Creating test collection: {self.collection_name}")

            # Create collection with proper vector configuration
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

            self.logger.info(f"Created collection with dimension {dimension}")

        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            raise

    def load_enron_emails(self, enron_path: str, count: int = 100) -> list[dict[str, Any]]:
        """Load Enron emails for testing."""
        try:
            enron_path = Path(enron_path)
            if not enron_path.exists():
                self.logger.warning(f"Enron path not found: {enron_path}, creating {count} mock emails")
                return self._create_mock_emails(count)

            self.logger.info(f"Loading {count} Enron emails from {enron_path}")

            emails = []
            email_files = []

            # Find email files (Enron emails typically have no extension)
            for person_dir in enron_path.iterdir():
                if person_dir.is_dir():
                    for email_file in person_dir.rglob("*"):
                        if (email_file.is_file() and email_file.suffix == "" and
                            email_file.name != "." and email_file.name != ".."):
                            email_files.append(email_file)
                            if len(email_files) >= count:
                                break
                    if len(email_files) >= count:
                        break

            # If we didn't find enough real emails, supplement with mock data
            if len(email_files) < count:
                self.logger.warning(
                    f"Only found {len(email_files)} real emails, "
                    f"creating {count - len(email_files)} mock emails"
                )
                mock_emails = self._create_mock_emails(count - len(email_files))
                emails.extend(mock_emails)

            # Load emails using EmailLoader
            for i, email_file in enumerate(email_files[:count]):
                try:
                    # Create a temporary .eml file for EmailLoader
                    temp_eml = email_file.with_suffix('.eml')
                    if not temp_eml.exists():
                        temp_eml.write_bytes(email_file.read_bytes())

                    loader = EmailLoader(str(temp_eml), logger=self.logger)
                    documents = loader.load()

                    if documents:
                        doc = documents[0]
                        original_id = f"enron_email_{i:04d}"
                        emails.append({
                            'id': str(uuid.uuid4()),
                            'original_id': original_id,
                            'content': doc.page_content,
                            'metadata': {
                                **doc.metadata,
                                'source_type': 'enron_email',
                                'original_path': str(email_file),
                                'test_id': f"enron_{i:04d}",
                                'original_id': original_id
                            },
                            'content_hash': hashlib.sha256(doc.page_content.encode()).hexdigest()
                        })

                    # Clean up temp file
                    if temp_eml.exists() and temp_eml != email_file:
                        temp_eml.unlink()

                except Exception as e:
                    self.logger.warning(f"Failed to load email {email_file}: {e}")
                    continue

            self.logger.info(f"Successfully loaded {len(emails)} Enron emails")
            return emails

        except Exception as e:
            self.logger.error(f"Failed to load Enron emails: {e}")
            return self._create_mock_emails(count)

    def _create_mock_emails(self, count: int) -> list[dict[str, Any]]:
        """Create mock email data for testing."""
        self.logger.info(f"Creating {count} mock emails for testing")

        mock_emails = []
        for i in range(count):
            content = f"""Subject: Test Email {i:04d}
From: test.user{i % 10}@enron.com
To: recipient{i % 5}@enron.com
Date: 2001-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}

Body:
This is a test email message number {i:04d} for the VectorSmuggle roundtrip test.
It contains sample content to test the embedding and retrieval pipeline.

The email discusses business matters including:
- Project status updates
- Financial information
- Meeting schedules
- Strategic planning

This email is part of the test dataset and contains {len(str(i)) * 50} characters of content.
"""

            original_id = f"mock_email_{i:04d}"
            mock_emails.append({
                'id': str(uuid.uuid4()),
                'original_id': original_id,
                'content': content,
                'metadata': {
                    'source_type': 'mock_email',
                    'subject': f"Test Email {i:04d}",
                    'from': f"test.user{i % 10}@enron.com",
                    'to': f"recipient{i % 5}@enron.com",
                    'test_id': f"mock_{i:04d}",
                    'original_id': original_id
                },
                'content_hash': hashlib.sha256(content.encode()).hexdigest()
            })

        return mock_emails

    def load_exfiltrate_docs(self, count: int = 10) -> list[dict[str, Any]]:
        """Load exfiltrate documents from sample_docs."""
        try:
            self.logger.info(f"Loading {count} exfiltrate documents")

            sample_docs_path = Path("sample_docs")
            if not sample_docs_path.exists():
                return self._create_mock_docs(count)

            factory = DocumentLoaderFactory(logger=self.logger)
            docs = []

            # Get all supported files from sample_docs
            doc_files = []
            for ext in factory.get_supported_formats():
                doc_files.extend(sample_docs_path.glob(f"*{ext}"))

            # Load documents
            for i, doc_file in enumerate(doc_files[:count]):
                try:
                    documents = factory.load_documents([str(doc_file)])

                    for j, doc in enumerate(documents):
                        original_id = f"exfil_doc_{i:02d}_{j:02d}"
                        docs.append({
                            'id': str(uuid.uuid4()),
                            'original_id': original_id,
                            'content': doc.page_content,
                            'metadata': {
                                **doc.metadata,
                                'source_type': 'exfiltrate_doc',
                                'original_file': str(doc_file),
                                'test_id': f"exfil_{i:02d}_{j:02d}",
                                'original_id': original_id
                            },
                            'content_hash': hashlib.sha256(doc.page_content.encode()).hexdigest()
                        })

                        if len(docs) >= count:
                            break

                    if len(docs) >= count:
                        break

                except Exception as e:
                    self.logger.warning(f"Failed to load document {doc_file}: {e}")
                    continue

            # Fill remaining with mock docs if needed
            if len(docs) < count:
                mock_docs = self._create_mock_docs(count - len(docs))
                docs.extend(mock_docs)

            self.logger.info(f"Successfully loaded {len(docs)} exfiltrate documents")
            return docs[:count]

        except Exception as e:
            self.logger.error(f"Failed to load exfiltrate docs: {e}")
            return self._create_mock_docs(count)

    def _create_mock_docs(self, count: int) -> list[dict[str, Any]]:
        """Create mock document data for testing."""
        self.logger.info(f"Creating {count} mock documents for testing")

        doc_types = ['financial_report', 'employee_handbook', 'api_documentation', 'budget_analysis']
        mock_docs = []

        for i in range(count):
            doc_type = doc_types[i % len(doc_types)]
            content = f"""Document Type: {doc_type.replace('_', ' ').title()}
Document ID: MOCK-{i:04d}
Classification: Internal Use Only
Date: 2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}

Content:
This is a mock {doc_type} document for testing the VectorSmuggle system.
It contains sensitive information that would typically be targeted for exfiltration.

Key Information:
- Revenue figures: ${(i + 1) * 100000:,}
- Employee count: {(i + 1) * 50}
- Project status: {'Active' if i % 2 == 0 else 'Pending'}
- Security level: {'High' if i % 3 == 0 else 'Medium'}

This document contains {len(str(i)) * 100} words of detailed content that would be
valuable for competitive intelligence or unauthorized disclosure.

The document includes technical specifications, financial data, and strategic
information that demonstrates the effectiveness of the embedding and retrieval system.
"""

            original_id = f"mock_doc_{i:04d}"
            mock_docs.append({
                'id': str(uuid.uuid4()),
                'original_id': original_id,
                'content': content,
                'metadata': {
                    'source_type': 'mock_document',
                    'document_type': doc_type,
                    'classification': 'Internal Use Only',
                    'test_id': f"mock_doc_{i:04d}",
                    'original_id': original_id
                },
                'content_hash': hashlib.sha256(content.encode()).hexdigest()
            })

        return mock_docs

    def embed_documents(self, documents: list[dict[str, Any]]) -> None:
        """Embed documents into Qdrant."""
        try:
            self.logger.info(f"Embedding {len(documents)} documents to Qdrant")

            points = []
            for doc in documents:
                try:
                    # Generate embedding
                    embedding = self.embeddings.embed_query(doc['content'])

                    # Create point for Qdrant
                    point = models.PointStruct(
                        id=doc['id'],
                        vector=embedding,
                        payload={
                            'content': doc['content'],
                            'metadata': doc['metadata'],
                            'content_hash': doc['content_hash'],
                            'embedded_at': datetime.now().isoformat()
                        }
                    )
                    points.append(point)

                    self.validation_results['successful_embeddings'] += 1

                except Exception as e:
                    self.logger.error(f"Failed to embed document {doc['id']}: {e}")
                    self.validation_results['errors'].append({
                        'type': 'embedding_error',
                        'document_id': doc['id'],
                        'error': str(e)
                    })

            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                self.logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1}")

            self.logger.info(f"Successfully embedded {len(points)} documents")

        except Exception as e:
            self.logger.error(f"Failed to embed documents: {e}")
            raise

    def retrieve_and_reassemble(self) -> list[dict[str, Any]]:
        """Retrieve all documents from Qdrant and reassemble them."""
        try:
            self.logger.info("Retrieving and reassembling documents from Qdrant")

            # Get all points from collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large limit to get all documents
                with_payload=True,
                with_vectors=False
            )

            retrieved_docs = []
            for point in scroll_result[0]:
                try:
                    doc = {
                        'id': point.id,
                        'original_id': point.payload['metadata']['original_id'],
                        'content': point.payload['content'],
                        'metadata': point.payload['metadata'],
                        'content_hash': point.payload['content_hash'],
                        'embedded_at': point.payload.get('embedded_at'),
                        'retrieved_at': datetime.now().isoformat()
                    }
                    retrieved_docs.append(doc)
                    self.validation_results['successful_retrievals'] += 1

                except Exception as e:
                    self.logger.error(f"Failed to process retrieved point {point.id}: {e}")
                    self.validation_results['errors'].append({
                        'type': 'retrieval_error',
                        'point_id': str(point.id),
                        'error': str(e)
                    })

            self.logger.info(f"Successfully retrieved {len(retrieved_docs)} documents")
            return retrieved_docs

        except Exception as e:
            self.logger.error(f"Failed to retrieve documents: {e}")
            raise

    def validate_roundtrip(self, original_docs: list[dict[str, Any]],
                          retrieved_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate that retrieved documents match originals."""
        try:
            self.logger.info("Validating roundtrip integrity")

            # Create lookup dictionaries using original_id for comparison
            original_by_id = {doc['original_id']: doc for doc in original_docs}
            retrieved_by_id = {doc['original_id']: doc for doc in retrieved_docs}

            validation_details = {
                'content_mismatches': [],
                'metadata_mismatches': [],
                'hash_mismatches': [],
                'missing_documents': [],
                'extra_documents': []
            }

            # Check for missing or extra documents
            original_ids = set(original_by_id.keys())
            retrieved_ids = set(retrieved_by_id.keys())

            missing_ids = original_ids - retrieved_ids
            extra_ids = retrieved_ids - original_ids

            for doc_id in missing_ids:
                validation_details['missing_documents'].append(doc_id)
                self.validation_results['errors'].append({
                    'type': 'missing_document',
                    'document_id': doc_id
                })

            for doc_id in extra_ids:
                validation_details['extra_documents'].append(doc_id)
                self.validation_results['errors'].append({
                    'type': 'extra_document',
                    'document_id': doc_id
                })

            # Validate matching documents
            common_ids = original_ids & retrieved_ids
            for doc_id in common_ids:
                original = original_by_id[doc_id]
                retrieved = retrieved_by_id[doc_id]

                # Check content match
                if original['content'] == retrieved['content']:
                    self.validation_results['content_matches'] += 1
                else:
                    validation_details['content_mismatches'].append({
                        'document_id': doc_id,
                        'original_length': len(original['content']),
                        'retrieved_length': len(retrieved['content'])
                    })

                # Check hash match
                if original['content_hash'] == retrieved['content_hash']:
                    self.validation_results['hash_matches'] += 1
                else:
                    validation_details['hash_mismatches'].append({
                        'document_id': doc_id,
                        'original_hash': original['content_hash'],
                        'retrieved_hash': retrieved['content_hash']
                    })

                # Check metadata match (excluding test-specific fields)
                original_meta = {k: v for k, v in original['metadata'].items()
                               if k not in ['embedded_at', 'retrieved_at']}
                retrieved_meta = {k: v for k, v in retrieved['metadata'].items()
                                if k not in ['embedded_at', 'retrieved_at']}

                if original_meta == retrieved_meta:
                    self.validation_results['metadata_matches'] += 1
                else:
                    validation_details['metadata_mismatches'].append({
                        'document_id': doc_id,
                        'differences': self._find_dict_differences(original_meta, retrieved_meta)
                    })

            # Calculate success rates
            total_docs = len(original_docs)
            self.validation_results.update({
                'total_documents': total_docs,
                'content_match_rate': (
                    self.validation_results['content_matches'] / total_docs if total_docs > 0 else 0
                ),
                'metadata_match_rate': (
                    self.validation_results['metadata_matches'] / total_docs if total_docs > 0 else 0
                ),
                'hash_match_rate': self.validation_results['hash_matches'] / total_docs if total_docs > 0 else 0,
                'overall_success_rate': (
                    self.validation_results['content_matches'] / total_docs if total_docs > 0 else 0
                )
            })

            success_rate = self.validation_results['overall_success_rate']
            self.logger.info(f"Validation complete. Success rate: {success_rate:.2%}")
            return validation_details

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise

    def _find_dict_differences(self, dict1: dict, dict2: dict) -> dict[str, Any]:
        """Find differences between two dictionaries."""
        differences = {}

        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            val1 = dict1.get(key, '<MISSING>')
            val2 = dict2.get(key, '<MISSING>')

            if val1 != val2:
                differences[key] = {'original': val1, 'retrieved': val2}

        return differences

    def export_results(self, output_dir: str, validation_details: dict[str, Any]) -> None:
        """Export test results to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export validation results
            results_file = output_path / f"qdrant_roundtrip_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'validation_results': self.validation_results,
                    'validation_details': validation_details,
                    'test_config': {
                        'collection_name': self.collection_name,
                        'qdrant_url': self.config.vector_store.qdrant_url,
                        'embedding_model': getattr(self.config.openai, 'model', 'unknown'),
                        'test_timestamp': timestamp
                    }
                }, f, indent=2, default=str)

            # Export reassembled documents
            docs_file = output_path / f"reassembled_documents_{timestamp}.json"
            with open(docs_file, 'w') as f:
                json.dump(self.retrieved_documents, f, indent=2, default=str)

            # Create summary report
            summary_file = output_path / f"test_summary_{timestamp}.md"
            with open(summary_file, 'w') as f:
                f.write(self._generate_summary_report(validation_details))

            self.logger.info(f"Results exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")

    def _generate_summary_report(self, validation_details: dict[str, Any]) -> str:
        """Generate a markdown summary report."""
        report = f"""# Qdrant Roundtrip Test Summary

## Test Configuration
- **Collection Name**: {self.collection_name}
- **Qdrant URL**: {self.config.vector_store.qdrant_url}
- **Embedding Model**: {getattr(self.config.openai, 'model', 'unknown')}
- **Test Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Results Summary
- **Total Documents**: {self.validation_results['total_documents']}
- **Successful Embeddings**: {self.validation_results['successful_embeddings']}
- **Successful Retrievals**: {self.validation_results['successful_retrievals']}
- **Content Matches**: {self.validation_results['content_matches']}
- **Metadata Matches**: {self.validation_results['metadata_matches']}
- **Hash Matches**: {self.validation_results['hash_matches']}

## Success Rates
- **Content Match Rate**: {self.validation_results['content_match_rate']:.2%}
- **Metadata Match Rate**: {self.validation_results['metadata_match_rate']:.2%}
- **Hash Match Rate**: {self.validation_results['hash_match_rate']:.2%}
- **Overall Success Rate**: {self.validation_results['overall_success_rate']:.2%}

## Issues Found
- **Content Mismatches**: {len(validation_details['content_mismatches'])}
- **Metadata Mismatches**: {len(validation_details['metadata_mismatches'])}
- **Hash Mismatches**: {len(validation_details['hash_mismatches'])}
- **Missing Documents**: {len(validation_details['missing_documents'])}
- **Extra Documents**: {len(validation_details['extra_documents'])}
- **Total Errors**: {len(self.validation_results['errors'])}

## Test Status
{'✅ PASSED' if self.validation_results['overall_success_rate'] >= 0.95 else '❌ FAILED'}

The test {'passed' if self.validation_results['overall_success_rate'] >= 0.95 else 'failed'} """
        f"""with an overall success rate of {self.validation_results['overall_success_rate']:.2%}.
"""
        return report

    def cleanup(self) -> None:
        """Clean up test resources."""
        try:
            if self.qdrant_client and self.collection_name:
                self.logger.info(f"Cleaning up test collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def run_test(self, enron_path: str, output_dir: str) -> bool:
        """Run the complete roundtrip test."""
        try:
            self.logger.info("Starting Qdrant roundtrip test")

            # Setup
            self.setup_qdrant_connection()
            self.setup_embeddings()
            self.create_test_collection()

            # Load test data
            self.logger.info("Loading test documents...")
            enron_emails = self.load_enron_emails(enron_path, count=100)
            exfil_docs = self.load_exfiltrate_docs(count=10)

            # Combine all documents
            all_documents = enron_emails + exfil_docs
            self.original_documents = all_documents

            self.logger.info(f"Loaded {len(all_documents)} total documents")
            self.logger.info(f"  - Enron emails: {len(enron_emails)}")
            self.logger.info(f"  - Exfiltrate docs: {len(exfil_docs)}")

            # Embed documents
            self.embed_documents(all_documents)

            # Retrieve and reassemble
            self.retrieved_documents = self.retrieve_and_reassemble()

            # Validate roundtrip
            validation_details = self.validate_roundtrip(
                self.original_documents,
                self.retrieved_documents
            )

            # Export results
            self.export_results(output_dir, validation_details)

            # Determine test success
            success = self.validation_results['overall_success_rate'] >= 0.95

            if success:
                self.logger.info("✅ Roundtrip test PASSED")
            else:
                self.logger.error("❌ Roundtrip test FAILED")

            return success

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return False
        finally:
            self.cleanup()


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('qdrant_roundtrip_test.log')
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Test Qdrant embedding and retrieval roundtrip")
    parser.add_argument(
        "--enron-path",
        default=os.getenv("ENRON_EMAIL_PATH", "/media/jascha/BKUP01/enron-emails/maildir/"),
        help="Path to Enron email archive"
    )
    parser.add_argument(
        "--output-dir",
        default="results/qdrant_roundtrip",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep test collection after completion (for debugging)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")

        # Create tester
        tester = QdrantRoundtripTester(config, logger)

        # Override cleanup behavior if requested
        if args.keep_collection:
            tester.cleanup = lambda: logger.info("Keeping collection as requested")

        # Run test
        success = tester.run_test(args.enron_path, args.output_dir)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
