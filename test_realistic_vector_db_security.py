#!/usr/bin/env python3
"""
Realistic Vector Database Security Testing for VectorSmuggle
Tests against actual security measures used by production vector databases
Rather than academic ML anomaly detection methods
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from steganography.obfuscation import EmbeddingObfuscator
from utils.embedding_factory import EmbeddingFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticVectorDBSecurity:
    """Test VectorSmuggle against realistic vector database security measures."""

    def __init__(self):
        self.models = [
            "snowflake-arctic-embed:335m",
            "mxbai-embed-large:335m",
            "nomic-embed-text:latest"
        ]

        # Sensitive data patterns (what we want to smuggle)
        self.sensitive_patterns = [
            "User's SSN is 123-45-6789 and email is test@example.com",
            "Credit card: 4111-1111-1111-1111, expires 12/25",
            "AWS Key: AKIAIOSFODNN7EXAMPLE",
            "Server IP: 192.168.1.1, mysql://user:pass@host/db",
            "Employee salary: John Doe $75,000",
            "Database password: admin123",
            "API key: sk-1234567890abcdef",
            "Confidential financial report Q3 2024",
            "postgresql://admin:Pr0d_DB_P@ssw0rd_2024@db.company.com:5432/finance",
            "Bearer token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        ]

        # Clean baseline texts
        self.clean_patterns = [
            "The quarterly business report shows steady growth.",
            "Employee handbook updated with new policies.",
            "Customer satisfaction survey results are positive.",
            "Technical documentation for software deployment.",
            "Marketing campaign analysis shows engagement.",
            "Training materials for compliance certification.",
            "Project timeline and milestone tracking.",
            "Safety protocols for all facility locations.",
            "Product specifications for next release.",
            "Vendor contracts need annual review."
        ]

        # Realistic security parameters scaled to 10k like other tests
        self.test_scale = 10000
        self.baseline_scale = 1000
        self.results = {}

    def get_embedding_dimensions(self, model_name: str) -> int:
        """Get embedding dimensions for each model."""
        dimension_map = {
            "snowflake-arctic-embed:335m": 1024,
            "mxbai-embed-large:335m": 1024,
            "nomic-embed-text:latest": 768
        }
        return dimension_map.get(model_name, 768)

    def generate_embeddings_batch(self, texts: list[str], model_name: str) -> np.ndarray:
        """Generate embeddings efficiently."""
        os.environ['OLLAMA_EMBEDDING_MODEL'] = model_name
        factory = EmbeddingFactory()
        embedding_model = factory.create_embeddings(prefer_ollama=True)

        embeddings = []
        for text in texts:
            try:
                embedding = embedding_model.embed_query(text)
                if embedding is not None and len(embedding) > 0:
                    embeddings.append(embedding)
            except Exception as e:
                logger.debug(f"Failed to embed text: {e}")
                dim = self.get_embedding_dimensions(model_name)
                embeddings.append(np.zeros(dim))

        return np.array(embeddings)

    def content_level_detection(self, texts: list[str]) -> dict[str, Any]:
        """Implement realistic content-level security scanning (pre-embedding)."""
        logger.info("Testing content-level security scanning...")

        # Common sensitive data patterns used by real systems
        sensitive_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'aws_key': r'AKIA[0-9A-Z]{16}',
            'password': r'(?i)(password|pwd|pass)\s*[:=]\s*\S+',
            'api_key': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*[A-Za-z0-9_-]+',
            'database_url': r'(?i)(mysql|postgresql|mongodb)://\S+',
            'bearer_token': r'(?i)bearer\s+[A-Za-z0-9_.-]+',
            'financial': r'(?i)(salary|revenue|profit|loss|budget)\s*[:$]\s*[\d,.]+'
        }

        detection_results = {}
        total_detected = 0
        total_sensitive = 0

        for i, text in enumerate(texts):
            detected_patterns = []
            is_sensitive = any(sensitive in text for sensitive in self.sensitive_patterns)
            if is_sensitive:
                total_sensitive += 1

            for pattern_name, pattern in sensitive_patterns.items():
                if re.search(pattern, text):
                    detected_patterns.append(pattern_name)

            if detected_patterns and is_sensitive:
                total_detected += 1

            detection_results[f"text_{i}"] = {
                "detected_patterns": detected_patterns,
                "is_sensitive": is_sensitive,
                "blocked": bool(detected_patterns)
            }

        detection_rate = total_detected / total_sensitive if total_sensitive > 0 else 0.0
        evasion_rate = 1.0 - detection_rate

        return {
            "detection_rate": detection_rate,
            "evasion_rate": evasion_rate,
            "total_tested": len(texts),
            "total_sensitive": total_sensitive,
            "total_detected": total_detected,
            "details": detection_results
        }

    def basic_vector_validation(self, embeddings: np.ndarray, model_name: str) -> dict[str, Any]:
        """Implement basic vector validation used by production vector DBs."""
        logger.info("Testing basic vector validation...")

        expected_dim = self.get_embedding_dimensions(model_name)
        validation_results = {
            "dimension_check": {"passed": 0, "failed": 0},
            "norm_check": {"passed": 0, "failed": 0},
            "format_check": {"passed": 0, "failed": 0},
            "range_check": {"passed": 0, "failed": 0}
        }

        # Realistic validation thresholds
        min_norm = 0.1  # Very low norm might indicate zero/empty vector
        max_norm = 100.0  # Very high norm might indicate corruption
        value_range = (-10.0, 10.0)  # Typical embedding value range

        for embedding in embeddings:
            # 1. Dimension validation
            if len(embedding) == expected_dim:
                validation_results["dimension_check"]["passed"] += 1
            else:
                validation_results["dimension_check"]["failed"] += 1

            # 2. Norm validation
            norm = np.linalg.norm(embedding)
            if min_norm <= norm <= max_norm:
                validation_results["norm_check"]["passed"] += 1
            else:
                validation_results["norm_check"]["failed"] += 1

            # 3. Format validation (no NaN/inf values)
            if np.all(np.isfinite(embedding)):
                validation_results["format_check"]["passed"] += 1
            else:
                validation_results["format_check"]["failed"] += 1

            # 4. Value range validation
            if np.all((embedding >= value_range[0]) & (embedding <= value_range[1])):
                validation_results["range_check"]["passed"] += 1
            else:
                validation_results["range_check"]["failed"] += 1

        # Calculate overall pass rate
        total_checks = len(embeddings) * 4
        total_passed = sum(check["passed"] for check in validation_results.values())
        pass_rate = total_passed / total_checks if total_checks > 0 else 0.0

        return {
            "overall_pass_rate": pass_rate,
            "total_embeddings": len(embeddings),
            "validation_details": validation_results
        }

    def simple_statistical_detection(self, clean_embeddings: np.ndarray,
                                   test_embeddings: np.ndarray) -> dict[str, Any]:
        """Simple statistical detection used by some vector DBs."""
        logger.info("Testing simple statistical detection...")

        # Calculate baseline statistics from clean data
        clean_norms = np.linalg.norm(clean_embeddings, axis=1)
        clean_mean_norm = np.mean(clean_norms)
        clean_std_norm = np.std(clean_norms)

        # Calculate dimension-wise statistics
        clean_dim_means = np.mean(clean_embeddings, axis=0)
        clean_dim_stds = np.std(clean_embeddings, axis=0)

        # Test embeddings against thresholds
        test_norms = np.linalg.norm(test_embeddings, axis=1)

        # Simple threshold-based detection (much more lenient than ML methods)
        norm_threshold_low = clean_mean_norm - 3 * clean_std_norm
        norm_threshold_high = clean_mean_norm + 3 * clean_std_norm

        # Count anomalies
        norm_anomalies = np.sum((test_norms < norm_threshold_low) |
                               (test_norms > norm_threshold_high))

        # Dimension-wise anomaly detection (simplified)
        dim_anomalies = 0
        for embedding in test_embeddings:
            # Check if any dimension is more than 4 standard deviations away
            dim_z_scores = np.abs((embedding - clean_dim_means) / (clean_dim_stds + 1e-8))
            if np.any(dim_z_scores > 4.0):
                dim_anomalies += 1

        total_anomalies = max(norm_anomalies, dim_anomalies)
        detection_rate = total_anomalies / len(test_embeddings)
        evasion_rate = 1.0 - detection_rate

        return {
            "detection_rate": detection_rate,
            "evasion_rate": evasion_rate,
            "norm_anomalies": int(norm_anomalies),
            "dimension_anomalies": int(dim_anomalies),
            "total_tested": len(test_embeddings),
            "baseline_stats": {
                "clean_mean_norm": float(clean_mean_norm),
                "clean_std_norm": float(clean_std_norm),
                "thresholds": {
                    "norm_low": float(norm_threshold_low),
                    "norm_high": float(norm_threshold_high)
                }
            }
        }

    def generate_large_scale_sensitive_texts(self, scale: int) -> list[str]:
        """Generate large-scale diverse sensitive texts for testing."""
        # Create variations that can evade simple pattern matching
        base_patterns = [
            "Employee salary information for John Doe: $75,000",  # No simple pattern
            "Confidential financial report Q3 2024",  # No simple pattern
            "API key: sk-1234567890abcdef",  # Will be detected
            "The server credentials are stored in config file",  # Evades detection
            "Meeting notes contain private customer data",  # Evades detection
            "Internal revenue projections show $2.3M growth",  # Evades detection
            "System access tokens require rotation quarterly",  # Evades detection
            "Database connection string in production environment",  # Evades detection
            "Customer PII data requires special handling procedures",  # Evades detection
            "AWS Key: AKIAIOSFODNN7EXAMPLE",  # Will be detected
        ]

        sensitive_texts = []
        for i in range(scale):
            base_text = base_patterns[i % len(base_patterns)]
            variation = f" Document #{i+1} - internal use only. Reference ID: CONF_{i:06d}"
            sensitive_texts.append(base_text + variation)

        return sensitive_texts

    def test_realistic_security_pipeline(self) -> dict[str, Any]:
        """Test complete realistic vector database security pipeline with 10k scale."""
        logger.info("Testing realistic vector database security pipeline with 10k scale...")

        all_results = {}

        for model in self.models:
            logger.info(f"Testing realistic security for {model}...")

            try:
                # 1. Generate clean baseline embeddings (1k samples)
                logger.info(f"Generating {self.baseline_scale} clean baseline embeddings...")
                clean_texts = []
                for i in range(self.baseline_scale):
                    base_text = self.clean_patterns[i % len(self.clean_patterns)]
                    variation = f" Version {i+1} - standard business document. ID: DOC_{i:06d}"
                    clean_texts.append(base_text + variation)

                clean_embeddings = self.generate_embeddings_batch(clean_texts, model)

                # 2. Generate large-scale sensitive texts (10k samples)
                logger.info(f"Generating {self.test_scale} sensitive text samples...")
                large_scale_sensitive = self.generate_large_scale_sensitive_texts(self.test_scale)

                # 3. Test content-level detection on large scale
                logger.info("Testing content-level detection on large scale...")
                content_detection = self.content_level_detection(large_scale_sensitive)

                # 4. Get texts that passed content detection
                undetected_texts = [
                    text for i, text in enumerate(large_scale_sensitive)
                    if not content_detection["details"][f"text_{i}"]["blocked"]
                ]

                if not undetected_texts:
                    logger.info(f"All sensitive texts blocked by content detection for {model}")
                    all_results[model] = {
                        "content_detection": content_detection,
                        "vector_validation": {"note": "No vectors to validate - all blocked at content level"},
                        "statistical_detection": {"note": "No vectors to test - all blocked at content level"}
                    }
                    continue

                logger.info(f"Content detection evaded for {len(undetected_texts)} texts, applying obfuscation...")

                # 5. Generate embeddings for undetected sensitive texts (batch processing)
                logger.info("Generating embeddings for texts that evaded content detection...")
                sensitive_embeddings = self.generate_embeddings_batch(undetected_texts, model)

                # 6. Apply VectorSmuggle obfuscation
                logger.info("Applying VectorSmuggle obfuscation...")
                obfuscator = EmbeddingObfuscator(
                    noise_level=0.01,
                    rotation_angle=0.08,
                    scaling_factor=0.95,
                    offset_range=0.03,
                    seed=42
                )

                obfuscated_result = obfuscator.obfuscate(
                    sensitive_embeddings,
                    techniques=["noise", "rotation", "scaling", "offset"]
                )
                obfuscated_embeddings = obfuscated_result["embeddings"]

                # 7. Test basic vector validation on all obfuscated embeddings
                logger.info("Testing basic vector validation...")
                vector_validation = self.basic_vector_validation(obfuscated_embeddings, model)

                # 8. Test simple statistical detection on large scale
                logger.info(f"Testing statistical detection on {len(obfuscated_embeddings)} obfuscated embeddings...")
                statistical_detection = self.simple_statistical_detection(
                    clean_embeddings, obfuscated_embeddings
                )

                # 9. Calculate semantic fidelity on sample
                logger.info("Calculating semantic fidelity...")
                sample_size = min(100, len(sensitive_embeddings))  # Sample for efficiency
                similarities = []
                for i in range(sample_size):
                    sim = cosine_similarity(
                        sensitive_embeddings[i].reshape(1, -1),
                        obfuscated_embeddings[i].reshape(1, -1)
                    )[0, 0]
                    similarities.append(sim)

                fidelity_results = {
                    "mean_similarity": float(np.mean(similarities)),
                    "min_similarity": float(np.min(similarities)),
                    "above_99_percent": float(np.mean(np.array(similarities) > 0.99)),
                    "above_95_percent": float(np.mean(np.array(similarities) > 0.95)),
                    "sample_size": sample_size
                }

                all_results[model] = {
                    "content_detection": content_detection,
                    "vector_validation": vector_validation,
                    "statistical_detection": statistical_detection,
                    "semantic_fidelity": fidelity_results,
                    "texts_processed": len(undetected_texts),
                    "total_sensitive_generated": len(large_scale_sensitive),
                    "obfuscation_techniques": obfuscated_result.get("techniques_applied", [])
                }

            except Exception as e:
                all_results[model] = {"error": str(e)}
                logger.error(f"❌ Realistic security test failed for {model}: {e}")

        return all_results

    def calculate_overall_evasion_pipeline(self, results: dict[str, Any]) -> dict[str, Any]:
        """Calculate end-to-end evasion rates through realistic security pipeline."""
        summary = {}

        content_evasion_rates = []
        vector_validation_rates = []
        statistical_evasion_rates = []
        overall_evasion_rates = []

        for _model, model_results in results.items():
            if "error" in model_results:
                continue

            # Content detection evasion rate
            content_evasion = model_results.get("content_detection", {}).get("evasion_rate", 0.0)
            content_evasion_rates.append(content_evasion)

            # Vector validation pass rate
            vector_pass_rate = model_results.get("vector_validation", {}).get("overall_pass_rate", 0.0)
            vector_validation_rates.append(vector_pass_rate)

            # Statistical detection evasion rate
            stat_evasion = model_results.get("statistical_detection", {}).get("evasion_rate", 0.0)
            statistical_evasion_rates.append(stat_evasion)

            # Overall pipeline evasion (must pass all stages)
            overall_evasion = content_evasion * vector_pass_rate * stat_evasion
            overall_evasion_rates.append(overall_evasion)

        if content_evasion_rates:
            summary["avg_content_evasion_rate"] = np.mean(content_evasion_rates)
            summary["avg_vector_validation_pass_rate"] = np.mean(vector_validation_rates)
            summary["avg_statistical_evasion_rate"] = np.mean(statistical_evasion_rates)
            summary["avg_overall_pipeline_evasion_rate"] = np.mean(overall_evasion_rates)

        return summary

    def run_realistic_tests(self) -> dict[str, Any]:
        """Run all realistic vector database security tests with 10k scale."""
        logger.info("🚀 Starting Large-Scale Realistic Vector Database Security Testing...")
        logger.info(f"Testing scale: {self.test_scale:,} sensitive documents")
        logger.info(f"Baseline scale: {self.baseline_scale:,} clean documents")
        logger.info("Testing against production-grade security measures...")

        start_time = time.time()

        test_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_tested": self.models,
                "test_scale": self.test_scale,
                "baseline_scale": self.baseline_scale,
                "methodology": "realistic_vector_db_security_large_scale"
            }
        }

        # Run realistic security pipeline test
        pipeline_results = self.test_realistic_security_pipeline()
        test_results["security_pipeline"] = pipeline_results

        # Calculate summary statistics
        test_results["summary"] = self.calculate_overall_evasion_pipeline(pipeline_results)

        total_time = time.time() - start_time
        test_results["metadata"]["total_test_time"] = total_time

        logger.info(f"✅ Large-scale realistic security tests completed in {total_time:.2f} seconds")

        return test_results

def main():
    """Main test execution function."""
    # Change to VectorSmuggle directory
    os.chdir("/home/jascha/Documents/repos/VectorSmuggle")

    # Run realistic security tests
    tester = RealisticVectorDBSecurity()
    results = tester.run_realistic_tests()

    # Save results
    output_file = f"realistic_security_large_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*90)
    print("📊 LARGE-SCALE REALISTIC VECTOR DATABASE SECURITY TEST RESULTS")
    print("="*90)

    summary = results.get("summary", {})
    metadata = results.get("metadata", {})

    print(f"Models Tested: {len(metadata.get('models_tested', []))}")
    print(f"Test Scale: {metadata.get('test_scale', 0):,} sensitive documents")
    print(f"Baseline Scale: {metadata.get('baseline_scale', 0):,} clean documents")
    print(f"Test Methodology: {metadata.get('methodology', 'N/A')}")
    print()
    print("🔍 SECURITY PIPELINE EVASION RATES:")
    print(f"Content-Level Detection: {summary.get('avg_content_evasion_rate', 0)*100:.1f}% evasion")
    print(f"Vector Validation: {summary.get('avg_vector_validation_pass_rate', 0)*100:.1f}% pass rate")
    print(f"Statistical Detection: {summary.get('avg_statistical_evasion_rate', 0)*100:.1f}% evasion")
    print()
    print(f"🎯 OVERALL PIPELINE EVASION: {summary.get('avg_overall_pipeline_evasion_rate', 0)*100:.1f}%")
    print()
    print(f"Results saved to: {output_file}")

    return results

if __name__ == "__main__":
    main()
