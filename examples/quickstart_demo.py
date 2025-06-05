#!/usr/bin/env python3
"""
VectorSmuggle Quickstart Demo

This script demonstrates the complete VectorSmuggle workflow:
1. Environment setup and configuration validation
2. Document loading from sample_docs/
3. Steganographic embedding with multiple techniques
4. Vector store upload (FAISS for simplicity)
5. Query execution and data reconstruction
6. Integrity verification and success metrics

Usage:
    python examples/quickstart_demo.py [--seed SEED] [--techniques TECHNIQUES]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import Config, get_config
from evasion import BehavioralCamouflage, DetectionAvoidance
from loaders import DocumentLoaderFactory
from steganography import EmbeddingObfuscator, MultiModelFragmenter
from utils.embedding_factory import create_embeddings
from utils.seed_manager import set_global_seed


class QuickstartDemo:
    """Comprehensive quickstart demonstration of VectorSmuggle capabilities."""

    def __init__(self, config: Config, logger: logging.Logger):
        """Initialize the demo with configuration and logger."""
        self.config = config
        self.logger = logger
        self.results = {
            "demo_start_time": time.time(),
            "steps_completed": [],
            "metrics": {},
            "errors": [],
            "success": False
        }

    def setup_environment(self) -> bool:
        """Step 1: Environment setup and configuration validation."""
        try:
            self.logger.info("=== Step 1: Environment Setup ===")

            # Validate configuration
            self.config.validate()
            self.logger.info("✓ Configuration validation passed")

            # Check API connectivity
            embeddings = create_embeddings(self.config, self.logger)
            test_embedding = embeddings.embed_query("test connectivity")
            if not test_embedding or len(test_embedding) == 0:
                raise ValueError("Embedding test failed")
            self.logger.info("✓ Embedding API connectivity verified")

            # Initialize evasion components
            if self.config.evasion.behavioral_camouflage_enabled:
                camouflage = BehavioralCamouflage(
                    legitimate_ratio=self.config.evasion.legitimate_ratio
                )
                camouflage.generate_cover_story("research and development project")
                self.logger.info("✓ Behavioral camouflage initialized")

            if self.config.evasion.detection_avoidance_enabled:
                DetectionAvoidance(
                    transformation_strength=self.config.evasion.content_transformation_strength
                )
                self.logger.info("✓ Detection avoidance initialized")

            self.results["steps_completed"].append("environment_setup")
            self.results["metrics"]["api_test_embedding_size"] = len(test_embedding)
            return True

        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            self.results["errors"].append(f"Environment setup: {e}")
            return False

    def load_sample_documents(self) -> list | None:
        """Step 2: Load documents from sample_docs/ directory."""
        try:
            self.logger.info("=== Step 2: Document Loading ===")

            # Initialize document factory
            factory = DocumentLoaderFactory(logger=self.logger)

            # Load all sample documents
            sample_docs_path = Path(__file__).parent.parent / "sample_docs"
            if not sample_docs_path.exists():
                raise FileNotFoundError(f"Sample docs directory not found: {sample_docs_path}")

            # Get all supported files in sample_docs
            supported_files = []
            for ext in factory.get_supported_formats():
                supported_files.extend(sample_docs_path.glob(f"*{ext}"))

            if not supported_files:
                raise ValueError("No supported documents found in sample_docs/")

            self.logger.info(f"Found {len(supported_files)} supported documents")

            # Load documents
            documents = factory.load_documents([str(f) for f in supported_files])
            self.logger.info(f"✓ Loaded {len(documents)} document objects")

            # Apply chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            self.logger.info(f"✓ Created {len(chunks)} text chunks")

            # Analyze document types
            format_stats = {}
            for doc in documents:
                file_type = doc.metadata.get('file_type', 'unknown')
                format_stats[file_type] = format_stats.get(file_type, 0) + 1

            self.logger.info(f"Document format distribution: {format_stats}")

            self.results["steps_completed"].append("document_loading")
            self.results["metrics"]["documents_loaded"] = len(documents)
            self.results["metrics"]["chunks_created"] = len(chunks)
            self.results["metrics"]["format_distribution"] = format_stats

            return chunks

        except Exception as e:
            self.logger.error(f"Document loading failed: {e}")
            self.results["errors"].append(f"Document loading: {e}")
            return None

    def apply_steganography(self, chunks: list, embeddings: Any) -> tuple[list, dict] | tuple[None, None]:
        """Step 3: Apply steganographic techniques."""
        try:
            self.logger.info("=== Step 3: Steganographic Processing ===")

            if not self.config.steganography.enabled:
                self.logger.info("Steganography disabled, using standard processing")
                self.results["steps_completed"].append("steganography_skipped")
                return chunks, {}

            self.logger.info(f"Applying techniques: {self.config.steganography.techniques}")

            # Extract text content
            texts = [chunk.page_content for chunk in chunks]

            # Initialize steganography components
            obfuscator = EmbeddingObfuscator(
                noise_level=self.config.steganography.noise_level,
                rotation_angle=self.config.steganography.rotation_angle,
                scaling_factor=self.config.steganography.scaling_factor
            )

            # Generate embeddings
            self.logger.info("Generating embeddings for chunks...")
            chunk_embeddings = []
            for i, text in enumerate(texts):
                if i % 10 == 0:  # Progress indicator
                    self.logger.info(f"Processing chunk {i+1}/{len(texts)}")
                embedding = embeddings.embed_query(text)
                chunk_embeddings.append(embedding)

            chunk_embeddings = np.array(chunk_embeddings)
            self.logger.info(f"✓ Generated {len(chunk_embeddings)} embeddings")

            # Apply obfuscation techniques
            obfuscation_result = obfuscator.obfuscate(
                chunk_embeddings,
                techniques=self.config.steganography.techniques
            )

            processed_embeddings = obfuscation_result["embeddings"]
            self.logger.info(f"✓ Applied obfuscation techniques: {self.config.steganography.techniques}")

            # Test fragmentation if enabled
            fragmentation_data = None
            if "fragmentation" in self.config.steganography.techniques:
                try:
                    fragmenter = MultiModelFragmenter(
                        fragment_strategy=self.config.steganography.fragment_strategy
                    )

                    # Fragment a sample text
                    sample_text = " ".join(texts[:3])  # Use first 3 chunks
                    fragmentation_data = fragmenter.fragment_and_embed(sample_text)
                    fragments_count = len(fragmentation_data['metadata'])
                    self.logger.info(f"✓ Fragmentation test successful: {fragments_count} fragments")

                except Exception as frag_e:
                    self.logger.warning(f"Fragmentation test failed: {frag_e}")

            # Store metadata
            steganography_metadata = {
                "obfuscation_result": obfuscation_result,
                "processed_embeddings": processed_embeddings,
                "techniques_applied": self.config.steganography.techniques,
                "fragmentation_data": fragmentation_data,
                "original_chunk_count": len(chunks),
                "processed_chunk_count": len(chunks)
            }

            self.results["steps_completed"].append("steganography_applied")
            self.results["metrics"]["obfuscation_techniques"] = self.config.steganography.techniques
            self.results["metrics"]["embeddings_processed"] = len(processed_embeddings)

            if fragmentation_data:
                self.results["metrics"]["fragmentation_models"] = len(fragmentation_data['metadata'])

            return chunks, steganography_metadata

        except Exception as e:
            self.logger.error(f"Steganographic processing failed: {e}")
            self.results["errors"].append(f"Steganography: {e}")
            return None, None

    def create_vector_store(self, chunks: list, embeddings: Any, steganography_metadata: dict) -> FAISS | None:
        """Step 4: Create FAISS vector store."""
        try:
            self.logger.info("=== Step 4: Vector Store Creation ===")

            if steganography_metadata and "processed_embeddings" in steganography_metadata:
                # Use steganographic embeddings
                processed_embeddings = steganography_metadata["processed_embeddings"]

                # Create FAISS index manually
                import faiss
                embeddings_array = np.array(processed_embeddings, dtype=np.float32)
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_array)

                # Create vector store
                from langchain_community.docstore.in_memory import InMemoryDocstore
                docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(chunks)})
                index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

                vector_store = FAISS(
                    embedding_function=embeddings.embed_query,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )

                self.logger.info("✓ Created FAISS vector store with steganographic embeddings")
            else:
                # Standard vector store creation
                vector_store = FAISS.from_documents(chunks, embeddings)
                self.logger.info("✓ Created FAISS vector store with standard embeddings")

            # Save to temporary location
            temp_index_path = Path("temp_quickstart_index")
            vector_store.save_local(str(temp_index_path))
            self.logger.info(f"✓ Saved vector store to: {temp_index_path}")

            # Save steganography metadata
            if steganography_metadata:
                metadata_path = temp_index_path / "steganography_metadata.json"
                serializable_metadata = self._serialize_metadata(steganography_metadata)
                with open(metadata_path, 'w') as f:
                    json.dump(serializable_metadata, f, indent=2)
                self.logger.info("✓ Saved steganography metadata")

            self.results["steps_completed"].append("vector_store_created")
            self.results["metrics"]["vector_store_size"] = len(chunks)
            self.results["metrics"]["index_path"] = str(temp_index_path)

            return vector_store

        except Exception as e:
            self.logger.error(f"Vector store creation failed: {e}")
            self.results["errors"].append(f"Vector store: {e}")
            return None

    def test_queries_and_reconstruction(self, vector_store: FAISS, embeddings: Any) -> bool:
        """Step 5: Test query execution and data reconstruction."""
        try:
            self.logger.info("=== Step 5: Query Testing and Reconstruction ===")

            # Test queries
            test_queries = [
                "financial data",
                "employee information",
                "API documentation",
                "database schema",
                "budget analysis"
            ]

            query_results = {}
            for query in test_queries:
                try:
                    results = vector_store.similarity_search(query, k=3)
                    query_results[query] = {
                        "num_results": len(results),
                        "sources": [doc.metadata.get('source', 'unknown') for doc in results]
                    }
                    self.logger.info(f"✓ Query '{query}': {len(results)} results")
                except Exception as qe:
                    self.logger.warning(f"Query '{query}' failed: {qe}")
                    query_results[query] = {"error": str(qe)}

            # Test semantic search capabilities
            semantic_query = "What sensitive information is available?"
            semantic_results = vector_store.similarity_search(semantic_query, k=5)
            self.logger.info(f"✓ Semantic search returned {len(semantic_results)} results")

            # Test data reconstruction from metadata
            reconstruction_success = False
            try:
                metadata_path = Path("temp_quickstart_index/steganography_metadata.json")
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    if "fragmentation_data" in metadata and metadata["fragmentation_data"]:
                        # Test reconstruction from fragmentation
                        frag_data = metadata["fragmentation_data"]
                        if "metadata" in frag_data and frag_data["metadata"]:
                            reconstructed_fragments = []
                            for fragment_meta in sorted(frag_data["metadata"], key=lambda x: x["fragment_id"]):
                                reconstructed_fragments.append(fragment_meta["fragment_text"])

                            reconstructed_text = "".join(reconstructed_fragments)
                            if len(reconstructed_text) > 0:
                                reconstruction_success = True
                                chars_count = len(reconstructed_text)
                                msg = f"✓ Successfully reconstructed {chars_count} characters from fragments"
                                self.logger.info(msg)

            except Exception as reconstruction_error:
                msg = f"Reconstruction test failed: {reconstruction_error}"
                self.logger.warning(msg)

            self.results["steps_completed"].append("query_testing")
            self.results["metrics"]["test_queries"] = query_results
            self.results["metrics"]["semantic_results"] = len(semantic_results)
            self.results["metrics"]["reconstruction_success"] = reconstruction_success

            return True

        except Exception as e:
            self.logger.error(f"Query testing failed: {e}")
            self.results["errors"].append(f"Query testing: {e}")
            return False

    def verify_integrity_and_metrics(self) -> bool:
        """Step 6: Integrity verification and success metrics."""
        try:
            self.logger.info("=== Step 6: Integrity Verification ===")

            # Calculate success metrics
            total_steps = 6
            completed_steps = len(self.results["steps_completed"])
            success_rate = completed_steps / total_steps

            # Verify no critical errors
            critical_errors = [e for e in self.results["errors"] if "failed" in e.lower()]

            # Check data integrity
            integrity_checks = {
                "configuration_valid": "environment_setup" in self.results["steps_completed"],
                "documents_loaded": "document_loading" in self.results["steps_completed"],
                "steganography_applied": ("steganography_applied" in self.results["steps_completed"] or
                                        "steganography_skipped" in self.results["steps_completed"]),
                "vector_store_created": "vector_store_created" in self.results["steps_completed"],
                "queries_successful": "query_testing" in self.results["steps_completed"],
                "no_critical_errors": len(critical_errors) == 0
            }

            all_checks_passed = all(integrity_checks.values())

            # Final metrics
            self.results["demo_end_time"] = time.time()
            self.results["total_duration"] = self.results["demo_end_time"] - self.results["demo_start_time"]
            self.results["success_rate"] = success_rate
            self.results["integrity_checks"] = integrity_checks
            self.results["success"] = all_checks_passed and success_rate >= 0.8

            # Log final status
            if self.results["success"]:
                self.logger.info("✓ VectorSmuggle quickstart demo completed successfully!")
                self.logger.info(f"✓ Success rate: {success_rate:.1%}")
                self.logger.info(f"✓ Total duration: {self.results['total_duration']:.2f} seconds")
            else:
                self.logger.warning(f"Demo completed with issues. Success rate: {success_rate:.1%}")
                if critical_errors:
                    self.logger.warning(f"Critical errors: {critical_errors}")

            self.results["steps_completed"].append("integrity_verification")
            return self.results["success"]

        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            self.results["errors"].append(f"Integrity verification: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            import shutil
            temp_path = Path("temp_quickstart_index")
            if temp_path.exists():
                shutil.rmtree(temp_path)
                self.logger.info("✓ Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def _serialize_metadata(self, metadata: dict) -> dict:
        """Convert numpy arrays to lists for JSON serialization."""
        serializable = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._serialize_metadata(value)
            else:
                serializable[key] = value
        return serializable

    def run_demo(self) -> dict:
        """Run the complete quickstart demonstration."""
        try:
            self.logger.info("Starting VectorSmuggle Quickstart Demo")
            self.logger.info("=" * 50)

            # Step 1: Environment setup
            if not self.setup_environment():
                return self.results

            # Step 2: Load documents
            chunks = self.load_sample_documents()
            if chunks is None:
                return self.results

            # Create embeddings
            embeddings = create_embeddings(self.config, self.logger)

            # Step 3: Apply steganography
            processed_chunks, steganography_metadata = self.apply_steganography(chunks, embeddings)
            if processed_chunks is None:
                return self.results

            # Step 4: Create vector store
            vector_store = self.create_vector_store(processed_chunks, embeddings, steganography_metadata)
            if vector_store is None:
                return self.results

            # Step 5: Test queries
            if not self.test_queries_and_reconstruction(vector_store, embeddings):
                return self.results

            # Step 6: Verify integrity
            self.verify_integrity_and_metrics()

            return self.results

        except Exception as e:
            self.logger.error(f"Demo execution failed: {e}")
            self.results["errors"].append(f"Demo execution: {e}")
            return self.results
        finally:
            self.cleanup()


def setup_logging() -> logging.Logger:
    """Set up logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point for the quickstart demo."""
    parser = argparse.ArgumentParser(description="VectorSmuggle Quickstart Demo")
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic results"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=["noise", "rotation", "scaling", "offset", "fragmentation", "interleaving"],
        help="Steganographic techniques to demonstrate"
    )
    parser.add_argument(
        "--disable-steganography",
        action="store_true",
        help="Run demo without steganographic techniques"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging()

    try:
        # Set random seed if provided
        if args.seed:
            set_global_seed(args.seed)
            logger.info(f"Set random seed to: {args.seed}")

        # Load configuration
        config = get_config()

        # Override configuration from arguments
        if args.disable_steganography:
            config.steganography.enabled = False
            logger.info("Steganography disabled via command line")

        if args.techniques:
            config.steganography.techniques = args.techniques
            logger.info(f"Using techniques: {args.techniques}")

        # Run the demo
        demo = QuickstartDemo(config, logger)
        results = demo.run_demo()

        # Display results
        logger.info("\n" + "=" * 50)
        logger.info("QUICKSTART DEMO RESULTS")
        logger.info("=" * 50)

        if results["success"]:
            logger.info("🎉 Demo completed successfully!")
        else:
            logger.warning("⚠️  Demo completed with issues")

        logger.info(f"Steps completed: {len(results['steps_completed'])}/6")
        logger.info(f"Success rate: {results.get('success_rate', 0):.1%}")
        logger.info(f"Duration: {results.get('total_duration', 0):.2f} seconds")

        if results["errors"]:
            logger.warning(f"Errors encountered: {len(results['errors'])}")
            for error in results["errors"]:
                logger.warning(f"  - {error}")

        # Key metrics
        metrics = results.get("metrics", {})
        if metrics:
            logger.info("\nKey Metrics:")
            if "documents_loaded" in metrics:
                logger.info(f"  Documents loaded: {metrics['documents_loaded']}")
            if "chunks_created" in metrics:
                logger.info(f"  Text chunks: {metrics['chunks_created']}")
            if "embeddings_processed" in metrics:
                logger.info(f"  Embeddings processed: {metrics['embeddings_processed']}")
            if "vector_store_size" in metrics:
                logger.info(f"  Vector store size: {metrics['vector_store_size']}")

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {args.output}")

        # Exit with appropriate code
        sys.exit(0 if results["success"] else 1)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
