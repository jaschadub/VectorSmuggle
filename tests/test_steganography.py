#!/usr/bin/env python3
"""
Steganography technique effectiveness testing for VectorSmuggle research validation.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from steganography.decoys import DecoyGenerator
from steganography.fragmentation import MultiModelFragmenter
from steganography.obfuscation import EmbeddingObfuscator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_obfuscation_techniques(obfuscator, logger):
    """Test effectiveness of embedding obfuscation techniques."""
    logger.info("Testing embedding obfuscation techniques")

    start_time = time.time()
    results = {
        "technique": "obfuscation",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_cases": []
    }

    try:
        # Generate test embeddings
        test_embeddings = np.random.normal(0, 1, (100, 128)).astype(np.float32)

        # Test different obfuscation methods
        techniques = ["noise", "rotation", "scaling", "offset", "fragmentation"]

        for technique in techniques:
            try:
                test_start = time.time()

                if technique == "noise":
                    obfuscated = obfuscator.inject_noise(test_embeddings)
                elif technique == "rotation":
                    obfuscated, rotation_matrix = obfuscator.apply_rotation(test_embeddings)
                elif technique == "scaling":
                    obfuscated = obfuscator.apply_scaling(test_embeddings)
                elif technique == "offset":
                    obfuscated = obfuscator.apply_offset(test_embeddings)
                elif technique == "fragmentation":
                    fragments = obfuscator.fragment_embeddings(test_embeddings)
                    obfuscated = np.concatenate(fragments, axis=0)

                test_time = time.time() - test_start

                # Calculate effectiveness metrics
                mse = np.mean((test_embeddings - obfuscated[:len(test_embeddings)]) ** 2)
                max_change = np.max(np.abs(test_embeddings - obfuscated[:len(test_embeddings)]))

                test_case = {
                    "technique": technique,
                    "success": True,
                    "processing_time": test_time,
                    "mse": float(mse),
                    "max_change": float(max_change),
                    "output_shape": obfuscated.shape
                }

                results["test_cases"].append(test_case)
                logger.info(f"  {technique}: SUCCESS (MSE: {mse:.6f}, time: {test_time:.3f}s)")

            except Exception as e:
                logger.error(f"  {technique}: ERROR - {e}")
                results["test_cases"].append({
                    "technique": technique,
                    "success": False,
                    "error": str(e)
                })

        # Calculate overall metrics
        successful_tests = [t for t in results["test_cases"] if t["success"]]

        if successful_tests:
            results["metrics"] = {
                "success_rate": len(successful_tests) / len(techniques),
                "avg_processing_time": sum(t["processing_time"] for t in successful_tests) / len(successful_tests),
                "avg_mse": sum(t["mse"] for t in successful_tests) / len(successful_tests),
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {
                "success_rate": 0.0,
                "total_test_time": time.time() - start_time,
                "error": "No successful tests"
            }

        logger.info(f"  Overall success rate: {results['metrics']['success_rate']:.2%}")

    except Exception as e:
        logger.error(f"Failed to test obfuscation: {e}")
        results["metrics"] = {"error": str(e), "success_rate": 0.0}

    return results


def test_fragmentation_system(fragmenter, logger):
    """Test effectiveness of multi-model fragmentation."""
    logger.info("Testing multi-model fragmentation system")

    start_time = time.time()
    results = {
        "technique": "fragmentation",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_cases": []
    }

    try:
        # Test different text sizes
        test_texts = [
            "Short test message",
            "Medium length test message with more content to fragment across models",
            "Long test message with substantial content that will be fragmented across multiple embedding models to test the effectiveness of the multi-model fragmentation system for steganographic data hiding purposes."
        ]

        for i, text in enumerate(test_texts):
            try:
                test_start = time.time()

                # Fragment and embed
                fragmented_data = fragmenter.fragment_and_embed(text)

                # Reconstruct
                reconstructed_text = fragmenter.reconstruct_from_fragments(fragmented_data)

                test_time = time.time() - test_start

                # Calculate metrics
                success = reconstructed_text == text
                compression_ratio = len(str(fragmented_data)) / len(text) if text else 0

                test_case = {
                    "test_size": f"text_{i+1}",
                    "original_length": len(text),
                    "success": success,
                    "processing_time": test_time,
                    "compression_ratio": compression_ratio,
                    "num_fragments": fragmented_data.get("metadata", {}).get("num_fragments", 0)
                }

                results["test_cases"].append(test_case)
                logger.info(f"  Text {i+1}: {'SUCCESS' if success else 'FAILED'} "
                          f"(fragments: {test_case['num_fragments']}, time: {test_time:.3f}s)")

            except Exception as e:
                logger.error(f"  Text {i+1}: ERROR - {e}")
                results["test_cases"].append({
                    "test_size": f"text_{i+1}",
                    "success": False,
                    "error": str(e)
                })

        # Calculate overall metrics
        successful_tests = [t for t in results["test_cases"] if t["success"]]

        if successful_tests:
            results["metrics"] = {
                "success_rate": len(successful_tests) / len(test_texts),
                "avg_processing_time": sum(t["processing_time"] for t in successful_tests) / len(successful_tests),
                "avg_compression_ratio": sum(t["compression_ratio"] for t in successful_tests) / len(successful_tests),
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {
                "success_rate": 0.0,
                "total_test_time": time.time() - start_time,
                "error": "No successful tests"
            }

        logger.info(f"  Overall success rate: {results['metrics']['success_rate']:.2%}")

    except Exception as e:
        logger.error(f"Failed to test fragmentation: {e}")
        results["metrics"] = {"error": str(e), "success_rate": 0.0}

    return results


def test_decoy_generation(decoy_generator, logger):
    """Test effectiveness of decoy generation system."""
    logger.info("Testing decoy generation system")

    start_time = time.time()
    results = {
        "technique": "decoy_generation",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_cases": []
    }

    try:
        # Test decoy text generation
        try:
            test_start = time.time()
            decoy_text = decoy_generator.generate_decoy_text(count=10)
            text_time = time.time() - test_start

            results["test_cases"].append({
                "test_type": "text_generation",
                "success": len(decoy_text) > 0,
                "processing_time": text_time,
                "output_count": len(decoy_text)
            })
            logger.info(f"  Text generation: SUCCESS ({len(decoy_text)} texts, {text_time:.3f}s)")
        except Exception as e:
            logger.error(f"  Text generation: ERROR - {e}")
            results["test_cases"].append({
                "test_type": "text_generation",
                "success": False,
                "error": str(e)
            })

        # Test decoy embedding generation
        try:
            test_start = time.time()
            decoy_embeddings = decoy_generator.generate_decoy_embeddings(count=50, dimensions=128)
            embedding_time = time.time() - test_start

            results["test_cases"].append({
                "test_type": "embedding_generation",
                "success": len(decoy_embeddings) > 0,
                "processing_time": embedding_time,
                "output_count": len(decoy_embeddings),
                "embedding_shape": decoy_embeddings[0].shape if decoy_embeddings else None
            })
            logger.info(f"  Embedding generation: SUCCESS ({len(decoy_embeddings)} embeddings, {embedding_time:.3f}s)")
        except Exception as e:
            logger.error(f"  Embedding generation: ERROR - {e}")
            results["test_cases"].append({
                "test_type": "embedding_generation",
                "success": False,
                "error": str(e)
            })

        # Calculate overall metrics
        successful_tests = [t for t in results["test_cases"] if t["success"]]

        if successful_tests:
            results["metrics"] = {
                "success_rate": len(successful_tests) / len(results["test_cases"]),
                "avg_processing_time": sum(t["processing_time"] for t in successful_tests) / len(successful_tests),
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {
                "success_rate": 0.0,
                "total_test_time": time.time() - start_time,
                "error": "No successful tests"
            }

        logger.info(f"  Overall success rate: {results['metrics']['success_rate']:.2%}")

    except Exception as e:
        logger.error(f"Failed to test decoy generation: {e}")
        results["metrics"] = {"error": str(e), "success_rate": 0.0}

    return results


def run_steganography_tests(technique=None):
    """Run steganography technique tests."""
    logger = setup_logging()
    logger.info("Starting steganography technique effectiveness tests")

    # Define available techniques
    techniques = {
        "obfuscation": ("Embedding Obfuscation", EmbeddingObfuscator),
        "fragmentation": ("Multi-Model Fragmentation", MultiModelFragmenter),
        "decoys": ("Decoy Generation", DecoyGenerator),
    }

    # Filter techniques if specific one requested
    if technique and technique in techniques:
        techniques = {technique: techniques[technique]}
    elif technique:
        logger.error(f"Unknown technique: {technique}")
        return False

    all_results = {
        "test_type": "steganography_effectiveness",
        "timestamp": datetime.utcnow().isoformat(),
        "techniques_tested": list(techniques.keys()),
        "results": {}
    }

    # Test each technique
    for tech_key, (tech_name, tech_class) in techniques.items():
        try:
            if tech_key == "obfuscation":
                instance = tech_class()
                result = test_obfuscation_techniques(instance, logger)
            elif tech_key == "fragmentation":
                instance = tech_class()
                result = test_fragmentation_system(instance, logger)
            elif tech_key == "decoys":
                instance = tech_class()
                result = test_decoy_generation(instance, logger)

            all_results["results"][tech_key] = result
        except Exception as e:
            logger.error(f"Failed to test {tech_name}: {e}")
            all_results["results"][tech_key] = {
                "technique": tech_key,
                "error": str(e),
                "metrics": {"success_rate": 0.0}
            }

    # Calculate summary statistics
    successful_techniques = [
        r for r in all_results["results"].values()
        if r.get("metrics", {}).get("success_rate", 0) > 0
    ]

    all_results["summary"] = {
        "total_techniques": len(techniques),
        "successful_techniques": len(successful_techniques),
        "overall_success_rate": len(successful_techniques) / len(techniques) if techniques else 0,
        "best_technique": None,
        "worst_technique": None
    }

    if successful_techniques:
        # Find best and worst performing techniques
        best = max(successful_techniques, key=lambda x: x["metrics"]["success_rate"])
        worst = min(successful_techniques, key=lambda x: x["metrics"]["success_rate"])

        all_results["summary"]["best_technique"] = {
            "name": best["technique"],
            "success_rate": best["metrics"]["success_rate"]
        }

        all_results["summary"]["worst_technique"] = {
            "name": worst["technique"],
            "success_rate": worst["metrics"]["success_rate"]
        }

    # Save results
    results_dir = os.getenv("RESULTS_DIR")
    if results_dir:
        results_file = Path(results_dir) / "steganography_effectiveness_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Steganography effectiveness tests completed")
    logger.info(f"Overall success rate: {all_results['summary']['overall_success_rate']:.2%}")

    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Steganography technique effectiveness test")
    parser.add_argument("--technique", choices=["obfuscation", "fragmentation", "decoys"],
                       help="Test specific technique only")

    args = parser.parse_args()

    success = run_steganography_tests(technique=args.technique)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
