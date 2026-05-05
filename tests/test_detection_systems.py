#!/usr/bin/env python3
"""
Detection system effectiveness testing for VectorSmuggle research validation.
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

from analysis.detection_signatures import PatternSignatureGenerator, StatisticalSignatureGenerator
from analysis.forensic_tools import ArtifactAnalyzer


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_statistical_detection(detector, test_vectors, logger):
    """Test effectiveness of statistical detection methods."""
    logger.info("Testing statistical detection system")

    start_time = time.time()
    results = {
        "detector": "statistical",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_results": []
    }

    try:
        # Establish baseline with clean vectors
        clean_vectors = test_vectors["clean_vectors"]["vectors"]
        detector.establish_baseline(np.array(clean_vectors), "test_baseline")

        # Test detection on different vector sets
        for test_case in test_vectors.values():
            case_name = test_case["name"]
            vectors = np.array(test_case["vectors"])
            expected_result = test_case["contains_steganography"]

            try:
                detect_start = time.time()
                detection_result = detector.detect_anomalies(vectors, "test_baseline")
                detect_time = time.time() - detect_start

                # Determine if steganography was detected
                anomaly_score = detection_result.get("anomaly_score", 0.0)
                predicted_positive = anomaly_score > 0.5  # Threshold for detection
                confidence = anomaly_score

                correct_prediction = (predicted_positive == expected_result)

                test_result = {
                    "test_case": case_name,
                    "expected": expected_result,
                    "predicted": predicted_positive,
                    "confidence": confidence,
                    "anomaly_score": anomaly_score,
                    "correct": correct_prediction,
                    "detection_time": detect_time,
                    "vector_count": len(vectors)
                }

                results["test_results"].append(test_result)
                logger.info(f"  {case_name}: {'CORRECT' if correct_prediction else 'INCORRECT'} "
                          f"(score: {anomaly_score:.3f}, time: {detect_time:.3f}s)")

            except Exception as e:
                logger.error(f"  {case_name}: ERROR - {e}")
                results["test_results"].append({
                    "test_case": case_name,
                    "error": str(e),
                    "correct": False
                })

        # Calculate metrics
        correct_predictions = [t for t in results["test_results"] if t.get("correct", False)]
        total_tests = len(results["test_results"])

        if total_tests > 0:
            accuracy = len(correct_predictions) / total_tests
            avg_detection_time = sum(t.get("detection_time", 0) for t in results["test_results"]) / total_tests
            avg_confidence = sum(t.get("confidence", 0) for t in results["test_results"]) / total_tests

            results["metrics"] = {
                "accuracy": accuracy,
                "avg_detection_time": avg_detection_time,
                "avg_confidence": avg_confidence,
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {"error": "No valid test results", "accuracy": 0.0}

        logger.info(f"  Accuracy: {results['metrics'].get('accuracy', 0):.2%}")

    except Exception as e:
        logger.error(f"Failed to test statistical detection: {e}")
        results["metrics"] = {"error": str(e), "accuracy": 0.0}

    return results


def test_pattern_detection(detector, test_vectors, logger):
    """Test effectiveness of pattern-based detection."""
    logger.info("Testing pattern detection system")

    start_time = time.time()
    results = {
        "detector": "pattern",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_results": []
    }

    try:
        # Test pattern analysis on different techniques
        techniques = ["noise", "rotation", "scaling", "fragmentation"]

        for technique in techniques:
            # Use modified vectors as test case for this technique
            test_vectors_array = np.array(test_vectors["modified_vectors"]["vectors"])

            try:
                detect_start = time.time()
                pattern_analysis = detector.analyze_steganographic_patterns(test_vectors_array, technique)
                detect_time = time.time() - detect_start

                # Extract detection confidence
                confidence = pattern_analysis.get("confidence", 0.0)
                detected = confidence > 0.5

                test_result = {
                    "technique": technique,
                    "detected": detected,
                    "confidence": confidence,
                    "detection_time": detect_time,
                    "pattern_indicators": len(pattern_analysis.get("indicators", []))
                }

                results["test_results"].append(test_result)
                logger.info(f"  {technique}: {'DETECTED' if detected else 'NOT DETECTED'} "
                          f"(confidence: {confidence:.3f}, time: {detect_time:.3f}s)")

            except Exception as e:
                logger.error(f"  {technique}: ERROR - {e}")
                results["test_results"].append({
                    "technique": technique,
                    "error": str(e),
                    "detected": False
                })

        # Calculate metrics
        successful_detections = [t for t in results["test_results"] if t.get("detected", False)]
        total_tests = len(results["test_results"])

        if total_tests > 0:
            detection_rate = len(successful_detections) / total_tests
            avg_detection_time = sum(t.get("detection_time", 0) for t in results["test_results"]) / total_tests
            avg_confidence = sum(t.get("confidence", 0) for t in results["test_results"]) / total_tests

            results["metrics"] = {
                "detection_rate": detection_rate,
                "avg_detection_time": avg_detection_time,
                "avg_confidence": avg_confidence,
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {"error": "No valid test results", "detection_rate": 0.0}

        logger.info(f"  Detection rate: {results['metrics'].get('detection_rate', 0):.2%}")

    except Exception as e:
        logger.error(f"Failed to test pattern detection: {e}")
        results["metrics"] = {"error": str(e), "detection_rate": 0.0}

    return results


def test_forensic_analysis(analyzer, logger):
    """Test effectiveness of forensic analysis tools."""
    logger.info("Testing forensic analysis system")

    start_time = time.time()
    results = {
        "detector": "forensic",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "test_results": []
    }

    try:
        # Generate test embeddings with potential artifacts
        test_embeddings = np.random.normal(0, 1, (100, 128)).astype(np.float32)

        # Add some artificial artifacts
        test_embeddings[50:60] += 0.5  # Scaling artifacts
        test_embeddings[70:80] = np.roll(test_embeddings[70:80], 5, axis=1)  # Rotation artifacts

        try:
            detect_start = time.time()
            artifact_analysis = analyzer.analyze_embedding_artifacts(test_embeddings)
            detect_time = time.time() - detect_start

            # Extract analysis results
            anomalies = artifact_analysis.get("anomalies", [])
            indicators = artifact_analysis.get("steganographic_indicators", [])

            test_result = {
                "analysis_type": "embedding_artifacts",
                "anomalies_found": len(anomalies),
                "indicators_found": len(indicators),
                "detection_time": detect_time,
                "success": len(anomalies) > 0 or len(indicators) > 0
            }

            results["test_results"].append(test_result)
            logger.info(f"  Embedding analysis: {len(anomalies)} anomalies, {len(indicators)} indicators "
                      f"(time: {detect_time:.3f}s)")

        except Exception as e:
            logger.error(f"  Embedding analysis: ERROR - {e}")
            results["test_results"].append({
                "analysis_type": "embedding_artifacts",
                "error": str(e),
                "success": False
            })

        # Test network artifact analysis
        test_network_data = [
            {"timestamp": "2024-01-01T10:00:00", "source_ip": "192.168.1.100", "dest_ip": "10.0.0.1", "bytes": 1024},
            {"timestamp": "2024-01-01T10:01:00", "source_ip": "192.168.1.100", "dest_ip": "10.0.0.1", "bytes": 2048},
            {"timestamp": "2024-01-01T10:02:00", "source_ip": "192.168.1.100", "dest_ip": "10.0.0.1", "bytes": 4096}
        ]

        try:
            detect_start = time.time()
            network_analysis = analyzer.analyze_network_artifacts(test_network_data)
            detect_time = time.time() - detect_start

            suspicious_patterns = network_analysis.get("suspicious_patterns", [])
            timing_analysis = network_analysis.get("timing_analysis", {})

            test_result = {
                "analysis_type": "network_artifacts",
                "suspicious_patterns": len(suspicious_patterns),
                "timing_anomalies": len(timing_analysis.get("anomalies", [])),
                "detection_time": detect_time,
                "success": len(suspicious_patterns) > 0
            }

            results["test_results"].append(test_result)
            logger.info(f"  Network analysis: {len(suspicious_patterns)} patterns "
                      f"(time: {detect_time:.3f}s)")

        except Exception as e:
            logger.error(f"  Network analysis: ERROR - {e}")
            results["test_results"].append({
                "analysis_type": "network_artifacts",
                "error": str(e),
                "success": False
            })

        # Calculate metrics
        successful_analyses = [t for t in results["test_results"] if t.get("success", False)]
        total_tests = len(results["test_results"])

        if total_tests > 0:
            success_rate = len(successful_analyses) / total_tests
            avg_detection_time = sum(t.get("detection_time", 0) for t in results["test_results"]) / total_tests

            results["metrics"] = {
                "success_rate": success_rate,
                "avg_detection_time": avg_detection_time,
                "total_test_time": time.time() - start_time
            }
        else:
            results["metrics"] = {"error": "No valid test results", "success_rate": 0.0}

        logger.info(f"  Success rate: {results['metrics'].get('success_rate', 0):.2%}")

    except Exception as e:
        logger.error(f"Failed to test forensic analysis: {e}")
        results["metrics"] = {"error": str(e), "success_rate": 0.0}

    return results


def generate_test_vectors():
    """Generate test vector sets for detection evaluation."""
    test_vectors = {}

    # Clean vectors (no steganography)
    clean_vectors = []
    for _i in range(50):
        vector = np.random.normal(0, 1, 128).astype(np.float32)
        clean_vectors.append(vector)

    test_vectors["clean_vectors"] = {
        "name": "clean_vectors",
        "vectors": clean_vectors,
        "contains_steganography": False
    }

    # Modified vectors (simulating steganography)
    modified_vectors = []
    for vector in clean_vectors[:25]:
        modified = vector.copy()
        # Add small modifications
        indices = np.random.choice(len(modified), size=10, replace=False)
        modified[indices] += np.random.normal(0, 0.2, 10)
        modified_vectors.append(modified)

    test_vectors["modified_vectors"] = {
        "name": "modified_vectors",
        "vectors": modified_vectors,
        "contains_steganography": True
    }

    # Heavily modified vectors
    heavily_modified = []
    for vector in clean_vectors[:15]:
        modified = vector.copy()
        # More significant modifications
        indices = np.random.choice(len(modified), size=20, replace=False)
        modified[indices] += np.random.normal(0, 0.8, 20)
        heavily_modified.append(modified)

    test_vectors["heavily_modified_vectors"] = {
        "name": "heavily_modified_vectors",
        "vectors": heavily_modified,
        "contains_steganography": True
    }

    return test_vectors


def run_detection_tests(full_evaluation=False):
    """Run detection system effectiveness tests."""
    logger = setup_logging()
    logger.info("Starting detection system effectiveness tests")

    # Define available detectors
    detectors = {
        "statistical": ("Statistical Signature", StatisticalSignatureGenerator),
        "pattern": ("Pattern Analysis", PatternSignatureGenerator),
        "forensic": ("Forensic Analysis", ArtifactAnalyzer)
    }

    # Generate test vectors
    logger.info("Generating test vector sets...")
    test_vectors = generate_test_vectors()

    all_results = {
        "test_type": "detection_effectiveness",
        "timestamp": datetime.utcnow().isoformat(),
        "detectors_tested": list(detectors.keys()),
        "test_vector_sets": len(test_vectors),
        "results": {}
    }

    # Test each detector
    for detector_key, (detector_name, detector_class) in detectors.items():
        try:
            detector_instance = detector_class()

            if detector_key == "statistical":
                result = test_statistical_detection(detector_instance, test_vectors, logger)
            elif detector_key == "pattern":
                result = test_pattern_detection(detector_instance, test_vectors, logger)
            elif detector_key == "forensic":
                result = test_forensic_analysis(detector_instance, logger)

            all_results["results"][detector_key] = result
        except Exception as e:
            logger.error(f"Failed to test {detector_name}: {e}")
            all_results["results"][detector_key] = {
                "detector": detector_key,
                "error": str(e),
                "metrics": {"accuracy": 0.0}
            }

    # Calculate summary statistics
    successful_detectors = []
    for result in all_results["results"].values():
        metrics = result.get("metrics", {})
        if any(key in metrics for key in ["accuracy", "detection_rate", "success_rate"]) and not metrics.get("error"):
            successful_detectors.append(result)

    all_results["summary"] = {
        "total_detectors": len(detectors),
        "successful_detectors": len(successful_detectors),
        "best_detector": None,
        "worst_detector": None
    }

    if successful_detectors:
        # Find best and worst performing detectors
        def get_performance_score(detector_result):
            metrics = detector_result.get("metrics", {})
            return metrics.get("accuracy", metrics.get("detection_rate", metrics.get("success_rate", 0)))

        best = max(successful_detectors, key=get_performance_score)
        worst = min(successful_detectors, key=get_performance_score)

        all_results["summary"]["best_detector"] = {
            "name": best["detector"],
            "performance_score": get_performance_score(best)
        }

        all_results["summary"]["worst_detector"] = {
            "name": worst["detector"],
            "performance_score": get_performance_score(worst)
        }

    # Save results
    results_dir = os.getenv("RESULTS_DIR")
    if results_dir:
        results_file = Path(results_dir) / "detection_effectiveness_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Detection effectiveness tests completed")
    if successful_detectors:
        avg_performance = sum(get_performance_score(d) for d in successful_detectors) / len(successful_detectors)
        logger.info(f"Average detection performance: {avg_performance:.2%}")

    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Detection system effectiveness test")
    parser.add_argument("--full-evaluation", action="store_true",
                       help="Run full evaluation with extended tests")

    args = parser.parse_args()

    success = run_detection_tests(full_evaluation=args.full_evaluation)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
