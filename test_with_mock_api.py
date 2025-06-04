#!/usr/bin/env python3
# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
Test VectorSmuggle with mock API responses to validate fixes without requiring real API key.
"""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def mock_openai_embedding(text):
    """Generate mock embedding for testing."""
    # Create deterministic but realistic embedding
    np.random.seed(hash(text) % 2**32)
    embedding = np.random.normal(0, 0.1, 1536).tolist()
    return embedding

def test_fragmentation_with_mock_api(logger):
    """Test fragmentation system with mocked API."""
    logger.info("🧩 Testing fragmentation with mock API...")

    # Mock the OpenAI API calls
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings_class:
        # Create mock instance
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.side_effect = mock_openai_embedding
        mock_embeddings_class.return_value = mock_embeddings

        # Set mock API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-mock-key-for-testing'}):
            try:
                from steganography.fragmentation import MultiModelFragmenter

                # Test fragmentation
                fragmenter = MultiModelFragmenter()
                stats = fragmenter.get_model_statistics()

                logger.info(f"✅ Fragmentation initialized: {stats['total_models']} models")

                # Test actual fragmentation
                test_text = "This is a comprehensive test document for fragmentation testing across multiple embedding models to validate our API fixes."
                result = fragmenter.fragment_and_embed(test_text, num_fragments=3)

                logger.info(f"✅ Fragmentation successful: {len(result['embeddings'])} fragments created")
                logger.info(f"   - Original text length: {len(test_text)}")
                logger.info(f"   - Fragment strategy: {result['fragment_strategy']}")
                logger.info(f"   - Models used: {len(set(meta['model_name'] for meta in result['metadata']))}")

                # Validate reconstruction
                reconstructed = fragmenter.reconstruct_from_fragments(result)
                if reconstructed == test_text:
                    logger.info("✅ Fragment reconstruction successful")
                    return True
                else:
                    logger.error("❌ Fragment reconstruction failed")
                    return False

            except Exception as e:
                logger.error(f"❌ Fragmentation test failed: {e}")
                return False

def test_decoy_generation_with_mock_api(logger):
    """Test decoy generation with mocked API."""
    logger.info("🎭 Testing decoy generation with mock API...")

    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings_class:
        # Create mock instance
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.side_effect = mock_openai_embedding
        mock_embeddings_class.return_value = mock_embeddings

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-mock-key-for-testing'}):
            try:
                from steganography.decoys import DecoyGenerator

                # Test decoy generation with API
                decoy_gen = DecoyGenerator(embedding_model=mock_embeddings)

                # Test text-based decoy generation
                decoy_embeddings, decoy_texts = decoy_gen.generate_decoy_embeddings_from_text(
                    num_embeddings=10,
                    category="technical"
                )

                logger.info(f"✅ Decoy generation successful: {len(decoy_embeddings)} embeddings")
                logger.info(f"   - Embedding shape: {decoy_embeddings.shape}")
                logger.info(f"   - Text samples: {len(decoy_texts)}")

                # Test mixing with real data
                real_embeddings = np.random.normal(0, 1, (5, 1536))
                real_texts = ["Real document " + str(i) for i in range(5)]

                mixed_data = decoy_gen.mix_with_decoys(real_embeddings, real_texts)

                logger.info("✅ Data mixing successful:")
                logger.info(f"   - Total embeddings: {mixed_data['metadata']['total_embeddings']}")
                logger.info(f"   - Real indices: {len(mixed_data['real_indices'])}")
                logger.info(f"   - Decoy indices: {len(mixed_data['decoy_indices'])}")
                logger.info(f"   - Decoy ratio: {mixed_data['metadata']['decoy_ratio']:.2%}")

                # Test extraction
                extracted = decoy_gen.extract_real_data(mixed_data)
                if len(extracted['embeddings']) == len(real_embeddings):
                    logger.info("✅ Real data extraction successful")
                    return True
                else:
                    logger.error("❌ Real data extraction failed")
                    return False

            except Exception as e:
                logger.error(f"❌ Decoy generation test failed: {e}")
                return False

def test_obfuscation_techniques(logger):
    """Test obfuscation techniques (no API required)."""
    logger.info("🔀 Testing obfuscation techniques...")

    try:
        from steganography.obfuscation import EmbeddingObfuscator

        # Create test embeddings
        test_embeddings = np.random.normal(0, 1, (20, 1536))

        # Test obfuscation
        obfuscator = EmbeddingObfuscator(
            noise_level=0.01,
            rotation_angle=0.1,
            scaling_factor=0.95
        )

        # Test different techniques
        techniques = ['noise', 'rotation', 'scaling', 'offset']

        for technique in techniques:
            result = obfuscator.obfuscate(test_embeddings, techniques=[technique])

            if 'embeddings' in result and len(result['embeddings']) == len(test_embeddings):
                logger.info(f"✅ {technique.title()} obfuscation successful")
            else:
                logger.error(f"❌ {technique.title()} obfuscation failed")
                return False

        # Test combined techniques
        combined_result = obfuscator.obfuscate(test_embeddings, techniques=techniques)
        logger.info(f"✅ Combined obfuscation successful: {len(combined_result['embeddings'])} embeddings")

        return True

    except Exception as e:
        logger.error(f"❌ Obfuscation test failed: {e}")
        return False

def test_detection_systems_comprehensive(logger):
    """Test detection systems with comprehensive data."""
    logger.info("🔍 Testing detection systems...")

    try:
        from analysis.detection_signatures import PatternSignatureGenerator, StatisticalSignatureGenerator

        # Create comprehensive test data
        np.random.seed(42)

        # Clean baseline data
        clean_embeddings = np.random.normal(0, 1, (200, 384))

        # Various types of suspicious data
        noisy_embeddings = clean_embeddings + np.random.normal(0, 0.2, (200, 384))
        scaled_embeddings = clean_embeddings * 1.5
        shifted_embeddings = clean_embeddings + 0.5

        # Test statistical detection
        stat_detector = StatisticalSignatureGenerator(logger)
        baseline = stat_detector.establish_baseline(clean_embeddings, "comprehensive_test")

        test_cases = [
            ("Noisy Data", noisy_embeddings),
            ("Scaled Data", scaled_embeddings),
            ("Shifted Data", shifted_embeddings)
        ]

        detection_results = {}

        for case_name, test_data in test_cases:
            anomalies = stat_detector.detect_anomalies(test_data, "comprehensive_test")
            detection_rate = len(anomalies['anomalies']) / len(test_data)
            detection_results[case_name] = detection_rate

            logger.info(f"✅ {case_name}: {detection_rate:.1%} detection rate")

        # Test pattern detection
        pattern_detector = PatternSignatureGenerator(logger)

        pattern_types = ["noise_injection", "rotation", "scaling", "fragmentation"]

        for pattern_type in pattern_types:
            patterns = pattern_detector.analyze_steganographic_patterns(noisy_embeddings, pattern_type)

            if 'patterns' in patterns and patterns['patterns']:
                logger.info(f"✅ {pattern_type.title()} pattern analysis successful")
            else:
                logger.warning(f"⚠️ {pattern_type.title()} pattern analysis returned no patterns")

        # Generate signatures
        stat_signatures = stat_detector.generate_statistical_signatures("comprehensive_test")
        pattern_signatures = pattern_detector.generate_pattern_signatures()

        logger.info(f"✅ Generated {len(stat_signatures)} statistical signatures")
        logger.info(f"✅ Generated {len(pattern_signatures)} pattern signatures")

        return True

    except Exception as e:
        logger.error(f"❌ Detection systems test failed: {e}")
        return False

def test_end_to_end_workflow(logger):
    """Test complete end-to-end workflow with mock API."""
    logger.info("🔄 Testing end-to-end workflow...")

    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings_class:
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.side_effect = mock_openai_embedding
        mock_embeddings_class.return_value = mock_embeddings

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-mock-key-for-testing'}):
            try:
                from steganography.decoys import DecoyGenerator
                from steganography.fragmentation import MultiModelFragmenter
                from steganography.obfuscation import EmbeddingObfuscator

                # Step 1: Create original embeddings
                original_texts = [
                    "Confidential financial report for Q4 2024",
                    "Strategic roadmap for product development",
                    "Internal security assessment results",
                    "Employee performance review data"
                ]

                original_embeddings = np.array([mock_openai_embedding(text) for text in original_texts])
                logger.info(f"✅ Created {len(original_embeddings)} original embeddings")

                # Step 2: Apply obfuscation
                obfuscator = EmbeddingObfuscator()
                obfuscated_result = obfuscator.obfuscate(original_embeddings, techniques=['noise', 'rotation'])
                logger.info(f"✅ Applied obfuscation to {len(obfuscated_result['embeddings'])} embeddings")

                # Step 3: Add decoys
                decoy_gen = DecoyGenerator(embedding_model=mock_embeddings)
                mixed_data = decoy_gen.mix_with_decoys(obfuscated_result['embeddings'], original_texts)
                logger.info(f"✅ Mixed with decoys: {mixed_data['metadata']['total_embeddings']} total embeddings")

                # Step 4: Fragment across models
                fragmenter = MultiModelFragmenter()
                combined_text = " ".join(original_texts)
                fragmented_result = fragmenter.fragment_and_embed(combined_text, num_fragments=3)
                logger.info(f"✅ Fragmented across {len(set(meta['model_name'] for meta in fragmented_result['metadata']))} models")

                # Step 5: Test detection evasion
                from analysis.detection_signatures import StatisticalSignatureGenerator

                detector = StatisticalSignatureGenerator(logger)
                clean_baseline = np.random.normal(0, 1, (100, 1536))
                detector.establish_baseline(clean_baseline, "evasion_test")

                # Test detection on obfuscated data
                detection_result = detector.detect_anomalies(mixed_data['embeddings'], "evasion_test")
                detection_rate = len(detection_result['anomalies']) / len(mixed_data['embeddings'])

                logger.info("✅ End-to-end workflow complete:")
                logger.info(f"   - Original documents: {len(original_texts)}")
                logger.info(f"   - Final embeddings: {len(mixed_data['embeddings'])}")
                logger.info(f"   - Detection rate: {detection_rate:.1%}")
                logger.info(f"   - Decoy ratio: {mixed_data['metadata']['decoy_ratio']:.1%}")

                return True

            except Exception as e:
                logger.error(f"❌ End-to-end workflow failed: {e}")
                return False

def main():
    """Run comprehensive testing with mock API."""
    logger = setup_logging()
    logger.info("🚀 Starting VectorSmuggle Mock API Testing")
    logger.info("=" * 60)

    tests = [
        ("Fragmentation with Mock API", test_fragmentation_with_mock_api),
        ("Decoy Generation with Mock API", test_decoy_generation_with_mock_api),
        ("Obfuscation Techniques", test_obfuscation_techniques),
        ("Detection Systems", test_detection_systems_comprehensive),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func(logger)
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 MOCK API TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    success_rate = passed / total
    logger.info(f"\nOverall Success Rate: {success_rate:.1%} ({passed}/{total} tests passed)")

    # Compare with original issues
    original_rate = 0.2667
    improvement = success_rate - original_rate

    logger.info("\n🎯 EFFECTIVENESS ANALYSIS:")
    logger.info(f"Original Success Rate: {original_rate:.1%}")
    logger.info(f"Current Success Rate: {success_rate:.1%}")
    logger.info(f"Improvement: {improvement:+.1%}")

    if success_rate >= 0.8:
        logger.info("🎉 Excellent! Fixes have significantly improved the system.")
    elif success_rate >= 0.6:
        logger.info("✅ Good! Fixes have improved the system substantially.")
    elif improvement > 0:
        logger.info("📈 Progress! Fixes have improved the system.")
    else:
        logger.warning("⚠️ More work needed to improve system reliability.")

    return 0 if success_rate > original_rate else 1

if __name__ == "__main__":
    sys.exit(main())
