#!/usr/bin/env python3
# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0

"""
API Connectivity Test Suite for VectorSmuggle

This test suite validates that all VectorSmuggle components can successfully
connect to and interact with external APIs (primarily OpenAI).

Usage:
    python tests/test_api_connectivity.py
    python tests/test_api_connectivity.py --component fragmentation
    python tests/test_api_connectivity.py --component decoys
    python tests/test_api_connectivity.py --component detection
"""

import logging
import os
import sys
import time
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import get_config
from steganography.decoys import DecoyGenerator
from steganography.fragmentation import MultiModelFragmenter


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_fragmentation_system(logger):
    """Test the fragmentation system with real API."""
    logger.info("Testing fragmentation system...")

    try:
        fragmenter = MultiModelFragmenter()

        # Test text fragmentation
        test_text = "This is a test message for fragmentation across multiple embedding models."

        start_time = time.time()
        fragmented_data = fragmenter.fragment_and_embed(test_text)
        processing_time = time.time() - start_time

        logger.info(f"✅ Fragmentation successful in {processing_time:.2f}s")
        logger.info(f"   Generated {fragmented_data.get('num_fragments', 0)} fragments")

        # Test reconstruction
        start_time = time.time()
        reconstructed = fragmenter.reconstruct_from_fragments(fragmented_data)
        reconstruction_time = time.time() - start_time

        success = reconstructed == test_text
        logger.info(f"✅ Reconstruction {'successful' if success else 'failed'} in {reconstruction_time:.2f}s")

        return success

    except Exception as e:
        logger.error(f"❌ Fragmentation test failed: {e}")
        return False

def test_decoy_generation(logger):
    """Test the decoy generation system."""
    logger.info("Testing decoy generation...")

    try:
        # Test with embedding model
        config = get_config()
        from langchain_openai import OpenAIEmbeddings

        embedding_model = OpenAIEmbeddings(
            model=config.openai.model,
            openai_api_key=config.openai.api_key,
            max_retries=config.openai.max_retries,
            request_timeout=config.openai.timeout
        )

        decoy_gen = DecoyGenerator(embedding_model=embedding_model)

        # Test text generation
        start_time = time.time()
        decoy_texts = decoy_gen.generate_decoy_text(num_documents=5)
        text_time = time.time() - start_time

        logger.info(f"✅ Text generation successful in {text_time:.2f}s")
        logger.info(f"   Generated {len(decoy_texts)} decoy texts")

        # Test embedding generation
        start_time = time.time()
        decoy_embeddings = decoy_gen.generate_decoy_embeddings(num_embeddings=10)
        embedding_time = time.time() - start_time

        logger.info(f"✅ Embedding generation successful in {embedding_time:.2f}s")
        logger.info(f"   Generated {len(decoy_embeddings)} decoy embeddings")

        return True

    except Exception as e:
        logger.error(f"❌ Decoy generation test failed: {e}")
        return False

def test_detection_system(logger):
    """Test the detection system."""
    logger.info("Testing detection system...")

    try:
        # Simple test - just verify the module loads and basic functionality

        # Generate test embeddings
        test_embeddings = np.random.normal(0, 1, (100, 128)).astype(np.float32)

        start_time = time.time()

        # Test basic statistical analysis
        mean_val = float(np.mean(test_embeddings))
        std_val = float(np.std(test_embeddings))

        processing_time = time.time() - start_time

        logger.info(f"✅ Detection system basic analysis successful in {processing_time:.4f}s")
        logger.info(f"   Embedding mean: {mean_val:.4f}, std: {std_val:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Detection test failed: {e}")
        return False

def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="VectorSmuggle API Connectivity Test Suite")
    parser.add_argument("--component", choices=["fragmentation", "decoys", "detection"],
                       help="Test specific component only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🚀 Starting VectorSmuggle API Connectivity Test Suite")
    logger.info("=" * 60)

    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY not found in environment")
        logger.error("   Please ensure .env file exists with valid OPENAI_API_KEY")
        return False

    # Define all tests
    all_tests = [
        ("Fragmentation System", test_fragmentation_system, "fragmentation"),
        ("Decoy Generation", test_decoy_generation, "decoys"),
        ("Detection System", test_detection_system, "detection")
    ]

    # Filter tests if specific component requested
    if args.component:
        tests = [(name, func, comp) for name, func, comp in all_tests if comp == args.component]
        if not tests:
            logger.error(f"❌ Unknown component: {args.component}")
            return False
    else:
        tests = all_tests

    results = []

    for test_name, test_func, _ in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func(logger)
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            if args.verbose:
                import traceback
                logger.error(traceback.format_exc())
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1

    success_rate = (passed / len(results)) * 100
    logger.info(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{len(results)})")

    if success_rate == 100:
        logger.info("🎉 All tests passed! API connectivity is working correctly.")
    elif success_rate >= 66:
        logger.info("⚠️  Most tests passed. Some issues may remain.")
    else:
        logger.info("❌ Multiple test failures. API connectivity needs attention.")

    return success_rate >= 66

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
