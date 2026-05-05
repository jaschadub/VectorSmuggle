#!/usr/bin/env python3
"""
Basic baseline generation test for VectorSmuggle testing framework validation.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.baseline_generator import BaselineDatasetGenerator


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_baseline_test(comprehensive=False):
    """Run baseline generation test."""
    logger = setup_logging()
    logger.info("Starting baseline generation test")

    try:
        # Initialize generator
        generator = BaselineDatasetGenerator(logger)

        # Generate small test dataset
        num_users = 5 if not comprehensive else 20
        days = 1 if not comprehensive else 3

        logger.info(f"Generating baseline dataset: {num_users} users, {days} days")

        dataset = generator.generate_baseline_dataset(
            num_users=num_users,
            days=days
        )

        # Validate dataset
        assert dataset is not None, "Dataset generation failed"
        assert "metadata" in dataset, "Missing metadata"
        assert "activities" in dataset, "Missing activities"
        assert "statistics" in dataset, "Missing statistics"

        # Check data quality
        activities = dataset["activities"]
        assert len(activities) > 0, "No activities generated"

        stats = dataset["statistics"]["activity_statistics"]
        assert stats["total_activities"] == len(activities), "Activity count mismatch"
        assert stats["success_rate"] > 0.8, "Success rate too low"

        # Save results if in test environment
        results_dir = os.getenv("RESULTS_DIR")
        if results_dir:
            results_file = Path(results_dir) / "baseline_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "test_type": "baseline_generation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "comprehensive": comprehensive,
                    "dataset_summary": {
                        "users": num_users,
                        "days": days,
                        "total_activities": len(activities),
                        "success_rate": stats["success_rate"]
                    },
                    "validation_passed": True
                }, f, indent=2)
            logger.info(f"Results saved to {results_file}")

        logger.info("Baseline generation test PASSED")
        logger.info(f"Generated {len(activities)} activities with {stats['success_rate']:.2%} success rate")

        return True

    except Exception as e:
        logger.error(f"Baseline generation test FAILED: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Baseline generation test")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test with more data")

    args = parser.parse_args()

    success = run_baseline_test(comprehensive=args.comprehensive)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
