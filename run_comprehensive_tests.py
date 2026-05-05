#!/usr/bin/env python3
"""
Comprehensive Test Runner for VectorSmuggle

This script provides a unified interface for running all types of tests:
- Unit tests (pytest-based)
- Integration tests
- Security tests
- Performance tests
- Research validation tests
- Legacy research tests

Usage:
    python run_comprehensive_tests.py --suite unit
    python run_comprehensive_tests.py --suite all --coverage
    python run_comprehensive_tests.py --suite performance --benchmark
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class VectorSmuggleTestRunner:
    """Comprehensive test runner for VectorSmuggle."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "suites": {},
            "overall_status": "unknown",
            "total_time": 0
        }

    def run_unit_tests(self, coverage: bool = True, verbose: bool = True) -> dict[str, Any]:
        """Run unit tests with pytest."""
        print("🧪 Running Unit Tests")
        print("=" * 50)

        cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v"]
        # Note: also run security and research suites which are pytest-based

        if coverage:
            cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])

        cmd.extend([
            "--tb=short",
            "--durations=10",
            "-m", "not slow"
        ])

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        execution_time = time.time() - start_time

        success = result.returncode == 0

        return {
            "name": "unit_tests",
            "success": success,
            "execution_time": execution_time,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }

    def run_integration_tests(self, verbose: bool = True) -> dict[str, Any]:
        """Run integration tests."""
        print("\n🔗 Running Integration Tests")
        print("=" * 50)

        cmd = [
            sys.executable, "-m", "pytest", "tests/integration/",
            "-v", "--tb=short", "-m", "integration"
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        execution_time = time.time() - start_time

        success = result.returncode == 0

        return {
            "name": "integration_tests",
            "success": success,
            "execution_time": execution_time,
            "output": result.stdout,
            "errors": result.stderr,
            "command": " ".join(cmd)
        }

    def run_security_tests(self) -> dict[str, Any]:
        """Run security tests.

        Bandit and safety are informational; only pytest failures fail the suite.
        Bandit excludes venv/tests directories to avoid scanning third-party code.
        """
        print("\n🛡️ Running Security Tests")
        print("=" * 50)

        # Run bandit security scanner (excluding venv and tests)
        print("Running Bandit security scanner...")
        bandit_cmd = [
            sys.executable, "-m", "bandit", "-r", ".",
            "-x", "./venv,./tests,./.venv",
            "-f", "json", "-o", "security_report.json"
        ]
        bandit_result = subprocess.run(
            bandit_cmd, capture_output=True, text=True, cwd=self.project_root, timeout=120
        )

        # Run safety dependency checker (informational only - command is deprecated)
        print("Running Safety dependency checker (informational)...")
        safety_cmd = [sys.executable, "-m", "safety", "check", "--json"]
        try:
            safety_result = subprocess.run(
                safety_cmd, capture_output=True, text=True,
                cwd=self.project_root, timeout=60
            )
            safety_output = safety_result.stdout
            safety_errors = safety_result.stderr
        except subprocess.TimeoutExpired:
            safety_output = ""
            safety_errors = "Safety check timed out (auth or network issue)"

        # Run pytest security tests
        print("Running pytest security tests...")
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "security"]
        pytest_start = time.time()
        pytest_result = subprocess.run(pytest_cmd, capture_output=True, text=True, cwd=self.project_root)
        pytest_time = time.time() - pytest_start

        # Aggregate results
        # Bandit: 0 = no issues, 1 = issues found (informational, not a test failure)
        bandit_success = bandit_result.returncode in [0, 1]
        # Safety: deprecated command, treat as informational only
        safety_success = True
        # Pytest: 0 = passed, 5 = no tests collected (acceptable for security marker)
        pytest_success = pytest_result.returncode in [0, 5]

        # Only pytest determines actual pass/fail; scanners are informational
        overall_success = pytest_success and bandit_success

        return {
            "name": "security_tests",
            "success": overall_success,
            "execution_time": pytest_time,
            "bandit": {
                "success": bandit_success,
                "output": bandit_result.stdout[:1000],  # Truncate large JSON
                "errors": bandit_result.stderr
            },
            "safety": {
                "success": safety_success,
                "output": safety_output[:1000],
                "errors": safety_errors,
                "note": "Informational only - safety check command is deprecated"
            },
            "pytest": {
                "success": pytest_success,
                "output": pytest_result.stdout,
                "errors": pytest_result.stderr
            }
        }

    def run_performance_tests(self, benchmark: bool = True) -> dict[str, Any]:
        """Run performance tests."""
        print("\n⚡ Running Performance Tests")
        print("=" * 50)

        cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "performance"]

        if benchmark:
            cmd.extend(["--benchmark-only", "--benchmark-json=benchmark_results.json"])

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        execution_time = time.time() - start_time

        success = result.returncode == 0

        # Load benchmark results if available
        benchmark_data = None
        if benchmark and (self.project_root / "benchmark_results.json").exists():
            try:
                with open(self.project_root / "benchmark_results.json") as f:
                    benchmark_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load benchmark results: {e}")

        return {
            "name": "performance_tests",
            "success": success,
            "execution_time": execution_time,
            "output": result.stdout,
            "errors": result.stderr,
            "benchmark_data": benchmark_data,
            "command": " ".join(cmd)
        }

    def run_research_validation_tests(self) -> dict[str, Any]:
        """Run research validation tests."""
        print("\n📊 Running Research Validation Tests")
        print("=" * 50)

        # Run new pytest-based research tests
        pytest_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "research", "--tb=short"]
        pytest_start = time.time()
        pytest_result = subprocess.run(pytest_cmd, capture_output=True, text=True, cwd=self.project_root)
        pytest_time = time.time() - pytest_start

        # Run legacy research tests if they exist
        legacy_time = 0
        legacy_result = None
        if (self.project_root / "run_research_tests.sh").exists():
            print("Running legacy research validation tests...")
            legacy_cmd = ["./run_research_tests.sh", "--suite", "baseline", "--suite", "steganography"]
            legacy_start = time.time()
            legacy_result = subprocess.run(legacy_cmd, capture_output=True, text=True, cwd=self.project_root)
            legacy_time = time.time() - legacy_start

        pytest_success = pytest_result.returncode == 0
        # Legacy Docker-based tests are informational only - they require Docker
        # setup and may not be available in all environments
        legacy_success = legacy_result is None or legacy_result.returncode == 0
        overall_success = pytest_success  # Only pytest determines pass/fail

        return {
            "name": "research_validation",
            "success": overall_success,
            "execution_time": pytest_time + legacy_time,
            "pytest": {
                "success": pytest_success,
                "output": pytest_result.stdout,
                "errors": pytest_result.stderr,
                "time": pytest_time
            },
            "legacy": {
                "success": legacy_success,
                "output": legacy_result.stdout if legacy_result else "",
                "errors": legacy_result.stderr if legacy_result else "",
                "time": legacy_time
            } if legacy_result else None
        }

    def run_all_tests(
        self,
        coverage: bool = True,
        benchmark: bool = False,
        include_slow: bool = False
    ) -> dict[str, Any]:
        """Run all test suites."""
        print("🚀 Running Complete Test Suite")
        print("=" * 50)

        start_time = time.time()

        # Run all test suites
        suites = [
            self.run_unit_tests(coverage=coverage),
            self.run_integration_tests(),
            self.run_security_tests(),
            self.run_performance_tests(benchmark=benchmark),
            self.run_research_validation_tests()
        ]

        total_time = time.time() - start_time

        # Aggregate results
        all_success = all(suite["success"] for suite in suites)

        return {
            "name": "all_tests",
            "success": all_success,
            "execution_time": total_time,
            "suites": suites,
            "summary": {
                "total_suites": len(suites),
                "passed_suites": sum(1 for suite in suites if suite["success"]),
                "failed_suites": sum(1 for suite in suites if not suite["success"])
            }
        }

    def generate_report(self, results: dict[str, Any], output_file: str = None) -> None:
        """Generate test execution report."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"test_report_{timestamp}.json"

        # Save detailed JSON report
        with open(self.project_root / output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary to console
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)

        if "suites" in results:
            # Multiple suites
            for suite in results["suites"]:
                status = "✅ PASS" if suite["success"] else "❌ FAIL"
                print(f"{suite['name']:<25} {status} ({suite['execution_time']:.2f}s)")

            summary = results["summary"]
            print(f"\nTotal: {summary['passed_suites']}/{summary['total_suites']} suites passed")
        else:
            # Single suite
            status = "✅ PASS" if results["success"] else "❌ FAIL"
            print(f"{results['name']:<25} {status} ({results['execution_time']:.2f}s)")

        print(f"Total execution time: {results['execution_time']:.2f}s")
        print(f"Detailed report saved to: {output_file}")

        # Coverage report location
        if (self.project_root / "htmlcov" / "index.html").exists():
            print(f"Coverage report: {self.project_root}/htmlcov/index.html")

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        # Map package name -> importable module name
        required_packages = {
            "pytest": "pytest",
            "pytest-cov": "pytest_cov",
            "pytest-mock": "pytest_mock",
            "pytest-benchmark": "pytest_benchmark",
            "bandit": "bandit",
        }

        missing_packages = []
        for package_name, module_name in required_packages.items():
            try:
                __import__(module_name)
            except ImportError:
                missing_packages.append(package_name)

        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install -r requirements-test.txt")
            return False

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for VectorSmuggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_tests.py --suite unit --coverage
  python run_comprehensive_tests.py --suite integration
  python run_comprehensive_tests.py --suite security
  python run_comprehensive_tests.py --suite performance --benchmark
  python run_comprehensive_tests.py --suite research
  python run_comprehensive_tests.py --suite all
        """
    )

    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "security", "performance", "research", "all"],
        default="unit",
        help="Test suite to run (default: unit)"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report (for unit tests)"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests (for performance suite)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test report"
    )

    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if required dependencies are installed"
    )

    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow-running tests"
    )

    args = parser.parse_args()

    runner = VectorSmuggleTestRunner()

    if args.check_deps:
        deps_ok = runner.check_dependencies()
        if not deps_ok:
            sys.exit(1)
        print("✅ All required dependencies are installed")
        return

    if not runner.check_dependencies():
        sys.exit(1)

    # Run selected test suite
    if args.suite == "unit":
        results = runner.run_unit_tests(coverage=args.coverage)
    elif args.suite == "integration":
        results = runner.run_integration_tests()
    elif args.suite == "security":
        results = runner.run_security_tests()
    elif args.suite == "performance":
        results = runner.run_performance_tests(benchmark=args.benchmark)
    elif args.suite == "research":
        results = runner.run_research_validation_tests()
    elif args.suite == "all":
        results = runner.run_all_tests(
            coverage=args.coverage,
            benchmark=args.benchmark,
            include_slow=args.include_slow
        )

    # Generate report
    runner.generate_report(results, args.output)

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
