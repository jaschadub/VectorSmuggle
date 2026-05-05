# VectorSmuggle Testing Framework

Automated testing and validation framework for VectorSmuggle research effectiveness.

## Overview

This framework provides comprehensive automated testing for steganographic techniques, detection systems, and baseline generation capabilities. All tests run in containerized environments for reproducibility.

## Quick Start

```bash
# Run all test suites
./run_research_tests.sh --suite baseline --suite steganography --suite detection

# Run specific test suite
./run_research_tests.sh --suite baseline

# Generate reports only
./run_research_tests.sh --generate-report
```

## Test Suites

- **baseline** - Validates activity generation (93% success rate achieved)
- **steganography** - Tests embedding techniques (obfuscation, fragmentation, decoys)
- **detection** - Evaluates detection system effectiveness
- **performance** - Benchmarks processing speed and resource usage
- **forensics** - Tests forensic analysis capabilities
- **integration** - End-to-end system validation

## Output Structure

```
results/
├── baseline_test_results.json
├── steganography_effectiveness_results.json
├── detection_effectiveness_results.json
├── comprehensive_effectiveness_summary.json
└── reports/
    ├── vectorsmuggle_effectiveness_report.md
    └── VectorSmuggle_Research_Effectiveness_Report.md
```

## Key Features

- **Docker Containerization** - Reproducible test environments
- **Automated Reporting** - Publication-ready effectiveness analysis
- **Multi-dimensional Testing** - Baseline, steganography, detection validation
- **Performance Metrics** - Quantified success rates, MSE, processing times
- **Research Integration** - Designed for academic publication workflows

## Requirements

- Docker and Docker Compose v2
- Python 3.11+ (for local execution)
- 4GB+ available memory for full test suite

## Notes

- Results directory is excluded from git repository
- All test outputs are containerized and isolated
- Framework designed for research validation and publication preparation