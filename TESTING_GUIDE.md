# VectorSmuggle Testing Guide

This guide explains how to use VectorSmuggle's testing framework to ensure code quality, security, and research validity.

## Quick Start

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Run unit tests with coverage
python run_comprehensive_tests.py --suite unit --coverage

# Run all tests
python run_comprehensive_tests.py --suite all

# Verify dependencies are installed
python run_comprehensive_tests.py --check-deps
```

## Test Suites

### 1. Unit Tests (`--suite unit`)

Tests individual functions and classes in isolation.

```bash
# Run with coverage report
python run_comprehensive_tests.py --suite unit --coverage

# Run specific test files
pytest tests/unit/test_config.py -v

# Run with specific markers
pytest tests/unit/ -m "not slow" -v
```

**Coverage:**

- Configuration loading and validation
- Steganography techniques (noise, rotation, scaling, offset, fragmentation, decoy interleaving)
- Document loaders for all 15+ formats
- Embedding generation and fallback (OpenAI plus any locally pulled Ollama embedding model)
- Query algorithms and optimization

### 2. Integration Tests (`--suite integration`)

Tests interactions between VectorSmuggle components.

```bash
# Run integration tests
python run_comprehensive_tests.py --suite integration

# Run with external services
docker-compose up -d qdrant
python run_comprehensive_tests.py --suite integration
```

**Coverage:**

- End-to-end document → embedding → storage pipeline
- Vector store connectivity for the cross-backend study (FAISS-flat / HNSW / IVF-PQ, Chroma, Qdrant float32 / int8) plus legacy Pinecone support in `scripts/embed.py` and `scripts/query.py`
- API integration with external services (OpenAI, Ollama)
- Error propagation and recovery
- Performance under load

### 3. Security Tests (`--suite security`)

Validates security research capabilities and prevents regressions in input handling.

```bash
# Run security test suite
python run_comprehensive_tests.py --suite security

# Individual security tools
bandit -r . -x ./venv,./tests        # Security scanner
safety check                          # Dependency vulnerabilities
pytest tests/ -m security -v          # Security-marked pytest tests
```

**Coverage:**

- Input sanitization and injection prevention
- Malicious document handling
- API key and credential protection
- Steganographic technique validation
- Attack simulation and detection

### 4. Performance Tests (`--suite performance`)

Ensures VectorSmuggle performs well at scale.

```bash
# Run performance tests with benchmarking
python run_comprehensive_tests.py --suite performance --benchmark

# Memory profiling
pytest tests/ -m performance --verbose

# Specific benchmarks
pytest tests/unit/test_steganography_obfuscation.py::test_performance_large_embeddings
```

**Coverage:**

- Document processing throughput
- Embedding generation speed
- Query latency and scalability
- Memory usage patterns
- Steganography overhead

### 5. Research Validation (`--suite research`)

Validates research claims and generates publication data.

```bash
# Run research validation
python run_comprehensive_tests.py --suite research

# Legacy research tests (require Docker)
./run_research_tests.sh --suite baseline --suite steganography
```

**Coverage:**

- Steganographic technique effectiveness against Isolation Forest and One-Class SVM defenders
- Detection resistance validation across the rotation parameter sweep
- Closed-form and empirical payload-capacity for the rotation channel (`scripts/payload_capacity.py`)
- Cross-model generalisation across `text-embedding-3-large` plus four local Ollama embedding models (`scripts/multi_model_study.py`); extensible to any other Ollama embedding model
- Cross-corpus generalisation on BEIR NFCorpus and a Quora subset (`scripts/multi_corpus_study.py`)
- Adaptive white-box detector evasion (`scripts/adaptive_attacker.py`)
- Paraphrased-query retrieval benchmark (`scripts/paraphrased_retrieval.py`)
- Reproducibility under fixed seeds and timestamped result directories

## Advanced Usage

### Running Specific Test Categories

```bash
# Unit tests only (fastest)
pytest tests/unit/ -v

# Integration tests with external services
pytest tests/integration/ -v -m integration

# Security tests only
pytest tests/ -m security -v

# Performance tests only
pytest tests/ -m performance -v

# Research validation tests
pytest tests/ -m research -v

# Slow tests (usually skipped)
pytest tests/ -m slow -v
```

### Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/unit/ --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html

# Coverage with specific thresholds
pytest tests/unit/ --cov=. --cov-fail-under=80

# Coverage for specific modules
pytest tests/unit/ --cov=steganography --cov-report=term-missing
```

### Parallel Test Execution

```bash
# Run tests in parallel
pytest tests/unit/ -n auto

# Control number of workers
pytest tests/unit/ -n 4

# Distribute tests across workers
pytest tests/unit/ --dist=worksteal
```

### Benchmarking

```bash
# Run benchmark tests only
pytest tests/ --benchmark-only

# Save benchmark results
pytest tests/ --benchmark-json=benchmark.json

# Compare benchmark results
pytest-benchmark compare benchmark1.json benchmark2.json
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

Key settings:

- Test discovery patterns
- Coverage reporting
- Parallel execution
- Custom markers
- Timeout settings

### Environment Variables

```bash
# Test configuration
export PYTEST_TIMEOUT=300
export PYTEST_WORKERS=auto

# External service URLs
export QDRANT_URL=http://localhost:6333
export PINECONE_API_KEY=test-key

# Coverage settings
export COVERAGE_FAIL_UNDER=80
```

### Test Markers

Use markers to categorize tests:

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.security       # Security tests
@pytest.mark.performance    # Performance tests
@pytest.mark.slow           # Slow-running tests
@pytest.mark.research       # Research validation
@pytest.mark.docker         # Requires Docker
@pytest.mark.external       # Requires external services
```

## Writing Tests

### Test Structure

```python
# tests/unit/test_my_module.py
import pytest
from my_module import MyClass


class TestMyClass:
    """Test MyClass functionality."""

    @pytest.fixture
    def my_instance(self):
        """Create test instance."""
        return MyClass(param="test")

    @pytest.mark.unit
    def test_basic_functionality(self, my_instance):
        """Test basic functionality."""
        result = my_instance.do_something()
        assert result == "expected"

    @pytest.mark.unit
    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
    ])
    def test_multiple_inputs(self, my_instance, input, expected):
        """Test with multiple inputs."""
        result = my_instance.process(input)
        assert result == expected
```

### Using Fixtures

Common fixtures are available in `tests/conftest.py`:

```python
def test_document_processing(sample_documents, sample_config):
    """Test using common fixtures."""
    # sample_documents: list of test Document objects
    # sample_config: mock configuration object


def test_embeddings(sample_embeddings, assert_helpers):
    """Test using embedding fixtures."""
    # sample_embeddings: NumPy array of test embeddings
    # assert_helpers: common assertion functions
    assert_helpers.assert_embeddings_valid(sample_embeddings)
```

### Mocking External Dependencies

```python
def test_with_mocked_openai(mock_openai_embeddings):
    """Test with mocked OpenAI API."""
    embeddings = create_embeddings()
    result = embeddings.embed_query("test")
    assert result == [0.1] * 1536


def test_with_mocked_vector_store(mock_vector_store):
    """Test with mocked vector store."""
    results = mock_vector_store.similarity_search("test query")
    assert len(results) > 0
```

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline runs automatically on:

- Pull requests to `main` and `develop`
- Pushes to `main`
- Nightly scheduled runs
- Release tags

Workflow stages:

1. **Unit tests** — fast feedback on all PRs
2. **Integration tests** — with external services
3. **Security tests** — vulnerability scanning
4. **Performance tests** — benchmark tracking
5. **Research validation** — research claims validation
6. **Code quality** — linting, formatting, type checking

### Local CI Simulation

```bash
# Simulate CI pipeline locally
python run_comprehensive_tests.py --suite all --coverage --benchmark

# Code quality checks
ruff check .
mypy . --ignore-missing-imports
```

## Performance Monitoring

### Benchmark Tracking

```bash
# Run benchmarks
pytest tests/ --benchmark-only --benchmark-json=benchmark.json

# Compare against baseline
pytest-benchmark compare baseline.json current.json

# Performance regression detection
pytest tests/ --benchmark-compare=baseline.json --benchmark-compare-fail=min:5%
```

### Memory Profiling

```bash
# Profile memory usage of a single test
pytest tests/unit/test_steganography_obfuscation.py::test_memory_efficiency -s -v

# Detailed memory profiling
mprof run pytest tests/unit/test_large_dataset.py
mprof plot
```

## Debugging Tests

### Running Specific Tests

```bash
# Run a single test method
pytest tests/unit/test_config.py::TestConfig::test_config_defaults -v

# Run tests matching a pattern
pytest tests/ -k "test_noise" -v

# Run failed tests only
pytest --lf

# Run failed tests first
pytest --ff
```

### Debug Mode

```bash
# Drop into debugger on failure
pytest tests/unit/test_config.py --pdb

# Capture print output
pytest tests/unit/test_config.py -s

# Verbose output
pytest tests/unit/test_config.py -vv

# Show local variables on failure
pytest tests/unit/test_config.py -l
```

## Best Practices

### Test Design

- One assertion per test where practical
- Descriptive test names that explain what is being tested
- Arrange-Act-Assert pattern
- Independent tests that do not depend on order
- Fixtures for shared test data

### Performance

- Mark slow tests with `@pytest.mark.slow`
- Use mocking for external dependencies
- Run independent tests in parallel
- Profile test performance regularly

### Security

- Test edge cases and malicious inputs
- Validate that security controls behave as expected
- Ensure error handling does not leak sensitive information
- Use synthetic data; never real secrets

### Research Validation

- Statistical significance testing
- Reproducible results with fixed seeds
- Baseline comparisons against established methods
- Document methodology in test comments

## Troubleshooting

### Import Errors

```bash
# Ensure VectorSmuggle is on the Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

### External Service Failures

```bash
# Start required services
docker-compose up -d qdrant

# Check service health
curl http://localhost:6333/health
```

### Permission Errors

```bash
# Make the runner executable
chmod +x run_comprehensive_tests.py

# Install with user permissions
pip install --user -r requirements-test.txt
```

### Memory Issues

```bash
# Reduce parallel workers
pytest tests/unit/ -n 2

# Skip memory-intensive tests
pytest tests/unit/ -m "not memory_intensive"
```

### Getting Help

- Inspect detailed test logs in `test_report_*.json`
- Review the coverage report in `htmlcov/`
- Run individual tests with `-vv` for verbose output
- Use `--pdb` to debug failing tests
- Check CI logs for environment-specific issues
