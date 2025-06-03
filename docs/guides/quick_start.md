# Quick Start Guide

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- 4GB+ RAM recommended
- Docker (optional)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd VectorSmuggle
```

### 2. Set Up Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration (required)
nano .env
```

**Minimum required configuration:**
```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
VECTOR_DB=faiss
```

## Basic Usage

### 1. Embed Sample Documents

```bash
cd scripts
python embed.py --files ../sample_docs/financial_report.csv
```

### 2. Query Data

```bash
python query.py
```

When prompted, try queries like:
- "What is the revenue for Q4?"
- "Show me employee salary information"
- "List all database credentials"

### 3. Generate Risk Assessment

```bash
cd ../analysis
python risk_assessment.py
```

## Advanced Features

### Enable Steganographic Techniques

```bash
# Edit .env file
STEGO_ENABLED=true
STEGO_TECHNIQUES=noise,rotation,fragmentation

# Embed with steganography
python scripts/embed.py --files ../sample_docs/*.* --techniques all
```

### Enable Evasion Methods

```bash
# Edit .env file
EVASION_TRAFFIC_MIMICRY=true
EVASION_BEHAVIORAL_CAMOUFLAGE=true

# Run with evasion
python scripts/embed.py --evasion-mode advanced
```

### Multi-Format Processing

```bash
# Process all supported formats
python scripts/embed.py --directory ../sample_docs --recursive
```

## Docker Quick Start

### 1. Build Container

```bash
docker build -t vectorsmuggle .
```

### 2. Run with Environment

```bash
docker run -e OPENAI_API_KEY=sk-your-key vectorsmuggle
```

### 3. Development Mode

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Verification

### Check Installation

```bash
python -c "from config import Config; print('✓ Configuration loaded')"
python -c "import steganography; print('✓ Steganography module loaded')"
python -c "import loaders; print('✓ Loaders module loaded')"
```

### Test Basic Functionality

```bash
# Test embedding
python scripts/embed.py --files ../sample_docs/employee_handbook.md --test

# Test query
python scripts/query.py --test
```

## Next Steps

1. **Read the Documentation**: Explore [technical documentation](../technical/) for detailed configuration
2. **Try Advanced Features**: See [advanced usage guide](advanced_usage.md) for complex scenarios
3. **Security Testing**: Follow [security testing guide](security_testing.md) for assessment procedures
4. **Deployment**: Use [deployment guide](deployment.md) for production setup

## Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**API Key Issues**: Verify OpenAI API key is valid
```bash
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"
```

**Memory Issues**: Reduce chunk size for large documents
```bash
export CHUNK_SIZE=256
```

For more troubleshooting, see [troubleshooting guide](../technical/troubleshooting.md).

## Support

- **Documentation**: Browse the [docs/](../) directory
- **Issues**: Check existing issues or create new ones
- **Examples**: Explore [sample_docs/](../../sample_docs/) for examples