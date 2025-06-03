# Configuration Guide

## Overview

VectorSmuggle uses environment variables and configuration files for flexible deployment across different environments.

## Environment Variables

### Core Configuration

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-...                    # Required: OpenAI API key
OPENAI_MODEL=text-embedding-ada-002      # Embedding model to use

# Vector Database Configuration
VECTOR_DB=faiss                          # Options: faiss, qdrant, pinecone
QDRANT_URL=http://localhost:6333         # Qdrant server URL
PINECONE_API_KEY=...                     # Pinecone API key
PINECONE_ENVIRONMENT=us-west1-gcp        # Pinecone environment

# Document Processing
CHUNK_SIZE=512                           # Text chunk size for processing
CHUNK_OVERLAP=50                         # Overlap between chunks
```

### Steganography Settings

```bash
# Enable/disable steganographic techniques
STEGO_ENABLED=true
STEGO_TECHNIQUES=noise,rotation,fragmentation
STEGO_NOISE_LEVEL=0.01
STEGO_ROTATION_ANGLE=15.0
STEGO_FRAGMENTATION_RATIO=0.3

# Multi-model fragmentation
STEGO_FRAGMENT_MODELS=text-embedding-ada-002,text-embedding-3-small
STEGO_FRAGMENT_STRATEGY=round_robin
```

### Evasion Configuration

```bash
# Traffic mimicry
EVASION_TRAFFIC_MIMICRY=true
EVASION_BASE_INTERVAL=300.0
EVASION_JITTER_FACTOR=0.2

# Behavioral camouflage
EVASION_BEHAVIORAL_CAMOUFLAGE=true
EVASION_LEGITIMATE_RATIO=0.8
EVASION_USER_PROFILE=researcher

# Network evasion
EVASION_PROXY_ROTATION=false
EVASION_USER_AGENT_ROTATION=true
EVASION_RATE_LIMITING=true
```

### Query Enhancement

```bash
# Query engine settings
QUERY_CACHE_ENABLED=true
QUERY_CACHE_SIZE=1000
QUERY_MULTI_STEP_REASONING=true
QUERY_CONTEXT_RECONSTRUCTION=true

# Performance optimization
QUERY_BATCH_SIZE=10
QUERY_TIMEOUT=30.0
QUERY_MAX_RETRIES=3
```

### Logging Configuration

```bash
# Logging settings
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json                          # json, text
LOG_FILE=vectorsmuggle.log
LOG_RETENTION_DAYS=30
```

## Configuration Files

### Main Configuration (`config.py`)

The main configuration is handled through the `Config` class which automatically loads environment variables:

```python
from config import Config

config = Config()
print(f"Vector DB: {config.vector_db}")
print(f"Chunk size: {config.chunk_size}")
```

### Environment File (`.env`)

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your specific settings
```

## Advanced Configuration

### Custom Steganography Techniques

```python
from config import Config

config = Config()
config.steganography.techniques = ["noise", "rotation", "custom"]
config.steganography.custom_params = {"param1": "value1"}
```

### Multi-Environment Setup

#### Development Environment

```bash
# .env.development
LOG_LEVEL=DEBUG
STEGO_ENABLED=true
EVASION_TRAFFIC_MIMICRY=false
```

#### Production Environment

```bash
# .env.production
LOG_LEVEL=WARNING
STEGO_ENABLED=true
EVASION_TRAFFIC_MIMICRY=true
EVASION_BEHAVIORAL_CAMOUFLAGE=true
```

### Docker Configuration

Environment variables can be passed through Docker:

```bash
docker run -e OPENAI_API_KEY=sk-... -e VECTOR_DB=qdrant vectorsmuggle
```

### Kubernetes Configuration

Use ConfigMaps and Secrets for Kubernetes deployment:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vectorsmuggle-config
data:
  VECTOR_DB: "qdrant"
  CHUNK_SIZE: "512"
  STEGO_ENABLED: "true"
```

## Validation

Configuration validation is performed at startup:

```python
from config import Config

try:
    config = Config()
    config.validate()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Security Considerations

- Store sensitive values (API keys) in secure secret management systems
- Use environment-specific configuration files
- Validate all configuration values
- Implement proper access controls for configuration files
- Audit configuration changes

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENAI_API_KEY` is set
2. **Invalid Vector DB**: Check `VECTOR_DB` value is supported
3. **Connection Issues**: Verify network connectivity to external services
4. **Permission Errors**: Check file system permissions for log files

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python scripts/embed.py --debug
```

## Examples

### Minimal Configuration

```bash
export OPENAI_API_KEY=sk-...
export VECTOR_DB=faiss
python scripts/embed.py
```

### Full Steganography Setup

```bash
export OPENAI_API_KEY=sk-...
export VECTOR_DB=qdrant
export QDRANT_URL=https://your-qdrant-instance.com
export STEGO_ENABLED=true
export STEGO_TECHNIQUES=noise,rotation,fragmentation
export EVASION_TRAFFIC_MIMICRY=true
python scripts/embed.py --techniques all