# API Reference

## Overview

This document provides comprehensive API reference for VectorSmuggle modules and components.

## Core Modules

### Configuration (`config.py`)

#### Classes

##### `Config`
Main configuration class with environment variable support.

```python
from config import Config

config = Config()
```

##### `SteganographyConfig`
Configuration for steganographic techniques.

##### `EvasionConfig`
Configuration for evasion methods.

##### `QueryConfig`
Configuration for query enhancement features.

### Steganography Module

#### `steganography.obfuscation`

##### `EmbeddingObfuscator`
Applies various obfuscation techniques to embeddings.

```python
from steganography.obfuscation import EmbeddingObfuscator

obfuscator = EmbeddingObfuscator(noise_level=0.01)
obfuscated = obfuscator.obfuscate(embeddings, techniques=["noise", "rotation"])
```

##### `MultiModelFragmenter`
Fragments data across multiple embedding models.

#### `steganography.timing`

##### `TimingController`
Controls timing patterns for covert operations.

### Document Loaders

#### `loaders.document_factory`

##### `DocumentLoaderFactory`
Factory for creating appropriate document loaders.

```python
from loaders.document_factory import DocumentLoaderFactory

factory = DocumentLoaderFactory()
documents = factory.load_documents(file_paths)
```

### Evasion Module

#### `evasion.behavioral_camouflage`

##### `BehavioralCamouflage`
Implements behavioral camouflage techniques.

#### `evasion.traffic_mimicry`

##### `TrafficMimicry`
Mimics legitimate traffic patterns.

### Query Enhancement

#### `query.advanced_engine`

##### `AdvancedQueryEngine`
Enhanced query engine with multiple strategies.

#### `query.context_reconstruction`

##### `ContextReconstructor`
Reconstructs document context from fragments.

### Analysis Tools

#### `analysis.risk_assessment`

##### `VectorExfiltrationRiskAssessor`
Comprehensive risk assessment for vector exfiltration.

#### `analysis.forensic_tools`

##### `EvidenceCollector`
Collects digital evidence from vector stores.

## Error Handling

All modules implement comprehensive error handling with custom exceptions:

- `ConfigurationError`: Configuration-related errors
- `SteganographyError`: Steganography operation errors
- `EvasionError`: Evasion technique errors
- `QueryError`: Query processing errors

## Logging

All modules use structured logging with configurable levels:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Operation completed", extra={"operation": "embed", "status": "success"})
```

## Type Hints

All public APIs include comprehensive type hints for better IDE support and code clarity.

## Examples

See individual module documentation and the main README for usage examples.