# Multi-Database Effectiveness Testing Setup

This document provides setup instructions for the multi-database effectiveness testing system.

## Overview

The multi-database effectiveness testing system allows you to test VectorSmuggle's steganographic techniques across multiple vector databases simultaneously, providing comparative analysis and performance metrics.

## Supported Databases

### Currently Implemented
- **Qdrant**: High-performance vector database with advanced filtering
- **Faiss**: Facebook's similarity search library for dense vectors
- **Pinecone**: Managed vector database service

### Docker Compose Available (Future Implementation)
- **ChromaDB**: Open-source embedding database
- **Weaviate**: Vector search engine with semantic capabilities
- **Milvus**: Cloud-native vector database
- **pgvector**: PostgreSQL extension for vector similarity search

## Quick Start

### 1. Basic Setup

```bash
# Ensure you're in the VectorSmuggle directory
cd /path/to/VectorSmuggle

# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Database Configuration

#### Qdrant (Local)
```bash
# Using Docker
docker run -p 6334:6333 qdrant/qdrant
```

#### Faiss (Local)
No additional setup required - uses local file storage.

#### Pinecone (Cloud)
```bash
# Set environment variables
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="your-environment"
```

### 3. Run Multi-Database Tests

```bash
# Test all available databases
python generate_multi_db_effectiveness_report.py

# Test specific databases
python generate_multi_db_effectiveness_report.py --databases qdrant,faiss

# Custom configuration
python generate_multi_db_effectiveness_report.py --config custom_config.json
```

## Configuration

### Environment Variables

```bash
# Qdrant
export QDRANT_HOST="localhost"
export QDRANT_PORT="6334"

# Pinecone
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="your-environment"

# Faiss
export FAISS_INDEX_PATH="./faiss_indexes"
```

### Configuration File Format

```json
{
  "databases": ["qdrant", "faiss", "pinecone"],
  "test_parameters": {
    "num_vectors": 1000,
    "vector_dimension": 1536,
    "test_iterations": 3
  },
  "output": {
    "format": "json",
    "file": "multi_db_results.json",
    "include_charts": true
  }
}
```

## Architecture

### Plugin System
The system uses a plugin-based architecture where each database has its own adapter:

```
database_adapters/
├── __init__.py           # Package initialization
├── base_adapter.py       # Abstract base class
├── registry.py           # Plugin registry system
├── qdrant_adapter.py     # Qdrant implementation
├── faiss_adapter.py      # Faiss implementation
└── pinecone_adapter.py   # Pinecone implementation
```

### Adding New Database Adapters

1. Create a new adapter file in `database_adapters/`
2. Inherit from `BaseDatabaseAdapter`
3. Implement required methods:
   - `is_available()`
   - `setup_database()`
   - `run_effectiveness_test()`
   - `cleanup()`

Example:
```python
from .base_adapter import BaseDatabaseAdapter

class MyDatabaseAdapter(BaseDatabaseAdapter):
    def __init__(self):
        super().__init__("mydatabase")
    
    def is_available(self) -> bool:
        # Check if database is accessible
        return True
    
    def setup_database(self, config: dict) -> bool:
        # Initialize database connection
        return True
    
    def run_effectiveness_test(self, config: dict) -> dict:
        # Run the effectiveness test
        return {"success": True, "results": {}}
    
    def cleanup(self):
        # Clean up resources
        pass
```

## Output Format

### JSON Report Structure
```json
{
  "summary": {
    "total_databases": 3,
    "successful_tests": 2,
    "failed_tests": 1,
    "execution_time": "45.2s"
  },
  "database_results": {
    "qdrant": {
      "status": "success",
      "effectiveness_score": 0.95,
      "performance_metrics": {
        "insertion_time": "2.3s",
        "query_time": "0.1s",
        "memory_usage": "256MB"
      }
    }
  },
  "comparative_analysis": {
    "best_performance": "qdrant",
    "best_effectiveness": "faiss",
    "recommendations": [
      "Qdrant shows best overall performance",
      "Faiss provides highest steganographic effectiveness"
    ]
  }
}
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check if database service is running
   - Verify connection parameters
   - Check firewall settings

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify virtual environment activation

3. **Permission Errors**
   - Check file system permissions
   - Verify database access credentials
   - Ensure proper environment variables

### Debug Mode

```bash
# Enable verbose logging
python generate_multi_db_effectiveness_report.py --debug

# Test single database
python generate_multi_db_effectiveness_report.py --databases qdrant --debug
```

## Performance Considerations

- **Parallel Execution**: Tests run in parallel where possible
- **Resource Management**: Automatic cleanup of temporary resources
- **Timeout Handling**: 5-minute timeout per database test
- **Memory Optimization**: Streaming results for large datasets

## Security Notes

- Database credentials are handled securely through environment variables
- Temporary files are automatically cleaned up
- No sensitive data is logged in debug mode
- All subprocess calls use secure parameter passing