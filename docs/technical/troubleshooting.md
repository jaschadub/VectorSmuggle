# Troubleshooting Guide

## Common Issues

### Installation and Setup

#### Python Version Issues

**Problem**: `ImportError` or compatibility issues
**Solution**: Ensure Python 3.11+ is installed
```bash
python --version  # Should be 3.11 or higher
pip install --upgrade pip
```

#### Dependency Installation Failures

**Problem**: Package installation fails
**Solution**: 
```bash
# Clear pip cache
pip cache purge

# Install with verbose output
pip install -r requirements.txt -v

# For specific package issues
pip install --no-cache-dir package_name
```

#### Virtual Environment Issues

**Problem**: Packages not found or wrong versions
**Solution**:
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration Issues

#### Missing API Keys

**Problem**: `ConfigurationError: OPENAI_API_KEY not found`
**Solution**:
```bash
# Set environment variable
export OPENAI_API_KEY=sk-your-key-here

# Or add to .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

#### Invalid Vector Database Configuration

**Problem**: `ConnectionError` or `VectorStoreError`
**Solution**:
```bash
# For FAISS (local)
export VECTOR_DB=faiss

# For Qdrant
export VECTOR_DB=qdrant
export QDRANT_URL=http://localhost:6334

# For Pinecone
export VECTOR_DB=pinecone
export PINECONE_API_KEY=your-key
export PINECONE_ENVIRONMENT=us-west1-gcp
```

### Runtime Errors

#### Memory Issues

**Problem**: `MemoryError` or system slowdown
**Solution**:
```bash
# Reduce chunk size
export CHUNK_SIZE=256

# Reduce batch size
export QUERY_BATCH_SIZE=5

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

#### Network Connectivity Issues

**Problem**: `ConnectionTimeout` or `RequestException`
**Solution**:
```bash
# Test connectivity
curl -I https://api.openai.com/v1/models

# Check proxy settings
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port

# Increase timeout
export QUERY_TIMEOUT=60.0
```

#### File Permission Issues

**Problem**: `PermissionError` when writing files
**Solution**:
```bash
# Check permissions
ls -la logs/
ls -la faiss_index/

# Fix permissions
chmod 755 logs/
chmod 644 logs/*.log
```

### Steganography Issues

#### Embedding Obfuscation Failures

**Problem**: `SteganographyError` during obfuscation
**Solution**:
```bash
# Reduce noise level
export STEGO_NOISE_LEVEL=0.005

# Disable problematic techniques
export STEGO_TECHNIQUES=noise,rotation

# Check embedding dimensions
python -c "from config import Config; print(Config().embedding_dimension)"
```

#### Multi-Model Fragmentation Issues

**Problem**: Model compatibility errors
**Solution**:
```bash
# Use compatible models only
export STEGO_FRAGMENT_MODELS=text-embedding-ada-002

# Check model availability
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"
```

### Query Issues

#### Poor Query Results

**Problem**: Irrelevant or no results returned
**Solution**:
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Try different retrieval strategies
python scripts/query.py --strategy semantic

# Increase result count
python scripts/query.py --top-k 20
```

#### Query Performance Issues

**Problem**: Slow query responses
**Solution**:
```bash
# Enable caching
export QUERY_CACHE_ENABLED=true

# Reduce context reconstruction
export QUERY_CONTEXT_RECONSTRUCTION=false

# Use batch processing
python scripts/query.py --batch-size 5
```

### Docker Issues

#### Container Build Failures

**Problem**: Docker build fails
**Solution**:
```bash
# Clean build cache
docker system prune -f

# Build with no cache
docker build --no-cache -t vectorsmuggle .

# Check build logs
docker build -t vectorsmuggle . 2>&1 | tee build.log
```

#### Container Runtime Issues

**Problem**: Container exits or crashes
**Solution**:
```bash
# Check container logs
docker logs vectorsmuggle

# Run with debug
docker run -it --entrypoint /bin/bash vectorsmuggle

# Check resource limits
docker stats vectorsmuggle
```

### Kubernetes Issues

#### Pod Startup Failures

**Problem**: Pods in `CrashLoopBackOff` or `Error` state
**Solution**:
```bash
# Check pod logs
kubectl logs -f deployment/vectorsmuggle

# Describe pod for events
kubectl describe pod vectorsmuggle-xxx

# Check resource constraints
kubectl top pods
```

#### Service Connectivity Issues

**Problem**: Services not accessible
**Solution**:
```bash
# Check service endpoints
kubectl get endpoints

# Test service connectivity
kubectl exec -it pod-name -- curl http://service-name:port

# Check network policies
kubectl get networkpolicies
```

## Debugging Techniques

### Enable Debug Logging

```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=text
python scripts/embed.py --debug
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile embedding operation
cProfile.run('embed_documents()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### Network Debugging

```bash
# Monitor network traffic
sudo tcpdump -i any host api.openai.com

# Check DNS resolution
nslookup api.openai.com

# Test SSL/TLS
openssl s_client -connect api.openai.com:443
```

## Error Codes

### Configuration Errors (1000-1099)

- `1001`: Missing required environment variable
- `1002`: Invalid configuration value
- `1003`: Configuration validation failed

### Steganography Errors (2000-2099)

- `2001`: Embedding obfuscation failed
- `2002`: Fragmentation error
- `2003`: Timing control error

### Evasion Errors (3000-3099)

- `3001`: Traffic mimicry failed
- `3002`: Behavioral camouflage error
- `3003`: Network evasion error

### Query Errors (4000-4099)

- `4001`: Query processing failed
- `4002`: Context reconstruction error
- `4003`: Cross-reference analysis failed

## Getting Help

### Log Analysis

Always include relevant log entries when seeking help:

```bash
# Get recent logs
tail -n 100 logs/vectorsmuggle.log

# Filter for errors
grep ERROR logs/vectorsmuggle.log

# Export logs for analysis
python -c "
import json
from pathlib import Path
logs = Path('logs/vectorsmuggle.log').read_text()
print(json.dumps({'logs': logs.split('\n')[-100:]}, indent=2))
" > debug_logs.json
```

### System Information

Collect system information for bug reports:

```bash
# System info
python -c "
import sys, platform, psutil
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'CPU: {psutil.cpu_count()} cores')
print(f'Memory: {psutil.virtual_memory().total // 1024**3} GB')
"

# Package versions
pip list | grep -E "(langchain|openai|numpy|torch)"
```

### Performance Metrics

```bash
# Generate performance report
python analysis/performance_report.py --output performance.json

# Check resource usage
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"
```

## Prevention

### Best Practices

1. **Always use virtual environments**
2. **Pin dependency versions in requirements.txt**
3. **Validate configuration before deployment**
4. **Monitor resource usage**
5. **Implement proper error handling**
6. **Use structured logging**
7. **Regular health checks**

### Monitoring Setup

```bash
# Set up log rotation
echo "logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}" > /etc/logrotate.d/vectorsmuggle

# Monitor disk space
df -h
du -sh logs/