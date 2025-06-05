# VectorSmuggle Examples

This directory contains practical examples demonstrating VectorSmuggle's capabilities for educational and research purposes.

## Quickstart Demo

The [`quickstart_demo.py`](quickstart_demo.py) script provides a comprehensive demonstration of the complete VectorSmuggle workflow.

### Prerequisites

1. **Python Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **API Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Optional - Ollama Fallback**:
   ```bash
   # Install Ollama for local embedding fallback
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull nomic-embed-text:latest
   ollama serve
   ```

### Quick Start

Run the complete demonstration:

```bash
cd examples
python quickstart_demo.py
```

### Command Line Options

```bash
# Run with deterministic seed for reproducible results
python quickstart_demo.py --seed 42

# Test specific steganographic techniques
python quickstart_demo.py --techniques noise rotation fragmentation

# Run without steganography (baseline comparison)
python quickstart_demo.py --disable-steganography

# Save detailed results to file
python quickstart_demo.py --output results.json
```

## What the Demo Demonstrates

### 1. Environment Setup and Validation
- Configuration validation with cross-dependency checks
- API connectivity testing
- Evasion component initialization
- Deterministic seeding for reproducible results

### 2. Multi-Format Document Loading
- Processes all sample documents from `sample_docs/`
- Supports 15+ document formats (PDF, Office, CSV, JSON, etc.)
- Content chunking and preprocessing
- Format distribution analysis

### 3. Steganographic Embedding Techniques
- **Noise Injection**: Adds statistical noise to embeddings
- **Rotation**: Applies geometric transformations
- **Scaling**: Modifies embedding magnitudes
- **Fragmentation**: Distributes data across multiple models
- **Detection Avoidance**: Content transformation and obfuscation

### 4. Vector Store Operations
- FAISS index creation with steganographic embeddings
- Metadata preservation for reconstruction
- Temporary storage and cleanup

### 5. Query Execution and Data Reconstruction
- Semantic similarity search testing
- Multi-query validation across document types
- Fragment reconstruction verification
- Data integrity validation

### 6. Success Metrics and Verification
- Step completion tracking
- Performance metrics collection
- Integrity verification
- Error analysis and reporting

## Expected Output

### Successful Run
```
Starting VectorSmuggle Quickstart Demo
==================================================
=== Step 1: Environment Setup ===
✓ Configuration validation passed
✓ Embedding API connectivity verified
✓ Behavioral camouflage initialized
✓ Detection avoidance initialized

=== Step 2: Document Loading ===
Found 6 supported documents
✓ Loaded 6 document objects
✓ Created 45 text chunks
Document format distribution: {'csv': 1, 'json': 1, 'yaml': 1, 'md': 1, 'eml': 1, 'html': 1}

=== Step 3: Steganographic Processing ===
Applying techniques: ['noise', 'rotation', 'scaling']
Processing chunk 1/45
Processing chunk 11/45
...
✓ Generated 45 embeddings
✓ Applied obfuscation techniques: ['noise', 'rotation', 'scaling']

=== Step 4: Vector Store Creation ===
✓ Created FAISS vector store with steganographic embeddings
✓ Saved vector store to: temp_quickstart_index
✓ Saved steganography metadata

=== Step 5: Query Testing and Reconstruction ===
✓ Query 'financial data': 3 results
✓ Query 'employee information': 3 results
✓ Query 'API documentation': 3 results
✓ Query 'database schema': 3 results
✓ Query 'budget analysis': 3 results
✓ Semantic search returned 5 results

=== Step 6: Integrity Verification ===
✓ VectorSmuggle quickstart demo completed successfully!
✓ Success rate: 100.0%
✓ Total duration: 12.34 seconds
✓ Cleaned up temporary files

==================================================
QUICKSTART DEMO RESULTS
==================================================
🎉 Demo completed successfully!
Steps completed: 6/6
Success rate: 100.0%
Duration: 12.34 seconds

Key Metrics:
  Documents loaded: 6
  Text chunks: 45
  Embeddings processed: 45
  Vector store size: 45
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   ```
   Error: OPENAI_API_KEY is required
   ```
   **Solution**: Set your OpenAI API key in `.env` file or environment variable.

2. **Missing Dependencies**:
   ```
   ImportError: No module named 'langchain'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

3. **Sample Documents Not Found**:
   ```
   FileNotFoundError: Sample docs directory not found
   ```
   **Solution**: Run from project root or ensure `sample_docs/` directory exists.

4. **Fragmentation Requires Multiple Models**:
   ```
   ValueError: Fragmentation technique requires at least 2 embedding models
   ```
   **Solution**: Configure multiple models in `OPENAI_FALLBACK_MODELS` or disable fragmentation.

5. **Ollama Connection Issues**:
   ```
   Failed to initialize Ollama embeddings: Connection refused
   ```
   **Solution**: Start Ollama service: `ollama serve`

### Debug Mode

For detailed debugging, set environment variables:
```bash
export LOG_LEVEL=DEBUG
export OPENAI_FALLBACK_ENABLED=true
python quickstart_demo.py
```

### Validation Steps

If the demo fails, check these components:

1. **Configuration**: Ensure all required environment variables are set
2. **API Access**: Test OpenAI API connectivity manually
3. **File Permissions**: Verify read access to `sample_docs/` directory
4. **Dependencies**: Confirm all Python packages are installed
5. **System Resources**: Ensure sufficient memory for embedding operations

## Understanding the Results

### Success Metrics
- **Success Rate**: Percentage of steps completed successfully
- **Documents Loaded**: Number of sample documents processed
- **Text Chunks**: Number of text segments created for embedding
- **Embeddings Processed**: Number of vector embeddings generated
- **Vector Store Size**: Number of documents stored in the vector database

### Steganography Effectiveness
The demo validates that:
- Embeddings can be successfully obfuscated using multiple techniques
- Data can be fragmented across different embedding models
- Original information remains retrievable through semantic search
- Integrity checks pass for reconstructed data

### Performance Indicators
- **Duration**: Total execution time (typically 10-30 seconds)
- **Query Success**: All test queries return relevant results
- **Reconstruction**: Fragment reconstruction maintains data integrity
- **No Critical Errors**: All major operations complete successfully

## Next Steps

After running the quickstart demo:

1. **Explore Advanced Features**: Try the full embedding and query scripts in `scripts/`
2. **Custom Documents**: Test with your own document sets
3. **Production Deployment**: Use Docker and Kubernetes configurations
4. **Security Analysis**: Run the risk assessment and forensic tools
5. **Research Applications**: Adapt techniques for your specific use case

## Educational Value

This demo illustrates:
- **Attack Vectors**: How sensitive data can be exfiltrated through embeddings
- **Detection Challenges**: Why traditional DLP tools miss semantic data leaks
- **Steganographic Techniques**: Methods for hiding data in vector spaces
- **Defensive Strategies**: What security teams should monitor and detect

Remember: This tool is for educational and authorized security testing only. Always obtain proper authorization before testing on any systems you don't own.