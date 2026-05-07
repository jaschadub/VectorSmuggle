![Vector Smuggle](logo-vs2.png "Vector Smuggle")

# VectorSmuggle

A research framework demonstrating vector-based data exfiltration techniques in AI/ML environments. This project examines covert data leakage in retrieval-augmented generation (RAG) systems and provides tools for both offensive analysis and defensive countermeasure development.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Security Research](https://img.shields.io/badge/purpose-security%20research-red.svg)](https://github.com/jaschadub/VectorSmuggle)
[![Educational](https://img.shields.io/badge/use-educational%20only-green.svg)](https://github.com/jaschadub/VectorSmuggle)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20058256.svg)](https://doi.org/10.5281/zenodo.20058256)

> **Companion preprint:** Wanger, J. (2026). *VectorSmuggle: Steganographic Exfiltration in Embedding Stores and a Cryptographic Provenance Defense*. Zenodo. <https://doi.org/10.5281/zenodo.20058256>
>
> **Constructive defense:** [VectorPin](https://github.com/ThirdKeyAI/VectorPin) — cryptographic provenance for embedding stores, locked to the threat model evaluated here.

## Overview

VectorSmuggle investigates how sensitive information can be hidden within seemingly legitimate vector embeddings, surfacing risks that traditional Data Loss Prevention systems are not designed to detect. The framework supports security researchers, red and blue team exercises, and academic study of attack and defense in AI/ML pipelines.

### Key Capabilities

- **Steganographic techniques**: Embedding obfuscation through noise injection, rotation, scaling, offset, fragmentation, and decoy interleaving
- **Multi-format ingestion**: Support for 15+ document formats (PDF, Office, email, structured data, databases)
- **Multi-backend evaluation**: FAISS Flat / HNSW / PQ, Chroma HNSW, Qdrant float32 and int8 — used for the cross-backend bit-channel survival study
- **Multi-model evaluation**: OpenAI plus four local Ollama models (`nomic-embed-text`, `embeddinggemma`, `snowflake-arctic-embed`, `mxbai-embed-large`)
- **Detection battery**: Isolation Forest and One-Class SVM detectors used as defenders inside every empirical script
- **Reproducibility**: deterministic per-run seeding and timestamped result directories under `results/`

## Architecture

```mermaid
graph LR
    A[Documents] --> B[Loaders]
    B --> C[Preprocessor]
    C --> D[Embedder]
    D --> E[Obfuscator]
    E --> F[Vector Backend]
    F --> G[Query Engine]
    G --> H[Recovery / Analysis]

    subgraph defenders [Defenders]
        I[Isolation Forest]
        J[One-Class SVM]
    end
    D -.->|clean baseline| I
    D -.->|clean baseline| J
    E -.->|obfuscated| I
    E -.->|obfuscated| J
```

See [`docs/architecture.md`](docs/architecture.md) for the module-by-module description.

## Quick Start

### Prerequisites

- Python 3.11 or later
- OpenAI API key (for the paper's headline numbers) or Ollama with at least one local embedding model
- Docker (optional, for the containerized run path)

### Installation

```bash
git clone https://github.com/jaschadub/VectorSmuggle.git
cd VectorSmuggle

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env to provide API keys and runtime settings
```

### Basic Usage

```bash
# Embed documents using steganographic techniques
python scripts/embed.py --files sample_docs/*.pdf --techniques noise,rotation,fragmentation

# Query and reconstruct embedded data
python scripts/query.py --mode recovery --export results.json
```

### Interactive Demonstration

The `examples/quickstart_demo.py` script walks through the complete workflow end to end:

```bash
cd examples
python quickstart_demo.py

# Reproducible run
python quickstart_demo.py --seed 42

# Specific techniques only
python quickstart_demo.py --techniques noise rotation fragmentation
```

The demo covers:

- End-to-end workflow: document loading, steganographic embedding, vector storage, and query reconstruction
- Multiple techniques: noise injection, rotation, scaling, and cross-model fragmentation
- Real sample data from `sample_docs/` (financial, HR, technical files)
- Integrity verification of encoding and decoding
- Performance metrics: processing time, success rate, and data statistics

Expected runtime: 10–30 seconds. Sample output: 6 documents to 45 chunks to 45 steganographic embeddings.

See [`examples/README.md`](examples/README.md) for detailed setup instructions, troubleshooting, and expected outputs.

## Documentation

The research narrative — threat model, technique catalog, empirical results, and the VectorPin defense — lives in the preprint at <https://doi.org/10.5281/zenodo.20058256>. The repository documentation covers how to run, configure, and extend the framework.

- [Architecture](docs/architecture.md) — module layout and pipeline
- [Configuration](docs/configuration.md) — environment variables and runtime settings
- [Usage](docs/usage.md) — quick-start, paper reproduction, library use
- [API reference](docs/api_reference.md) — public module APIs
- [Deployment](docs/deployment.md) — running in Docker and bringing up local backends
- [Troubleshooting](docs/troubleshooting.md) — common failure modes

For testing, see [TEST_PLAN.md](TEST_PLAN.md) and [TESTING_GUIDE.md](TESTING_GUIDE.md).

## Library use

The pipeline modules are importable. A typical custom experiment looks like:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loaders.document_factory import DocumentLoaderFactory
from steganography.obfuscation import EmbeddingObfuscator
from analysis.detectors import IsolationForestDetector
from analysis.detectors.isolation_forest_detector import evaluate

docs = DocumentLoaderFactory().load_documents(["sample_docs/financial_report_q3_2024.md"])
chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_documents(docs)

embedder = OpenAIEmbeddings(model="text-embedding-3-large")
clean = embedder.embed_documents([c.page_content for c in chunks])

obfuscator = EmbeddingObfuscator(seed=42, noise_level=0.01)
rotated, rotation_matrix = obfuscator.apply_rotation(clean)

detector = IsolationForestDetector(n_estimators=200, contamination=0.05).fit(clean[:34])
metrics = evaluate(detector, clean[34:], rotated[34:])
print(metrics["auc"])  # ~0.5 — rotation is not detected by distribution-based methods
```

See [`docs/api_reference.md`](docs/api_reference.md) for the public module APIs.

## Deployment

VectorSmuggle is a research framework, so "deployment" usually means running the experiments inside a container for reproducibility:

```bash
docker build -t vectorsmuggle:latest .
docker run --rm -e OPENAI_API_KEY=sk-... -v "$PWD/results:/app/results" \
  vectorsmuggle:latest python scripts/empirical_study.py
```

The cross-backend study expects local Qdrant and Chroma; bring them up with:

```bash
cd test_vector_dbs_docker && docker compose up -d
```

See [`docs/deployment.md`](docs/deployment.md) for the full workflow.

## Configuration

The minimum settings to run the empirical study against OpenAI:

```bash
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
VECTOR_DB=faiss
CHUNK_SIZE=512
```

For local embedding via Ollama, install a model and run the multi-model study:

```bash
ollama pull nomic-embed-text
python scripts/multi_model_study.py
```

The full set of environment variables — including the steganography intensity parameters used to reproduce specific paper measurements — is documented in [`docs/configuration.md`](docs/configuration.md).

## Testing and Code Quality

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run the full pytest suite
python -m pytest tests/

# Run a specific suite via the harness
python run_comprehensive_tests.py --suite unit --coverage
python run_comprehensive_tests.py --suite integration
python run_comprehensive_tests.py --suite security
python run_comprehensive_tests.py --suite research

# Linting and security checks
ruff check .
bandit -r . -x ./venv,./tests
```

## Research

The empirical study, threat model, technique catalog, and the constructive defense (VectorPin) are described in the companion preprint at <https://doi.org/10.5281/zenodo.20058256>. The numbers in the paper are produced by the scripts in `scripts/` against the corpus in `sample_docs/`; see [`docs/usage.md`](docs/usage.md) for the reproduction workflow.

Headline findings:

- Distribution-shifting perturbations (noise, scaling, offset, combinations) are detectable in our corpus by an off-the-shelf Isolation Forest or One-Class SVM trained on a clean baseline.
- Orthogonal rotation defeats distribution-based detection at every parameter setting we swept, and the finding generalizes across five embedding models (OpenAI `text-embedding-3-large` plus four local Ollama models).
- The bit channel survives normal vector-database deployments: cross-backend round-trips, int8 scalar quantization, and FAISS PQ all leave enough signal for end-to-end recovery.
- White-box adaptive attackers drive both detector AUCs to near-zero, confirming that statistical detection is not a load-bearing control.

The constructive defense, VectorPin, signs each embedding to its source content and producing model with Ed25519 over a canonical byte representation. Any post-embedding modification — including all studied techniques — breaks signature verification. See <https://github.com/ThirdKeyAI/VectorPin>.

## Educational use

VectorSmuggle is intended for use in red-team exercises, blue-team detector development, and academic security research. Sample applications include studying how steganographic perturbations interact with deployed RAG defenses, evaluating new detector designs against the technique catalog, and probing how vector backends transform embeddings under quantization.

## Contributing

Contributions are welcome. Open a pull request with a clear description of the change, ensure `ruff check .` and the test suite pass, and update documentation for user-visible behavior changes. For larger changes, open an issue first so we can discuss the approach.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Legal Disclaimer

This repository and its contents are intended for educational and ethical security research only.

- Any actions or activities related to this material are solely your responsibility
- Misuse of these tools or techniques to access unauthorized data is illegal and unethical
- The authors assume no liability for any misuse or damage caused by this material
- Always obtain proper authorization before performing any security testing

## Contact

For questions, suggestions, or responsible disclosure of security issues:

- General questions: open an issue on GitHub
- Research collaboration: contact the maintainer

## Citation

If you reference VectorSmuggle in your research, please cite the companion
preprint (which describes the threat model, empirical results, and the
VectorPin defense) and the software framework itself.

### Preprint (preferred)

> Wanger, J. (2026). *VectorSmuggle: Steganographic Exfiltration in Embedding Stores and a Cryptographic Provenance Defense*. Zenodo. <https://doi.org/10.5281/zenodo.20058256>

```bibtex
@misc{wanger2026vectorsmuggle,
  title  = {{VectorSmuggle}: Steganographic Exfiltration in Embedding Stores and a Cryptographic Provenance Defense},
  author = {Wanger, Jascha},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.20058256},
  url    = {https://doi.org/10.5281/zenodo.20058256}
}
```

### Software framework

```bibtex
@software{vectorsmuggle-framework,
  title  = {{VectorSmuggle}: A research framework for vector-based data exfiltration},
  author = {Wanger, Jascha},
  organization = {Tarnover, LLC},
  year   = {2025},
  url    = {https://github.com/jaschadub/VectorSmuggle},
  note   = {Apache-2.0; companion to \href{https://doi.org/10.5281/zenodo.20058256}{the Zenodo preprint}}
}
```

### APA

```
Wanger, J. (2026). VectorSmuggle: Steganographic exfiltration in embedding stores and a cryptographic provenance defense. Zenodo. https://doi.org/10.5281/zenodo.20058256
```

### IEEE

```
J. Wanger, "VectorSmuggle: Steganographic exfiltration in embedding stores and a cryptographic provenance defense," Zenodo, 2026. [Online]. Available: https://doi.org/10.5281/zenodo.20058256
```

---

This tool is intended to improve security through education and research. Use responsibly and ethically.

*Portions of this code are generated, tested, and audited using advanced AI models. [ThirdKey](https://thirdkey.ai) can help secure your AI infrastructure.*
