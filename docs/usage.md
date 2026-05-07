# Usage

This guide covers the three things you are most likely to do with VectorSmuggle: run the quick-start demo, reproduce the empirical study from the paper, and use the framework as a library inside your own experiments.

## Prerequisites

VectorSmuggle requires Python 3.11 or newer. Install dependencies into a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you intend to use OpenAI embeddings, set `OPENAI_API_KEY` in your environment or `.env` file. For local embeddings, install Ollama and pull at least one embedding model:

```bash
ollama pull nomic-embed-text
```

## Quick-start demo

The fastest way to confirm the install works is the quick-start demo:

```bash
python quickstart_demo.py --seed 42
```

This loads six small documents from `sample_docs/`, embeds them, applies a default mix of steganographic techniques, writes the result to a local FAISS index, and queries it to confirm round-trip recovery. Expected runtime is 10–30 seconds. The script prints a one-page summary that includes how many chunks were processed, which techniques fired, and end-to-end recovery accuracy.

To restrict the techniques applied:

```bash
python quickstart_demo.py --techniques noise rotation fragmentation
```

The available technique names are `noise`, `rotation`, `scaling`, `offset`, `fragmentation`, `decoys`, `timing`. They can be combined arbitrarily.

## Reproducing the paper

The empirical study in the paper is driven by a small set of scripts in `scripts/`. Each script writes its output to a timestamped directory under `results/` and is independent — you can run them in any order, and each one re-embeds the corpus from `sample_docs/`. Re-embedding is a few seconds for local models and a few minutes for OpenAI, so each script is self-contained.

### Core empirical study (Table 4)

```bash
python scripts/empirical_study.py
```

Sweeps four noise levels and seven technique combinations, measuring cosine fidelity, KS statistic, pairwise-similarity correlation, MSE versus original, and entropy. Output: `results/empirical/<timestamp>/{summary.json, per_technique.csv}`.

### Detection AUC and quantization resilience (Tables 5, 7, 8)

```bash
python scripts/preprint_extensions.py
```

Fits an Isolation Forest and a One-Class SVM on a clean-baseline subset of the corpus, then evaluates each technique against both detectors at the held-out half. Also measures bit-error rate under int8 round-trip and retrieval utility under paraphrased queries. Output: `results/preprint/<timestamp>/{summary.json, report.md}`.

### Rotation parameter sweep (Figure 3)

```bash
python scripts/rotation_sweep.py
```

Sweeps the rotation angle from near-zero to π and confirms detection AUC remains at 0.5 across the full range. Output: `results/rotation_sweep/<timestamp>/{summary.json, sweep.csv}`.

### Cross-backend bit-channel survival (Table 6)

```bash
python scripts/cross_backend_study.py
```

Embeds the payload through every backend in `vector_backends/` (FAISS Flat / HNSW / PQ8 / PQ16, Chroma HNSW, Qdrant float32 / int8) and measures how much of the bit channel survives a round trip. Output: `results/cross_backend/<timestamp>/{summary.json, per_backend.csv}`.

### Cross-model study

```bash
python scripts/multi_model_study.py
```

Runs the core fidelity + detection battery against four local Ollama embedding models (`nomic-embed-text`, `embeddinggemma:300m`, `snowflake-arctic-embed:335m`, `mxbai-embed-large:335m`). Confirms the rotation undetectability finding generalizes across embedding architectures. Requires Ollama to be running locally on `localhost:11434`. Output: `results/multi_model/<timestamp>/{summary.json, per_model.csv, report.md}`.

### Adaptive-attacker variant (§5.10)

```bash
python scripts/adaptive_attacker.py
```

White-box adaptive attacker that tunes its perturbation parameters using gradient feedback from the deployed detectors. Drives both detector AUCs to near-zero on noise / scaling / offset, confirming that statistical detection is not a load-bearing control. Output: `results/adaptive/<timestamp>/{summary.json, attack_log.csv}`.

### Paraphrased-query retrieval

```bash
python scripts/paraphrased_retrieval.py
```

Measures top-k retrieval recall when the user's query has been paraphrased away from the original document phrasing. Confirms that obfuscation does not detectably degrade retrieval quality at typical operating points. Output: `results/paraphrased/<timestamp>/{summary.json, recall.csv}`.

### Aggregating results into the paper

```bash
python scripts/empirical_report.py results/empirical/<timestamp>/summary.json
```

Renders a single Markdown report from the JSON outputs of the above scripts. The paper's tables are derived from these reports.

## Using VectorSmuggle as a library

The pipeline modules are importable. A typical custom experiment looks like:

```python
from config import Config
from loaders.document_factory import DocumentLoaderFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from steganography.obfuscation import EmbeddingObfuscator
from analysis.detectors import IsolationForestDetector

config = Config()

factory = DocumentLoaderFactory()
docs = factory.load_documents(["my_corpus/doc1.pdf", "my_corpus/doc2.txt"])
chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_documents(docs)

embedder = OpenAIEmbeddings(model="text-embedding-3-large")
clean = embedder.embed_documents([c.page_content for c in chunks])

obfuscator = EmbeddingObfuscator(seed=42, noise_level=0.01)
obfuscated, rotation_matrix = obfuscator.apply_rotation(clean)

detector = IsolationForestDetector(n_estimators=200, contamination=0.05).fit(clean[:50])
auc = detector.score(obfuscated[50:]).mean()
```

The detectors and obfuscator both consume plain numpy arrays, so you can substitute any embedding source — including the local Ollama models used by `multi_model_study.py`.

## Document corpus

The repository ships with `sample_docs/`: 68 chunks of synthetic PII spread across financial, HR, and technical documents. This is the corpus used throughout the paper. Replacing it with your own corpus is a matter of dropping files into `sample_docs/` (or any directory passed to the loader). Supported formats are listed in `config.py:DocumentConfig.supported_formats`.

For experiments that need a much larger corpus (the paper does not), a separate ingestion driver in `scripts/setup_large_scale_test.sh` will load 100 000 Enron emails as a haystack against which 1 000 sensitive needle documents can be tested. That script is left over from earlier exploratory work and is not part of the paper's reported measurements.

## Troubleshooting

If a script fails before producing output, check `LOG_LEVEL=DEBUG` first. The most common causes are:

- **Missing API key**: scripts that use OpenAI embeddings exit with a clear error if `OPENAI_API_KEY` is not set.
- **Ollama not running**: `multi_model_study.py` will surface a connection error against `localhost:11434`. Run `ollama serve` (or restart the desktop app) and try again.
- **Vector backend connection**: the cross-backend study expects local Qdrant and Chroma instances if those backends are enabled. The repo's `docker-compose.yml` brings them up.

For deeper debugging see [troubleshooting.md](troubleshooting.md).
