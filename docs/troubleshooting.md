# Troubleshooting

This document is a short list of the failure modes most commonly hit when reproducing VectorSmuggle's experiments. For each, the symptom is followed by what to check first.

## Installation

**`ImportError` for a langchain or numpy package**: the most common cause is a stale virtual environment. Recreate it:

```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

VectorSmuggle requires Python 3.11 or newer. Older interpreters fail at import time because the codebase uses PEP 604 union syntax (`int | None`).

**A dependency fails to build from source**: verify pip is current (`pip install --upgrade pip`) before retrying.

## Configuration

**`ConfigurationError: OPENAI_API_KEY not found`**: the script you ran requires an OpenAI API key. Either export it (`export OPENAI_API_KEY=sk-...`), add it to `.env`, or run a script that uses local Ollama embeddings instead (see `scripts/multi_model_study.py`).

**`ConnectionError` against the vector backend**: confirm the backend is reachable.

```bash
# Qdrant
curl -s http://localhost:6333/collections

# Ollama
curl -s http://localhost:11434/api/tags
```

The repository's `docker-compose.yml` brings up local Qdrant and Chroma instances if you do not already have them running.

## Embedding

**Out-of-memory while embedding**: lower `CHUNK_SIZE` (try `256`) so the per-chunk model context is smaller. The default of `512` fits comfortably on a laptop with 8 GB RAM.

**OpenAI rate limits**: scripts that re-embed the corpus run sequentially and produce only 68 requests, so this is rare. If it does happen, the OpenAI client retries automatically; the run will be slower but will complete.

**Ollama connection refused on `localhost:11434`**: start the daemon with `ollama serve` (or restart the desktop app). The multi-model script verifies connectivity at startup and exits with a clear message if Ollama is unreachable.

## Steganography

**Recovery cosine far below 1.0 for `rotation` or `scaling`**: rotation and scaling are mathematically lossless, so a recovery cosine below ≈0.9999 means the obfuscator's metadata (rotation matrix, scaling factor) was not threaded through to the deobfuscator. Inspect the dict returned by `obfuscate(...)` — it must be passed unchanged to `deobfuscate(...)`. The empirical scripts do this correctly; if you wrote your own driver, this is the first thing to check.

**Detector AUC near 0.5 for noise**: this is expected behavior at low noise levels (`STEGO_NOISE_LEVEL <= 0.005`). Raise the noise level if you want a detectable signal, or accept that small perturbations are below the detector's noise floor — that is one of the paper's findings.

## Vector backends

**Slow round-trip in the cross-backend study**: the int8 Qdrant variant is the slowest because of the per-vector quantization step. Expected runtime is a few minutes for 68 chunks across all seven backend configurations on a laptop.

**FAISS PQ recovery near zero**: PQ16 recovers more bits than PQ8 because it has more sub-quantizers per vector. Both lose information by design — that loss is what §5.6 of the paper measures. If you need lossless storage, use FAISS Flat or HNSW.

## Detection and reproducibility

**AUC values shift between runs**: the empirical scripts seed every RNG, but the OpenAI embedding API is not deterministic — running `empirical_study.py` twice can produce slightly different embeddings, which propagates into slightly different AUCs. The paper's reported numbers are within the run-to-run noise floor (typically ±0.01 AUC). For perfect reproducibility, embed once and cache the array (`np.save embeddings.npy`), then run the detection battery against the cached array.

**Numbers do not match the paper exactly**: the paper reports `text-embedding-3-large` numbers; the `.env.example` ships with `text-embedding-3-small` for cost reasons. Set `OPENAI_EMBEDDING_MODEL=text-embedding-3-large` to match the paper.

## Debugging

Enable debug logging to trace per-chunk processing:

```bash
LOG_LEVEL=DEBUG python scripts/empirical_study.py
```

For a snapshot of the running environment when filing a bug report:

```bash
python -c "import sys, platform; print(sys.version); print(platform.platform())"
pip list | grep -E "(langchain|openai|numpy|sklearn|qdrant|chromadb|faiss|ollama)"
```

Include both alongside the failing script's last 50 lines of output.
